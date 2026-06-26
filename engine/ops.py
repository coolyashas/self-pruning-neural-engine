"""Elementwise binary ops: add, sub, mul, div.

Each op does three things when it runs:
  1. Forward: compute `out.data` from the inputs' `.data` (NumPy broadcasting
     applies exactly as it would for plain ndarrays).
  2. Record the graph edge: `out._prev = {a, b}`, `out._op = "..."`.
  3. Attach `out._backward`: a closure that, given `out.grad` (already
     populated by whoever called backward() on a downstream node), computes
     each input's *local* gradient and accumulates it into that input's
     `.grad`.

None of this actually runs end-to-end yet -- the topological-sort traversal
that seeds `out.grad` and calls these closures in the right order is
commit 5. What's testable now: build `a op b`, manually set `out.grad`,
call `out._backward()` once, and check `a.grad` / `b.grad` match the
hand-derived formula. See the manual checks run against this file before
committing.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def _as_tensor(x) -> Tensor:
    """Wrap a plain Python/NumPy value as a (non-grad-requiring) Tensor.

    Lets code write `t + 1.0` or `2.0 * t` without every caller having to
    box constants themselves.
    """
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce `grad` (shaped like the broadcasted op output) back down to
    `shape` (the shape of the original, pre-broadcast input).

    NumPy broadcasting only ever does two things to turn `shape` into the
    output shape: (a) it prepends new leading axes, and (b) it stretches
    existing axes that were size 1. The backward direction of "stretch" is
    "sum" -- every element that was copied along a broadcast axis during
    the forward pass must have its incoming gradients summed back together,
    otherwise dL/dinput silently double-counts contributions that came from
    a single original element.

    This is the #1 place naive autodiff implementations get backward wrong
    for ops with broadcasting (e.g. `(N, D) + (D,)` bias add): forgetting
    this step gives a gradient with the *output's* shape, not the input's,
    which then errors out or silently mis-accumulates a few ops later.
    """
    # (a) Sum away any extra leading axes broadcasting added.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # (b) For axes that were size-1 in the original shape but got
    # stretched in the output, sum back over that axis (keepdims so the
    # axis count still lines up with `shape` for the next iteration / the
    # final shape check).
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    assert grad.shape == shape, (
        f"unbroadcast failed: got {grad.shape}, expected {shape}"
    )
    return grad


def add(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="add",
    )

    def _backward():
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            b.accumulate_grad(_unbroadcast(out.grad, b.shape))

    out._backward = _backward
    return out


def sub(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(
        a.data - b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="sub",
    )

    def _backward():
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            # d(a - b)/db = -1, so upstream grad flows to b negated.
            b.accumulate_grad(_unbroadcast(-out.grad, b.shape))

    out._backward = _backward
    return out


def mul(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="mul",
    )

    def _backward():
        # Product rule: d(a*b)/da = b, d(a*b)/db = a. Multiplying by
        # out.grad (broadcasted ndarray-style, by NumPy) before
        # unbroadcasting gives the correct local-times-upstream chain rule
        # term in the *output's* shape; unbroadcast then folds it down to
        # each input's own shape.
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad * b.data, a.shape))
        if b.requires_grad:
            b.accumulate_grad(_unbroadcast(out.grad * a.data, b.shape))

    out._backward = _backward
    return out


def div(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(
        a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="div",
    )

    def _backward():
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2.
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad / b.data, a.shape))
        if b.requires_grad:
            b.accumulate_grad(
                _unbroadcast(-out.grad * a.data / (b.data**2), b.shape)
            )

    out._backward = _backward
    return out


# --- Wire these onto Tensor as operators -----------------------------------
# Done here (not in tensor.py) to keep the core data structure (commit 2)
# free of any actual math, per the commit-by-commit split. `engine/__init__`
# imports this module so the operators are always available wherever
# `Tensor` is imported.
Tensor.__add__ = lambda self, other: add(self, other)
Tensor.__radd__ = lambda self, other: add(other, self)
Tensor.__sub__ = lambda self, other: sub(self, other)
Tensor.__rsub__ = lambda self, other: sub(other, self)
Tensor.__mul__ = lambda self, other: mul(self, other)
Tensor.__rmul__ = lambda self, other: mul(other, self)
Tensor.__truediv__ = lambda self, other: div(self, other)
Tensor.__rtruediv__ = lambda self, other: div(other, self)
