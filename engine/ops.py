"""Elementwise binary ops: add, sub, mul, div."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def _as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Undo NumPy broadcasting: sum grad back down to the input's shape.

    Broadcasting either adds leading axes or stretches size-1 axes, so the
    backward of "stretch" is "sum" — each broadcast-copied element's
    gradients need to land back on the one original element they came from.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    assert grad.shape == shape, f"unbroadcast failed: got {grad.shape}, expected {shape}"
    return grad


def add(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data + b.data, a.requires_grad or b.requires_grad, (a, b), "add")

    def _backward():
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            b.accumulate_grad(_unbroadcast(out.grad, b.shape))

    out._backward = _backward
    return out


def sub(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data - b.data, a.requires_grad or b.requires_grad, (a, b), "sub")

    def _backward():
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            b.accumulate_grad(_unbroadcast(-out.grad, b.shape))

    out._backward = _backward
    return out


def mul(a, b) -> Tensor:
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data * b.data, a.requires_grad or b.requires_grad, (a, b), "mul")

    def _backward():
        # product rule: d(ab)/da = b, d(ab)/db = a
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad * b.data, a.shape))
        if b.requires_grad:
            b.accumulate_grad(_unbroadcast(out.grad * a.data, b.shape))

    out._backward = _backward
    return out


def div(a, b) -> Tensor:
    """a / b, elementwise, with broadcasting.

    Numerical stability: unlike mul/add/sub, this has no protection near
    b == 0 -- d(a/b)/da = 1/b and d(a/b)/db = -a/b^2 both blow up to
    inf/nan there, and forward division by exact zero does too. No
    epsilon is added here deliberately: this op isn't on this project's
    actual forward path (Linear/softmax_cross_entropy never call it;
    softmax's own division is computed directly in NumPy, not through
    this graph node), so there's no real input distribution to tune an
    epsilon against, and clamping would silently change the result for
    any caller that DOES pass values near zero on purpose. If this op
    is ever used somewhere a small denominator is plausible, that call
    site -- not this generic op -- is where a guard belongs.
    """
    a, b = _as_tensor(a), _as_tensor(b)
    out = Tensor(a.data / b.data, a.requires_grad or b.requires_grad, (a, b), "div")

    def _backward():
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        if a.requires_grad:
            a.accumulate_grad(_unbroadcast(out.grad / b.data, a.shape))
        if b.requires_grad:
            b.accumulate_grad(_unbroadcast(-out.grad * a.data / (b.data**2), b.shape))

    out._backward = _backward
    return out


# Wired here, not in tensor.py, to keep the core node type math-free.
Tensor.__add__ = lambda self, other: add(self, other)
Tensor.__radd__ = lambda self, other: add(other, self)
Tensor.__sub__ = lambda self, other: sub(self, other)
Tensor.__rsub__ = lambda self, other: sub(other, self)
Tensor.__mul__ = lambda self, other: mul(self, other)
Tensor.__rmul__ = lambda self, other: mul(other, self)
Tensor.__truediv__ = lambda self, other: div(self, other)
Tensor.__rtruediv__ = lambda self, other: div(other, self)
