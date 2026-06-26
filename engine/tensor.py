"""Core Tensor type for the hand-written autodiff engine.

This module defines only the data structure: an ndarray wrapper that also
tracks (a) which other Tensors produced it ("parents"), (b) a per-node
backward closure that will know how to push gradient to those parents, and
(c) accumulated gradient storage. Actual math (add, mul, matmul, ...) is
added in later commits as methods/functions that *populate* `_backward` and
`_prev` on the Tensor they return -- this file has no arithmetic yet.
"""

from __future__ import annotations

import numpy as np


class Tensor:
    """An ndarray plus the bookkeeping needed for reverse-mode autodiff.

    Each Tensor is a node in a computation graph. Leaf tensors (model
    parameters, input data) are created directly by user code. Non-leaf
    tensors are created by ops (commit 3+), which record:
      - `_prev`: the parent Tensors this one was computed from.
      - `_backward`: a zero-argument closure that, given `self.grad`
        already populated, accumulates the appropriate gradient into each
        parent's `.grad`. Defaults to a no-op until an op sets it.
    `backward()` itself (topological sort + the traversal that calls each
    node's `_backward` in the right order) is commit 5 -- not here.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        # float64 (not float32): finite-difference gradient checking
        # (commit 6) needs enough precision that rounding error doesn't
        # swamp the O(h) approximation error -- float32 makes that check
        # unreliable.
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)

        # requires_grad lets later code (e.g. the mask in commit 16, or
        # plain input data) opt out of gradient bookkeeping. Ops will set
        # this to True on their output whenever *any* input requires it,
        # mirroring how every other autodiff system propagates the flag.
        self.requires_grad = requires_grad

        # Gradient accumulator. Left as None until backward() actually
        # needs to write into it (commit 5) -- most tensors created during
        # a forward pass never have backward() called on the graph they
        # belong to (e.g. during inference), so eagerly allocating a
        # same-shape zeros array for every single one would be wasted work.
        self.grad: np.ndarray | None = None

        # _prev is a set, not a list/tuple: the same parent Tensor can
        # appear more than once in one op's inputs (e.g. x + x, or a
        # weight reused at two points in the graph). The topological sort
        # in backward() needs each *node* visited once regardless of how
        # many edges point to it; a set gives that for free and avoids
        # double-counting a parent's contribution from its own perspective
        # when we later walk "which nodes consume me".
        self._prev: set["Tensor"] = set(_children)

        # Filled in by whichever op produced this tensor; no-op for leaves.
        self._backward = lambda: None

        # Purely descriptive (debugging/printing), e.g. "add", "matmul".
        self._op = _op

    def accumulate_grad(self, grad: np.ndarray) -> None:
        """Add `grad` into `self.grad`, never overwrite it.

        A Tensor can be a parent of more than one downstream op (e.g.
        `b = a + a`, or a weight reused at two points in the graph), so
        each contribution must be summed, not replace the previous one --
        overwriting would silently drop every contribution but the last.
        Lazily allocates `self.grad` as zeros on the first call instead of
        upfront in __init__, since most tensors created during a forward
        pass (e.g. plain input data) never have backward() reach them.
        """
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, "
            f"op={self._op!r})"
        )
