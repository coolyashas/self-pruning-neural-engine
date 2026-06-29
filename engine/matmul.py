"""Matmul: 2D matrices, or batched stacks of them (3D+, NumPy-style batch
broadcasting on every axis before the last two). 1D operands (vector-matrix/
matrix-vector) are out of scope and rejected.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def _unbroadcast_batch(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Like ops.py's _unbroadcast, but only over the BATCH axes (everything
    before the last two). The last two are the matrix dimensions and must
    never be summed over.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis in range(len(shape) - 2):  # batch axes only, never the last two
        if shape[axis] == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    assert grad.shape == shape, f"unbroadcast failed: got {grad.shape}, expected {shape}"
    return grad


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Y = A @ B. Both must have ndim >= 2: the last two axes are the matrix
    dimensions ((N,K) and (K,M)); anything before is a batch dim, broadcast
    like NumPy's `@`. The plain 2D case is just the zero-batch-dims special
    case.

    Backward (standard batched chain rule):
        dA = dY @ B^T
        dB = A^T @ dY
    ^T swaps only the LAST TWO axes (np.swapaxes(-1,-2)), not `.T`, which
    would also reverse the batch axes. Each gradient is computed at the
    broadcast batch shape, then reduced back to its operand's shape via
    _unbroadcast_batch.
    """
    if a.data.ndim < 2 or b.data.ndim < 2:
        raise ValueError(
            f"matmul requires both operands to have ndim >= 2 (1D operands are out "
            f"of scope). Got shapes {a.data.shape} and {b.data.shape}"
        )
    out = Tensor(a.data @ b.data, a.requires_grad or b.requires_grad, (a, b), "matmul")

    def _backward():
        if a.requires_grad:
            b_t = np.swapaxes(b.data, -1, -2)
            a.accumulate_grad(_unbroadcast_batch(out.grad @ b_t, a.shape))
        if b.requires_grad:
            a_t = np.swapaxes(a.data, -1, -2)
            b.accumulate_grad(_unbroadcast_batch(a_t @ out.grad, b.shape))

    out._backward = _backward
    return out


Tensor.__matmul__ = lambda self, other: matmul(self, other)
