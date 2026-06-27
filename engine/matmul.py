"""Matmul: 2D matrices, or batched stacks of them (3D+, NumPy-style
batch broadcasting on every axis before the last two). Vector-matrix/
matrix-vector (1D operands) is a different operation -- no batch
dimension to broadcast, and a different gradient-shape convention --
and stays explicitly out of scope, rejected loudly.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def _unbroadcast_batch(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Like ops.py's _unbroadcast, but for batched matmul's BATCH axes
    only -- everything before the last two, which are the real matrix
    dimensions and must never be summed over (that would corrupt the
    matrix math itself, not just undo a broadcast). Batch axes can be
    broadcast exactly like any elementwise op's axes (added as new
    leading axes, or stretched from size 1); the matrix axes (N,K /
    K,M) never broadcast against each other in matmul -- the shared K
    dimension must match exactly, that's matmul's own contract, not
    something relaxed here.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis in range(len(shape) - 2):  # batch axes only, never the last two
        if shape[axis] == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    assert grad.shape == shape, f"unbroadcast failed: got {grad.shape}, expected {shape}"
    return grad


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Y = A @ B. Both A and B must have ndim >= 2: the last two axes
    are the actual matrix dimensions ((N,K) and (K,M)); everything
    before that is a batch dimension, broadcast against each other
    exactly like NumPy's own `@` already does (e.g. (5,N,K) @ (K,M) is
    valid, broadcasting one (K,M) matrix across a batch of 5 -- the
    plain 2D case used everywhere in this repo today is just the
    zero-batch-dims special case of this, unchanged).

    Forward: out = A @ B, shape (*batch, N, M), where *batch is A's
    and B's batch dims broadcast together (NumPy handles this
    natively, including raising its own clear error if the batch
    shapes aren't broadcast-compatible -- not re-implemented here).

    Backward, the standard batched-matmul chain rule:
        dA = dY @ B^T
        dB = A^T @ dY
    where ^T here means swapping only the LAST TWO axes
    (np.swapaxes(-1,-2)), not `.T`, which reverses every axis
    including the batch ones -- using `.T` on a >2D array would
    silently transpose the batch dims too, producing a wrong-shaped,
    wrong-valued gradient rather than a clean error (confirmed by
    direct execution against the old 2D-only implementation: this is
    exactly what made `.T` unsafe to reuse as-is once batch dims were
    allowed in). Each of dA/dB is computed at the broadcast batch
    shape first (a plain matmul, batch-broadcasting automatically),
    then reduced back down to its own operand's original batch shape
    via _unbroadcast_batch -- the same "sum over what was broadcast"
    principle as ops.py's elementwise _unbroadcast, just restricted to
    the batch axes.

    Numerical stability: matmul introduces no new numerical risk
    beyond any individual multiply-accumulate (no division, no
    exp/log) -- whatever precision/overflow behavior NumPy's BLAS
    backend has is inherited as-is, batched or not.

    Complexity: O(batch_size * N*K*M) forward and backward, same as
    the underlying BLAS call(s) -- one batched call, not a Python loop
    over the batch.
    """
    if a.data.ndim < 2 or b.data.ndim < 2:
        raise ValueError(
            f"matmul requires both operands to have ndim >= 2 (2D matrices, or "
            f"batched stacks of them) -- vector-matrix/matrix-vector (1D operands) "
            f"is a different operation, explicitly out of scope. Got shapes "
            f"{a.data.shape} and {b.data.shape}"
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
