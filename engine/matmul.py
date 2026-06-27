"""2D matmul. Only 2D — that's all Linear layers need, no batched matmul."""

from __future__ import annotations

from engine.tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    # O(N*D*H) forward and backward, same as the matmul itself.
    #
    # This is a hard SCOPE boundary, not a debug-only sanity check: no
    # batched matmul, no vector-matrix/matrix-vector, no >2D inputs --
    # `assert` would be wrong here because `python -O` strips it
    # entirely, and the failure mode it's protecting against isn't a
    # clean crash, it's worse. Confirmed by direct execution: NumPy's
    # `@` happily computes batched matmul for 3D inputs (forward
    # "succeeds", silently, with no error at all if backward is never
    # called -- e.g. an inference-only use). If backward IS called, the
    # `.T` below means something different for ndim>2 (reverses ALL
    # axes, not just the last two), so backward then crashes with a
    # confusing low-level NumPy shape-mismatch error far from the real
    # problem, instead of a clear message about this engine's own
    # documented contract. Raising loudly and immediately, before any
    # of that, is strictly better than either failure mode.
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(
            f"matmul only supports 2D tensors (no batched matmul, no vector-matrix/"
            f"matrix-vector), got shapes {a.data.shape} and {b.data.shape}"
        )
    out = Tensor(a.data @ b.data, a.requires_grad or b.requires_grad, (a, b), "matmul")

    def _backward():
        # Y = A @ B  =>  dA = dY @ B.T, dB = A.T @ dY
        if a.requires_grad:
            a.accumulate_grad(out.grad @ b.data.T)
        if b.requires_grad:
            b.accumulate_grad(a.data.T @ out.grad)

    out._backward = _backward
    return out


Tensor.__matmul__ = lambda self, other: matmul(self, other)
