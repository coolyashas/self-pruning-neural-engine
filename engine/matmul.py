"""2D matmul. Only 2D — that's all Linear layers need, no batched matmul."""

from __future__ import annotations

from engine.tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    # O(N*D*H) forward and backward, same as the matmul itself.
    assert a.data.ndim == 2 and b.data.ndim == 2, "matmul only supports 2D tensors"
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
