"""sum and mean reductions."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def _normalize_axis(axis, ndim) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(ax % ndim for ax in axis)


def _grad_to_input_shape(grad: np.ndarray, shape: tuple[int, ...], axis, keepdims: bool) -> np.ndarray:
    """Reverse of a reduction: re-insert the reduced axes, then broadcast."""
    if not keepdims:
        for ax in sorted(_normalize_axis(axis, len(shape))):
            grad = np.expand_dims(grad, ax)
    return np.broadcast_to(grad, shape).copy()


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), a.requires_grad, (a,), "sum")

    def _backward():
        if a.requires_grad:
            a.accumulate_grad(_grad_to_input_shape(out.grad, a.shape, axis, keepdims))

    out._backward = _backward
    return out


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), a.requires_grad, (a,), "mean")
    # backward divides by the number of averaged elements (unlike sum's).
    axes = _normalize_axis(axis, a.data.ndim)
    n = 1
    for ax in axes:
        n *= a.shape[ax]

    def _backward():
        if a.requires_grad:
            a.accumulate_grad(_grad_to_input_shape(out.grad / n, a.shape, axis, keepdims))

    out._backward = _backward
    return out


Tensor.sum = lambda self, axis=None, keepdims=False: sum(self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis, keepdims)
