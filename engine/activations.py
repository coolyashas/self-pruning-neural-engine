"""ReLU and tanh."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def relu(a: Tensor) -> Tensor:
    out = Tensor(np.maximum(a.data, 0.0), a.requires_grad, (a,), "relu")

    def _backward():
        if a.requires_grad:
            # subgradient at x=0 taken as 0 (the usual convention)
            a.accumulate_grad(out.grad * (a.data > 0))

    out._backward = _backward
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = Tensor(t, a.requires_grad, (a,), "tanh")

    def _backward():
        # d(tanh)/dx = 1 - tanh(x)^2
        if a.requires_grad:
            a.accumulate_grad(out.grad * (1 - t**2))

    out._backward = _backward
    return out


Tensor.relu = lambda self: relu(self)
Tensor.tanh = lambda self: tanh(self)
