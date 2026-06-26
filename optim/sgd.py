"""SGD with classic ("heavy ball") momentum."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class SGD:
    def __init__(self, parameters: list[Tensor], lr: float, momentum: float = 0.9) -> None:
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for p, v in zip(self.parameters, self.velocity):
            if p.grad is None:  # never touched by backward() this step
                continue
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None
