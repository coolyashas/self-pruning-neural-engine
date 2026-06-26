"""Adam: per-parameter first/second moment EMAs, bias-corrected."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Adam:
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0  # shared step count, used for bias correction

    def step(self) -> None:
        self.t += 1  # must start at 1: (1 - beta**0) = 0 would divide by zero
        for p, m, v in zip(self.parameters, self.m, self.v):
            if p.grad is None:
                continue
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * (p.grad**2)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None
