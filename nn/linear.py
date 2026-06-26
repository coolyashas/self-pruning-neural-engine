"""Linear (affine) layer: y = x @ W + b, with He initialization for ReLU."""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        # He init: std = sqrt(2/fan_in). ReLU zeros about half its inputs,
        # so it halves variance going forward; the extra factor of 2 here
        # (vs. Xavier's sqrt(1/fan_in)) cancels that out and keeps
        # activation variance roughly constant across layers.
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * std, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
