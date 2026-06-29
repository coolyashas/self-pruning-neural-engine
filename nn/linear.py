"""Linear (affine) layer: y = x @ (W * mask) + b, with He init for ReLU.

The mask is always part of the forward graph (all-ones until pruned), so
dense and pruned training share one forward path. dL/dW for a masked entry
is exactly 0 because mul()'s backward multiplies grad by the mask.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        # He init: std = sqrt(2/fan_in), the factor of 2 compensating for
        # ReLU halving activation variance going forward.
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * std, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        # Only weights are pruned (bias isn't a "connection"); mask is a
        # non-trainable constant from autodiff's view.
        self.mask = Tensor(np.ones_like(self.weight.data), requires_grad=False)
        # bias_mask tracks whether an output neuron still has any active weight
        # (synced by prune.mask.set_mask). Lets Adam freeze a structurally-dead
        # neuron's bias too -- otherwise its momentum keeps nudging the bias
        # since nothing masks bias's own forward contribution (z = bias alone).
        self.bias_mask = Tensor(np.ones(out_features), requires_grad=False)

    def __call__(self, x: Tensor) -> Tensor:
        # Stored on self so prune/criteria.py can read w_eff.grad after
        # backward(): the dense/unmasked signal "how much would loss change if
        # this connection were active", unlike weight.grad (always 0 if masked).
        self.w_eff = self.weight * self.mask
        return x @ self.w_eff + self.bias

    def masked_parameters(self) -> list[tuple[Tensor, Tensor | None]]:
        return [(self.weight, self.mask), (self.bias, self.bias_mask)]

    def parameters(self) -> list[Tensor]:
        return [p for p, _ in self.masked_parameters()]
