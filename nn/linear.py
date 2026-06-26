"""Linear (affine) layer: y = x @ (W * mask) + b, with He init for ReLU.

The mask is always part of the forward graph (all-ones until something
prunes it), not a separate masked/unmasked code path -- so dense and
pruned training run through the exact same forward. dL/dW for a masked
entry comes out exactly 0 because mul()'s existing backward multiplies the
upstream gradient by the mask -- nothing here special-cases it.
"""

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
        # requires_grad=False: a constant from autodiff's view, like any
        # other non-trainable input. Only weights get pruned, not bias --
        # standard practice; bias isn't a "connection" to remove.
        self.mask = Tensor(np.ones_like(self.weight.data), requires_grad=False)

    def __call__(self, x: Tensor) -> Tensor:
        w_eff = self.weight * self.mask
        return x @ w_eff + self.bias

    def masked_parameters(self) -> list[tuple[Tensor, Tensor | None]]:
        return [(self.weight, self.mask), (self.bias, None)]

    def parameters(self) -> list[Tensor]:
        return [p for p, _ in self.masked_parameters()]
