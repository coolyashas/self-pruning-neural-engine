"""Adam: per-parameter first/second moment EMAs, bias-corrected.

Mask-aware: an optional `masks` list (parallel to `parameters`, None for
params that aren't pruned, e.g. bias) restricts the ENTIRE update -- not
just the weight, but the m/v EMA updates too -- to active entries. Zero
gradient alone is not enough to freeze a masked entry: `m = beta1*m +
(1-beta1)*0` still decays old m toward zero instead of holding it, and
leftover m keeps nudging the weight for several more steps even after
masking. Restricting m/v's own update (not just relying on grad=0) is what
actually freezes a masked connection.
"""

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
        masks: list[Tensor | None] | None = None,
    ) -> None:
        self.parameters = parameters
        self.masks = masks if masks is not None else [None] * len(parameters)
        assert len(self.masks) == len(self.parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0  # shared step count, used for bias correction

    def step(self) -> None:
        self.t += 1  # must start at 1: (1 - beta**0) = 0 would divide by zero
        for p, m, v, mask in zip(self.parameters, self.m, self.v, self.masks):
            if p.grad is None:
                continue
            # read mask.data fresh every step, not cached at __init__: the
            # mask can change between steps (pruning happens between
            # training steps), and Tensor.mask.data may even be reassigned
            # to a new array by set_mask(), so we must dereference it now.
            active = np.ones_like(p.data, dtype=bool) if mask is None else (mask.data != 0)

            m_new = self.beta1 * m + (1 - self.beta1) * p.grad
            v_new = self.beta2 * v + (1 - self.beta2) * (p.grad**2)
            m[active] = m_new[active]
            v[active] = v_new[active]

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data[active] -= update[active]

    def reset_state(self, param: Tensor, indices) -> None:
        """Zero m and v at `indices` for `param`. Call this exactly when
        those entries flip from masked to active (revival) -- otherwise
        they inherit stale momentum from before they were pruned, causing
        an oversized or wrong-direction first step.
        """
        i = self.parameters.index(param)
        self.m[i][indices] = 0.0
        self.v[i][indices] = 0.0

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None
