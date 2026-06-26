"""Fused softmax + cross-entropy.

Fused (one op, one backward formula) rather than composed from separate
softmax/log/CE nodes: lets us compute log_softmax directly via the
log-sum-exp trick instead of log(softmax(x)), and gives the simple combined
gradient (probs - one_hot) instead of chaining three local Jacobians.
"""

from __future__ import annotations

import numpy as np

from engine.tensor import Tensor


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """logits: (N, C) scores. labels: (N,) int class indices, not a Tensor
    — they're not differentiable w.r.t. anything. Returns mean loss over N.
    """
    n = logits.shape[0]
    labels = np.asarray(labels)

    # subtract the row max before exp(): shifts every row by a constant,
    # which softmax is invariant to, but keeps exp() from overflowing.
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    sum_exp = exp.sum(axis=1, keepdims=True)
    probs = exp / sum_exp
    log_probs = shifted - np.log(sum_exp)  # log-sum-exp, not log(probs)
    loss_per_example = -log_probs[np.arange(n), labels]

    out = Tensor(loss_per_example.mean(), logits.requires_grad, (logits,), "softmax_ce")

    def _backward():
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(n), labels] -= 1.0
            grad /= n  # loss is the mean over the batch
            logits.accumulate_grad(out.grad * grad)

    out._backward = _backward
    return out


Tensor.softmax_cross_entropy = lambda self, labels: softmax_cross_entropy(self, labels)
