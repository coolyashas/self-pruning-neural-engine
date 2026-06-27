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
    n, n_classes = logits.shape
    labels = np.asarray(labels)

    # These three checks are real input-validation, not debug-only
    # sanity checks -- they MUST survive `python -O` (which strips every
    # `assert` statement, silently re-enabling the exact wrong-answer
    # paths they exist to prevent, confirmed by running this function
    # under -O before this fix). Plain `if: raise` instead of `assert`.

    # Integer dtype, checked first: a float label array like
    # array([0.0, 1.0, 2.0]) would otherwise reach the fancy-indexing
    # below and die there with NumPy's own raw
    # "IndexError: arrays used as indices must be of integer (or
    # boolean) type" -- not wrong, but a leaky API boundary that should
    # fail with a message about THIS function's contract, not a
    # downstream NumPy implementation detail.
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be an integer array, got dtype={labels.dtype}")

    # labels MUST be 1D, shape (n,) -- a common, easy mistake is passing
    # (n, 1) (e.g. straight out of a CSV column or a one-hot-decode that
    # forgot to .squeeze()). NumPy's fancy indexing below does NOT raise
    # for that shape: log_probs[arange(n), labels] with labels.shape ==
    # (n, 1) broadcasts arange(n) (shape (n,)) against labels (shape
    # (n, 1)) into an (n, n) index pair, silently selecting an n x n
    # block instead of n single entries, and .mean() then quietly
    # averages the wrong object -- a different, wrong loss AND gradient,
    # confirmed by direct execution, with no exception raised anywhere.
    # Catching the shape here, before that indexing happens, turns a
    # silent wrong answer into a loud, immediate error.
    if labels.shape != (n,):
        raise ValueError(f"labels must have shape ({n},), got {labels.shape}")

    # fancy-indexing with an out-of-[0, C) label doesn't always raise: a
    # label of -1 is valid NumPy negative indexing and silently selects
    # the LAST class instead of erroring -- a wrong loss/gradient, not a
    # crash. Catch it here rather than downstream.
    if not np.all((labels >= 0) & (labels < n_classes)):
        raise ValueError(f"labels must be in [0, {n_classes}), got min={labels.min()}, max={labels.max()}")

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
