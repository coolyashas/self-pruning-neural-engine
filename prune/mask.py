"""Pruning mask mechanics: flipping which connections are active.

The mask Tensor itself lives on the Linear layer (nn/linear.py) since the
forward pass needs it; this module owns setting it. Which weights to prune
(magnitude, saliency) and the sparsity schedule are later commits -- this
just sets an arbitrary mask safely.
"""

from __future__ import annotations

import numpy as np

from nn.linear import Linear


def set_mask(layer: Linear, keep: np.ndarray) -> None:
    """keep: boolean (or 0/1) array matching layer.weight's shape; True/1 = active."""
    assert keep.shape == layer.weight.shape, (keep.shape, layer.weight.shape)
    layer.mask.data = keep.astype(np.float64)


def _top_k_keep_mask(scores: np.ndarray, n_keep: int) -> np.ndarray:
    """Boolean mask, same shape as `scores`, keeping exactly the n_keep
    highest-scoring entries. Top-k via argsort, not a percentile/threshold
    cutoff: a threshold can over- or undershoot the target count when
    scores tie, top-k can't.
    """
    flat_keep = np.zeros(scores.size, dtype=bool)
    if n_keep > 0:
        top_indices = np.argsort(scores.ravel())[scores.size - n_keep :]
        flat_keep[top_indices] = True
    return flat_keep.reshape(scores.shape)


def keep_mask_from_scores(scores: np.ndarray, sparsity: float) -> np.ndarray:
    """Boolean keep-mask removing exactly `sparsity` fraction of entries
    -- the lowest-scoring ones. Criterion-agnostic: works for magnitude
    scores, saliency scores, or anything else that scores "how much would
    removing this hurt".
    """
    assert 0.0 <= sparsity <= 1.0
    n_keep = round((1 - sparsity) * scores.size)
    return _top_k_keep_mask(scores, n_keep)


def prune_to_sparsity(layer: Linear, scores: np.ndarray, target_sparsity: float) -> None:
    """Prune `layer` toward `target_sparsity`, monotonically: only ever
    removes more connections, never revives one. Already-pruned entries
    are forced to score -inf so top-k ranking can never select them again.

    If the discretely-achieved sparsity from an earlier call already
    meets or exceeds this call's (continuous) target, this is a no-op,
    not an error: the cubic schedule's target is continuous but counts
    are integers, so rounding can occasionally overshoot ahead of
    schedule, and the next call's target needs a moment to catch back up.
    """
    current_mask = layer.mask.data
    n_total = current_mask.size
    n_active = int(current_mask.sum())
    n_keep_target = min(round((1 - target_sparsity) * n_total), n_active)
    if n_keep_target == n_active:
        return

    adjusted_scores = np.where(current_mask == 0, -np.inf, scores)
    set_mask(layer, _top_k_keep_mask(adjusted_scores, n_keep_target))
