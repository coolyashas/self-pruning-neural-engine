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


def keep_mask_from_scores(scores: np.ndarray, sparsity: float) -> np.ndarray:
    """Boolean keep-mask, same shape as `scores`, that removes exactly
    `sparsity` fraction of entries -- the lowest-scoring ones. Top-k via
    argsort, not a percentile/threshold cutoff: a threshold can over- or
    undershoot the target count when many scores tie, top-k can't.
    Criterion-agnostic: works for magnitude scores, saliency scores, or
    anything else that scores "how much would removing this hurt".
    """
    assert 0.0 <= sparsity <= 1.0
    n = scores.size
    n_keep = round((1 - sparsity) * n)

    flat_keep = np.zeros(n, dtype=bool)
    if n_keep > 0:
        # argsort ascending; the highest-scoring n_keep indices are the
        # last n_keep entries of that order
        top_indices = np.argsort(scores.ravel())[n - n_keep :]
        flat_keep[top_indices] = True
    return flat_keep.reshape(scores.shape)


def prune_to_sparsity(layer: Linear, scores: np.ndarray, target_sparsity: float) -> None:
    """Prune `layer` to exactly `target_sparsity`, monotonically: only
    ever removes more connections, never revives one. Already-pruned
    entries are forced to score -inf so keep_mask_from_scores's top-k
    ranking can never select them again, without duplicating its logic.
    """
    current_mask = layer.mask.data
    current_sparsity = 1.0 - current_mask.mean()
    assert target_sparsity >= current_sparsity - 1e-9, (
        "target sparsity must not decrease -- this path has no regrowth"
    )
    adjusted_scores = np.where(current_mask == 0, -np.inf, scores)
    set_mask(layer, keep_mask_from_scores(adjusted_scores, target_sparsity))
