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


def keep_neurons_from_scores(scores: np.ndarray, n_prune: int) -> np.ndarray:
    """Boolean keep-mask over neurons (shape (out_features,)), pruning
    exactly the n_prune lowest-scoring neurons. Mirrors _top_k_keep_mask's
    top-k approach, parameterized by an exact count rather than a
    fraction since callers here pass a neuron count, not a sparsity.
    """
    n_total = scores.size
    n_keep = n_total - n_prune
    return _top_k_keep_mask(scores, n_keep)


def prune_neurons_to_count(layer: Linear, scores: np.ndarray, target_active: int) -> None:
    """Prune layer's OUTPUT neurons (whole columns of layer.mask) down to
    exactly target_active active neurons, monotonically -- mirrors
    prune_to_sparsity's never-revives contract via the same -inf trick,
    just column-wise: a neuron that's already fully pruned (all-zero
    mask column) is forced to -inf so top-k can't reactivate it.

    "Active" means the mask column has any nonzero entry -- a neuron
    only partly pruned by earlier unstructured pruning still counts as
    active until this function zeros out the rest of its column.

    Deliberately not built on top of prune_to_sparsity: that function's
    -inf trick operates per-entry (current_mask == 0), not per-column, so
    reusing it as-is would let a half-dead column escape full-column
    zeroing. prune_to_sparsity itself is left untouched.
    """
    n_out = layer.weight.shape[1]
    column_alive = layer.mask.data.any(axis=0)
    n_active = int(column_alive.sum())
    target_active = min(target_active, n_active)  # never revive
    if target_active == n_active:
        return

    adjusted_scores = np.where(column_alive, scores, -np.inf)
    n_prune = n_out - target_active
    keep_neurons = keep_neurons_from_scores(adjusted_scores, n_prune)

    new_mask = layer.mask.data.copy()
    new_mask[:, ~keep_neurons] = 0.0
    set_mask(layer, new_mask)
