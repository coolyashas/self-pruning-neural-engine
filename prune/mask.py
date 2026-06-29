"""Pruning mask mechanics: flipping which connections are active.

The mask Tensor lives on the Linear layer (nn/linear.py); this module owns
setting it safely. Criteria and schedules live in sibling modules.
"""

from __future__ import annotations

import numpy as np

from nn.linear import Linear


def set_mask(layer: Linear, keep: np.ndarray) -> None:
    #keep: boolean (or 0/1) array matching layer.weight's shape; True/1 = active.

    # `if: raise` not `assert`, so a shape mismatch can't slip through `-O`.
    if keep.shape != layer.weight.shape:
        raise ValueError(f"keep.shape {keep.shape} != layer.weight.shape {layer.weight.shape}")
    new_column_alive = keep.astype(bool).any(axis=0)
    # bias_mask still mirrors the OLD mask here (this is the only writer).
    was_column_alive = layer.bias_mask.data.astype(bool)
    newly_dead = was_column_alive & ~new_column_alive

    layer.mask.data = keep.astype(np.float64)
    layer.bias_mask.data = new_column_alive.astype(np.float64)
    layer.bias.data[newly_dead] = 0.0


def _top_k_keep_mask(scores: np.ndarray, n_keep: int) -> np.ndarray:
    """Boolean mask, same shape as `scores`, keeping exactly the n_keep
    highest-scoring entries. Top-k via argsort, not a threshold cutoff, which
    can over/undershoot the target count on ties.
    """
    
    if np.isnan(scores).any():
        raise ValueError("NaN score would be silently treated as highest-ranked by argsort")
    flat_keep = np.zeros(scores.size, dtype=bool)
    if n_keep > 0:
        top_indices = np.argsort(scores.ravel())[scores.size - n_keep :]
        flat_keep[top_indices] = True
    return flat_keep.reshape(scores.shape)


def keep_mask_from_scores(scores: np.ndarray, sparsity: float) -> np.ndarray:
    """Boolean keep-mask removing the lowest-scoring `sparsity` fraction of
    entries. Criterion-agnostic (magnitude, saliency, anything).
    """
    assert 0.0 <= sparsity <= 1.0
    n_keep = round((1 - sparsity) * scores.size)
    return _top_k_keep_mask(scores, n_keep)


def prune_to_sparsity(layer: Linear, scores: np.ndarray, target_sparsity: float) -> None:
    """Prune `layer` toward `target_sparsity`, monotonically: only removes,
    never revives. Already-pruned entries score -inf so top-k can't reselect
    them."""
    current_mask = layer.mask.data
    n_total = current_mask.size
    n_active = int(current_mask.sum())
    n_keep_target = min(round((1 - target_sparsity) * n_total), n_active)
    if n_keep_target == n_active:
        return

    adjusted_scores = np.where(current_mask == 0, -np.inf, scores)
    set_mask(layer, _top_k_keep_mask(adjusted_scores, n_keep_target))


def keep_neurons_from_scores(scores: np.ndarray, n_prune: int) -> np.ndarray:
    """Boolean keep-mask over neurons (shape (out_features,)), pruning the
    n_prune lowest-scoring neurons. Count-parameterized, not fraction.
    """
    n_total = scores.size
    n_keep = n_total - n_prune
    return _top_k_keep_mask(scores, n_keep)


def prune_neurons_to_count(layer: Linear, scores: np.ndarray, target_active: int) -> None:
    """Prune layer's OUTPUT neurons (whole mask columns) down to target_active
    active neurons, monotonically -- prune_to_sparsity's -inf never-revives
    trick applied column-wise.
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


def revive_to_count(layer: Linear, scores: np.ndarray, n_revive: int) -> np.ndarray:
    """Activate the n_revive highest-scoring CURRENTLY-MASKED entries -- the
    grow half of DST. prune_to_sparsity's -inf trick inverted: already-active
    entries score -inf so top-k only selects among masked-off ones.
    """
    current_mask = layer.mask.data
    n_inactive = int((current_mask == 0).sum())
    n_revive = min(n_revive, n_inactive)
    if n_revive == 0:
        return np.zeros_like(current_mask, dtype=bool)

    adjusted_scores = np.where(current_mask != 0, -np.inf, scores)
    revived = _top_k_keep_mask(adjusted_scores, n_revive)

    new_mask = current_mask.astype(bool) | revived
    set_mask(layer, new_mask)
    return revived
