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
    """keep: boolean (or 0/1) array matching layer.weight's shape; True/1 = active.

    Also resyncs layer.bias_mask from this new mask (any nonzero entry in
    a column -> that neuron's bias stays active). This is the single
    place layer.mask is ever written, so it's the right place to keep
    bias_mask consistent regardless of which caller changed the mask
    (per-weight or per-neuron pruning both end up here).
    """
    assert keep.shape == layer.weight.shape, (keep.shape, layer.weight.shape)
    layer.mask.data = keep.astype(np.float64)
    layer.bias_mask.data = layer.mask.data.any(axis=0).astype(np.float64)


def _top_k_keep_mask(scores: np.ndarray, n_keep: int) -> np.ndarray:
    """Boolean mask, same shape as `scores`, keeping exactly the n_keep
    highest-scoring entries. Top-k via argsort, not a percentile/threshold
    cutoff: a threshold can over- or undershoot the target count when
    scores tie, top-k can't.

    Every caller in this module (prune_to_sparsity, prune_neurons_to_count,
    revive_to_count, and prune.dst's run_exchange_cycle) funnels through
    here, so this is the one place to guard against NaN scores reaching
    argsort: NumPy sorts NaN to the END (treats it as the LARGEST value),
    so a NaN-scored entry would be silently KEPT by every pruning/revival
    decision regardless of its true importance -- a wrong mask, not a
    crash. NaN scores would only arise from an already-corrupted
    upstream computation (e.g. a diverged loss during a scoring sweep),
    but failing loudly here is cheap insurance against a much harder to
    diagnose silent failure several layers downstream.
    """
    assert not np.isnan(scores).any(), "NaN score would be silently treated as highest-ranked by argsort"
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

    Also zeros the bias entry of every newly-dead neuron. set_mask already
    resyncs layer.bias_mask so mask-aware Adam freezes that bias going
    forward (see nn/linear.py) -- but freezing alone holds it at whatever
    value it had at the exact moment it died, not necessarily 0. A
    structurally-dead neuron's bias must be exactly 0, or the dense
    forward still emits ReLU(0 + bias) for it, a nonzero constant that
    compress_model (prune/compress.py) would wrongly treat as no
    contribution at all. The two pieces together -- explicit zero here,
    frozen going forward by bias_mask -- are both required: this caught a
    real bug where bias kept drifting after pruning even though the
    weight was correctly frozen (the exact same zero-gradient-isn't-
    frozen-state lesson mask-aware Adam already encodes for weight,
    rediscovered for bias).

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
    layer.bias.data[~keep_neurons] = 0.0


def revive_to_count(layer: Linear, scores: np.ndarray, n_revive: int) -> np.ndarray:
    """Activate exactly the n_revive highest-scoring CURRENTLY-MASKED
    entries in layer -- the grow half of dynamic sparse training.
    Mirrors prune_to_sparsity's -inf trick, inverted: already-ACTIVE
    entries are forced to -inf so top-k can only ever select among
    masked-off ones, guaranteeing this never re-selects something that's
    already active (the grow-side analogue of prune_to_sparsity's
    never-revives guarantee).

    Returns the boolean index array (same shape as layer.mask) of
    entries that were just revived -- callers need this to (a) call
    optimizer.reset_state on exactly these indices, since a revived
    entry must not inherit stale pre-pruning momentum, and (b) exclude
    them from the same DST cycle's subsequent drop step. Does NOT touch
    the optimizer itself -- mask mechanics here, optimizer-state
    mechanics in optim/adam.py, same division of responsibility
    prune_to_sparsity already has.

    Clamps n_revive to however many masked-off entries actually exist,
    same spirit as prune_to_sparsity clamping to n_active.
    """
    current_mask = layer.mask.data
    n_inactive = int((current_mask == 0).sum())
    n_revive = min(n_revive, n_inactive)
    if n_revive == 0:
        return np.zeros_like(current_mask, dtype=bool)

    adjusted_scores = np.where(current_mask != 0, -np.inf, scores)
    revived = _top_k_keep_mask(adjusted_scores, n_revive)  # top n_revive among masked-off entries

    new_mask = current_mask.astype(bool) | revived
    set_mask(layer, new_mask)
    return revived
