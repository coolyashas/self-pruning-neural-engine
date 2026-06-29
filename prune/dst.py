"""Dynamic sparse training: grow+drop exchange cycles that reallocate a
layer's fixed active-connection budget instead of only shrinking it. 
"""

from __future__ import annotations

import numpy as np

from nn.linear import Linear
from prune.criteria import accumulate_dense_gradients, accumulate_gradients
from prune.mask import _top_k_keep_mask, prune_to_sparsity, revive_to_count, set_mask
from prune.schedule import cubic_sparsity


def run_exchange_cycle(
    layer: Linear,
    drop_scores: np.ndarray,
    grow_scores: np.ndarray,
    n_exchange: int,
    optimizer,
) -> None:
    """One grow+drop exchange cycle on `layer`, net-zero active-count change:
    revive up to n_exchange masked entries by grow_scores (the dense
    w_eff.grad regrowth signal), then drop exactly as many ORIGINALLY-active
    entries by drop_scores as were actually revived (revive_to_count can clamp
    below n_exchange).
    """
    n_active_before = int(layer.mask.data.sum())
    if n_active_before == 0:
        return

    bias_active_before = layer.bias_mask.data.astype(bool).copy()
    revived = revive_to_count(layer, grow_scores, n_exchange)
    if revived.any():
        optimizer.reset_state(layer.weight, revived)
        bias_newly_active = layer.bias_mask.data.astype(bool) & ~bias_active_before
        if bias_newly_active.any():
            optimizer.reset_state(layer.bias, bias_newly_active)

    inactive_remaining = layer.mask.data == 0  # still off after revive
    adjusted_drop_scores = np.where(
        revived, np.inf, np.where(inactive_remaining, -np.inf, drop_scores)
    )
    keep = _top_k_keep_mask(adjusted_drop_scores, n_active_before)
    set_mask(layer, keep)


def dst_step(
    model,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    step: int,
    prune_start_step: int,
    prune_end_step: int,
    final_sparsity: float,
    drop_score_fn,
    exchange_fraction: float = 0.1,
) -> None:

    prunable = [layer for layer in model.layers if hasattr(layer, "mask")]

    if step < prune_end_step:
        target = cubic_sparsity(step, prune_start_step, prune_end_step, final_sparsity)
        accumulate_gradients(model, X, y, batch_size)
        for layer in prunable:
            prune_to_sparsity(layer, drop_score_fn(layer), target)
    else:
        dense_grads = accumulate_dense_gradients(model, X, y, batch_size)
        for layer in prunable:
            n_active = int(layer.mask.data.sum())
            n_exchange = max(1, round(exchange_fraction * n_active))
            run_exchange_cycle(layer, drop_score_fn(layer), dense_grads[layer], n_exchange, optimizer)
