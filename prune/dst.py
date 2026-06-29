"""Dynamic sparse training: grow+drop exchange cycles that reallocate a
layer's fixed active-connection budget instead of only shrinking it. The
prune_* primitives are monotonic by design; this is a higher-level policy
composing revive_to_count with a drop step.
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

    The drop step needs THREE-way scoring: revived entries forced to +inf
    (kept this cycle, to prove themselves), still-inactive entries forced to
    -inf (never reactivated here), and only originally-active entries scored
    by drop_scores. Lumping revived and still-inactive together would let
    top-k keep still-inactive entries instead of real candidates.

    Calls optimizer.reset_state(layer.weight, revived) so revived entries
    don't inherit stale pre-pruning momentum (the optimizer must be the one
    built from this model's masked_parameters()). Also resets bias state for
    any neuron whose bias_mask flips 0->1 here: set_mask un-freezes that bias,
    and without a reset Adam would apply momentum from before the neuron died.

    Guarded no-op at n_active_before == 0: the drop target is n_active_before,
    so at 0 even a +inf-forced entry gets dropped -- a fully-dead layer would
    revive then immediately re-drop. Nothing to exchange anyway.
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
    """One scheduled DST maintenance step (called from on_step_end). Two phases:

    1. Ramp (step < prune_end_step): accumulate_gradients + prune_to_sparsity
       toward the cubic target, monotonic, no regrowth yet -- nothing to
       exchange while sparsity is still climbing.
    2. Maintenance (step >= prune_end_step): grow+drop exchange cycles per
       layer, sized exchange_fraction * n_active (>= 1), growing by
       w_eff.grad and dropping by drop_score_fn. Net active count stays
       constant.

    Maintenance needs only ONE sweep: accumulate_dense_gradients's backward()
    populates weight.grad (which drop_score_fn reads for saliency) as a side
    effect of producing w_eff.grad, so a second sweep would redo identical work.
    """
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
