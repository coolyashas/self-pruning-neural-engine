"""Dynamic sparse training: grow+drop exchange cycles that reallocate a
layer's fixed active-connection budget instead of only ever shrinking it.
prune_to_sparsity/prune_neurons_to_count are monotonic by design (and
must stay that way -- see their own never-revives tests); this is a
separate, higher-level policy that composes revive_to_count with a drop
step, kept in its own module since it's about orchestrating a cycle, not
a single mask primitive.
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
    """One grow+drop exchange cycle on `layer`, net-zero active-count
    change: revive up to n_exchange currently-masked entries by
    grow_scores (the dense w_eff.grad-derived regrowth signal), then drop
    exactly as many of the ORIGINALLY-active entries by drop_scores (the
    existing saliency/magnitude criterion) as were actually revived --
    not the requested n_exchange, since revive_to_count can clamp if
    fewer than n_exchange masked entries exist.

    The drop step needs THREE-way scoring, not two -- this is the part
    most likely to be subtly wrong (a single "excluded -> +inf" bucket
    lumping revived and still-inactive entries together was the first,
    wrong, draft of this function): revived entries must be FORCED into
    the kept set (+inf, so top-k can't drop them this cycle -- they need
    a chance to prove themselves first), still-inactive entries must be
    FORCED OUT of it (-inf, so top-k can never reactivate them -- this
    is not a revival path), and only the untouched originally-active
    entries get real drop_scores. Lumping revived and still-inactive
    together at +inf would let top-k satisfy "keep N highest" using
    still-inactive entries instead of real candidates whenever there are
    more excluded entries than the keep budget -- silently wrong, not a
    crash.

    Calls optimizer.reset_state(layer.weight, revived) for every revived
    entry so it doesn't inherit stale pre-pruning momentum -- the
    optimizer passed in must be the one built from this model's
    masked_parameters(), so layer.weight is actually in its parameter
    list (reset_state looks it up via .index()).

    Also resets bias state for any neuron whose bias_mask flips 0->1 as
    a SIDE EFFECT of this revival. set_mask (called inside
    revive_to_count) resyncs bias_mask from mask.any(axis=0) on every
    call -- so reviving even one weight into a column that had been
    driven fully to zero (whether by plain unstructured drift or by an
    earlier structured prune) silently un-freezes that neuron's bias,
    letting Adam resume updating it. If that bias's m/v were never
    reset, the very next step() applies real momentum/variance computed
    from before the neuron died -- exactly the stale-momentum failure
    mode the whole mask-aware-Adam design exists to prevent, just for
    bias instead of weight. revive_to_count only resets the WEIGHT side
    (it has no bias-side knowledge by design), so this function -- the
    one place that already watches mask transitions during revival --
    is where the bias-side reset belongs.

    Guarded no-op at n_active_before == 0: the drop step's "keep N
    highest" target IS n_active_before, so at 0 it would keep nothing
    regardless of score -- even a +inf-forced "must survive" entry gets
    discarded, since _top_k_keep_mask(scores, n_keep=0) returns all-False
    unconditionally. Without this guard, a fully-dead layer would revive
    an entry and then immediately drop it again in the same cycle,
    silently violating the "revived entries survive their first cycle"
    guarantee above. There's nothing to "exchange" when nothing is
    active to begin with, so skipping entirely is also the correct
    semantics, not just a crash-avoidance hack.
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
    """One scheduled DST maintenance step -- called from on_step_end at
    the same cadence/spirit as run_part3's existing pruning hook. Two
    phases:

    1. Ramp phase (step < prune_end_step): identical to run_part3's
       existing non-regrowth behavior -- accumulate_gradients +
       prune_to_sparsity toward the cubic schedule's target, monotonic,
       no regrowth yet. Reusing prune_to_sparsity here (not
       run_exchange_cycle) during the ramp matters: there's nothing to
       "exchange" yet while sparsity is still increasing toward target.
    2. Maintenance phase (step >= prune_end_step): final_sparsity is
       already reached; switch to grow+drop exchange cycles per
       prunable layer, sized as exchange_fraction * n_active (rounded,
       at least 1), using accumulate_dense_gradients' w_eff.grad signal
       to decide what to grow and drop_score_fn (the existing
       saliency/magnitude criterion) to decide what to drop. Net
       active-connection count per layer stays constant in this phase,
       by run_exchange_cycle's own contract.

    The maintenance phase needs only ONE dataset sweep, not two:
    accumulate_dense_gradients's backward() calls populate weight.grad as
    a side effect of the same computation that produces w_eff.grad --
    mul()'s backward writes both from one pass over the data -- so
    drop_score_fn (which reads weight.grad for saliency) is already
    correctly populated by the time accumulate_dense_gradients returns.
    A second accumulate_gradients call here would silently redo
    identical work for zero benefit (verified bit-for-bit identical
    against a standalone accumulate_gradients call on the same data).
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
