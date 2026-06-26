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
from prune.mask import _top_k_keep_mask, revive_to_count, set_mask


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
    """
    n_active_before = int(layer.mask.data.sum())

    revived = revive_to_count(layer, grow_scores, n_exchange)
    if revived.any():
        optimizer.reset_state(layer.weight, revived)

    inactive_remaining = layer.mask.data == 0  # still off after revive
    adjusted_drop_scores = np.where(
        revived, np.inf, np.where(inactive_remaining, -np.inf, drop_scores)
    )
    keep = _top_k_keep_mask(adjusted_drop_scores, n_active_before)
    set_mask(layer, keep)
