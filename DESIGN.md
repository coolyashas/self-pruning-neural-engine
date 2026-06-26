# DESIGN.md

Four things worth defending in detail: where the saliency criterion comes
from, why the masked-gradient/mask-aware-Adam design is built the way it
is, what pruning here actually costs (and doesn't save), and what would
need to change to ship it.

## 1. Importance criterion: why `|w·g|`, not just `|w|`

Pruning connection `i` means setting `w_i → 0`, i.e. `Δw_i = -w_i`. A
first-order Taylor expansion of the loss around the current weights gives:

```
ΔL ≈ g_i · Δw_i = g_i · (-w_i) = -w_i · g_i
```

So the loss increase from removing connection `i`, to first order, is
`|w_i · g_i|`. That's the saliency criterion (`prune/criteria.py:saliency_scores`):
it estimates *how much removing this connection would actually hurt*.

Magnitude (`|w_i|` alone) is the same expression with `g_i` implicitly
assumed equal across all connections — i.e. it's saliency with the
gradient term thrown away. That's why it's the baseline, not the main
criterion: a large weight with a near-zero gradient (the network doesn't
currently care what it does) is a fine prune; magnitude alone can't see
that, saliency can.

Two practical consequences that show up directly in the code:

- A single mini-batch's gradient is noisy, so saliency needs an
  *accumulated* gradient estimate over a representative sweep of the data
  (`accumulate_gradients`, commit 20) before scoring — using the last
  batch's gradient alone would make the criterion itself noisy, defeating
  the point of using gradient information at all.
- The criteria are genuinely different, not just differently-scaled: a
  connection can rank highest on magnitude and lowest on saliency
  simultaneously (`test_saliency_and_magnitude_can_disagree`), and the
  real Pareto sweep shows this divergence matters in practice, not just
  in a contrived test (see §3 below and `results/CLAIM.md`).

## 2. Masked gradient and mask-aware Adam

**The forward design.** Pruning is implemented as `w_eff = weight * mask`,
a literal multiplication node in the autodiff graph (`nn/linear.py`), not
a post-hoc edit to a computed gradient. Mechanically, this needed *zero*
new engine code: `mul()`'s backward (written in commit 3, long before
pruning existed) already computes `grad_a = grad_out * b.data`. With
`a = weight`, `b = mask`, and `mask.requires_grad = False`, that's exactly
`dL/dweight = dL/dw_eff * mask` — the masked-gradient-is-zero property
falls out of the chain rule that was already there. This is the
difference between "the math gives us this for free" and "we remembered
to patch the gradient afterward" — the second kind of fix is exactly what
breaks the moment the graph is extended (e.g. weight decay, a different
loss, weight reuse) and nobody remembers to re-apply the patch.

**Why mask-aware Adam needs more than zero gradient.** This is the
project's central correctness requirement, and the part most
implementations get wrong. Zero gradient does *not* imply a frozen
optimizer state. Adam's moments are exponential moving averages:

```
m = β1·m + (1-β1)·g
v = β2·v + (1-β2)·g²
```

If `g = 0` (correctly, from the masked weight), this still computes
`m = β1·m`, `v = β2·v` — geometric *decay*, not freezing. A masked
weight's leftover momentum from before it was pruned keeps nudging it for
several more steps, and the weight only stops moving once `m` has decayed
numerically to zero. `optim/adam.py` instead restricts the entire update —
`m`, `v`, *and* `w` — to active entries via the layer's mask, gated
explicitly, not inferred from the gradient being zero. This is verified
directly: a masked entry's `m`/`v` stay *bitwise unchanged*, not merely
small, across many steps with a constant nonzero incoming gradient
(`tests/test_masked_adam.py`).

**Revival.** `reset_state(param, indices)` zeros `m` and `v` for entries
that flip from masked back to active. Without it, a revived connection's
first update uses momentum accumulated from a completely different
training regime (whatever was happening before it got pruned) — `tests/test_masked_adam.py::test_without_reset_state_revival_inherits_stale_momentum`
demonstrates this produces a measurably different (and generally
wrong-direction or oversized) first step compared to the reset version of
the identical scenario.

**An explicit trade-off, not an oversight.** Adam here uses one *shared*
step counter `t` for bias correction, not a per-parameter or per-element
one. On revival, a freshly-zeroed `m, v` gets bias-corrected against
whatever `t` the rest of training has reached (e.g. `t=2000`), not against
a true `t=1`. Since `(1 - β^2000) ≈ 1`, this gives almost no
bias-correction boost, unlike a genuinely fresh Adam parameter at `t=1`.
This does not violate the stated correctness bar (masked `m`/`v` exactly
frozen; revived `m`/`v` exactly zero) — it's a deliberate simplification
that avoids per-element step bookkeeping that masking would *also* need
to gate. Flagging this proactively, rather than waiting to be asked "is
your bias correction exactly right on revival?", is the point of writing
it down here.

## 3. The bottleneck: pruning here saves parameters, not time

`evaluation/cost.py` deliberately reports two different numbers:
`active_param_count` (what a sparse implementation would need to store)
and `model_flops(..., active_only=False)` (what this implementation
*actually* computes). They diverge as soon as anything is pruned, and
that divergence is the whole point: `x @ (weight * mask)` is a normal
dense matmul. Multiplying by a zero is still a multiply — NumPy (and any
dense BLAS routine underneath it) has no way to know an entry is zero and
skip it.

This isn't a theoretical caveat; it's measured.
`test_dense_matmul_speed_unaffected_by_sparsity` times a 0%-sparse and a
90%-sparse layer of identical shape and finds no speedup (ratio ≈0.93,
within normal timing noise). The bottleneck for a *real* speedup isn't
the pruning logic, it's that dense linear-algebra kernels don't skip
zeros at all — getting wall-clock benefit needs one of:

- **A genuine sparse matmul kernel** (CSR/CSC storage, sparse BLAS). These
  typically only pay off at very high sparsity (often >90-95%) because
  the indexing overhead of sparse formats is itself non-trivial, and
  unstructured (per-weight) sparsity patterns are hard for vectorized
  hardware (CPU SIMD, GPU tensor cores) to exploit efficiently regardless
  of format.
- **Structured sparsity** (pruning whole neurons, channels, or blocks
  instead of individual weights) — which a normal dense kernel can
  exploit directly, because the result is just a *smaller* dense matrix,
  no special kernel required at all.

Neither is implemented here (explicitly out of scope, see project brief).
The contribution of this repo is the *correctness* infrastructure
(mask-as-graph-node, mask-aware Adam, exact-budget scheduling) and honest
accounting of what pruning has and hasn't bought — not a speed demo.

## 4. What it would take to actually serve this

Two separate benefits get conflated when people say "pruning helps
deployment," and they have different requirements:

- **Memory.** Active-parameter count is a real, achievable win
  *independent of any speedup* — store only the active weights (e.g. COO
  or CSR format) and the model genuinely takes less memory at 90%+
  sparsity. This works today, with the masks already produced by this
  repo, with zero additional engineering.
- **Compute/latency.** This needs the sparse-kernel or structured-sparsity
  work described in §3, neither of which exists here. Unstructured
  sparsity at the levels this project reaches (90-95%) is usually *not*
  enough on its own to win on commodity dense hardware — the practical
  threshold where naive sparse formats start beating dense compute is
  often higher (sometimes 95-98%+, and pattern-dependent), and even then
  requires a runtime that actually has a sparse code path.

Given the Pareto sweep's actual result — saliency pruning stays accurate
*and stable* across seeds even at 95% sparsity (`results/CLAIM.md`) — the
natural next step toward something deployable isn't a sparse kernel for
this specific unstructured mask. It's converting the criterion/schedule to
score and prune *structured* units: e.g. if every incoming (or outgoing)
connection to a hidden unit in the 128-wide layers ends up pruned, that
unit can be dropped entirely and the layer's actual matrix dimensions
shrink — a regular dense matmul on a smaller matrix, no new kernel needed.
That's a concrete, scoped extension; it's not implemented here because the
locked scope for this project is correctness and honest accounting, not a
serving pipeline.
