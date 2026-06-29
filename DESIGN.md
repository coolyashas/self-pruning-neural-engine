# DESIGN.md

Six things worth defending in detail: where the saliency criterion comes
from, why the masked-gradient/mask-aware-Adam design is built the way it
is, why this network is initialized the way it is, what unstructured
pruning here actually costs (and doesn't save), how structured
(neuron-level) pruning closes that gap with a real measured speedup, and
what dynamic sparse training (regrowth) actually buys — measured, not
assumed.

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
  in a contrived test (see `results/CLAIM.md`).

**Structured (neuron-level) saliency: sum the signed terms, then abs.**
Removing an output *neuron* means zeroing its whole incoming column at
once, not one connection at a time. The same first-order Taylor argument
applied to the group gives `ΔL ≈ -Σ_i w_i·g_i` over the column, so the
loss increase is `|Σ_i w_i·g_i|` — sum the *signed* per-connection
saliencies, then take the absolute value
(`prune/criteria.py:neuron_saliency_scores`). This is deliberately *not*
`Σ_i |w_i·g_i|` (abs-then-sum): by the triangle inequality that L1 norm
upper-bounds the real estimate and equals it only when every connection
in the column shares a sign. The two diverge exactly when a column's
contributions cancel — and that cancellation is real to first order: a
neuron whose incoming saliencies sum to zero has near-zero net effect on
the loss when removed, so it *should* score low, which abs-then-sum would
wrongly rank high (`test_neuron_saliency_sums_signed_then_abs_not_abs_then_sum`).
Squaring instead of taking the absolute value (`(Σ w_i·g_i)²`, Molchanov
et al. 2019) gives an identical ranking.

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

**An explicit trade-off, not an oversight — and the direction is an
oversized step, not a conservative one.** Adam here uses one *shared*
step counter `t` for bias correction, not a per-parameter or per-element
one. On revival, a freshly-zeroed `m, v` gets bias-corrected against
whatever `t` the rest of training has reached, not against a true `t=1`.
An earlier version of this note claimed that since `(1-β^t) ≈ 1` at large
`t`, this "gives almost no bias-correction boost" — i.e. a conservative,
under-sized first update. Measuring it directly (not just reasoning about
the boost factor in isolation) shows the opposite:

```
t=     1  update/lr = 1.0000   <- a true fresh Adam step
t=    50  update/lr = 0.7021
t=   500  update/lr = 1.9840
t=  2000  update/lr = 2.9407   <- this project's actual step counts land here
t= 20000  update/lr = 3.1623   <- asymptotes to (1-β1)/sqrt(1-β2)
```

The reason: `β1=0.9`'s correction saturates almost immediately (`m_hat ≈
m` by `t≈50`), while `β2=0.999`'s saturates roughly 10x slower. At a
stale, large `t`, `m_hat` gets no boost but `v_hat` is *still*
under-corrected (too small) — and since `v_hat` sits under a square root
in the denominator, an under-corrected `v_hat` makes the update *larger*,
not smaller. At the `t` values this project's real runs actually reach
(hundreds to thousands of steps before any revival happens), a revived
connection's first update is **roughly 2–3x oversized** relative to a
true fresh Adam step, not conservative. This still does not violate the
stated correctness bar (masked `m`/`v` exactly frozen; revived `m`/`v`
exactly zero) — it's a deliberate simplification that avoids per-element
step bookkeeping that masking would *also* need to gate — but the
practical consequence is a bounded overshoot on the first post-revival
step, not a muted one. `tests/test_masked_adam.py::test_revival_first_update_is_oversized_not_conservative_at_realistic_t`
pins the measured ratio so this claim can't silently drift from the code
again.

## 3. Weight initialization: why He, not Xavier

`nn/linear.py` initializes each `Linear`'s weight with
`std = sqrt(2 / in_features)`. The factor of 2 (vs. Xavier/Glorot's
`sqrt(1 / in_features)`) is specifically because this network uses ReLU,
not because it's a generically "better" default. ReLU zeros out roughly
half its inputs (everything below 0), which roughly halves the variance
of the activations it produces relative to its pre-activation input. He
initialization's extra factor of 2 is chosen to exactly cancel that
halving, so that activation variance stays roughly constant layer to
layer instead of shrinking geometrically with depth (which would push
deeper layers' activations toward 0 and starve their gradients). Xavier's
derivation assumes a roughly linear or symmetric activation (e.g. tanh
near 0) where no such halving happens — using it under ReLU would
under-scale the weights and is a known, common mismatch. With only two
hidden layers here the effect is mild, but the reasoning is the same
regardless of depth, and it's the kind of detail worth being able to
justify on demand rather than copying a "standard" initializer without
checking it matches the activation actually in use.

## 4. The bottleneck: unstructured pruning here saves parameters, not time

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

## 5. Structured (neuron-level) pruning: closing the "real speedup" gap, with a real measured number

Two separate benefits get conflated when people say "pruning helps
deployment," and they have different requirements:

- **Memory.** Active-parameter count is a real, achievable win
  *independent of any speedup* — store only the active weights (e.g. COO
  or CSR format) and the model genuinely takes less memory at 90%+
  sparsity. This works today, with the masks §4 already produces, with
  zero additional engineering.
- **Compute/latency.** Unstructured masking can't deliver this (§4) — a
  dense matmul never skips a zero. Structured (neuron-level) pruning can,
  because dropping a whole output column genuinely shrinks the matrix
  itself, and a completely ordinary dense matmul on a smaller matrix does
  less work, no sparse kernel required.

`prune/criteria.py`'s `neuron_magnitude_scores`/`neuron_saliency_scores`
score whole output columns (`axis=0` of `weight`, since
`weight.shape == (in_features, out_features)`); `prune/mask.py`'s
`prune_neurons_to_count` zeros entire columns (and the corresponding bias
entries — see §2's bias_mask discussion) by the same `-inf`-forces-never-
revives top-k trick `prune_to_sparsity` already uses, just at column
granularity. `prune/compress.py::compress_model` then builds a genuinely
smaller dense model: each `Linear`'s dead output columns make the *next*
`Linear`'s corresponding input rows dead too (`ReLU(0) == 0` passes
deadness through unchanged), so alive-neuron index sets thread layer to
layer, row-slicing and column-slicing real NumPy arrays — no `Tensor`, no
mask multiply left to skip, because there's nothing left to skip.

**The speedup is real, but the first measurement of it overstated it** —
caught in review, not before: comparing `model(Tensor(x))` (the
Tensor-wrapped path, pays for autodiff bookkeeping that's never used in a
pure forward pass) against `compressed(x)` (plain NumPy, no such
bookkeeping) conflates genuine FLOP reduction with the unrelated cost of
the autodiff engine's own per-call overhead disappearing. Decomposing it
directly: of the originally-measured ~4x ratio, about 28% was autodiff
overhead, not matrix size. The fair, apples-to-apples number — plain
NumPy at full size vs. plain NumPy at compressed size, `evaluation/cost.py::time_dense_numpy_vs_compressed_forward`
— is **~2.8x** at 75% structured sparsity on the 128-wide hidden layers.
Still a real, substantial, measured win; just not the original number.
This is exactly the standard §4 holds unstructured masking to ("never
claim a speedup from multiplying by zero") applied a level deeper: a
benchmark comparing two code paths that differ in more than one way can
overstate a real effect just as easily as dense×mask can fake one
entirely.

Accuracy cost: a real comparison sweep (`train/run_structured_vs_unstructured.py`,
3 sparsities × 3 seeds) found structured and unstructured saliency
pruning statistically indistinguishable on this task (~99.7% both,
std≈0.001) up to 85% sparsity — the expected "structured is coarser, so
it should cost more accuracy at matched sparsity" penalty didn't show up,
consistent with this task's known overparameterization
(`results/CLAIM.md`). Reported as measured, not as the expected result.

## 6. Dynamic sparse training (regrowth): a real implementation, with a real negative result

The masking design's `w_eff = weight * mask` graph node (§2) has a second
use beyond making masked-gradient-is-zero fall out for free: `w_eff.grad`,
populated by `matmul`'s backward, is *not* mask-gated — masking only
happens one step later, inside `mul()`'s own backward
(`weight.grad += w_eff.grad * mask.data`, see `engine/ops.py::mul`). So
`w_eff.grad` already carries the dense, unmasked "how much would this
matter if it were active" signal for *every* entry, including ones whose
`weight.grad` is exactly 0 — the same first-order Taylor logic
`saliency_scores` already uses, with zero new engine math required.
`prune/criteria.py::accumulate_dense_gradients` sums this signal across a
sweep (it needs its own accumulation logic, unlike `weight.grad`: `w_eff`
is a fresh `Tensor` every forward call, so its `.grad` doesn't accumulate
across batches on its own the way a persistent `Tensor`'s does).

`prune/mask.py::revive_to_count` mirrors `prune_to_sparsity`'s `-inf`
trick inverted (already-active entries forced to `-inf` so top-k can only
select among masked-off ones — the grow-side analogue of "never
revives"). `prune/dst.py::run_exchange_cycle` composes it with a drop
step into a net-zero-active-count grow+drop cycle. **The drop step's
scoring needs three buckets, not two — a real bug caught before commit,
not after a failing test**: a first draft lumped revived and
still-inactive entries into a single "excluded → `+inf`" bucket. That's
backwards for the still-inactive half: it should be `-inf` (never
reactivate via the drop step — that's not a revival path), while only the
just-revived entries should be `+inf` (must survive this cycle, since
they haven't had a chance to prove themselves yet). Lumping both at
`+inf` silently breaks the moment there are more excluded entries than
the keep budget, since top-k then can't distinguish "force this in" from
"force this out" — wrong masks, not a crash. Caught by manually tracing
the failure mode against a constructed adversarial case before writing
any test, then proven directly:
`tests/test_dst.py::test_run_exchange_cycle_excludes_just_revived_from_drop`.

**Measured result: regrowth substantially hurts accuracy and stability on
this task, not the literature's usual finding.** `train/run_dst_comparison.py`
(5 seeds, 90% and 95% sparsity, otherwise identical to the non-regrowth
run) found:

| sparsity | without regrowth | with regrowth (DST) |
|---|---|---|
| 90% | 99.67% ± 0.07% | 94.89% ± 4.68% |
| 95% | 99.53% ± 0.19% | 91.02% ± 11.59% (worst seed: 68.3%) |

Plausible explanation, not independently verified: plain saliency pruning
already converges to a near-optimal fixed mask quickly on this small,
heavily-overparameterized task (§5/`results/CLAIM.md`), so continuously
disturbing that mask via exchange cycles adds churn the small amount of
training between cycles can't recover from — the opposite of the regime
DST is usually evaluated in (a large/hard task where the early fixed mask
is itself suboptimal and benefits from reallocation). Reported honestly
because it's what was measured, not what was expected going in.

A real, narrow correctness gap was found and fixed at the boundary
between this feature and §2's mask-aware Adam, exactly where the two
weren't jointly tested before: `set_mask` resyncs `bias_mask` from
`mask.any(axis=0)` on every call, so `revive_to_count` reviving a single
weight into a column that had been driven fully to zero silently
un-freezes that neuron's bias — but the revival path only ever reset the
*weight's* Adam state. Without an explicit check for this transition, the
bias would resume updating on the very next step using real, pre-death
momentum — the exact stale-optimizer-state failure mode §2 exists to
prevent, just rediscovered for bias instead of weight. Fixed in
`run_exchange_cycle` by snapshotting `bias_mask` before reviving and
resetting state for any index that flips 0→1 as a side effect.
