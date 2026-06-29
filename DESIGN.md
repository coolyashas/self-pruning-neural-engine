# DESIGN.md

This document answers four questions about the pruning engine, in order:

1. How the importance criterion is derived, and why it approximates the loss change from removing a connection (§1).
2. What the engine computes as the gradient of a masked weight, and why that is the right choice (§2).
3. Where the autodiff engine bottlenecks, and how to optimize it (§3).
4. How to serve a self-pruned model in a multi-tenant inference service at scale (§4).

Two additional design notes follow: weight initialization (§5) and a measured negative result for dynamic sparse training (§6).

The task throughout is 3-class classification on synthetic interleaved spirals (`train/dataset.py:make_spirals`, 2D inputs), and the model is a small MLP: `Linear(2, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 3)` (`train/run_part3.py`). It is heavily overparameterized for this task, which is why accuracies sit near ~99.7% and why the §3/§6 results read the way they do — keep that in mind whenever the text says "this task" or "the 128-wide hidden layers."

## 1. Importance criterion: `|w·g|`, not `|w|`

Pruning connection `i` sets `w_i → 0`, i.e. `Δw_i = -w_i`. A first-order Taylor expansion of the loss around the current weights gives:

```
ΔL ≈ g_i · Δw_i = -w_i · g_i
```

So the loss increase from removing connection `i` is `|w_i · g_i|` to first order. That is the saliency criterion (`prune/criteria.py:saliency_scores`): an estimate of how much removing the connection would hurt.

The first-order term vanishes at a true minimum, which is why scoring happens during training, not after. `ΔL ≈ -w_i·g_i` is only the leading term, and it goes to zero as `g_i → 0`. At an exact minimum every gradient vanishes, so saliency would be ~0 for every connection and the ranking would be noise. The criterion is therefore scored on a gradient accumulated from a model that is still moving (`accumulate_gradients`), never on a converged one. The converged regime is exactly where the second-order curvature term `½·w_i²·H_ii` of OBD/OBS earns its cost, since it does not vanish when the gradient does. This engine stays first-order on purpose: pruning runs during training where the gradient signal is live, so Hessian-vector products are not worth their cost here.

Magnitude (`|w_i|` alone) is the same expression with `g_i` assumed equal across all connections — saliency with the gradient term discarded. That makes it the baseline, not the main criterion. A large weight with a near-zero gradient is a good prune; magnitude can't see that, saliency can.

Two consequences show up directly in the code:

- A single mini-batch gradient is noisy, so saliency needs an accumulated gradient estimate over a representative data sweep (`accumulate_gradients`) before scoring. Using one batch's gradient would make the criterion itself noisy.
- The criteria are genuinely different, not differently scaled: a connection can rank highest on magnitude and lowest on saliency at once (`test_saliency_and_magnitude_can_disagree`), and the Pareto sweep shows the divergence matters in practice (`results/CLAIM.md`).

### Structured saliency: sum signed terms, then take abs

Removing an output *neuron* zeros its whole incoming column at once. The same Taylor argument applied to the group gives `ΔL ≈ -Σ_i w_i·g_i` over the column, so the loss increase is `|Σ_i w_i·g_i|` — sum the signed per-connection saliencies, then take the absolute value (`prune/criteria.py:neuron_saliency_scores`).

This is deliberately not `Σ_i |w_i·g_i|` (abs-then-sum). By the triangle inequality that L1 form upper-bounds the real estimate and equals it only when every connection in the column shares a sign. The two diverge when a column's contributions cancel, and that cancellation is real to first order: a neuron whose incoming saliencies sum to zero has near-zero net effect when removed, so it should score low — which abs-then-sum would wrongly rank high (`test_neuron_saliency_sums_signed_then_abs_not_abs_then_sum`). Squaring instead of abs (`(Σ w_i·g_i)²`, Molchanov et al. 2019) gives an identical ranking.

## 2. The gradient of a masked weight

Pruning is implemented as `w_eff = weight * mask`, a multiplication node in the autodiff graph (`nn/linear.py`), not a post-hoc edit to a computed gradient. This needed no new engine code: `mul()`'s backward already computes `grad_a = grad_out * b.data`. With `a = weight`, `b = mask`, and `mask.requires_grad = False`, that is exactly:

```
dL/dweight = dL/dw_eff * mask
```

So the gradient of a masked weight is the upstream gradient times its mask entry — zero for pruned connections, unchanged for active ones. This is the right choice because it falls out of the chain rule rather than being patched in afterward. A post-hoc gradient patch is the kind of fix that breaks the moment the graph is extended (weight decay, a different loss, weight reuse) and nobody re-applies it. Here the property holds automatically for any graph built on the same `mul` node.

### Mask-aware Adam: zero gradient is not enough

Zero gradient does not imply a frozen optimizer state. Adam's moments are exponential moving averages:

```
m = β1·m + (1-β1)·g
v = β2·v + (1-β2)·g²
```

If `g = 0`, this still computes `m = β1·m`, `v = β2·v` — geometric decay, not freezing. A pruned weight's leftover momentum keeps nudging it for several steps until `m` decays to zero numerically. `optim/adam.py` instead gates the entire update — `m`, `v`, and `w` — to active entries via the layer's mask, explicitly rather than inferring it from `g = 0`. Verified directly: a masked entry's `m`/`v` stay bitwise unchanged across many steps under a constant nonzero incoming gradient (`tests/test_masked_adam.py`).

### Revival

`reset_state(param, indices)` zeros `m` and `v` for entries that flip from masked back to active. Without it, a revived connection's first update uses momentum from a different training regime (`test_without_reset_state_revival_inherits_stale_momentum`), producing a measurably wrong first step.

One known trade-off: revival gives an oversized first step, not a conservative one. Adam here uses a single shared step counter `t` for bias correction, not per-element. On revival, freshly-zeroed `m, v` are bias-corrected against the current `t`, not a true `t=1`. Because `β1=0.9`'s correction saturates almost immediately (`m_hat ≈ m` by `t≈50`) while `β2=0.999`'s saturates ~10x slower, at a stale large `t` the `v_hat` term is under-corrected (too small), and since it sits under a square root in the denominator the update grows rather than shrinks:

```
t=     1   update/lr = 1.0000   (a true fresh Adam step)
t=    50   update/lr = 0.7021
t=  2000   update/lr = 2.9407   (this project's real step counts land here)
t= 20000   update/lr = 3.1623   (asymptote: (1-β1)/sqrt(1-β2))
```

So a revived connection's first update is ~2–3x oversized at realistic `t`. This is a deliberate simplification that avoids per-element step bookkeeping the mask would also have to gate. It does not violate the correctness bar — masked `m`/`v` stay frozen, revived `m`/`v` stay zero — the cost is a bounded first-step overshoot, pinned by `test_revival_first_update_is_oversized_not_conservative_at_realistic_t`.

## 3. The bottleneck: unstructured pruning saves parameters, not time

`evaluation/cost.py` reports two numbers on purpose: `active_param_count` (what a sparse implementation would store) and `model_flops(..., active_only=False)` (what this implementation actually computes). They diverge as soon as anything is pruned. `x @ (weight * mask)` is a normal dense matmul, and multiplying by zero is still a multiply: NumPy and the dense BLAS underneath it have no way to know an entry is zero and skip it.

This is measured, not assumed. `test_dense_matmul_speed_unaffected_by_sparsity` times a 0%-sparse and a 90%-sparse layer of identical shape and finds no speedup (ratio ≈0.93, within timing noise). The bottleneck for a real speedup is not the pruning logic; it is that dense kernels never skip zeros. Wall-clock benefit needs one of:

- **A sparse matmul kernel** (CSR/CSC storage, sparse BLAS). These usually only pay off above ~90–95% sparsity, because sparse-format indexing overhead is non-trivial and unstructured per-weight patterns are hard for SIMD/tensor cores to exploit.
- **Structured sparsity** (whole neurons, channels, or blocks). A normal dense kernel exploits this directly, because the result is just a smaller dense matrix.

The sparse-kernel route is left unbuilt. Structured sparsity is the optimization actually implemented, below.

### Optimization: structured pruning with a measured speedup

There are two distinct benefits, with different requirements:

- **Memory.** Active-parameter count is a real win independent of any speedup: store only active weights (COO/CSR) and the model takes less memory at high sparsity. This works today with the masks §1 produces.
- **Compute/latency.** Unstructured masking can't deliver this. Structured pruning can, because dropping a whole output column shrinks the matrix itself, and a dense matmul on a smaller matrix does less work — no sparse kernel.

`neuron_magnitude_scores`/`neuron_saliency_scores` score whole output columns (`axis=0` of `weight`, since `weight.shape == (in_features, out_features)`); `prune/mask.py:prune_neurons_to_count` zeros entire columns and their bias entries by the same `-inf`-forces-never-revive top-k trick as `prune_to_sparsity`, at column granularity. `prune/compress.py:compress_model` then builds a genuinely smaller dense model: a layer's dead output columns make the next layer's corresponding input rows dead too (`ReLU(0) == 0` passes deadness through), so alive-neuron index sets thread layer to layer, row- and column-slicing real NumPy arrays. No `Tensor`, no mask multiply left to skip.

The speedup is real, but the first measurement overstated it (caught in review). Comparing `model(Tensor(x))` against `compressed(x)` conflates genuine FLOP reduction with the autodiff bookkeeping that only the Tensor path pays. Decomposed, about 28% of the original ~4x ratio was autodiff overhead, not matrix size. The apples-to-apples number — plain NumPy at full size vs. plain NumPy at compressed size, `evaluation/cost.py:time_dense_numpy_vs_compressed_forward` — is **2.8x** at 75% structured sparsity on the 128-wide hidden layers. This is the same discipline this section holds unstructured masking to, applied a level deeper: a benchmark whose two paths differ in more than one way can overstate a real effect as easily as dense×mask can fake one.

Accuracy cost: a comparison sweep (`train/run_structured_vs_unstructured.py`, 3 sparsities × 3 seeds) found structured and unstructured saliency pruning statistically indistinguishable on this task (~99.7% both, std≈0.001) up to 85% sparsity. The expected "structured is coarser, so it costs more accuracy at matched sparsity" penalty did not appear, consistent with this task's overparameterization (`results/CLAIM.md`).

## 4. Serving a self-pruned model in a multi-tenant service at scale

At serve time the masks are frozen, so the question is purely how sparsity becomes cheaper inference when one fleet hosts many tenants' models at thousands of requests per second.

**Ship the compressed model, not the masked one.** §3 is the binding constraint: `x @ (weight * mask)` is a dense GEMM that never skips a zero, so unstructured masks buy nothing on the latency path, and per-weight sparsity is hostile to the batched GEMMs and tensor cores serving throughput depends on. The serving artifact is the output of `prune/compress.py:compress_model`: a smaller dense network (§3's measured ~2.8x), plain matmuls, no mask multiply, no Tensor overhead. This pushes the trade-off toward structured pruning for anything served, even where unstructured reaches slightly higher accuracy at matched sparsity, because a smaller dense matrix is the only "sparsity" a batched serving kernel can exploit.

**Two wins, provisioned separately.** Memory (store only active params) decides how many tenant variants fit resident per node — the lever for density and cold-start avoidance. Compute (smaller GEMM) decides per-request latency and throughput. They scale independently: memory packs more tenants per box, compression makes each one run faster.

**Batching across tenants is the real problem.** Tenants pruned to different sparsities produce different matrix shapes, and a batched GEMM needs one shape. Three routes, increasing in effort:

- **Route by variant.** Group concurrent requests for the same pruned model and continuous-batch within each variant; keep hot variants resident and page cold ones.
- **Bucket to canonical sparsities** (e.g. 50/75/90%) so many tenants share one shape and one GEMM.
- **Pad to a common width** when latency SLOs forbid waiting to fill a per-variant batch, trading some FLOP win for batch occupancy.

Which one wins depends on request mix and tail-latency targets, and should be measured per deployment.

**What carries over from training.** The mask-aware-Adam and revival work (§2) is entirely training-side with no serving cost — `compress_model` bakes the final mask into dense arrays once, offline. The accounting discipline (§3: never quote a speedup from multiply-by-zero, never quote a benchmark that conflates two effects) is exactly what a capacity-planning number needs: the figure that decides how many GPUs to buy must be the apples-to-apples one (`time_dense_numpy_vs_compressed_forward`), not the flattering one.

## 5. Note: weight initialization (He, not Xavier)

`nn/linear.py` initializes each `Linear` weight with `std = sqrt(2 / in_features)`. The factor of 2 over Xavier/Glorot's `sqrt(1 / in_features)` exists because this network uses ReLU. ReLU zeros roughly half its inputs, which roughly halves the variance of the activations it produces relative to its pre-activation input. He init's factor of 2 cancels that halving, so activation variance stays roughly constant across layers instead of shrinking geometrically with depth (which would starve deeper layers' gradients). Xavier's derivation assumes a roughly linear or symmetric activation (e.g. tanh near 0) where no halving happens; using it under ReLU under-scales the weights. With two hidden layers the effect is mild, but the reasoning holds regardless of depth.

## 6. Note: dynamic sparse training (regrowth) — a measured negative result

The `w_eff = weight * mask` graph node (§2) has a second use: `w_eff.grad`, populated by `matmul`'s backward, is *not* mask-gated. Masking happens one step later, inside `mul()`'s backward (`weight.grad += w_eff.grad * mask.data`, `engine/ops.py:mul`). So `w_eff.grad` already carries the dense, unmasked "how much would this matter if active" signal for every entry, including ones whose `weight.grad` is exactly 0 — the same first-order Taylor logic as `saliency_scores`, with no new engine math. `prune/criteria.py:accumulate_dense_gradients` sums this across a sweep (it needs its own accumulation, because `w_eff` is a fresh `Tensor` each forward call and its `.grad` does not accumulate the way a persistent `Tensor`'s does).

`prune/mask.py:revive_to_count` mirrors `prune_to_sparsity`'s `-inf` trick inverted: active entries forced to `-inf` so top-k can only select among masked-off ones. `prune/dst.py:run_exchange_cycle` composes it with a drop step into a net-zero grow+drop cycle. The drop step's scoring needs three buckets, not two: just-revived entries forced to `+inf` (must survive — they haven't proven themselves), still-inactive entries forced to `-inf` (drop is not a revival path), and untouched active entries scored normally. Collapsing the first two into one `+inf` bucket produces wrong masks whenever excluded entries outnumber the keep budget, since top-k can no longer tell "force in" from "force out" (`tests/test_dst.py::test_run_exchange_cycle_excludes_just_revived_from_drop`).

Result: regrowth substantially hurt accuracy and stability on this task, the opposite of the literature's usual finding. `train/run_dst_comparison.py` (5 seeds, otherwise identical to the non-regrowth run):

| sparsity | without regrowth | with regrowth (DST) |
|---|---|---|
| 90% | 99.67% ± 0.07% | 94.89% ± 4.68% |
| 95% | 99.53% ± 0.19% | 91.02% ± 11.59% (worst seed: 68.3%) |

Plausible explanation, not independently verified: plain saliency pruning already converges to a near-optimal fixed mask quickly on this small, overparameterized task (§3, `results/CLAIM.md`), so continuously disturbing that mask via exchange cycles adds churn the small amount of training between cycles can't recover from — the opposite of DST's usual regime (a large/hard task where the early fixed mask is itself suboptimal). Reported as measured.

One narrow correctness gap lives at the boundary with §2's mask-aware Adam: `set_mask` resyncs `bias_mask` from `mask.any(axis=0)` on every call, so reviving a single weight into a fully-dead column un-freezes that neuron's bias — but the revival path resets only the weight's Adam state, so the bias would otherwise resume with stale pre-death momentum (§2's failure mode, for bias). `run_exchange_cycle` snapshots `bias_mask` before reviving and resets state for any index that flips 0→1.
