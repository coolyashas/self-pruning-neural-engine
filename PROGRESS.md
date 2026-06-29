# Progress

Tracks the commit plan for the AQUA self-pruning network (originally 27
commits, since extended through 54). One row per commit. Mark `[x]` only
once the commit is actually made (code + tests passing + committed), not
when merely started. The table is split into sections; each section
repeats the header row so every fragment renders as its own table.


| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 1 | chore: repo scaffold | [x] | `5be28ad` |
| 2 | feat(engine): Tensor wrapping ndarray, parent tracking, grad storage | [x] | `b83d4ff` |
| 3 | feat(engine): elementwise add/sub/mul/div with unbroadcast backward | [x] | `cb3f42c`. Forward + broadcast unbroadcast + shared-node accumulation verified vs finite-diff |
| 4 | feat(engine): matmul + sum/mean reductions | [x] | `c9e2ae5`. Matmul + sum/mean (axis/keepdims) verified vs finite-diff |
| 5 | feat(engine): backward() with topo-sort + grad accumulation | [x] | `bf362a1`. Full pipeline, diamond accumulation, and repeated backward() verified |
| 6 | test(engine): finite-difference gradient checks for all ops | [x] | `tests/test_engine_ops.py`, 23 tests across all ops + broadcast/skip cases |
| 7 | feat(engine): ReLU + tanh | [x] | `engine/activations.py` + gradcheck (26 total) |
| 8 | feat(engine): fused numerically-stable softmax cross-entropy | [x] | `engine/loss.py`; verified vs naive reference + overflow stress test |
| 9 | test(engine): grad-check softmax-CE + broadcasting bias case | [x] | `tests/test_softmax_ce.py`: gradcheck + large-logit stress + bias-broadcast pipeline (31 total) |
| 10 | feat(nn): Linear layer + He init | [x] | `nn/linear.py`; end-to-end gradcheck + He-init std check (35 total) |
| 11 | feat(nn): Sequential MLP container | [x] | `nn/sequential.py` + ReLU module; full MLP gradcheck (38 total) |
| 12 | feat(optim): SGD-with-momentum | [x] | `optim/sgd.py`; velocity recursion verified + real loop reduces loss (43 total) |
| 13 | feat(optim): Adam from scratch (bias-corrected) | [x] | `optim/adam.py`: single shared `t` (revival trade-off flagged for commit 17). t=1 + multi-step tests (48 total) |
| 14 | feat(train): mini-batch loop + spirals dataset | [x] | `train/dataset.py`, `train/loop.py`. Real 2-128-128-3 run: 99.7% accuracy (50 total) |
| 15 | feat(train): learning-curve logging + Part-2 reproduce script | [x] | `on_epoch_end` callback; `run_part2.py` reproduces 99.7%. Artifacts untracked until commit 25 (52 total) |
| 16 | feat(prune): boolean mask as graph node; w_eff = w * mask | [x] | Mask baked into Linear (`w_eff = weight*mask`), reusing `mul()`'s backward — zero new engine code (57 total) |
| 17 | feat(optim): mask-aware Adam — skip masked m/v/w, reset on revive | [x] | Critical-correctness commit. `Adam` gained optional `masks` + `reset_state`; added `masked_parameters()` |
| 18 | test(prune): masked weight stays 0, moments don't drift, revive resets m/v | [x] | Most important test in the repo. Frozen weight/moments + revive vs hand-derived step + bug-contrast test (63 total) |
| 19 | feat(prune): magnitude criterion (baseline) | [x] | `magnitude_scores` + `keep_mask_from_scores` (exact-budget top-k). Sparsity-count + tie tests (68 total) |
| 20 | feat(prune): \|w·g\| saliency criterion with batch accumulation | [x] | `saliency_scores` + `accumulate_gradients`. Key test: saliency and magnitude can rank oppositely (74 total) |
| 21 | feat(prune): cubic sparsity schedule + hard budget enforcement | [x] | `cubic_sparsity` (Zhu & Gupta 2017) + `prune_to_sparsity` (monotonic, never revives) (82 total) |
| 22 | feat(train): Part-3 self-pruning run to target sparsity + report | [x] | `run_part3.py`: 90% sparsity, 99.78% accuracy. Fixed a rounding-overshoot bug in `prune_to_sparsity` (85 total) |
| 23 | feat(eval): active-param/FLOP counter + honest dense×mask cost note | [x] | New `evaluation/cost.py`. Measured: dense×mask gives no speedup (ratio ≈0.93) (91 total) |
| 24 | feat(eval): Pareto sweep + magnitude-vs-saliency, N seeds, mean±std | [x] | `run_pareto_sweep.py`. At 95% magnitude destabilizes (std≈0.030), saliency stays flat (95 total) |
| 25 | docs: commit plots + raw CSVs; falsifiable claim | [x] | Committed `results/` artifacts + `CLAIM.md` (saliency stability vs magnitude instability at 95%) |
| 26 | docs: DESIGN.md | [x] | Criterion derivation, mask-aware-Adam design, no-speedup bottleneck, serving |
| 27 | docs: README run commands, uv setup | [x] | Run commands + layout + uv/venv setup. Re-runs reproduced identical numbers. All 27 commits done |

Commits 28-39 close two gaps found in a later re-check against the original spec: a real (not just disclosed) cost measurement via structured/neuron-level pruning + compressed inference, and full dynamic sparse training (regrowth).

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 28 | feat(prune): structured (neuron-level) magnitude/saliency criteria | [x] | `neuron_magnitude_scores`/`neuron_saliency_scores` (axis=0 per column). Non-square + rank-disagreement tests (99 total) |
| 29 | feat(prune): structured masking — prune_neurons_to_count | [x] | `keep_neurons_from_scores`/`prune_neurons_to_count` (column-wise `-inf`). Exact-budget + never-revives tests (105 total) |
| 30 | feat(prune): compress_model — cross-layer dense compression | [x] | New `prune/compress.py`: inference-only `Compressed*` classes threading alive indices across layers. Exact vs masked dense, no aliasing (112 total) |
| 31 | feat(eval): wall-clock benchmark proving structured compression is actually faster | [x] | `time_dense_vs_compressed_forward`: ~4x at 75% structured sparsity. Real Part-4 cost evidence (113 total) |
| 32 | feat(train): Part-4 structured self-pruning run + structured-vs-unstructured comparison | [x] | New structured run + comparison scripts. Fixed a bias-drift bug via `bias_mask` on Linear. Structured/unstructured indistinguishable up to 85% (122 total) |

Commits 33-38 build dynamic sparse training (regrowth), the second gap from the same re-check: a `revive_to_count` primitive existed only conceptually until now -- these commits make it real and wire it into an actual training run.

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 33 | feat(nn): store w_eff on Linear for external gradient inspection | [x] | `w_eff` now `self.w_eff`, so its `.grad` (dense signal for regrowth) survives. Test: nonzero where `weight.grad` is 0 (123 total) |
| 34 | feat(prune): accumulate_dense_gradients — multi-batch w_eff.grad accumulation | [x] | Explicitly sums `layer.w_eff.grad` each batch (rebuilt fresh per forward, so won't self-accumulate) (128 total) |
| 35 | feat(prune): revive_to_count — mask regrowth primitive | [x] | Inverted `-inf` trick; returns revived indices, doesn't touch optimizer/weights. Adversarial + budget + clamping tests (135 total) |
| 36 | feat(prune): run_exchange_cycle — grow+drop, net-zero active count | [x] | New `prune/dst.py`. Caught a pre-commit bug: drop needs three-way scoring (revived +inf / inactive -inf / active real). 5-cycle stability test (139 total) |
| 37 | feat(prune): dst_step — ramp-then-maintain scheduling | [x] | Ramps like run_part3, then periodic `run_exchange_cycle`. Byte-exact ramp cross-check + constant active count (141 total) |
| 38 | feat(train): enable_regrowth opt-in + DST comparison run | [x] | `run_part3` opt-in (default off, byte-identical when omitted) + comparison run. Regrowth reduced accuracy here (90%: 99.67% vs 94.89%) (146 total) |

Commits 39+ fix concrete, executed-and-verified issues found in a hostile "Staff AI Platform Review" pass over the whole repo (re-derived every op's math by hand, ran adversarial repros against the actual code rather than trusting comments/tests/docs). Each row cites the exact repro that proved the issue real before it was fixed.

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 39 | fix(prune): remove redundant gradient sweep in dst_step maintenance phase | [x] | Maintenance ran two equivalent sweeps; verified `weight.grad` identical, removed the waste. Regression test forbids it (147 total) |
| 40 | fix(prune): guard run_exchange_cycle no-op when a layer is fully dead | [x] | At 0 active, top-k(n_keep=0) drops even a `+inf` must-survive entry. Fixed with an early return (148 total) |
| 41 | fix(prune): reset bias Adam state when revival un-freezes a dead neuron's bias | [x] | Most serious finding: reviving into a dead column flipped `bias_mask` active but never reset bias Adam state, so it resumed on stale momentum. Fixed in `run_exchange_cycle` (149 total) |
| 42 | fix(docs): correct the direction of the Adam revival bias-correction trade-off | [x] | DESIGN.md had the direction backwards: stale `t` makes the first revival step ~3x LARGER, not smaller. Code always correct; doc fixed + pinning test (150 total) |
| 44 | fix(numerical): reject NaN scores in top-k pruning/revival + reject out-of-range labels | [x] | (1) argsort silently kept NaN scores — guarded the shared chokepoint. (2) softmax-CE accepted negative labels — added range check (155 total) |
| -- | fix(misc): document div()'s numerical-stability gap + guard compress_model against an empty model | [x] | Documented `div`'s near-zero behavior; `compress_model` on a Linear-free model now raises clearly (156 total) |
| 45 | docs: DESIGN.md §3/§5/§6 + README — structured pruning, DST, weight-init, dependency-surface fixes | [x] | DESIGN.md gained He-vs-Xavier, rewritten §5 (structured pipeline, ~2.8x), §6 (DST). README fixed sklearn line + new run commands (156 total) |

Commits 46-48 fix three further issues found in a second adversarial review pass (same hostile, execution-verified standard as commits 39-45), surfaced via specific line-numbered code review rather than a fresh from-scratch audit.

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 46 | fix(prune): weight per-batch gradient contributions by batch size, not equally | [x] | Summed un-weighted batch-MEANs, so the size-4 last batch got 16x influence. Fixed by weighting each batch `n_batch/n`; buggy-assertion tests rewritten (160 total) |
| 47 | fix(engine): reject non-1D labels in softmax_cross_entropy | [x] | `(N,1)` labels silently broadcast into an `(n,n)` index → wrong loss, no error. Fixed with a shape assertion (160 total) |
| 48 | fix(prune): zero a neuron's bias in set_mask whenever ANY caller drives its column fully dead | [x] | Unstructured pruning froze `bias_mask` but never zeroed the value, so a dead neuron kept emitting `ReLU(0+bias)`. Moved zero-on-death into `set_mask` (160 total) |

Commits 49-50 fix a third round of review findings: the input/invariant guards added in commits 41/44/46-48 were all implemented with `assert`, which `python -O` strips entirely, silently re-enabling the exact wrong-answer paths they existed to prevent; a missing label-dtype check; and a generic-helper robustness gap.

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 49 | fix(robustness): convert input/invariant guards from assert to explicit raise | [x] | `assert` guards were stripped by `python -O`, reviving the wrong-answer paths. Converted all five to `if: raise` + added the missing label-dtype check. `-O`-subprocess tests (166 total) |
| 50 | fix(prune): skip parameters with no gradient this sweep instead of crashing on None | [x] | Accumulators used `p.grad` unconditionally; a disconnected parameter would crash. Guarded with `if p.grad is not None` (168 total) |

Commits 51-52 fix a fourth round of review findings: the engine-level 2D-only matmul and scalar-only no-arg-backward scope boundaries were still `assert`-based (same `-O`-stripping issue as commits 41/44/46-49, just in the core engine rather than prune/), and broadcasting/reduction gradcheck coverage was narrower than the review's full checklist.

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 51 | fix(engine): convert matmul's 2D-only and backward's scalar-only guards from assert to raise | [x] | Under `-O` a 3D matmul silently batched and non-scalar `backward()` silently seeded grad=1.0. Converted both to `if: raise` + `-O`-subprocess tests (217 total) |
| 52 | test(engine): broaden broadcasting/reduction gradcheck coverage to the full review checklist | [x] | Added missing broadcast shapes, composite chains, and multi-axis/3D reductions. Fixed a NEP 50 scalar-promotion bug in the gradcheck util (217 total) |

Commits 53-54 extend matmul to real batched matmul (previously an intentional 2D-only scope limit) and remove a leaked RuntimeWarning from a deliberately-pathological test, both requested directly rather than found by review.

| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 53 | feat(engine): batched matmul with NumPy-style batch broadcasting | [x] | Extended `matmul` to ndim>=2 with batch broadcasting (2D is the unchanged special case). New `_unbroadcast_batch`; backward uses `swapaxes(-1,-2)`. 7 finite-diff cases (224 total) |
| 54 | test(train): stop a real, expected RuntimeWarning from leaking out of the forced-divergence test | [x] | Wrapped the deliberate `inf-inf=nan` divergence in `pytest.warns` (asserted, not leaked or globally suppressed). Confirmed under `-W error::RuntimeWarning` (224 total) |

## Workflow per commit

1. Implement following the engine conventions captured in
   [DESIGN.md](DESIGN.md) (forward + backward + shape reasoning +
   numerical-stability notes for every op; explicit unbroadcasting; `+=`
   accumulation never overwrite; no destructive in-place mutation of
   parameters unless asked).
2. Write/extend tests for the feature (gradcheck where applicable) and run
   them — don't mark done on faith.
3. Commit, then update this file's status to `[x]` with the commit hash.
