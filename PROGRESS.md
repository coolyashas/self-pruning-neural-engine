# Progress

Tracks the 27-commit plan for the AQUA self-pruning network. One row per
commit. Mark `[x]` only once the commit is actually made (code + tests
passing + committed), not when merely started.


| # | Commit | Status | Notes |
|---|--------|--------|-------|
| 1 | chore: repo scaffold | [x] | `5be28ad`; gap (missing seed docstring) closed in `53ce2d4` |
| 2 | feat(engine): Tensor wrapping ndarray, parent tracking, grad storage | [x] | `b83d4ff` |
| 3 | feat(engine): elementwise add/sub/mul/div with unbroadcast backward | [x] | `cb3f42c`; comments trimmed in `786f8f9`. Verified: forward, bias-broadcast unbroadcast, shared-node (`x+x`) accumulation, finite-diff cross-check |
| 4 | feat(engine): matmul + sum/mean reductions | [x] | `c9e2ae5`. Verified: matmul backward vs finite-diff, sum over axis=None/0/1 + keepdims, mean divide-by-N vs finite-diff, chained matmul->mean backward |
| 5 | feat(engine): backward() with topo-sort + grad accumulation | [x] | `bf362a1`. Verified: full multi-op pipeline vs finite-diff, diamond/shared-node accumulation, requires_grad=False branch (no crash, no spurious grad), repeated backward() accumulates |
| 6 | test(engine): finite-difference gradient checks for all ops | [x] | `tests/test_engine_ops.py`, 23 tests passing: add/sub/mul/div (plain + 2 broadcast shapes), matmul, sum/mean (axis+keepdims variants), shared-node accumulation, requires_grad=False skip, repeated-backward accumulation |
| 7 | feat(engine): ReLU + tanh | [ ] | |
| 8 | feat(engine): fused numerically-stable softmax cross-entropy | [ ] | |
| 9 | test(engine): grad-check softmax-CE + broadcasting bias case | [ ] | |
| 10 | feat(nn): Linear layer + He init | [ ] | |
| 11 | feat(nn): Sequential MLP container | [ ] | |
| 12 | feat(optim): SGD-with-momentum | [ ] | |
| 13 | feat(optim): Adam from scratch (bias-corrected) | [ ] | |
| 14 | feat(train): mini-batch loop + spirals dataset | [ ] | |
| 15 | feat(train): learning-curve logging + Part-2 reproduce script | [ ] | |
| 16 | feat(prune): boolean mask as graph node; w_eff = w * mask | [ ] | |
| 17 | feat(optim): mask-aware Adam — skip masked m/v/w, reset on revive | [ ] | The critical-correctness commit |
| 18 | test(prune): masked weight stays 0, moments don't drift, revive resets m/v | [ ] | The single most important test in the repo |
| 19 | feat(prune): magnitude criterion (baseline) | [ ] | |
| 20 | feat(prune): \|w·g\| saliency criterion with batch accumulation | [ ] | |
| 21 | feat(prune): cubic sparsity schedule + hard budget enforcement | [ ] | |
| 22 | feat(train): Part-3 self-pruning run to target sparsity + report | [ ] | |
| 23 | feat(eval): active-param/FLOP counter + honest dense×mask cost note | [ ] | |
| 24 | feat(eval): Pareto sweep + magnitude-vs-saliency, N seeds, mean±std | [ ] | |
| 25 | docs: commit plots + raw CSVs; falsifiable claim | [ ] | |
| 26 | docs: DESIGN.md | [ ] | |
| 27 | docs: README run commands, uv setup | [ ] | |

## Workflow per commit

1. Check [Common_pittfall.md](Common_pittfall.md) for the section(s)
   relevant to the feature about to be built.
2. Implement following [CODING_GUIDELINES.md](CODING_GUIDELINES.md)
   (forward + backward + shape reasoning + numerical-stability notes for
   every op; explicit unbroadcasting; `+=` accumulation never overwrite;
   no destructive in-place mutation of parameters unless asked).
3. Write/extend tests for the feature (gradcheck where applicable) and run
   them — don't mark done on faith.
4. Commit, then update this file's status to `[x]` with the commit hash.
