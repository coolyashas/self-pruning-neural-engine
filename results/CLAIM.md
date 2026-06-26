# The falsifiable claim

**Claim:** at extreme sparsity (95% of weights pruned) on this spirals
task, gradient-informed saliency pruning (`|w*g|`) is more *stable*
across random seeds than magnitude-only pruning (`|w|`) — even though
their mean accuracy is close. Magnitude occasionally cuts a
small-magnitude but high-gradient connection that actually matters;
saliency doesn't, because it scores importance using the gradient, not
just the weight's size.

## Evidence

From `pareto_summary.csv` / `pareto_raw.csv` (2-128-128-3 MLP, Adam
lr=0.01, 200 epochs, cubic schedule, seeds 0-4, identical for both
criteria):

| criterion | sparsity | mean accuracy | std accuracy | worst seed |
|-----------|----------|----------------|---------------|------------|
| magnitude | 0.95     | 0.9716         | 0.0305        | 0.9211 (seed 3) |
| saliency  | 0.95     | 0.9969         | 0.0008        | 0.9956 (seed 1) |

Below 90% sparsity both criteria are statistically indistinguishable
(mean ~99.7%, std < 0.001 for every sparsity level up to and including
0.90 — see `pareto_summary.csv`). The task is heavily overparameterized
for a 2-128-128-3 MLP, so the criteria only diverge once pruning is
aggressive enough to actually bite.

## How to falsify this claim

Re-run:

```bash
python -m train.run_pareto_sweep
```

Every (criterion, sparsity, seed) combination calls `utils.seed.set_seed`
before training, so this should reproduce `pareto_summary.csv`
deterministically (modulo platform-level floating-point differences,
which should be negligible at these tolerances). If a re-run shows
magnitude's std at 95% sparsity is *not* meaningfully larger than
saliency's, this claim is refuted.

## Scope

This is one task (3-class 2D spirals), one architecture (2-128-128-3
MLP), one optimizer (Adam), and one sparsity schedule (cubic, 5 seeds).
It is not a general claim that saliency always beats magnitude pruning —
a larger or harder task, a different architecture, or a different
schedule could behave differently.
