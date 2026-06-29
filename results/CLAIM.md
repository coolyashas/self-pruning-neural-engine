# The falsifiable claim

**Claim:** at high sparsity on this spirals task, gradient-informed
saliency pruning (`|w*g|`) beats magnitude-only pruning (`|w|`) on both
mean accuracy and seed-to-seed stability. Specifically — **at 95%
sparsity, saliency retains 99.40% ± 0.11% accuracy versus 98.62% ± 0.45%
for magnitude, averaged across 5 seeds** — a 0.78-point gap that sits
well outside seed noise. Magnitude occasionally cuts a small-magnitude
but high-gradient connection that actually matters; saliency scores
importance using the gradient, not just the weight's size, so it doesn't.

## Setup

`train/run_pareto_sweep.py`: one self-pruning run (`run_part3`) per
(criterion, target sparsity, seed). 2-128-128-3 MLP, Adam lr=1e-3, 100
epochs, batch size 32, 900-point spirals (noise 0.2), cubic sparsity
schedule (prune from step 400 to 80% of training, every 20 steps),
grad-clip 1.0, seeds 0-4 — identical for both criteria. Numbers below come
straight from `pareto_summary.csv` / `pareto_raw.csv`.

## Evidence

| criterion | sparsity | mean accuracy | std accuracy | worst seed |
|-----------|----------|---------------|--------------|------------|
| magnitude | 0.90     | 0.9940        | 0.0013       | 0.9933 |
| saliency  | 0.90     | 0.9971        | 0.0005       | 0.9967 |
| magnitude | 0.95     | 0.9862        | 0.0045       | 0.9789 (seed 3) |
| saliency  | 0.95     | 0.9940        | 0.0011       | 0.9922 (seed 2) |

Below 90% sparsity the two criteria are statistically indistinguishable
(both ~99.7%, std < 0.001 at every level up to and including 0.75 — see
`pareto_summary.csv`). The task is heavily overparameterized for a
2-128-128-3 MLP, so the criteria only diverge once pruning is aggressive
enough to actually bite (>=90%).

### Is the difference real or noise?

At 95% sparsity the 0.78-point gap is ~1.7x magnitude's own seed std
(0.45 pt) and ~7x saliency's (0.11 pt). A two-sample (Welch) t-test on the
five-seed accuracies gives t ≈ 3.4, and the two seed distributions barely
overlap: saliency's *worst* seed (99.22%) ties magnitude's *best*
(99.22%). So the gap is not seed noise. Saliency is also markedly more
stable — its std at 95% is ~4x smaller than magnitude's, because magnitude
has heavy-tailed bad seeds (its worst seed drops to 97.89%) while
saliency's five seeds stay within a 0.33-point band. At 90% the smaller
0.31-point gap (99.71% vs 99.40%) likewise exceeds the seed spread.

## How to falsify this claim

Re-run:

```bash
python -m train.run_pareto_sweep
```

Every (criterion, sparsity, seed) combination calls `utils.seed.set_seed`
before training, so this reproduces `pareto_summary.csv` deterministically
(modulo platform-level floating-point differences, which should be
negligible at these tolerances). If a re-run shows magnitude's mean
accuracy at 95% sparsity landing within saliency's mean ± seed noise — i.e.
the 0.78-point gap collapses — this claim is refuted.

## Scope

This is one task (3-class 2D spirals), one architecture (2-128-128-3
MLP), one optimizer (Adam), and one sparsity schedule (cubic, 5 seeds).
It is not a general claim that saliency always beats magnitude pruning —
a larger or harder task, a different architecture, or a different
schedule could behave differently.
