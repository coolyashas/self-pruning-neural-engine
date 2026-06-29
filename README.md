# aqua-selfprune

A minimal **self-pruning neural network** built from scratch in pure Python +
NumPy. We hand-write a reverse-mode autodiff engine, a small over-parameterized
MLP, an Adam optimizer, and a training loop that prunes its own weights toward a
target sparsity on a 2D spirals dataset — then compares pruning criteria on a
sparsity↔accuracy Pareto curve.

No PyTorch / TensorFlow / JAX, and no scikit-learn either — every line of
autodiff, backward pass, optimizer, pruning, training loop, *and* the spirals
dataset itself (`train/dataset.py::make_spirals`, generated from scratch with
plain NumPy, not loaded from any library) is implemented by hand in this repo.
NumPy for numerics, Matplotlib for plots; that's the entire dependency surface
(see `pyproject.toml`).

## Install (uv)

```bash
uv venv
uv pip install -e ".[dev]"
```

If `uv` isn't available, plain venv + pip works identically:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Repository layout

| Dir           | Contents                                              |
|---------------|--------------------------------------------------------|
| `engine/`     | Tensor + ops + backward (autodiff core)               |
| `nn/`         | Linear, activations, MLP container, softmax-CE loss   |
| `optim/`      | SGD-momentum, Adam (mask-aware)                       |
| `prune/`      | importance criteria, schedule, masking                |
| `train/`      | training loop, spirals dataset, experiment runners    |
| `evaluation/` | active-param/FLOP cost accounting, Pareto sweep       |
| `tests/`      | gradient-check suite + masked-weight correctness test |
| `results/`    | plots + raw CSVs + the falsifiable claim              |
| `utils/`      | shared helpers (seeding)                              |

## Run commands

- **Gradient checks (full test suite, ~5s):**
  ```bash
  pytest
  ```
  Or just the finite-difference gradcheck files:
  ```bash
  pytest tests/test_engine_ops.py tests/test_activations.py tests/test_softmax_ce.py tests/test_linear.py tests/test_sequential.py
  ```
  The single most important test in the repo (masked weight stays exactly
  0, moments don't drift while masked, revive resets m/v cleanly):
  ```bash
  pytest tests/test_masked_adam.py -v
  ```

- **Part 2 — train the dense MLP** (2-128-128-3, He init, ReLU, Adam; logs
  a learning curve to `results/part2_learning_curve.{csv,png}`):
  ```bash
  python -m train.run_part2
  ```

- **Part 3 — self-pruning run** (cubic schedule + saliency criterion +
  mask-aware Adam, to 90% sparsity by default; logs loss/accuracy/sparsity
  to `results/part3_self_pruning.{csv,png}`):
  ```bash
  python -m train.run_part3
  ```

- **Part 4 — Pareto sweep** (magnitude vs. saliency, 5 seeds each, across
  sparsity levels 0–95%; ~2-3 minutes; writes `results/pareto_{raw,summary}.csv`
  and `results/pareto_sweep.png`):
  ```bash
  python -m train.run_pareto_sweep
  ```
  See [`results/CLAIM.md`](results/CLAIM.md) for the falsifiable claim
  this sweep's actual output supports, and how to falsify it by re-running
  the command above.

- **Extension — structured (neuron-level) self-pruning** (closes the "real
  cost measurement" gap: compresses to a genuinely smaller dense model,
  not just a masked one — see `DESIGN.md` §5 for the measured speedup):
  ```bash
  python -m train.run_part4_structured
  ```
  Structured-vs-unstructured accuracy comparison at matched sparsity
  (writes `results/structured_vs_unstructured.{csv,png}`):
  ```bash
  python -m train.run_structured_vs_unstructured
  ```

- **Dynamic sparse training (regrowth) comparison** (with-vs-without
  regrowth at matched final sparsity, 5 seeds; writes
  `results/dst_comparison.{csv,png}` — see `DESIGN.md` §6 for the
  measured result, which goes against the literature's usual finding on
  this small task):
  ```bash
  python -m train.run_dst_comparison
  ```

See [`DESIGN.md`](DESIGN.md) for the criterion derivation, the
masked-gradient/mask-aware-Adam design rationale, weight-init
justification, the honest cost/bottleneck story, the structured-pruning
speedup (measured, with a caveat about how the first measurement of it
overstated it), and the dynamic-sparse-training result.

## Reproducibility

All randomness is seeded via `utils.seed.set_seed`. Results reproduce from a
clean clone.
