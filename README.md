# aqua-selfprune

A minimal **self-pruning neural network** built from scratch in pure Python +
NumPy. We hand-write a reverse-mode autodiff engine, a small over-parameterized
MLP, an Adam optimizer, and a training loop that prunes its own weights toward a
target sparsity on a 2D spirals dataset — then compares pruning criteria on a
sparsity↔accuracy Pareto curve.

No PyTorch / TensorFlow / JAX. The autodiff, backward pass, optimizer, pruning,
and training loop are all implemented by hand. NumPy for numerics, Matplotlib
for plots, scikit-learn only ever for loading data (never for models or grads).

## Install (uv)

```bash
uv venv
uv pip install -e ".[dev]"
```

## Repository layout

| Dir        | Contents                                             |
|------------|------------------------------------------------------|
| `engine/`  | Tensor + ops + backward (autodiff core)              |
| `nn/`      | Linear, activations, MLP container, softmax-CE loss  |
| `optim/`   | SGD-momentum, Adam (mask-aware)                      |
| `prune/`   | importance criteria, schedule, masking               |
| `train/`   | training loop, spirals dataset, experiment runners   |
| `tests/`   | gradient-check suite + masked-weight correctness test|
| `results/` | plots + raw CSVs                                      |
| `utils/`   | shared helpers (seeding)                              |

## Run commands

_Filled in as the corresponding commits land._

- **Gradient checks:** _TODO_
- **Part 2 — train the dense MLP:** _TODO_
- **Part 3 — self-pruning run:** _TODO_
- **Pareto sweep (magnitude vs. saliency):** _TODO_

## Reproducibility

All randomness is seeded via `utils.seed.set_seed`. Results reproduce from a
clean clone.
