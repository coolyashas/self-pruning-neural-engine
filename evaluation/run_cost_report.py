"""Commit a cost-measurement artifact for PART 4's cost claim, which otherwise
lived only in DESIGN.md prose and bound-checking tests. Per sparsity level it
writes active/total parameter counts, dense vs. theoretical-sparse FLOPs, and
the NumPy wall-clock speedup, using the 2-128-128-3 MLP from tests/test_cost.py.
"""

from __future__ import annotations

import csv
import os
import platform

import numpy as np

from evaluation.cost import (
    active_param_count,
    model_flops,
    time_dense_numpy_vs_compressed_forward,
    total_param_count,
    weight_sparsity,
)
from nn import Linear, ReLU, Sequential
from prune.compress import compress_model
from prune.criteria import neuron_magnitude_scores
from prune.mask import prune_neurons_to_count
from utils.seed import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
HIDDEN = 128


def measure(target_sparsity: float, batch_size: int = 64, repeats: int = 300) -> dict:
    """Build a structurally-pruned 2-128-128-3 MLP at the given hidden-neuron
    sparsity, then return its exact cost counts and measured speedup.
    """
    set_seed(0)
    mlp = Sequential(Linear(2, HIDDEN), ReLU(), Linear(HIDDEN, HIDDEN), ReLU(), Linear(HIDDEN, 3))
    target_active = round((1.0 - target_sparsity) * HIDDEN)
    for layer in (mlp.layers[0], mlp.layers[2]):
        prune_neurons_to_count(layer, neuron_magnitude_scores(layer), target_active)

    compressed = compress_model(mlp)
    x = np.random.randn(batch_size, 2)
    t_dense, t_compressed = time_dense_numpy_vs_compressed_forward(mlp, compressed, x, repeats=repeats)

    return {
        "target_sparsity": target_sparsity,
        "achieved_sparsity": round(weight_sparsity(mlp), 6),
        "active_params": active_param_count(mlp),
        "total_params": total_param_count(mlp),
        "dense_flops": model_flops(mlp, batch_size, active_only=False),
        "theoretical_sparse_flops": model_flops(mlp, batch_size, active_only=True),
        "dense_numpy_sec": round(t_dense, 6),
        "compressed_numpy_sec": round(t_compressed, 6),
        "speedup": round(t_dense / t_compressed, 3),
    }


def main(results_dir: str = RESULTS_DIR, sparsities=(0.5, 0.75, 0.9)) -> list[dict]:
    rows = [measure(s) for s in sparsities]

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "cost_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    env = f"{platform.platform()} | Python {platform.python_version()} | NumPy {np.__version__}"
    md_path = os.path.join(results_dir, "cost_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Cost report (PART 4 — real cost measurement)\n\n")
        f.write(
            "Structured (neuron-level) pruning on the two hidden layers of the "
            "2-128-128-3 MLP, then `prune/compress.py:compress_model` into a "
            "genuinely smaller dense model. FLOP and parameter counts are exact; "
            "the speedup is a microbenchmark (`>1x` is the durable claim, the "
            "exact multiple is machine/BLAS-dependent).\n\n"
        )
        f.write(f"Environment: {env}\n\n")
        f.write(
            "| target sparsity | achieved | active params | total params | "
            "dense FLOPs | sparse FLOPs | dense NumPy (s) | compressed (s) | speedup |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['target_sparsity']:.2f} | {r['achieved_sparsity']:.4f} | "
                f"{r['active_params']} | {r['total_params']} | {r['dense_flops']} | "
                f"{r['theoretical_sparse_flops']} | {r['dense_numpy_sec']:.4f} | "
                f"{r['compressed_numpy_sec']:.4f} | {r['speedup']:.2f}x |\n"
            )
        f.write(
            "\nReproduce: `python -m evaluation.run_cost_report`. Note that "
            "`x @ (W*mask)` is a full dense matmul, so the FLOP/param savings "
            "are realized only by the compressed model, not by the masked one "
            "(see `evaluation/cost.py:HONEST_COST_NOTE` and DESIGN.md §3).\n"
        )

    for r in rows:
        print(
            f"sparsity={r['achieved_sparsity']:.4f} active={r['active_params']} "
            f"dense_flops={r['dense_flops']} sparse_flops={r['theoretical_sparse_flops']} "
            f"speedup={r['speedup']:.2f}x"
        )
    print(f"wrote {csv_path} and {md_path}")
    return rows


if __name__ == "__main__":
    main()
