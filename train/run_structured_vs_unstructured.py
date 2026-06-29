"""Structured vs. unstructured pruning at matched connection-count sparsity:
does pruning whole neurons cost more accuracy than pruning individual weights?
Structured is coarser (all-or-nothing per neuron), so it should cost more;
this measures whether it actually does.
"""

from __future__ import annotations

import csv
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.run_part3 import main as run_unstructured
from train.run_part4_structured import main as run_structured

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_sweep(
    sparsities=(0.5, 0.7, 0.85),
    seeds=(0, 1, 2),
    epochs=200,
    n_per_class=300,
    batch_size=32,
    lr=0.01,
    prune_start_step=400,
    prune_every=20,
) -> list[dict]:
    """One run_part3 (unstructured) + one run_part4_structured (structured)
    call per (sparsity, seed), both saliency, both otherwise identical.
    """
    records = []
    with tempfile.TemporaryDirectory() as scratch:
        for sparsity in sparsities:
            for seed in seeds:
                history, _ = run_unstructured(
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    final_sparsity=sparsity,
                    prune_start_step=prune_start_step,
                    prune_every=prune_every,
                    criterion="saliency",
                    n_per_class=n_per_class,
                    results_dir=scratch,
                )
                _, achieved_sparsity, _, accuracy = history[-1]
                records.append(
                    {
                        "method": "unstructured",
                        "target_sparsity": sparsity,
                        "achieved_sparsity": achieved_sparsity,
                        "seed": seed,
                        "final_accuracy": accuracy,
                    }
                )

                history, _, _ = run_structured(
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    final_sparsity=sparsity,
                    prune_start_step=prune_start_step,
                    prune_every=prune_every,
                    criterion="saliency",
                    n_per_class=n_per_class,
                    results_dir=scratch,
                )
                _, achieved_sparsity, _, accuracy = history[-1]
                records.append(
                    {
                        "method": "structured",
                        "target_sparsity": sparsity,
                        "achieved_sparsity": achieved_sparsity,
                        "seed": seed,
                        "final_accuracy": accuracy,
                    }
                )
    return records


def summarize(records: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(r["method"], r["target_sparsity"]) for r in records})
    for method, sparsity in keys:
        matching = [r for r in records if r["method"] == method and r["target_sparsity"] == sparsity]
        accs = [r["final_accuracy"] for r in matching]
        achieved = [r["achieved_sparsity"] for r in matching]
        summary.append(
            {
                "method": method,
                "target_sparsity": sparsity,
                "mean_achieved_sparsity": float(np.mean(achieved)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "n_seeds": len(accs),
            }
        )
    return summary


def main(results_dir: str = RESULTS_DIR, **sweep_kwargs):
    records = run_sweep(**sweep_kwargs)
    summary = summarize(records)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "structured_vs_unstructured_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with open(os.path.join(results_dir, "structured_vs_unstructured.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted({s["method"] for s in summary}):
        rows = sorted((s for s in summary if s["method"] == method), key=lambda r: r["mean_achieved_sparsity"])
        x = [r["mean_achieved_sparsity"] for r in rows]
        y = [r["mean_accuracy"] for r in rows]
        yerr = [r["std_accuracy"] for r in rows]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)
    ax.set(xlabel="achieved sparsity", ylabel="accuracy", title="Structured vs. unstructured pruning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "structured_vs_unstructured.png"))
    plt.close(fig)

    return records, summary


if __name__ == "__main__":
    main()
