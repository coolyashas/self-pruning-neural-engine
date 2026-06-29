"""Pareto sweep: magnitude vs saliency pruning across target sparsities
and seeds, reporting mean +/- std accuracy at each point. Reuses
run_part3's single self-pruning run as the primitive -- this script's job
is orchestrating many such runs and aggregating them, not training itself.
"""

from __future__ import annotations

import csv
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.run_part3 import main as run_part3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_sweep(
    criteria=("magnitude", "saliency"),
    sparsities=(0.0, 0.5, 0.75, 0.9, 0.95),
    seeds=(0, 1, 2, 3, 4),
    epochs=100,
    n_per_class=300,
    batch_size=32,
    lr=1e-3,
    prune_start_step=400,
    prune_every=20,
) -> list[dict]:
    """One run_part3 call per (criterion, sparsity, seed); returns raw
    per-run records. Each run's own learning-curve artifacts go to a
    throwaway scratch dir -- only the final (sparsity, accuracy) matters
    here, the per-run CSV/plot would just be noise at this scale.
    """
    records = []
    with tempfile.TemporaryDirectory() as scratch:
        for criterion in criteria:
            for sparsity in sparsities:
                for seed in seeds:
                    history, _ = run_part3(
                        seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        final_sparsity=sparsity,
                        prune_start_step=prune_start_step,
                        prune_every=prune_every,
                        criterion=criterion,
                        n_per_class=n_per_class,
                        results_dir=scratch,
                    )
                    _, achieved_sparsity, final_loss, final_accuracy = history[-1]
                    records.append(
                        {
                            "criterion": criterion,
                            "target_sparsity": sparsity,
                            "achieved_sparsity": achieved_sparsity,
                            "seed": seed,
                            "final_loss": final_loss,
                            "final_accuracy": final_accuracy,
                        }
                    )
    return records


def summarize(records: list[dict]) -> list[dict]:
    """Aggregate raw per-seed records into mean +/- std accuracy per
    (criterion, target_sparsity)."""
    summary = []
    keys = sorted({(r["criterion"], r["target_sparsity"]) for r in records})
    for criterion, sparsity in keys:
        matching = [r for r in records if r["criterion"] == criterion and r["target_sparsity"] == sparsity]
        accs = [r["final_accuracy"] for r in matching]
        achieved = [r["achieved_sparsity"] for r in matching]
        summary.append(
            {
                "criterion": criterion,
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
    with open(os.path.join(results_dir, "pareto_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with open(os.path.join(results_dir, "pareto_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(7, 5))
    for criterion in sorted({s["criterion"] for s in summary}):
        rows = sorted(
            (s for s in summary if s["criterion"] == criterion),
            key=lambda r: r["mean_achieved_sparsity"],
        )
        x = [r["mean_achieved_sparsity"] for r in rows]
        y = [r["mean_accuracy"] for r in rows]
        yerr = [r["std_accuracy"] for r in rows]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=criterion)
    ax.set(xlabel="achieved sparsity", ylabel="accuracy", title="Pareto: sparsity vs accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pareto_sweep.png"))
    plt.close(fig)

    return records, summary


if __name__ == "__main__":
    main()
