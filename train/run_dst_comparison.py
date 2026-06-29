"""Dynamic sparse training (regrowth) vs. plain monotonic pruning at matched
final sparsity: does reallocating the active-connection budget (grow+drop
cycles, prune/dst.py) help accuracy? Orchestrates many run_part3 calls (with
vs without enable_regrowth) and aggregates them.
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
    sparsities=(0.9, 0.95),
    seeds=(0, 1, 2, 3, 4),
    epochs=200,
    n_per_class=300,
    batch_size=32,
    lr=0.01,
    prune_start_step=400,
    prune_every=20,
    exchange_fraction=0.1,
) -> list[dict]:
    """One run_part3 call per (sparsity, seed, with/without regrowth),
    both saliency, both otherwise identical.
    """
    records = []
    with tempfile.TemporaryDirectory() as scratch:
        for sparsity in sparsities:
            for seed in seeds:
                for enable_regrowth in (False, True):
                    history, _ = run_part3(
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
                        enable_regrowth=enable_regrowth,
                        exchange_fraction=exchange_fraction,
                    )
                    _, achieved_sparsity, _, accuracy = history[-1]
                    records.append(
                        {
                            "regrowth": enable_regrowth,
                            "target_sparsity": sparsity,
                            "achieved_sparsity": achieved_sparsity,
                            "seed": seed,
                            "final_accuracy": accuracy,
                        }
                    )
    return records


def summarize(records: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(r["regrowth"], r["target_sparsity"]) for r in records})
    for regrowth, sparsity in keys:
        matching = [r for r in records if r["regrowth"] == regrowth and r["target_sparsity"] == sparsity]
        accs = [r["final_accuracy"] for r in matching]
        achieved = [r["achieved_sparsity"] for r in matching]
        summary.append(
            {
                "regrowth": regrowth,
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
    with open(os.path.join(results_dir, "dst_comparison_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with open(os.path.join(results_dir, "dst_comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(7, 5))
    for regrowth in sorted({s["regrowth"] for s in summary}):
        rows = sorted((s for s in summary if s["regrowth"] == regrowth), key=lambda r: r["mean_achieved_sparsity"])
        x = [r["mean_achieved_sparsity"] for r in rows]
        y = [r["mean_accuracy"] for r in rows]
        yerr = [r["std_accuracy"] for r in rows]
        label = "with regrowth (DST)" if regrowth else "without regrowth (plain saliency)"
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)
    ax.set(xlabel="achieved sparsity", ylabel="accuracy", title="DST (regrowth) vs. plain pruning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "dst_comparison.png"))
    plt.close(fig)

    return records, summary


if __name__ == "__main__":
    main()
