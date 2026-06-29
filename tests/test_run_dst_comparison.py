import csv

import numpy as np

from train.run_dst_comparison import main, run_sweep, summarize


def _tiny_sweep_kwargs():
    # tiny: 2 sparsities x 2 seeds x 2 (with/without regrowth) = 8 runs, fast for CI
    return dict(
        sparsities=(0.5, 0.7),
        seeds=(0, 1),
        epochs=10,
        n_per_class=20,
        batch_size=8,
        prune_start_step=3,
        prune_every=2,
    )


def test_run_sweep_produces_one_record_per_regrowth_setting_per_combination():
    records = run_sweep(**_tiny_sweep_kwargs())
    assert len(records) == 2 * 2 * 2  # sparsities x seeds x (with/without regrowth)
    expected_keys = {"regrowth", "target_sparsity", "achieved_sparsity", "seed", "final_accuracy"}
    for r in records:
        assert set(r.keys()) == expected_keys
        assert r["regrowth"] in (True, False)
        assert 0.0 <= r["final_accuracy"] <= 1.0


def test_summarize_aggregates_mean_and_std_per_group():
    records = run_sweep(**_tiny_sweep_kwargs())
    summary = summarize(records)
    assert len(summary) == 2 * 2  # sparsities x regrowth settings, seeds collapsed

    for s in summary:
        matching = [
            r["final_accuracy"]
            for r in records
            if r["regrowth"] == s["regrowth"] and r["target_sparsity"] == s["target_sparsity"]
        ]
        assert s["n_seeds"] == len(matching) == 2
        assert s["mean_accuracy"] == np.mean(matching)
        assert s["std_accuracy"] == np.std(matching)


def test_main_writes_raw_and_summary_csv_and_plot(tmp_path):
    records, summary = main(results_dir=str(tmp_path), **_tiny_sweep_kwargs())

    raw_path = tmp_path / "dst_comparison_raw.csv"
    summary_path = tmp_path / "dst_comparison.csv"
    plot_path = tmp_path / "dst_comparison.png"
    assert raw_path.exists() and summary_path.exists() and plot_path.exists()

    with open(raw_path) as f:
        raw_rows = list(csv.DictReader(f))
    assert len(raw_rows) == len(records)

    with open(summary_path) as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == len(summary)
