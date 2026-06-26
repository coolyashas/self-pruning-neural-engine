import csv

import numpy as np

from train.run_part2 import main


def test_run_part2_produces_artifacts(tmp_path):
    # small/fast settings, redirected to tmp_path so this never touches the
    # real results/ directory (those are the actual Part-2 artifacts).
    history = main(epochs=5, n_per_class=30, results_dir=str(tmp_path))
    assert len(history) == 5

    csv_path = tmp_path / "part2_learning_curve.csv"
    png_path = tmp_path / "part2_learning_curve.png"
    assert csv_path.exists()
    assert png_path.exists()

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["epoch", "mean_loss", "accuracy"]
    assert len(rows) - 1 == 5

    _, final_loss, final_acc = history[-1]
    assert np.isfinite(final_loss)
    assert 0.0 <= final_acc <= 1.0
