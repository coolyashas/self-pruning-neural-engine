import csv

import numpy as np

from engine.tensor import Tensor
from train.run_part4_structured import main


def test_run_part4_structured_produces_artifacts_and_reaches_sparsity(tmp_path):
    history, mlp, compressed = main(
        epochs=20,
        n_per_class=30,
        batch_size=8,
        final_sparsity=0.5,
        prune_start_step=10,
        prune_every=5,
        results_dir=str(tmp_path),
    )
    assert len(history) == 20

    csv_path = tmp_path / "part4_structured_self_pruning.csv"
    png_path = tmp_path / "part4_structured_self_pruning.png"
    assert csv_path.exists()
    assert png_path.exists()

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "sparsity", "mean_loss", "accuracy"]
    assert len(rows) - 1 == 20

    final_step, final_sparsity_achieved, final_loss, final_acc = history[-1]
    assert np.isfinite(final_loss)
    assert 0.0 <= final_acc <= 1.0
    assert final_sparsity_achieved >= 0.4  # reached close to the 0.5 target


def test_run_part4_structured_never_prunes_final_layer_output(tmp_path):
    _, mlp, _ = main(
        epochs=15,
        n_per_class=20,
        batch_size=8,
        final_sparsity=0.6,
        prune_start_step=5,
        prune_every=3,
        results_dir=str(tmp_path),
    )
    final_layer = mlp.layers[-1]
    assert final_layer.mask.data.any(axis=0).sum() == final_layer.weight.shape[1]  # all 3 classes intact


def test_run_part4_structured_compressed_model_matches_dense_exactly(tmp_path):
    """Re-verify main()'s in-script sanity check here as a real pytest
    assertion, with a fresh batch.
    """
    _, mlp, compressed = main(
        epochs=15,
        n_per_class=20,
        batch_size=8,
        final_sparsity=0.7,
        prune_start_step=5,
        prune_every=3,
        results_dir=str(tmp_path),
    )
    x = np.random.randn(10, 2)
    dense_out = mlp(Tensor(x)).data
    compressed_out = compressed(x)
    assert np.allclose(dense_out, compressed_out, atol=1e-10)
