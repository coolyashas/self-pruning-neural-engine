import csv

import numpy as np

from engine.tensor import Tensor
from train.run_part3 import main


def test_run_part3_produces_artifacts_and_reaches_sparsity(tmp_path):
    # small/fast settings to tmp_path, never the real results/ directory
    history, mlp = main(
        epochs=20,
        n_per_class=30,
        batch_size=8,
        final_sparsity=0.5,
        prune_start_step=10,
        prune_every=5,
        results_dir=str(tmp_path),
    )
    assert len(history) == 20

    csv_path = tmp_path / "part3_self_pruning.csv"
    png_path = tmp_path / "part3_self_pruning.png"
    assert csv_path.exists()
    assert png_path.exists()

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["step", "sparsity", "mean_loss", "accuracy"]
    assert len(rows) - 1 == 20

    final_step, final_sparsity_achieved, final_loss, final_acc = history[-1]
    assert np.isfinite(final_loss)
    assert 0.0 <= final_acc <= 1.0
    assert final_sparsity_achieved >= 0.45  # reached close to the 0.5 target


def test_run_part3_pruned_model_respects_masked_grad(tmp_path):
    """After a real self-pruning run, masked entries must still get exactly
    zero gradient.
    """
    _, mlp = main(
        epochs=15,
        n_per_class=20,
        batch_size=8,
        final_sparsity=0.6,
        prune_start_step=5,
        prune_every=3,
        results_dir=str(tmp_path),
    )

    prunable_layers = [layer for layer in mlp.layers if hasattr(layer, "mask")]
    assert any((layer.mask.data == 0).any() for layer in prunable_layers)  # something got pruned

    for p in mlp.parameters():
        p.grad = None
    x = Tensor(np.random.randn(4, 2), requires_grad=True)
    mlp(x).sum().backward()

    for layer in prunable_layers:
        assert np.all(layer.weight.grad[layer.mask.data == 0] == 0.0)


def test_enable_regrowth_default_false_is_byte_for_byte_unchanged(tmp_path):
    """enable_regrowth/exchange_fraction must be true no-ops at their defaults:
    passing enable_regrowth=False must match omitting it entirely.
    """
    kwargs = dict(
        epochs=15,
        n_per_class=20,
        batch_size=8,
        final_sparsity=0.5,
        prune_start_step=5,
        prune_every=3,
        results_dir=str(tmp_path),
    )
    history_default, _ = main(**kwargs)
    history_explicit_false, _ = main(enable_regrowth=False, **kwargs)
    assert history_default == history_explicit_false


def test_enable_regrowth_true_runs_end_to_end_and_reaches_sparsity(tmp_path):
    history, mlp = main(
        epochs=20,
        n_per_class=30,
        batch_size=8,
        final_sparsity=0.5,
        prune_start_step=10,
        prune_every=5,
        results_dir=str(tmp_path),
        enable_regrowth=True,
        exchange_fraction=0.2,
    )
    assert len(history) == 20
    final_step, final_sparsity_achieved, final_loss, final_acc = history[-1]
    assert np.isfinite(final_loss)
    assert 0.0 <= final_acc <= 1.0
    assert final_sparsity_achieved >= 0.4  # reached close to the 0.5 target

    # FRESH forward+backward: the last exchange cycle may have changed the mask
    # after the training .grad was computed against the old mask.
    for p in mlp.parameters():
        p.grad = None
    x = Tensor(np.random.randn(4, 2), requires_grad=True)
    mlp(x).sum().backward()

    prunable_layers = [layer for layer in mlp.layers if hasattr(layer, "mask")]
    for layer in prunable_layers:
        assert np.all(layer.weight.grad[layer.mask.data == 0] == 0.0)
