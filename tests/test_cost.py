import time

import numpy as np

from engine.tensor import Tensor
from evaluation.cost import (
    active_param_count,
    cost_report,
    model_flops,
    total_param_count,
    weight_sparsity,
)
from nn import Linear, ReLU, Sequential
from prune.mask import set_mask
from utils.seed import set_seed

set_seed(0)


def _make_model():
    return Sequential(Linear(10, 20), ReLU(), Linear(20, 5))


def test_weight_sparsity_matches_mask_fraction():
    mlp = _make_model()
    layer = mlp.layers[0]
    keep = np.ones((10, 20))
    keep[:, :10] = 0.0  # exactly half of this layer pruned
    set_mask(layer, keep)
    # layer0: 200 entries, 100 pruned. layer2: 100 entries, 0 pruned.
    expected = 1.0 - (100 + 100) / (200 + 100)
    assert weight_sparsity(mlp) == expected


def test_active_params_counts_mask_ones_plus_bias():
    mlp = _make_model()
    layer0, layer2 = mlp.layers[0], mlp.layers[2]
    keep = np.ones((10, 20))
    keep[:, :10] = 0.0
    set_mask(layer0, keep)

    expected = (100 + layer0.bias.data.size) + (100 + layer2.bias.data.size)
    assert active_param_count(mlp) == expected


def test_total_params_unaffected_by_mask():
    mlp = _make_model()
    before = total_param_count(mlp)
    keep = np.zeros((10, 20))
    set_mask(mlp.layers[0], keep)  # prune everything in layer 0
    after = total_param_count(mlp)
    assert before == after  # total count doesn't change, only active does


def test_dense_flops_unaffected_by_sparsity_but_theoretical_sparse_flops_drops():
    mlp = _make_model()
    dense_before = model_flops(mlp, batch_size=4, active_only=False)
    sparse_before = model_flops(mlp, batch_size=4, active_only=True)
    assert dense_before == sparse_before  # nothing pruned yet, so they match

    set_mask(mlp.layers[0], np.zeros((10, 20)))  # prune all of layer 0
    dense_after = model_flops(mlp, batch_size=4, active_only=False)
    sparse_after = model_flops(mlp, batch_size=4, active_only=True)

    assert dense_after == dense_before  # our actual cost: unchanged by pruning
    assert sparse_after < sparse_before  # theoretical sparse cost: dropped


def test_cost_report_keys_and_consistency():
    mlp = _make_model()
    report = cost_report(mlp, batch_size=8)
    assert set(report.keys()) == {
        "weight_sparsity",
        "active_params",
        "total_params",
        "dense_flops",
        "theoretical_sparse_flops",
        "note",
    }
    assert report["active_params"] <= report["total_params"]
    assert report["dense_flops"] == report["theoretical_sparse_flops"]  # sparsity is 0 here
    assert "dense" in report["note"].lower()


def test_dense_matmul_speed_unaffected_by_sparsity():
    """The honest claim, measured: a 90%-sparse layer takes the same
    wall-clock time as a 0%-sparse layer of the same shape, because
    x @ (W*mask) is a full dense matmul regardless of mask content.
    """
    n_in, n_out, batch = 256, 256, 64

    def time_forward(sparsity: float, repeats: int = 200) -> float:
        layer = Linear(n_in, n_out)
        if sparsity > 0:
            n_total = layer.weight.data.size
            n_prune = int(sparsity * n_total)
            flat = np.ones(n_total)
            flat[:n_prune] = 0.0
            np.random.shuffle(flat)
            set_mask(layer, flat.reshape(n_in, n_out))
        x = Tensor(np.random.randn(batch, n_in))
        start = time.perf_counter()
        for _ in range(repeats):
            layer(x)
        return time.perf_counter() - start

    t_dense = time_forward(0.0)
    t_sparse = time_forward(0.9)

    # generous tolerance -- this is a noisy microbenchmark; the point is
    # there's no dramatic speedup, not that the ratio is exactly 1.0
    ratio = t_sparse / t_dense
    assert 0.5 < ratio < 2.0, f"expected no speedup from sparsity, got ratio={ratio:.2f}"
