import numpy as np

from engine.loss import softmax_cross_entropy
from engine.tensor import Tensor
from nn import Linear, ReLU, Sequential
from prune.criteria import accumulate_dense_gradients
from utils.seed import set_seed

set_seed(0)


def test_single_batch_matches_plain_backward():
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 3, size=10)

    totals = accumulate_dense_gradients(mlp, X, y, batch_size=10)  # one batch covers everything
    layer0 = mlp.layers[0]
    via_helper = totals[layer0].copy()

    for p in mlp.parameters():
        p.grad = None
    softmax_cross_entropy(mlp(Tensor(X)), y).backward()
    direct = mlp.layers[0].w_eff.grad

    assert np.allclose(via_helper, direct)


def test_multi_batch_sums_correctly():
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(8, 2)
    y = np.random.randint(0, 3, size=8)
    layer0 = mlp.layers[0]

    # hand-summed reference: each batch's w_eff.grad, added up independently
    expected = np.zeros_like(layer0.weight.data)
    for start in range(0, 8, 3):
        for p in mlp.parameters():
            p.grad = None
        idx = slice(start, start + 3)
        softmax_cross_entropy(mlp(Tensor(X[idx])), y[idx]).backward()
        expected += layer0.w_eff.grad

    totals = accumulate_dense_gradients(mlp, X, y, batch_size=3)
    assert np.allclose(totals[layer0], expected)


def test_returns_every_prunable_layer():
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(6, 2)
    y = np.random.randint(0, 3, size=6)

    totals = accumulate_dense_gradients(mlp, X, y, batch_size=6)
    prunable = [layer for layer in mlp.layers if hasattr(layer, "mask")]
    assert set(totals.keys()) == set(prunable)
    for layer in prunable:
        assert totals[layer].shape == layer.weight.shape


def test_is_unmasked_unlike_weight_grad():
    """The whole point of this function: at a masked entry, weight.grad
    is exactly 0 (mask-gated), but the accumulated dense total isn't.
    """
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 3, size=10)

    from prune.mask import set_mask

    layer0 = mlp.layers[0]
    keep = np.ones_like(layer0.weight.data)
    keep[0, 1] = 0.0
    set_mask(layer0, keep)

    totals = accumulate_dense_gradients(mlp, X, y, batch_size=10)

    for p in mlp.parameters():
        p.grad = None
    softmax_cross_entropy(mlp(Tensor(X)), y).backward()

    assert layer0.weight.grad[0, 1] == 0.0
    assert totals[layer0][0, 1] != 0.0


def test_leaves_weight_grad_dirty_on_exit():
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(6, 2)
    y = np.random.randint(0, 3, size=6)
    accumulate_dense_gradients(mlp, X, y, batch_size=6)
    assert mlp.layers[0].weight.grad is not None
