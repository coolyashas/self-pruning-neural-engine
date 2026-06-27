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


def test_multi_batch_matches_true_dataset_mean_with_even_batches():
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(8, 2)
    y = np.random.randint(0, 3, size=8)
    layer0 = mlp.layers[0]

    for p in mlp.parameters():
        p.grad = None
    softmax_cross_entropy(mlp(Tensor(X)), y).backward()
    true_mean = layer0.w_eff.grad.copy()

    totals = accumulate_dense_gradients(mlp, X, y, batch_size=4)  # divides 8 evenly
    assert np.allclose(totals[layer0], true_mean)


def test_multi_batch_matches_true_dataset_mean_with_uneven_final_batch():
    """batch_size=3 does not divide N=8 evenly (batches of 3, 3, 2).
    Each backward() computes a BATCH-mean gradient -- naively summing
    those un-weighted (the previous behavior here) overweights the
    smaller final batch's examples. The correct total must match a
    single direct backward() over the whole dataset, not "sum of each
    batch's w_eff.grad".
    """
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(8, 2)
    y = np.random.randint(0, 3, size=8)
    layer0 = mlp.layers[0]

    for p in mlp.parameters():
        p.grad = None
    softmax_cross_entropy(mlp(Tensor(X)), y).backward()
    true_mean = layer0.w_eff.grad.copy()

    # the buggy reference this test used to assert against, kept only to
    # prove it's now genuinely different from the true mean.
    buggy_sum = np.zeros_like(layer0.weight.data)
    for start in range(0, 8, 3):
        for p in mlp.parameters():
            p.grad = None
        idx = slice(start, start + 3)
        softmax_cross_entropy(mlp(Tensor(X[idx])), y[idx]).backward()
        buggy_sum += layer0.w_eff.grad
    assert not np.allclose(buggy_sum, true_mean)

    totals = accumulate_dense_gradients(mlp, X, y, batch_size=3)
    assert np.allclose(totals[layer0], true_mean)


def test_skips_a_parameter_never_touched_this_sweep():
    """Same generic-helper concern as accumulate_gradients: a parameter
    a future, conditionally-used architecture's forward never touches
    would leave p.grad as None after backward() -- unconditionally
    doing `batch_weight * p.grad` would crash instead of deliberately
    skipping it.
    """

    class _ModelWithDisconnectedParam:
        def __init__(self, real_layer, extra_param):
            self.real_layer = real_layer
            self.extra_param = extra_param
            self.layers = [real_layer]

        def __call__(self, x):
            return self.real_layer(x)

        def parameters(self):
            return self.real_layer.parameters() + [self.extra_param]

    real_layer = Linear(2, 3)
    extra_param = Tensor(np.ones((5, 5)), requires_grad=True)  # never used in the forward path
    model = _ModelWithDisconnectedParam(real_layer, extra_param)

    X = np.random.randn(10, 2)
    y = np.random.randint(0, 3, size=10)

    accumulate_dense_gradients(model, X, y, batch_size=4)  # uneven last batch too -- must not crash

    assert extra_param.grad is not None
    assert np.array_equal(extra_param.grad, np.zeros((5, 5)))
    assert real_layer.weight.grad is not None


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
