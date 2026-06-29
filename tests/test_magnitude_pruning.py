import numpy as np

from engine.tensor import Tensor
from nn.linear import Linear
from prune.criteria import magnitude_scores
from prune.mask import keep_mask_from_scores, set_mask
from utils.seed import set_seed

set_seed(0)


def test_magnitude_scores_is_abs_weight():
    layer = Linear(3, 4)
    layer.weight.data = np.array(
        [[-1.0, 2.0, -3.0, 0.5], [4.0, -5.0, 6.0, -0.5], [0.0, 1.0, -1.0, 2.0]]
    )
    assert np.allclose(magnitude_scores(layer), np.abs(layer.weight.data))


def test_keep_mask_exact_sparsity_count():
    scores = np.random.randn(10, 10) ** 2  # all positive, no ties expected
    for sparsity in [0.0, 0.1, 0.3, 0.5, 0.73, 0.9, 1.0]:
        keep = keep_mask_from_scores(scores, sparsity)
        n_keep_expected = round((1 - sparsity) * scores.size)
        assert keep.sum() == n_keep_expected, (sparsity, keep.sum(), n_keep_expected)
        assert keep.shape == scores.shape


def test_keep_mask_keeps_highest_scores_not_lowest():
    scores = np.arange(12).reshape(3, 4).astype(float)  # 0..11, all distinct
    keep = keep_mask_from_scores(scores, sparsity=0.5)
    assert keep.sum() == 6
    kept_scores = scores[keep]
    dropped_scores = scores[~keep]
    assert kept_scores.min() > dropped_scores.max()  # exactly the top half


def test_keep_mask_handles_ties_without_crashing():
    scores = np.ones((4, 4))  # every score identical
    keep = keep_mask_from_scores(scores, sparsity=0.5)
    assert keep.sum() == 8


def test_magnitude_pruning_end_to_end():
    """Real scoring + selection + set_mask: the kept connections are the
    largest-magnitude ones and the pruned layer still works end to end.
    """
    layer = Linear(5, 5)
    scores = magnitude_scores(layer)
    keep = keep_mask_from_scores(scores, sparsity=0.6)
    set_mask(layer, keep)

    assert layer.mask.data.sum() == round(0.4 * 25)
    kept_weights = np.abs(layer.weight.data[layer.mask.data == 1])
    pruned_weights = np.abs(layer.weight.data[layer.mask.data == 0])
    assert kept_weights.min() >= pruned_weights.max()

    x = Tensor(np.random.randn(3, 5), requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert np.all(layer.weight.grad[layer.mask.data == 0] == 0.0)
