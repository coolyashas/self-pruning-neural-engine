import numpy as np
import pytest

from engine.tensor import Tensor
from nn import Linear, ReLU, Sequential
from prune.criteria import accumulate_gradients, magnitude_scores, saliency_scores
from prune.mask import keep_mask_from_scores, set_mask
from utils.seed import set_seed

set_seed(0)


def test_saliency_scores_formula():
    layer = Linear(3, 3)
    layer.weight.grad = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0], [2.0, -2.0, 1.0]])
    expected = np.abs(layer.weight.data * layer.weight.grad)
    assert np.allclose(saliency_scores(layer), expected)


def test_saliency_requires_a_gradient():
    layer = Linear(3, 3)
    with pytest.raises(AssertionError):
        saliency_scores(layer)


def test_accumulate_gradients_single_batch_matches_plain_backward():
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 3, size=10)

    accumulate_gradients(mlp, X, y, batch_size=10)  # one batch covers everything
    grad_via_helper = mlp.layers[0].weight.grad.copy()

    for p in mlp.parameters():
        p.grad = None
    from engine.loss import softmax_cross_entropy

    softmax_cross_entropy(mlp(Tensor(X)), y).backward()
    grad_direct = mlp.layers[0].weight.grad

    assert np.allclose(grad_via_helper, grad_direct)


def test_accumulate_gradients_matches_true_dataset_mean_with_even_batches():
    """batch_size=4 divides N=8 evenly, so every batch is the same size
    -- the case where "sum of per-batch means" and "true dataset mean"
    happen to coincide. Confirms the weighted accumulation reduces to
    the obvious answer when there's nothing uneven to get wrong.
    """
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(8, 2)
    y = np.random.randint(0, 3, size=8)

    from engine.loss import softmax_cross_entropy

    for p in mlp.parameters():
        p.grad = None
    softmax_cross_entropy(mlp(Tensor(X)), y).backward()
    true_mean = mlp.layers[0].weight.grad.copy()

    accumulate_gradients(mlp, X, y, batch_size=4)
    assert np.allclose(mlp.layers[0].weight.grad, true_mean)


def test_accumulate_gradients_matches_true_dataset_mean_with_uneven_final_batch():
    """The case that actually matters: batch_size=3 does NOT divide
    N=8 evenly (batches of 3, 3, 2). Each backward() computes a
    BATCH-mean gradient -- naively summing those un-weighted (the
    previous behavior here) overweights the smaller final batch's
    examples relative to the full ones. The correct result must match
    a single direct backward() over the WHOLE dataset (which IS the
    true full-dataset-mean gradient by construction, since
    softmax_cross_entropy's own mean reduction divides by N), not "sum
    of each batch's mean gradient".
    """
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    X = np.random.randn(8, 2)
    y = np.random.randint(0, 3, size=8)

    from engine.loss import softmax_cross_entropy

    for p in mlp.parameters():
        p.grad = None
    softmax_cross_entropy(mlp(Tensor(X)), y).backward()
    true_mean = mlp.layers[0].weight.grad.copy()

    # the buggy reference this test used to assert against, kept here
    # only to prove it's now DIFFERENT from the true mean -- i.e. the
    # old behavior and the fixed behavior are genuinely distinguishable,
    # not just two ways of writing the same answer.
    buggy_sum_of_means = np.zeros_like(mlp.layers[0].weight.data)
    for start in range(0, 8, 3):
        for p in mlp.parameters():
            p.grad = None
        idx = slice(start, start + 3)
        softmax_cross_entropy(mlp(Tensor(X[idx])), y[idx]).backward()
        buggy_sum_of_means += mlp.layers[0].weight.grad
    assert not np.allclose(buggy_sum_of_means, true_mean)

    accumulate_gradients(mlp, X, y, batch_size=3)
    assert np.allclose(mlp.layers[0].weight.grad, true_mean)


def test_saliency_and_magnitude_can_disagree():
    """The whole point of having both criteria: a large weight with near-
    zero gradient should rank LOW on saliency despite ranking HIGH on
    magnitude, and vice versa for a small weight with a large gradient.
    """
    layer = Linear(2, 2)
    layer.weight.data = np.array([[10.0, 0.1], [0.1, 10.0]])
    layer.weight.grad = np.array([[0.001, 5.0], [5.0, 0.001]])

    mag = magnitude_scores(layer)
    sal = saliency_scores(layer)

    # magnitude ranks [0,0] and [1,1] highest
    assert mag[0, 0] == mag.max() or mag[1, 1] == mag.max()
    # saliency ranks [0,1] and [1,0] highest instead
    assert sal[0, 1] == sal.max() and sal[1, 0] == sal.max()
    assert np.argmax(mag) != np.argmax(sal)


def test_saliency_pruning_end_to_end():
    mlp = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    X = np.random.randn(40, 2)
    y = np.random.randint(0, 3, size=40)

    accumulate_gradients(mlp, X, y, batch_size=8)

    layer = mlp.layers[0]
    scores = saliency_scores(layer)
    keep = keep_mask_from_scores(scores, sparsity=0.5)
    set_mask(layer, keep)

    assert layer.mask.data.sum() == round(0.5 * layer.weight.data.size)

    for p in mlp.parameters():
        p.grad = None  # the saliency sweep above left stale accumulated grad
    x = Tensor(np.random.randn(3, 2), requires_grad=True)
    loss = mlp(x).sum()
    loss.backward()
    assert np.all(layer.weight.grad[layer.mask.data == 0] == 0.0)
