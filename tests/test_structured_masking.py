import numpy as np

from engine.tensor import Tensor
from nn.linear import Linear
from prune.criteria import magnitude_scores, neuron_magnitude_scores
from prune.mask import keep_neurons_from_scores, prune_neurons_to_count, prune_to_sparsity
from utils.seed import set_seed

set_seed(0)


def test_keep_neurons_from_scores_exact_count_and_keeps_highest():
    scores = np.arange(10).astype(float)  # 0..9, all distinct
    keep = keep_neurons_from_scores(scores, n_prune=4)
    assert keep.sum() == 6
    assert keep.shape == scores.shape
    kept_scores = scores[keep]
    dropped_scores = scores[~keep]
    assert kept_scores.min() > dropped_scores.max()  # exactly the top 6


def test_prune_neurons_to_count_exact_budget():
    layer = Linear(5, 12)  # non-square: rows=in_features, cols=out_features (neurons)
    scores = np.abs(np.random.randn(12))
    for target_active in [12, 9, 5, 1, 0]:
        prune_neurons_to_count(layer, scores, target_active)
        column_alive = layer.mask.data.any(axis=0)
        assert column_alive.sum() == target_active
        # a fully-pruned neuron must have its ENTIRE column zeroed, not partial
        for j in range(12):
            if not column_alive[j]:
                assert np.all(layer.mask.data[:, j] == 0.0)


def test_prune_neurons_to_count_never_revives():
    """Adversarial, mirrors test_prune_to_sparsity_never_revives at
    neuron granularity: give already-dead neurons enormous fake scores
    on a later call and confirm they still don't come back.
    """
    layer = Linear(4, 6)
    scores_step1 = np.abs(np.random.randn(6))
    prune_neurons_to_count(layer, scores_step1, target_active=3)
    alive_after_step1 = layer.mask.data.any(axis=0).copy()

    scores_step2 = np.abs(np.random.randn(6)) + 100.0
    scores_step2[~alive_after_step1] = 1e6
    prune_neurons_to_count(layer, scores_step2, target_active=4)  # would need a revive

    assert np.all(layer.mask.data[:, ~alive_after_step1] == 0.0)


def test_prune_neurons_to_count_increasing_target_is_noop():
    layer = Linear(4, 6)
    scores = np.abs(np.random.randn(6))
    prune_neurons_to_count(layer, scores, target_active=3)
    mask_after_first = layer.mask.data.copy()
    prune_neurons_to_count(layer, scores, target_active=5)  # "behind" -- would need a revive
    assert np.array_equal(layer.mask.data, mask_after_first)


def test_mixed_unstructured_then_structured_pruning_on_same_layer():
    """A neuron half-zeroed by unstructured pruning still counts as
    "active" (any nonzero in its column) until structured pruning
    explicitly finishes zeroing the rest of it -- this must not double-
    count or miscompute the active total.
    """
    layer = Linear(8, 6)
    mag_scores = magnitude_scores(layer)
    prune_to_sparsity(layer, mag_scores, target_sparsity=0.5)  # unstructured first

    column_alive_before = layer.mask.data.any(axis=0)
    assert column_alive_before.sum() <= 6  # sanity: still well-defined

    neuron_scores = neuron_magnitude_scores(layer)
    prune_neurons_to_count(layer, neuron_scores, target_active=3)
    column_alive_after = layer.mask.data.any(axis=0)
    assert column_alive_after.sum() == 3
    for j in range(6):
        if not column_alive_after[j]:
            assert np.all(layer.mask.data[:, j] == 0.0)


def test_structured_pruning_forward_backward_still_correct():
    """Structured pruning is still just per-entry mask gating under the
    hood -- the existing masked-grad machinery needs no changes.
    """
    layer = Linear(4, 6)
    scores = np.abs(np.random.randn(6))
    prune_neurons_to_count(layer, scores, target_active=3)

    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert np.all(layer.weight.grad[layer.mask.data == 0] == 0.0)
