import numpy as np

from nn.linear import Linear
from prune.mask import prune_to_sparsity, revive_to_count, set_mask
from utils.seed import set_seed

set_seed(0)


def test_revive_to_count_never_selects_already_active():
    """Adversarial, mirrors test_prune_to_sparsity_never_revives's spirit
    inverted: give already-active entries an enormous fake score and
    confirm revive_to_count still never touches them -- they aren't even
    candidates, since only masked-off entries are eligible.
    """
    layer = Linear(5, 5)
    keep = np.zeros((5, 5))
    keep[0, :] = 1.0  # row 0 active, everything else masked off
    set_mask(layer, keep)

    scores = np.zeros((5, 5))
    scores[0, :] = 1e6  # adversarial: the active row scores far higher than anything else
    revived = revive_to_count(layer, scores, n_revive=5)

    assert not np.any(revived[0, :])  # active row never "revived" -- it wasn't a candidate
    assert layer.mask.data[0, :].sum() == 5  # unchanged, still active


def test_revive_to_count_exact_budget():
    layer = Linear(6, 6)  # 36 entries
    prune_to_sparsity(layer, np.abs(np.random.randn(6, 6)), target_sparsity=0.8)  # 29 masked off
    assert (layer.mask.data == 0).sum() == 29

    scores = np.abs(np.random.randn(6, 6))
    for n_revive in [0, 1, 5, 10]:
        layer_copy_mask = layer.mask.data.copy()
        revived = revive_to_count(layer, scores, n_revive)
        newly_active = layer.mask.data.sum() - layer_copy_mask.sum()
        assert revived.sum() == n_revive
        assert newly_active == n_revive


def test_revive_to_count_clamps_when_insufficient_masked_entries():
    layer = Linear(4, 4)  # 16 entries
    prune_to_sparsity(layer, np.abs(np.random.randn(4, 4)), target_sparsity=0.5)  # 8 masked off
    n_inactive = int((layer.mask.data == 0).sum())
    assert n_inactive == 8

    revived = revive_to_count(layer, np.abs(np.random.randn(4, 4)), n_revive=100)
    assert revived.sum() == n_inactive  # clamped, not 100
    assert (layer.mask.data == 0).sum() == 0  # everything now active


def test_revive_to_count_returns_zero_array_when_nothing_to_revive():
    layer = Linear(3, 3)  # all-ones mask by default
    revived = revive_to_count(layer, np.abs(np.random.randn(3, 3)), n_revive=5)
    assert revived.sum() == 0
    assert revived.shape == layer.mask.shape


def test_revive_keeps_highest_scoring_masked_entries():
    layer = Linear(4, 4)
    keep = np.zeros((4, 4))
    set_mask(layer, keep)  # everything masked off
    scores = np.arange(16).reshape(4, 4).astype(float)  # 0..15, all distinct

    revived = revive_to_count(layer, scores, n_revive=4)
    revived_scores = scores[revived]
    not_revived_scores = scores[~revived]
    assert revived_scores.min() > not_revived_scores.max()  # exactly the top 4


def test_revive_does_not_change_weight_values():
    """Reviving only flips the mask -- the underlying weight value stays
    whatever it was frozen at (mask gates, never destroys/resets).
    """
    layer = Linear(3, 3)
    keep = np.zeros((3, 3))
    set_mask(layer, keep)
    weight_before = layer.weight.data.copy()

    revive_to_count(layer, np.abs(np.random.randn(3, 3)), n_revive=3)
    assert np.array_equal(layer.weight.data, weight_before)


def test_revive_to_count_does_not_break_prune_to_sparsity_contract():
    """Regression check: revive_to_count is additive -- prune_to_sparsity
    itself must remain untouched, including its never-revives guarantee.
    """
    layer = Linear(6, 6)
    scores_step1 = np.abs(np.random.randn(6, 6))
    prune_to_sparsity(layer, scores_step1, target_sparsity=0.5)
    pruned_after_step1 = layer.mask.data.copy()

    scores_step2 = np.abs(np.random.randn(6, 6)) + 100.0
    scores_step2[pruned_after_step1 == 0] = 1e6
    prune_to_sparsity(layer, scores_step2, target_sparsity=0.7)

    assert np.all(layer.mask.data[pruned_after_step1 == 0] == 0.0)
