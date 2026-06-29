import numpy as np
import pytest

from nn.linear import Linear
from prune.mask import keep_mask_from_scores, prune_to_sparsity, set_mask
from prune.schedule import cubic_sparsity
from utils.seed import set_seed

set_seed(0)


def test_cubic_sparsity_boundary_conditions():
    assert cubic_sparsity(step=0, start_step=10, end_step=50, final_sparsity=0.9) == 0.0
    assert cubic_sparsity(step=10, start_step=10, end_step=50, final_sparsity=0.9) == 0.0
    assert cubic_sparsity(step=50, start_step=10, end_step=50, final_sparsity=0.9) == 0.9
    assert cubic_sparsity(step=100, start_step=10, end_step=50, final_sparsity=0.9) == 0.9


def test_cubic_sparsity_matches_formula_at_midpoint():
    step, start, end, s_f, s_i = 30, 10, 50, 0.9, 0.0
    progress = (step - start) / (end - start)
    expected = s_f + (s_i - s_f) * (1 - progress) ** 3
    assert cubic_sparsity(step, start, end, s_f, s_i) == pytest.approx(expected)


def test_cubic_sparsity_prunes_faster_early_than_late():
    """The reason to use cubic over linear: most pruning happens early, while
    there's slack, tapering off near the target.
    """
    start, end, s_f = 0, 100, 0.9
    s_at_25 = cubic_sparsity(25, start, end, s_f)
    s_at_50 = cubic_sparsity(50, start, end, s_f)
    s_at_75 = cubic_sparsity(75, start, end, s_f)
    first_half_gain = s_at_50 - s_at_25
    second_half_gain = s_f - s_at_75
    assert first_half_gain > second_half_gain


def test_cubic_sparsity_monotonically_increasing():
    start, end, s_f = 0, 100, 0.9
    values = [cubic_sparsity(t, start, end, s_f) for t in range(start, end + 1, 5)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_prune_to_sparsity_hits_exact_target():
    layer = Linear(10, 10)  # 100 entries, so target fractions land on exact counts
    scores = np.abs(np.random.randn(10, 10))
    for target in [0.1, 0.3, 0.5, 0.7, 0.9]:
        prune_to_sparsity(layer, scores, target)
        n_keep_expected = round((1 - target) * 100)
        assert layer.mask.data.sum() == n_keep_expected


def test_prune_to_sparsity_never_revives():
    layer = Linear(6, 6)
    scores_step1 = np.abs(np.random.randn(6, 6))
    prune_to_sparsity(layer, scores_step1, target_sparsity=0.5)
    pruned_after_step1 = layer.mask.data.copy()

    # give previously-pruned entries huge scores: monotonic pruning won't revive
    scores_step2 = np.abs(np.random.randn(6, 6)) + 100.0
    scores_step2[pruned_after_step1 == 0] = 1e6
    prune_to_sparsity(layer, scores_step2, target_sparsity=0.7)

    assert np.all(layer.mask.data[pruned_after_step1 == 0] == 0.0)


def test_prune_to_sparsity_decreasing_target_is_a_noop_not_an_error():
    """A target behind the already-achieved sparsity is a no-op (clamped), not
    an error. See test_rounding_overshoot_does_not_break_schedule for why.
    """
    layer = Linear(4, 4)
    scores = np.abs(np.random.randn(4, 4))
    prune_to_sparsity(layer, scores, target_sparsity=0.5)
    mask_after_first = layer.mask.data.copy()
    prune_to_sparsity(layer, scores, target_sparsity=0.2)  # behind schedule
    assert np.array_equal(layer.mask.data, mask_after_first)  # unchanged


def test_rounding_overshoot_does_not_break_schedule():
    """Regression: a small layer's rounding can overshoot the continuous
    target, so the next scheduled target lands slightly below the achieved
    level even though the schedule is monotonic. That must not crash.
    """
    layer = Linear(16, 16)  # 256 entries, coarse granularity
    scores = np.abs(np.random.randn(16, 16))

    # target that rounds UP the number pruned, overshooting slightly
    prune_to_sparsity(layer, scores, target_sparsity=0.8359375)  # exact: 42/256 kept
    achieved = 1 - layer.mask.data.mean()

    # next target is larger but below the achieved sparsity -- must not raise
    prune_to_sparsity(layer, scores, target_sparsity=achieved - 0.0001)


def test_gradual_schedule_end_to_end():
    """Simulates a multi-step pruning run: at each step apply the target
    sparsity and check the layer stays masked correctly throughout.
    """
    layer = Linear(8, 8)
    start, end, s_f = 0, 80, 0.8
    n_total = layer.weight.data.size

    for step in range(0, end + 1, 10):
        target = cubic_sparsity(step, start, end, s_f)
        scores = np.abs(layer.weight.data)  # magnitude, recomputed each step
        prune_to_sparsity(layer, scores, target)
        n_keep_expected = round((1 - target) * n_total)
        assert layer.mask.data.sum() == n_keep_expected  # exact achievable count, not approximate

    from engine.tensor import Tensor

    x = Tensor(np.random.randn(3, 8), requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert np.all(layer.weight.grad[layer.mask.data == 0] == 0.0)
