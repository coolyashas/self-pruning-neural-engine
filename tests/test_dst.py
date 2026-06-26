"""Tests run_exchange_cycle -- the grow+drop exchange step at the heart
of dynamic sparse training. The repeated-thrashing test at the bottom is
the most important one here, in the same role test_masked_adam.py's
suite played for the first half of this project: it's the concrete
answer to "reason about the stability implications of regrowth."
"""

import numpy as np

from engine.tensor import Tensor
from nn.activations import ReLU
from nn.linear import Linear
from nn.sequential import Sequential
from optim.adam import Adam
from prune.criteria import accumulate_gradients, magnitude_scores
from prune.dst import dst_step, run_exchange_cycle
from prune.mask import prune_to_sparsity, set_mask
from prune.schedule import cubic_sparsity
from utils.seed import set_seed

set_seed(0)


class _DummyOptimizer:
    """Stands in where only mask mechanics are under test, not Adam's
    own moment-reset behavior (that's covered separately, with real
    Adam, below)."""

    def reset_state(self, param, indices):
        pass


def test_run_exchange_cycle_is_net_zero_active_count():
    layer = Linear(6, 6)
    prune_to_sparsity(layer, np.abs(np.random.randn(6, 6)), target_sparsity=0.5)
    n_active_before = int(layer.mask.data.sum())

    for n_exchange in [0, 1, 5, 100]:  # 100 forces the clamp path
        drop_scores = np.abs(np.random.randn(6, 6))
        grow_scores = np.abs(np.random.randn(6, 6))
        run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange, _DummyOptimizer())
        assert int(layer.mask.data.sum()) == n_active_before


def test_run_exchange_cycle_excludes_just_revived_from_drop():
    """The landmine this function is built around: a freshly-revived
    entry must survive its first cycle even if it would otherwise be the
    single lowest-scoring entry by drop_scores -- it hasn't had a chance
    to prove itself yet. A naive single-bucket "excluded" implementation
    would let still-inactive entries usurp its slot instead; this proves
    the three-way scoring (revived=+inf, still-inactive=-inf, original
    real score) gets it right.
    """
    layer = Linear(4, 4)
    keep = np.zeros((4, 4))
    keep[0, :] = 1.0  # row 0 active, everything else inactive
    set_mask(layer, keep)

    grow_scores = np.zeros((4, 4))
    grow_scores[2, 0] = 100.0  # (2,0) wins the grow competition

    drop_scores = np.full((4, 4), 5.0)
    drop_scores[2, 0] = -999.0  # lowest possible -- would be dropped first if eligible
    drop_scores[0, 0] = 1.0  # the real lowest-scoring ORIGINAL active entry

    run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange=1, optimizer=_DummyOptimizer())

    assert layer.mask.data[2, 0] == 1.0  # survives despite the worst drop score
    assert layer.mask.data[0, 0] == 0.0  # this one gets dropped instead
    assert layer.mask.data.sum() == 4.0  # net-zero


def test_revived_entry_gets_exactly_zero_fresh_moments_in_integrated_path():
    """Not just reset_state in isolation (already tested in
    test_masked_adam.py) -- the wiring through run_exchange_cycle itself:
    force a specific index to be the one revived, and check its m/v are
    exactly 0 right after the cycle.
    """
    layer = Linear(3, 3)
    pairs = layer.masked_parameters()
    opt = Adam([p for p, _ in pairs], lr=0.1, masks=[m for _, m in pairs])

    for _ in range(10):
        opt.zero_grad()
        layer(Tensor(np.random.randn(5, 3))).sum().backward()
        opt.step()

    keep = np.ones((3, 3))
    keep[1, 1] = 0.0  # mask this one off, but it already has real momentum
    set_mask(layer, keep)
    assert opt.m[0][1, 1] != 0.0  # stale momentum from before masking, still sitting there

    grow_scores = np.zeros((3, 3))
    grow_scores[1, 1] = 1.0  # only this masked-off entry is a candidate
    drop_scores = np.abs(np.random.randn(3, 3))
    run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange=1, optimizer=opt)

    assert layer.mask.data[1, 1] == 1.0  # confirmed revived
    assert opt.m[0][1, 1] == 0.0
    assert opt.v[0][1, 1] == 0.0


def test_repeated_prune_revive_prune_same_connection_does_not_corrupt_state():
    """The concrete answer to 'reason about the stability implications of
    regrowth': cycle one specific connection through prune -> revive ->
    prune -> revive several times in a row, and prove nothing accumulates
    incorrectly -- the underlying weight value never moves while masked
    (mask gates, never mutates), moments are exactly reset on every
    single revival (not just the first), nothing ever goes non-finite,
    and the state after thrashing is indistinguishable from a fresh
    single-revive scenario fed the same final gradient.
    """
    layer = Linear(3, 3)
    pairs = layer.masked_parameters()
    opt = Adam([p for p, _ in pairs], lr=0.1, masks=[m for _, m in pairs])
    target = (1, 1)

    for _ in range(10):  # build up real momentum everywhere first
        opt.zero_grad()
        layer(Tensor(np.random.randn(5, 3))).sum().backward()
        opt.step()

    for cycle in range(5):
        # prune: mask target off (give it the lowest drop score, everything else high)
        drop_scores = np.full((3, 3), 100.0)
        drop_scores[target] = -100.0
        keep = layer.mask.data.copy().astype(bool)
        keep[target] = False  # forced off directly via set_mask -- simulates a real prune call
        set_mask(layer, keep)

        # the value at THIS freeze, not the very first one: it's allowed
        # (and expected) to have moved during the previous cycle's active
        # training steps -- the invariant under test is "frozen while
        # masked," not "permanently pinned to its first-ever frozen value."
        weight_value_at_this_freeze = layer.weight.data[target]

        # a few real training steps while masked
        for _ in range(3):
            opt.zero_grad()
            layer(Tensor(np.random.randn(5, 3))).sum().backward()
            opt.step()

        assert np.isfinite(opt.m[0]).all() and np.isfinite(opt.v[0]).all()
        assert np.isfinite(layer.weight.data).all()
        assert layer.weight.data[target] == weight_value_at_this_freeze  # still frozen

        # revive via run_exchange_cycle, forcing target to win the grow competition
        grow_scores = np.zeros((3, 3))
        grow_scores[target] = 1.0
        drop_scores = np.abs(np.random.randn(3, 3))
        run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange=1, optimizer=opt)

        assert layer.mask.data[target] == 1.0  # confirmed revived this cycle
        assert opt.m[0][target] == 0.0  # reset, every single time, not just the first
        assert opt.v[0][target] == 0.0
        assert layer.weight.data[target] == weight_value_at_this_freeze  # revival itself doesn't change the value

        # a few real training steps while active -- the value is now
        # free to move again, that's normal training, not corruption
        for _ in range(3):
            opt.zero_grad()
            layer(Tensor(np.random.randn(5, 3))).sum().backward()
            opt.step()
        assert np.isfinite(opt.m[0]).all() and np.isfinite(opt.v[0]).all()
        assert np.isfinite(layer.weight.data).all()

    # final check: after all that thrashing, one more prune+revive cycle's
    # post-revival update must match a hand-derived FRESH Adam step exactly
    # -- the same exact-formula-match bar test_masked_adam.py uses, proving
    # no path-dependent residue survives the thrashing.
    keep = layer.mask.data.copy().astype(bool)
    keep[target] = False
    set_mask(layer, keep)
    grow_scores = np.zeros((3, 3))
    grow_scores[target] = 1.0
    run_exchange_cycle(layer, np.abs(np.random.randn(3, 3)), grow_scores, n_exchange=1, optimizer=opt)
    assert opt.m[0][target] == 0.0 and opt.v[0][target] == 0.0

    layer.weight.grad = np.zeros((3, 3))
    layer.weight.grad[target] = 2.0
    weight_before = layer.weight.data[target]
    opt.step()

    b1, b2, eps, lr, t = opt.beta1, opt.beta2, opt.eps, opt.lr, opt.t
    m_hat = ((1 - b1) * 2.0) / (1 - b1**t)
    v_hat = ((1 - b2) * 4.0) / (1 - b2**t)
    expected = weight_before - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(layer.weight.data[target], expected, atol=1e-10)


def _build_mlp_with_copied_weights(source: Sequential) -> Sequential:
    """A second model, identical weights/bias to `source`, so a dst_step
    call on one and the manual equivalent on the other can be compared
    bit-for-bit -- there's no built-in deep-copy for Linear/Sequential.
    """
    copy = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    for src, dst in zip(source.layers, copy.layers):
        if hasattr(src, "weight"):
            dst.weight.data = src.weight.data.copy()
            dst.bias.data = src.bias.data.copy()
    return copy


def test_dst_step_ramp_phase_matches_existing_prune_to_sparsity_behavior():
    """dst_step's ramp branch (step < prune_end_step) must produce a mask
    byte-identical to run_part3's existing non-regrowth on_step_end
    (accumulate_gradients + prune_to_sparsity directly) given the same
    inputs -- a regression-style cross-check between the two code paths.
    """
    mlp_a = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    mlp_b = _build_mlp_with_copied_weights(mlp_a)

    X = np.random.randn(20, 2)
    y = np.random.randint(0, 3, size=20)

    pairs_a = mlp_a.masked_parameters()
    opt_a = Adam([p for p, _ in pairs_a], lr=0.01, masks=[m for _, m in pairs_a])

    step, prune_start_step, prune_end_step, final_sparsity = 10, 5, 50, 0.6

    dst_step(
        mlp_a,
        opt_a,
        X,
        y,
        batch_size=10,
        step=step,
        prune_start_step=prune_start_step,
        prune_end_step=prune_end_step,
        final_sparsity=final_sparsity,
        drop_score_fn=magnitude_scores,
    )

    # manual equivalent of run_part3's existing non-regrowth on_step_end
    target = cubic_sparsity(step, prune_start_step, prune_end_step, final_sparsity)
    accumulate_gradients(mlp_b, X, y, batch_size=10)
    for layer in [layer for layer in mlp_b.layers if hasattr(layer, "mask")]:
        prune_to_sparsity(layer, magnitude_scores(layer), target)

    prunable_a = [layer for layer in mlp_a.layers if hasattr(layer, "mask")]
    prunable_b = [layer for layer in mlp_b.layers if hasattr(layer, "mask")]
    for layer_a, layer_b in zip(prunable_a, prunable_b):
        assert np.array_equal(layer_a.mask.data, layer_b.mask.data)


def test_dst_step_maintenance_phase_does_one_sweep_not_two(monkeypatch):
    """accumulate_dense_gradients's backward() pass already leaves
    weight.grad correctly populated (same backward call that produces
    w_eff.grad also writes weight.grad, via mul()'s backward) -- a
    second accumulate_gradients sweep would silently redo identical
    work. Pin this by making accumulate_gradients raise if dst_step
    ever calls it again during the maintenance phase.
    """
    import prune.dst as dst_module

    mlp = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    pairs = mlp.masked_parameters()
    opt = Adam([p for p, _ in pairs], lr=0.01, masks=[m for _, m in pairs])
    prunable = [layer for layer in mlp.layers if hasattr(layer, "mask")]

    X = np.random.randn(20, 2)
    y = np.random.randint(0, 3, size=20)

    accumulate_gradients(mlp, X, y, batch_size=10)
    for layer in prunable:
        prune_to_sparsity(layer, magnitude_scores(layer), target_sparsity=0.5)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("dst_step's maintenance phase must not sweep the dataset twice")

    monkeypatch.setattr(dst_module, "accumulate_gradients", _fail_if_called)

    dst_step(
        mlp,
        opt,
        X,
        y,
        batch_size=10,
        step=100,
        prune_start_step=5,
        prune_end_step=50,
        final_sparsity=0.5,
        drop_score_fn=magnitude_scores,
    )  # must not raise


def test_dst_step_maintenance_phase_keeps_sparsity_constant_over_many_cycles():
    """Once final_sparsity is reached, repeated dst_step calls must hold
    each layer's active count exactly constant -- the "roughly constant"
    DST claim, measured across many consecutive cycles, not just one.
    """
    mlp = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    pairs = mlp.masked_parameters()
    opt = Adam([p for p, _ in pairs], lr=0.01, masks=[m for _, m in pairs])
    prunable = [layer for layer in mlp.layers if hasattr(layer, "mask")]

    X = np.random.randn(20, 2)
    y = np.random.randint(0, 3, size=20)

    # pre-prune to final_sparsity directly, simulating "ramp already done"
    final_sparsity = 0.5
    accumulate_gradients(mlp, X, y, batch_size=10)
    for layer in prunable:
        prune_to_sparsity(layer, magnitude_scores(layer), final_sparsity)
    active_counts_before = [int(layer.mask.data.sum()) for layer in prunable]

    prune_start_step, prune_end_step = 5, 50
    for step in range(prune_end_step, prune_end_step + 10 * 3, 3):  # several maintenance cycles
        dst_step(
            mlp,
            opt,
            X,
            y,
            batch_size=10,
            step=step,
            prune_start_step=prune_start_step,
            prune_end_step=prune_end_step,
            final_sparsity=final_sparsity,
            drop_score_fn=magnitude_scores,
            exchange_fraction=0.2,
        )
        active_counts_now = [int(layer.mask.data.sum()) for layer in prunable]
        assert active_counts_now == active_counts_before
