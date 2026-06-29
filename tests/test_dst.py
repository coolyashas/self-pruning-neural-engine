"""Tests run_exchange_cycle -- the grow+drop exchange step of dynamic sparse
training. The repeated-thrashing test at the bottom is the key one: the
concrete answer to "reason about the stability implications of regrowth."
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
    moment-reset (covered separately with real Adam below)."""

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


def test_run_exchange_cycle_is_a_no_op_on_a_fully_dead_layer():
    """At n_active_before == 0 the drop keep-target is also 0, so even a
    +inf-forced entry gets dropped. Without the guard, a revive would be
    immediately re-dropped; confirm the guard makes it a true no-op.
    """
    layer = Linear(3, 3)
    set_mask(layer, np.zeros((3, 3)))  # fully dead
    assert layer.mask.data.sum() == 0.0

    grow_scores = np.ones((3, 3))
    drop_scores = np.abs(np.random.randn(3, 3))
    run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange=1, optimizer=_DummyOptimizer())

    assert layer.mask.data.sum() == 0.0  # no revive-then-drop churn


def test_run_exchange_cycle_excludes_just_revived_from_drop():
    """A freshly-revived entry must survive its first cycle even if it's the
    lowest-scoring by drop_scores. Proves the three-way scoring (revived=+inf,
    still-inactive=-inf, original real score) gets it right.
    """
    layer = Linear(4, 4)
    keep = np.zeros((4, 4))
    keep[0, :] = 1.0  # row 0 active, rest inactive
    set_mask(layer, keep)

    grow_scores = np.zeros((4, 4))
    grow_scores[2, 0] = 100.0  # (2,0) wins the grow competition

    drop_scores = np.full((4, 4), 5.0)
    drop_scores[2, 0] = -999.0  # would be dropped first if eligible
    drop_scores[0, 0] = 1.0  # real lowest-scoring original active entry

    run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange=1, optimizer=_DummyOptimizer())

    assert layer.mask.data[2, 0] == 1.0  # survives despite worst drop score
    assert layer.mask.data[0, 0] == 0.0  # dropped instead
    assert layer.mask.data.sum() == 4.0  # net-zero


def test_revived_entry_gets_exactly_zero_fresh_moments_in_integrated_path():
    """The wiring through run_exchange_cycle itself (not reset_state in
    isolation): force one index to be revived, check its m/v are exactly 0.
    """
    layer = Linear(3, 3)
    pairs = layer.masked_parameters()
    opt = Adam([p for p, _ in pairs], lr=0.1, masks=[m for _, m in pairs])

    for _ in range(10):
        opt.zero_grad()
        layer(Tensor(np.random.randn(5, 3))).sum().backward()
        opt.step()

    keep = np.ones((3, 3))
    keep[1, 1] = 0.0  # mask off, but it already has real momentum
    set_mask(layer, keep)
    assert opt.m[0][1, 1] != 0.0  # stale momentum from before masking

    grow_scores = np.zeros((3, 3))
    grow_scores[1, 1] = 1.0  # only this masked-off entry is a candidate
    drop_scores = np.abs(np.random.randn(3, 3))
    run_exchange_cycle(layer, drop_scores, grow_scores, n_exchange=1, optimizer=opt)

    assert layer.mask.data[1, 1] == 1.0  # confirmed revived
    assert opt.m[0][1, 1] == 0.0
    assert opt.v[0][1, 1] == 0.0


def test_revive_un_freezing_a_dead_neurons_bias_resets_its_stale_moments():
    """Reviving a weight into a fully-dead column flips its neuron's bias_mask
    back to active (set_mask resyncs bias_mask = mask.any(axis=0)). Without a
    reset, the bias's next update would use stale pre-death momentum -- the
    same bug mask-aware Adam prevents, on the bias side.
    """
    from prune.mask import prune_neurons_to_count

    layer = Linear(3, 3)
    pairs = layer.masked_parameters()
    opt = Adam([p for p, _ in pairs], lr=0.1, masks=[m for _, m in pairs])

    for _ in range(15):  # build up real momentum everywhere, including bias
        opt.zero_grad()
        layer(Tensor(np.random.randn(5, 3))).sum().backward()
        opt.step()
    stale_bias_m = opt.m[1][1]
    stale_bias_v = opt.v[1][1]
    assert stale_bias_m != 0.0 and stale_bias_v != 0.0

    # deterministic scores so neuron 1 is killed regardless of RNG/run order.
    forced_kill_scores = np.array([10.0, -10.0, 10.0])
    prune_neurons_to_count(layer, forced_kill_scores, target_active=2)  # kills neuron 1
    assert layer.bias_mask.data[1] == 0.0
    assert opt.m[1][1] == stale_bias_m  # frozen, untouched by the freeze

    grow_scores = np.zeros((3, 3))
    grow_scores[0, 1] = 100.0  # force-revive a weight feeding dead neuron 1
    run_exchange_cycle(layer, np.abs(np.random.randn(3, 3)), grow_scores, n_exchange=1, optimizer=opt)

    assert layer.bias_mask.data[1] == 1.0  # neuron 1 un-frozen by the revival
    assert opt.m[1][1] == 0.0  # clean reset, not the stale pre-death value
    assert opt.v[1][1] == 0.0


def test_repeated_prune_revive_prune_same_connection_does_not_corrupt_state():
    """The concrete answer to 'stability implications of regrowth': cycle one
    connection through prune->revive several times and prove nothing accumulates
    wrongly -- the weight never moves while masked, moments reset on every
    revival, nothing goes non-finite, and the final state matches a fresh
    single-revive fed the same gradient.
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
        # prune: mask target off
        drop_scores = np.full((3, 3), 100.0)
        drop_scores[target] = -100.0
        keep = layer.mask.data.copy().astype(bool)
        keep[target] = False
        set_mask(layer, keep)

        # value at THIS freeze (it may have moved during the previous active
        # cycle); the invariant is "frozen while masked", not "pinned forever".
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

        assert layer.mask.data[target] == 1.0  # revived this cycle
        assert opt.m[0][target] == 0.0  # reset every cycle, not just the first
        assert opt.v[0][target] == 0.0
        assert layer.weight.data[target] == weight_value_at_this_freeze  # revival doesn't move it

        # real training steps while active -- value is free to move again now
        for _ in range(3):
            opt.zero_grad()
            layer(Tensor(np.random.randn(5, 3))).sum().backward()
            opt.step()
        assert np.isfinite(opt.m[0]).all() and np.isfinite(opt.v[0]).all()
        assert np.isfinite(layer.weight.data).all()

    # final check: after thrashing, one more prune+revive's post-revival update
    # must match a hand-derived FRESH Adam step, proving no residue survives.
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
    """A second model with identical weights/bias to `source`, for bit-for-bit
    comparison (no built-in deep-copy for Linear/Sequential).
    """
    copy = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    for src, dst in zip(source.layers, copy.layers):
        if hasattr(src, "weight"):
            dst.weight.data = src.weight.data.copy()
            dst.bias.data = src.bias.data.copy()
    return copy


def test_dst_step_ramp_phase_matches_existing_prune_to_sparsity_behavior():
    """dst_step's ramp branch must produce a mask byte-identical to
    accumulate_gradients + prune_to_sparsity directly, given the same inputs.
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
    """accumulate_dense_gradients already populates weight.grad (same backward
    that produces w_eff.grad), so a second sweep is redundant. Pin this by
    making accumulate_gradients raise if the maintenance phase calls it.
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
    """Once final_sparsity is reached, repeated dst_step calls must hold each
    layer's active count exactly constant across many cycles.
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
