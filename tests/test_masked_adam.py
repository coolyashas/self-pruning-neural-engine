"""The single most important test in this repo: proves mask-aware Adam is
correct. A masked weight's gradient is exactly 0 (falls out of w*mask, see
test_mask.py), its Adam moments m/v are frozen -- not just decaying --
while masked, and a revived weight starts from a clean moment state
rather than inheriting stale momentum that would otherwise cause an
oversized or wrong-direction first step.
"""

import numpy as np

from engine.tensor import Tensor
from nn.linear import Linear
from optim.adam import Adam
from prune.mask import set_mask
from utils.seed import set_seed

set_seed(0)


def test_masked_weight_stays_exactly_zero_across_many_steps():
    p = Tensor(np.array([0.0, 5.0, -3.0]), requires_grad=True)
    mask = Tensor(np.array([0.0, 1.0, 1.0]), requires_grad=False)  # index 0 masked, starts at 0
    opt = Adam([p], lr=0.5, masks=[mask])

    for _ in range(50):
        p.grad = np.array([1.0, 1.0, 1.0])  # nonzero grad everywhere would normally move it
        opt.step()

    assert p.data[0] == 0.0  # exactly 0, not "close to"
    assert p.data[1] != 5.0 and p.data[2] != -3.0  # unmasked entries did move


def test_masked_weight_frozen_at_arbitrary_nonzero_value():
    """Pitfalls doc #8: masking must not force the value to 0, just freeze
    whatever it already was. Nonzero start rules out an implementation
    that secretly zeroes masked weights instead of gating them.
    """
    p = Tensor(np.array([7.5]), requires_grad=True)
    mask = Tensor(np.array([0.0]), requires_grad=False)
    opt = Adam([p], lr=0.5, masks=[mask])
    for _ in range(50):
        p.grad = np.array([10.0])
        opt.step()
    assert p.data[0] == 7.5


def test_masked_moments_do_not_drift():
    """Not just 'no NaN': m and v must stay bitwise unchanged while
    masked, not slowly decay toward zero via the EMA recursion.
    """
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    mask = Tensor(np.array([1.0, 1.0]), requires_grad=False)
    opt = Adam([p], lr=0.1, masks=[mask])

    for _ in range(10):  # build up real, nonzero momentum first
        p.grad = np.array([1.0, 1.0])
        opt.step()
    m_at_mask_time = opt.m[0].copy()
    v_at_mask_time = opt.v[0].copy()
    assert not np.allclose(m_at_mask_time, 0.0)

    mask.data[1] = 0.0  # mask off index 1 now that it has real momentum
    for _ in range(20):
        p.grad = np.array([1.0, 1.0])
        opt.step()

    assert opt.m[0][1] == m_at_mask_time[1]
    assert opt.v[0][1] == v_at_mask_time[1]
    assert opt.m[0][0] != m_at_mask_time[0]  # never-masked index kept evolving


def test_revived_weight_starts_from_clean_moment_state():
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    mask = Tensor(np.array([1.0, 1.0]), requires_grad=False)
    opt = Adam([p], lr=0.1, masks=[mask])
    for _ in range(10):
        p.grad = np.array([1.0, 1.0])
        opt.step()
    assert not np.allclose(opt.m[0], 0.0)

    mask.data[1] = 0.0
    for _ in range(10):
        p.grad = np.array([1.0, 1.0])
        opt.step()

    mask.data[1] = 1.0  # revive
    opt.reset_state(p, 1)
    assert opt.m[0][1] == 0.0
    assert opt.v[0][1] == 0.0

    # first post-revival update should match a *fresh* Adam step from m=v=0
    p.grad = np.array([1.0, 5.0])
    weight_before = p.data[1].copy()
    opt.step()

    b1, b2, eps, lr, t = opt.beta1, opt.beta2, opt.eps, opt.lr, opt.t
    m_hat = ((1 - b1) * 5.0) / (1 - b1**t)
    v_hat = ((1 - b2) * 25.0) / (1 - b2**t)
    expected_weight = weight_before - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(p.data[1], expected_weight, atol=1e-10)


def test_without_reset_state_revival_inherits_stale_momentum():
    """The bug being guarded against: skipping reset_state on revival lets
    old momentum leak into the first post-revival step, producing a
    DIFFERENT update than the clean-state version gets for the identical
    scenario.
    """

    def run(reset: bool):
        p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        mask = Tensor(np.array([1.0, 1.0]), requires_grad=False)
        opt = Adam([p], lr=0.1, masks=[mask])
        for _ in range(10):
            p.grad = np.array([1.0, -1.0])  # negative grad -> momentum builds up negative
            opt.step()
        mask.data[1] = 0.0
        for _ in range(10):
            p.grad = np.array([1.0, -1.0])
            opt.step()
        mask.data[1] = 1.0
        if reset:
            opt.reset_state(p, 1)
        p.grad = np.array([1.0, 5.0])  # gradient direction flips
        before = p.data[1]
        opt.step()
        return p.data[1] - before

    assert not np.isclose(run(reset=True), run(reset=False))


def test_structurally_dead_neurons_bias_stays_exactly_zero_under_continued_training():
    """Regression for a real bug found running an actual Part-4 training
    script, not a synthetic case: bias has no mask of its own degrading
    its weight column -- a structurally-pruned neuron's bias kept
    drifting via stale Adam momentum even after its weight was correctly
    frozen, so the dense forward still emitted ReLU(0 + bias) for a
    "dead" neuron instead of exactly 0 (which prune/compress.py's
    compress_model silently assumed). bias_mask (nn/linear.py, synced by
    prune.mask.set_mask) closes this the same way mask-aware Adam already
    closes it for weight.
    """
    from prune.criteria import neuron_magnitude_scores
    from prune.mask import prune_neurons_to_count

    layer = Linear(4, 6)
    pairs = layer.masked_parameters()
    params = [p for p, _ in pairs]
    masks = [m for _, m in pairs]
    opt = Adam(params, lr=0.1, masks=masks)

    for _ in range(10):  # build up real, nonzero bias momentum first
        opt.zero_grad()
        layer(Tensor(np.random.randn(5, 4))).sum().backward()
        opt.step()
    assert not np.allclose(layer.bias.data, 0.0)  # sanity: bias actually moved

    prune_neurons_to_count(layer, neuron_magnitude_scores(layer), target_active=4)  # kills 2 neurons
    dead = ~layer.mask.data.any(axis=0)
    assert dead.sum() == 2
    assert np.all(layer.bias.data[dead] == 0.0)  # zeroed immediately

    for _ in range(30):  # continued training, generic loss, no special-casing
        opt.zero_grad()
        layer(Tensor(np.random.randn(5, 4))).sum().backward()
        opt.step()

    assert np.all(layer.bias.data[dead] == 0.0)  # still exactly 0 -- no drift


def test_masked_adam_end_to_end_with_real_backward():
    """Combines masking (commit 16) with mask-aware Adam (commit 17)
    through a real Linear layer and real backward(), not hand-set grads.
    """
    layer = Linear(4, 3)
    keep = np.ones((4, 3))
    keep[1, 2] = 0.0
    set_mask(layer, keep)

    opt = Adam([layer.weight, layer.bias], lr=0.1, masks=[layer.mask, None])
    weight_at_masked_entry = layer.weight.data[1, 2]

    for _ in range(30):
        opt.zero_grad()
        loss = layer(Tensor(np.random.randn(5, 4))).sum()
        loss.backward()
        opt.step()

    assert layer.weight.data[1, 2] == weight_at_masked_entry
    assert opt.m[0][1, 2] == 0.0
    assert opt.v[0][1, 2] == 0.0
