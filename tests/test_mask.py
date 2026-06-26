import numpy as np
import pytest

from engine.tensor import Tensor
from nn.linear import Linear
from prune.mask import _top_k_keep_mask, keep_mask_from_scores, set_mask
from tests.gradcheck_utils import assert_grad_matches
from utils.seed import set_seed

set_seed(0)


def test_top_k_keep_mask_rejects_nan_scores():
    """NumPy's argsort sorts NaN to the END (treats it as the LARGEST
    value), so a NaN score would be silently KEPT by top-k regardless of
    its true importance -- a wrong mask, not a crash. Every pruning/
    revival decision in this module funnels through _top_k_keep_mask, so
    guarding it once here protects all of them.
    """
    scores = np.array([1.0, np.nan, 3.0, 2.0])
    with pytest.raises(AssertionError):
        _top_k_keep_mask(scores, n_keep=2)


def test_keep_mask_from_scores_surfaces_the_nan_guard_through_the_public_api():
    scores = np.array([[1.0, np.nan], [3.0, 2.0]])
    with pytest.raises(AssertionError):
        keep_mask_from_scores(scores, sparsity=0.5)


def test_default_mask_is_all_ones_and_excluded_from_parameters():
    layer = Linear(4, 3)
    assert np.allclose(layer.mask.data, 1.0)
    assert layer.mask.requires_grad is False
    assert layer.parameters() == [layer.weight, layer.bias]


def test_set_mask_resyncs_bias_mask_from_column_alive():
    layer = Linear(4, 3)
    assert np.allclose(layer.bias_mask.data, 1.0)  # default: every neuron alive

    keep = np.ones((4, 3))
    keep[:, 1] = 0.0  # column 1 entirely dead -- that neuron has no active weight
    keep[0, 2] = 0.0  # column 2 only partly pruned -- still alive
    set_mask(layer, keep)

    assert np.array_equal(layer.bias_mask.data, [1.0, 0.0, 1.0])


def test_w_eff_grad_is_the_unmasked_dense_gradient():
    """w_eff = weight*mask is itself a graph node; mul()'s backward only
    applies masking on the step FROM w_eff.grad TO weight.grad, so
    w_eff.grad (populated by matmul's backward, with no mask knowledge at
    all) is the dense signal -- "how much would loss change if this
    connection were fully active" -- even at entries where weight.grad is
    exactly 0. This is what regrowth scoring (prune/criteria.py) reads.
    """
    layer = Linear(4, 3)
    keep = np.ones((4, 3))
    keep[1, 2] = 0.0
    set_mask(layer, keep)

    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    layer(x).sum().backward()

    assert layer.weight.grad[1, 2] == 0.0  # masked entry: weight.grad is exactly 0
    assert layer.w_eff.grad[1, 2] != 0.0  # but the dense/unmasked signal isn't
    # away from the masked entry, weight.grad and w_eff.grad agree (mask=1 there)
    assert np.allclose(layer.weight.grad[0, 0], layer.w_eff.grad[0, 0])


def test_masked_weight_gradient_is_exactly_zero():
    """The core requirement: dL/dweight at a masked entry must be exactly
    0, falling out of mul()'s backward (grad * mask), not patched in.
    """
    layer = Linear(4, 3)
    keep = np.ones((4, 3))
    keep[1, 2] = 0.0
    keep[3, 0] = 0.0
    set_mask(layer, keep)

    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    layer(x).sum().backward()

    assert layer.weight.grad[1, 2] == 0.0
    assert layer.weight.grad[3, 0] == 0.0
    # an unmasked entry should NOT be zero (sanity: mask is doing something)
    assert layer.weight.grad[0, 0] != 0.0
    # mask itself never accumulates a gradient -- it's not trained
    assert layer.mask.grad is None


def test_masked_weight_does_not_affect_forward_even_if_changed():
    """Pitfalls doc #8: never destroy the underlying weight, only gate its
    contribution. Proof: changing a masked-off weight's value by a lot
    must not change the layer's output at all.
    """
    layer = Linear(4, 3)
    keep = np.ones((4, 3))
    keep[1, 2] = 0.0
    set_mask(layer, keep)

    x = Tensor(np.random.randn(5, 4))
    out_before = layer(x).data.copy()
    layer.weight.data[1, 2] += 1000.0  # huge change to a masked-off weight
    out_after = layer(x).data
    assert np.allclose(out_before, out_after)


def test_gradcheck_with_partial_mask_matches_finite_difference():
    """Stronger than asserting "implementation says 0": finite-difference
    on the *masked-off* entries should independently land at ~0 too, since
    perturbing a weight that's multiplied by mask=0 can't change the loss.
    """
    keep = np.ones((4, 3))
    keep[0, 1] = 0.0
    keep[2, 0] = 0.0
    X = np.random.randn(5, 4)
    W = np.random.randn(4, 3)
    B = np.random.randn(3)

    def tensor_op(Xt, Wt, Bt):
        Mt = Tensor(keep, requires_grad=False)
        return (Xt @ (Wt * Mt) + Bt).sum()

    def numpy_op(Xa, Wa, Ba):
        return (Xa @ (Wa * keep) + Ba).sum()

    assert_grad_matches(tensor_op, numpy_op, [X, W, B])


def test_linear_with_mask_end_to_end_backward():
    layer = Linear(4, 3)
    keep = np.ones((4, 3))
    keep[1, 2] = 0.0
    set_mask(layer, keep)

    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    layer(x).sum().backward()

    assert layer.weight.grad[1, 2] == 0.0
    assert layer.mask.requires_grad is False
    assert layer.mask.grad is None
    assert layer.parameters() == [layer.weight, layer.bias]
