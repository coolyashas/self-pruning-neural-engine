import numpy as np

from engine.tensor import Tensor
from nn.linear import Linear
from prune.mask import set_mask
from tests.gradcheck_utils import assert_grad_matches
from utils.seed import set_seed

set_seed(0)


def test_default_mask_is_all_ones_and_excluded_from_parameters():
    layer = Linear(4, 3)
    assert np.allclose(layer.mask.data, 1.0)
    assert layer.mask.requires_grad is False
    assert layer.parameters() == [layer.weight, layer.bias]


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
