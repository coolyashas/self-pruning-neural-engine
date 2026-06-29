from pathlib import Path

import numpy as np
import pytest

from engine.tensor import Tensor
from nn.linear import Linear
from prune.mask import _top_k_keep_mask, keep_mask_from_scores, set_mask
from tests.gradcheck_utils import assert_grad_matches
from utils.seed import set_seed

set_seed(0)


def test_top_k_keep_mask_rejects_nan_scores():
    """argsort sorts NaN to the END (as largest), so a NaN score would be
    silently KEPT. Every decision funnels through _top_k_keep_mask, so one
    guard here protects all of them.
    """
    scores = np.array([1.0, np.nan, 3.0, 2.0])
    with pytest.raises(ValueError):
        _top_k_keep_mask(scores, n_keep=2)


def test_keep_mask_from_scores_surfaces_the_nan_guard_through_the_public_api():
    scores = np.array([[1.0, np.nan], [3.0, 2.0]])
    with pytest.raises(ValueError):
        keep_mask_from_scores(scores, sparsity=0.5)


def test_set_mask_rejects_a_shape_mismatched_mask():
    layer = Linear(4, 3)
    with pytest.raises(ValueError):
        set_mask(layer, np.ones((3, 4)))  # transposed shape, easy mistake


def test_nan_guard_and_set_mask_shape_guard_survive_python_dash_O():
    """Both guards prevent a silent wrong answer (NaN ranked highest; a
    shape-mismatched mask broadcasting into garbage) and are `if: raise`, not
    `assert`. Spawn a real -O subprocess to confirm both still raise there.
    """
    import subprocess
    import sys

    nan_code = (
        "import numpy as np\n"
        "from prune.mask import _top_k_keep_mask\n"
        "_top_k_keep_mask(np.array([1.0, np.nan, 3.0, 2.0]), n_keep=2)\n"
    )
    shape_code = (
        "import numpy as np\n"
        "from nn.linear import Linear\n"
        "from prune.mask import set_mask\n"
        "set_mask(Linear(4, 3), np.ones((3, 4)))\n"
    )
    for code in (nan_code, shape_code):
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "expected a ValueError under -O, but the call succeeded silently"
        assert "ValueError" in result.stderr


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


def test_set_mask_zeros_bias_of_a_neuron_killed_by_plain_unstructured_pruning():
    """Plain per-weight pruning can zero an entire column, not just
    prune_neurons_to_count. set_mask must zero that neuron's bias VALUE too,
    not just freeze it -- otherwise it emits a constant ReLU(0 + stale_bias).
    """
    layer = Linear(3, 4)
    layer.bias.data[1] = 2.5  # nonzero on purpose; default-zero would hide the bug

    keep = np.ones((3, 4))
    keep[:, 1] = 0.0  # every incoming weight to neuron 1 pruned
    set_mask(layer, keep)

    assert layer.bias_mask.data[1] == 0.0  # frozen going forward
    assert layer.bias.data[1] == 0.0  # AND zeroed, not left at 2.5

    x = Tensor(np.random.randn(2, 3))
    out = layer(x)
    assert np.all(out.data[:, 1] == 0.0)  # neuron 1 contributes nothing


def test_w_eff_grad_is_the_unmasked_dense_gradient():
    """w_eff = weight*mask is a graph node; mul()'s backward applies masking
    only from w_eff.grad to weight.grad, so w_eff.grad is the dense signal even
    where weight.grad is 0. This is what regrowth scoring reads.
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
    """dL/dweight at a masked entry must be exactly 0, falling out of mul()'s
    backward (grad * mask), not patched in.
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
    """Never destroy the underlying weight, only gate its contribution:
    changing a masked-off weight's value must not change the output.
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
    """Finite-difference on the masked-off entries independently lands at ~0,
    since perturbing a weight multiplied by mask=0 can't change the loss.
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
