import numpy as np
import pytest

from engine.tensor import Tensor
from nn import Linear, ReLU, Sequential
from prune.compress import compress_model
from prune.criteria import neuron_magnitude_scores
from prune.mask import prune_neurons_to_count
from utils.seed import set_seed

set_seed(0)


def _make_mlp():
    return Sequential(Linear(2, 8), ReLU(), Linear(8, 5), ReLU(), Linear(5, 3))


def _dense_forward(mlp, x):
    return mlp(Tensor(x)).data


@pytest.mark.parametrize(
    "target_l1,target_l2",
    [
        (8, 5),  # 0% structurally pruned -- compression should be a faithful identity
        (3, 1),  # high sparsity, middle layer keeps only 1 neuron
        (5, 5),  # only the first hidden layer pruned
    ],
)
def test_compress_matches_masked_forward_exactly(target_l1, target_l2):
    mlp = _make_mlp()
    prune_neurons_to_count(mlp.layers[0], neuron_magnitude_scores(mlp.layers[0]), target_l1)
    prune_neurons_to_count(mlp.layers[2], neuron_magnitude_scores(mlp.layers[2]), target_l2)

    compressed = compress_model(mlp)

    for seed_offset in range(3):
        x = np.random.randn(7, 2)
        dense_out = _dense_forward(mlp, x)
        compressed_out = compressed(x)
        assert np.allclose(dense_out, compressed_out, atol=1e-12)


def test_compress_actually_shrinks_the_matrices():
    mlp = _make_mlp()
    prune_neurons_to_count(mlp.layers[0], neuron_magnitude_scores(mlp.layers[0]), 3)
    prune_neurons_to_count(mlp.layers[2], neuron_magnitude_scores(mlp.layers[2]), 2)

    compressed = compress_model(mlp)
    assert compressed.layers[0].weight.shape == (2, 3)  # input untouched, output shrunk to 3
    assert compressed.layers[1].__class__.__name__ == "CompressedReLU"
    assert compressed.layers[2].weight.shape == (3, 2)  # input shrunk to match layer 1's output
    assert compressed.layers[3].__class__.__name__ == "CompressedReLU"
    assert compressed.layers[4].weight.shape == (2, 3)  # input shrunk, output (logits) untouched


def test_compress_last_layer_output_never_sliced_even_if_masked():
    """Zeroing a whole output column on the FINAL layer must not shrink the
    output width -- dropping a class changes meaning, not just size.
    """
    mlp = _make_mlp()
    mlp.layers[-1].mask.data[:, 1] = 0.0  # simulate an accidental full-column prune on output

    compressed = compress_model(mlp)
    assert compressed.layers[-1].weight.shape[1] == 3
    assert compressed.layers[-1].bias.shape[0] == 3

    # masked weight is zero in w_eff, but bias (never masked) still applies
    x = np.random.randn(4, 2)
    assert np.allclose(_dense_forward(mlp, x), compressed(x), atol=1e-12)


def test_compress_matches_dense_forward_with_nonzero_bias():
    """Regression: a pruned neuron with NONZERO bias emits ReLU(0 + bias), not
    0. compress_model assumes a pruned neuron contributes nothing, which holds
    only because prune_neurons_to_count zeros its bias.
    """
    mlp = _make_mlp()
    for layer in mlp.layers:
        if hasattr(layer, "bias"):
            layer.bias.data = np.random.randn(*layer.bias.data.shape)  # nonzero, post-training-like

    prune_neurons_to_count(mlp.layers[0], neuron_magnitude_scores(mlp.layers[0]), 4)
    prune_neurons_to_count(mlp.layers[2], neuron_magnitude_scores(mlp.layers[2]), 3)

    compressed = compress_model(mlp)
    x = np.random.randn(6, 2)
    assert np.allclose(_dense_forward(mlp, x), compressed(x), atol=1e-10)


def test_compress_does_not_alias_original_weights():
    mlp = _make_mlp()
    prune_neurons_to_count(mlp.layers[0], neuron_magnitude_scores(mlp.layers[0]), 4)
    compressed = compress_model(mlp)

    assert not np.shares_memory(compressed.layers[0].weight, mlp.layers[0].weight.data)
    assert not np.shares_memory(compressed.layers[0].bias, mlp.layers[0].bias.data)

    original_weight = mlp.layers[0].weight.data.copy()
    compressed.layers[0].weight[:] = 999.0
    assert np.array_equal(mlp.layers[0].weight.data, original_weight)  # untouched

    original_compressed = compressed.layers[2].weight.copy()
    mlp.layers[2].weight.data[:] = -999.0
    assert np.array_equal(compressed.layers[2].weight, original_compressed)  # untouched


def test_compress_unsupported_layer_type_raises():
    class NotALayer:
        def __call__(self, x):
            return x

    mlp = Sequential(Linear(2, 4), NotALayer())
    with pytest.raises(NotImplementedError):
        compress_model(mlp)


def test_compress_model_with_no_linear_layers_raises_clearly():
    """A model with no Linear layers has nothing to compress; fail with a clear
    message rather than a bare IndexError from linear_layers[-1].
    """
    mlp = Sequential(ReLU())
    with pytest.raises(AssertionError, match="no Linear layers"):
        compress_model(mlp)
