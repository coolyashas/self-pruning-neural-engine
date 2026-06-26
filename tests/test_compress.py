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
    """A caller mistakenly zeroing a whole output column on the FINAL
    layer (e.g. via a bug upstream) must not shrink the output width --
    dropping a class changes the model's meaning, not just its size.
    """
    mlp = _make_mlp()
    mlp.layers[-1].mask.data[:, 1] = 0.0  # simulate an accidental full-column prune on output

    compressed = compress_model(mlp)
    assert compressed.layers[-1].weight.shape[1] == 3
    assert compressed.layers[-1].bias.shape[0] == 3

    # the masked-out weight contribution is baked into w_eff as zero already,
    # but bias (never masked) still applies -- compressed must match dense exactly
    x = np.random.randn(4, 2)
    assert np.allclose(_dense_forward(mlp, x), compressed(x), atol=1e-12)


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
