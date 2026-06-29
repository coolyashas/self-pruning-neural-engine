"""Compress a structurally-pruned model into a real, smaller dense model.

Unstructured masking (x @ (weight*mask)) never shrinks -- still a full-size
matmul with zero entries. Structured (neuron-level) pruning lets the matrix's
actual dimensions shrink once a whole output column is dead, so a normal
dense matmul does genuinely less work.
"""

from __future__ import annotations

import numpy as np

from nn.activations import ReLU
from nn.linear import Linear
from nn.sequential import Sequential


class CompressedLinear:
    """Inference-only affine layer: y = x @ weight + bias, plain ndarrays.
    weight/bias are already sliced to the surviving rows/columns, so this
    matmul is genuinely smaller -- no mask multiply at all.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias


class CompressedReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)


class CompressedSequential:
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


def compress_model(model: Sequential) -> CompressedSequential:
    """Walk model.layers, slicing out only currently-alive neurons into real
    smaller dense weight/bias arrays.
    """
    linear_layers = [layer for layer in model.layers if isinstance(layer, Linear)]
    assert linear_layers, "compress_model: model has no Linear layers to compress"
    compressed_layers = []
    alive_out = None  # alive-output-neuron index array from the previous Linear

    for layer in model.layers:
        if isinstance(layer, Linear):
            is_last_linear = layer is linear_layers[-1]
            w_eff = layer.weight.data * layer.mask.data  # bake in this layer's mask

            if alive_out is not None:
                w_eff = w_eff[alive_out, :]  # drop dead upstream input rows

            if is_last_linear:
                out_idx = np.arange(w_eff.shape[1])  # never slice the final output
            else:
                out_idx = np.flatnonzero(layer.mask.data.any(axis=0))

            w_eff = w_eff[:, out_idx]
            b_eff = layer.bias.data[out_idx]

            # .copy() makes no-aliasing explicit, not implied by fancy indexing.
            compressed_layers.append(CompressedLinear(w_eff.copy(), b_eff.copy()))
            alive_out = out_idx
        elif isinstance(layer, ReLU):
            compressed_layers.append(CompressedReLU())
        else:
            raise NotImplementedError(f"compress_model: unsupported layer type {type(layer)}")

    return CompressedSequential(compressed_layers)
