"""Compress a structurally-pruned model into a real, smaller dense model.

Unstructured masking (x @ (weight*mask)) never actually shrinks -- it's
still a full-size matmul with some zero entries (see evaluation/cost.py's
measured no-speedup result). Structured (neuron-level) pruning is
different: once an entire output column is dead, the matrix's actual
dimensions can shrink, and a normal dense matmul on the smaller matrix
does genuinely less work. This module builds that smaller model.

Inference-only: plain ndarrays, no Tensor/autodiff -- this is not a
training path. A model that was never structurally pruned compresses to
itself, trivially (no slicing happens).
"""

from __future__ import annotations

import numpy as np

from nn.activations import ReLU
from nn.linear import Linear
from nn.sequential import Sequential


class CompressedLinear:
    """Inference-only affine layer: y = x @ weight + bias, plain ndarrays.
    weight/bias are already-sliced to only the surviving rows/columns of
    the original layer, so this matmul is genuinely smaller -- no mask
    multiply at all, nothing left to skip.
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
    """Walk model.layers, slicing out only currently-alive neurons into
    real smaller dense weight/bias arrays.

    Cross-layer coupling: a Linear layer's dead OUTPUT columns make the
    corresponding INPUT rows of the NEXT Linear layer dead too -- a
    zeroed output column can never contribute a nonzero value downstream,
    and ReLU(0) == 0 passes that deadness through unchanged. So each
    Linear's input-side slicing is driven by the PREVIOUS Linear's
    output-side alive set, not recomputed from this layer's own mask.

    This relies on a dead neuron's PRE-activation being exactly 0, i.e.
    bias == 0 there too, not just its weight column -- prune.mask's
    prune_neurons_to_count and the bias_mask it maintains (nn/linear.py)
    are what guarantee that holds; this function doesn't re-check it.

    First Linear: no upstream deadness -- input features (e.g. raw 2D
    spiral coordinates) were never neurons, never sliced. Last Linear:
    output (e.g. logits) is never sliced either, even if its mask has
    some zero columns -- dropping an output class changes the model's
    meaning, not just its size. This is a defensive guard; structured
    pruning should simply never target the final layer's output to begin
    with (see train/run_part4_structured.py).
    """
    linear_layers = [layer for layer in model.layers if isinstance(layer, Linear)]
    compressed_layers = []
    alive_out = None  # alive-output-neuron index array from the previous Linear

    for layer in model.layers:
        if isinstance(layer, Linear):
            is_last_linear = layer is linear_layers[-1]
            w_eff = layer.weight.data * layer.mask.data  # bake this layer's own mask in first

            if alive_out is not None:
                w_eff = w_eff[alive_out, :]  # drop dead input rows inherited from upstream

            if is_last_linear:
                out_idx = np.arange(w_eff.shape[1])  # never slice the final output
            else:
                out_idx = np.flatnonzero(layer.mask.data.any(axis=0))

            w_eff = w_eff[:, out_idx]
            b_eff = layer.bias.data[out_idx]

            # .copy() makes the no-aliasing guarantee explicit rather than
            # relying on fancy indexing already returning a copy.
            compressed_layers.append(CompressedLinear(w_eff.copy(), b_eff.copy()))
            alive_out = out_idx
        elif isinstance(layer, ReLU):
            compressed_layers.append(CompressedReLU())
        else:
            raise NotImplementedError(f"compress_model: unsupported layer type {type(layer)}")

    return CompressedSequential(compressed_layers)
