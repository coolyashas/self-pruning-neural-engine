"""Active-parameter and FLOP counting -- and the honest limit of that number.

These counts are what a SPARSE kernel could theoretically achieve. Our actual
forward pass computes x @ (weight * mask): a dense matmul where masked entries
are multiplied by zero, not skipped, so wall-clock time does NOT improve with
sparsity (see test_dense_matmul_speed_unaffected_by_sparsity). A real speedup
needs a sparse kernel or structured sparsity, neither of which exists here.
"""

from __future__ import annotations

import time

import numpy as np

from engine.tensor import Tensor
from nn.activations import ReLU
from nn.linear import Linear
from nn.sequential import Sequential

HONEST_COST_NOTE = (
    "Active-parameter and FLOP counts are the cost a sparse kernel could "
    "theoretically achieve, not our actual runtime: x @ (W * mask) is a "
    "full dense matmul, so wall-clock time is unaffected by sparsity. "
    "Realizing a real speedup needs a sparse kernel or structured "
    "sparsity, neither implemented here."
)


def _prunable_layers(model: Sequential):
    return [layer for layer in model.layers if hasattr(layer, "mask")]


def weight_sparsity(model: Sequential) -> float:
    """Fraction of weight connections masked off (bias excluded, matching
    prune/'s definition and the schedule targets).
    """
    layers = _prunable_layers(model)
    active = sum(int(l.mask.data.sum()) for l in layers)
    total = sum(l.mask.data.size for l in layers)
    return 1.0 - active / total


def active_param_count(model: Sequential) -> int:
    """Active weight connections + bias (bias is never pruned) -- the
    parameter count a sparse implementation would actually need to store.
    """
    return sum(int(l.mask.data.sum()) + l.bias.data.size for l in _prunable_layers(model))


def total_param_count(model: Sequential) -> int:
    """Full weight + bias count, independent of pruning state."""
    return sum(l.weight.data.size + l.bias.data.size for l in _prunable_layers(model))


def model_flops(model: Sequential, batch_size: int, active_only: bool = False) -> int:
    """Theoretical multiply-add FLOPs for one forward pass (2 per connection,
    the standard MAC convention), plus one add per output element for bias.

    active_only=True: what a sparse kernel could skip to. False (default):
    what our dense path does, the same regardless of sparsity.
    """
    total = 0
    for layer in _prunable_layers(model):
        n_in, n_out = layer.weight.shape
        n_connections = int(layer.mask.data.sum()) if active_only else n_in * n_out
        total += 2 * batch_size * n_connections + batch_size * n_out
    return total


def time_dense_vs_compressed_forward(
    model: Sequential, compressed, x: np.ndarray, repeats: int = 200
) -> tuple[float, float]:
    """Real wall-clock time (seconds) for `repeats` forward passes:
    `model(Tensor(x))` (dense x @ (weight*mask)) vs. `compressed(x)`
    (prune.compress's sliced-matrix path), which genuinely does less work.

    CAVEAT: model(Tensor(x)) pays autodiff per-call bookkeeping (Tensor
    construction, _prev sets, _backward closures) even though nothing calls
    .backward(); compressed(x) is plain NumPy. So this ratio conflates the
    genuine FLOP reduction with that overhead vanishing (~28% of the gap is
    autodiff overhead). Use this for "is the compressed model faster than the
    current dense path"; use time_dense_numpy_vs_compressed_forward for the
    FLOP-isolated number.
    """
    x_t = Tensor(x)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x_t)
    t_dense = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        compressed(x)
    t_compressed = time.perf_counter() - start

    return t_dense, t_compressed


def _dense_numpy_forward(model: Sequential, x: np.ndarray) -> np.ndarray:
    """Forward through model's CURRENT masked weights in plain NumPy (no
    Tensor/autodiff). Lets time_dense_numpy_vs_compressed_forward compare
    apples-to-apples against compress_model's output: both arms differ ONLY in
    matrix size. Mirrors compress_model's layer-walking, without slicing.
    """
    for layer in model.layers:
        if isinstance(layer, Linear):
            x = x @ (layer.weight.data * layer.mask.data) + layer.bias.data
        elif isinstance(layer, ReLU):
            x = np.maximum(x, 0.0)
        else:
            raise NotImplementedError(f"_dense_numpy_forward: unsupported layer type {type(layer)}")
    return x


def time_dense_numpy_vs_compressed_forward(
    model: Sequential, compressed, x: np.ndarray, repeats: int = 200
) -> tuple[float, float]:
    """Apples-to-apples version of time_dense_vs_compressed_forward: both arms
    are plain NumPy (_dense_numpy_forward vs. compressed(x)), so the ratio
    reflects ONLY the FLOP reduction from a smaller matrix. This is the number
    to cite as "real speedup from structured compression".
    """
    start = time.perf_counter()
    for _ in range(repeats):
        _dense_numpy_forward(model, x)
    t_dense = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        compressed(x)
    t_compressed = time.perf_counter() - start

    return t_dense, t_compressed


def cost_report(model: Sequential, batch_size: int) -> dict:
    return {
        "weight_sparsity": weight_sparsity(model),
        "active_params": active_param_count(model),
        "total_params": total_param_count(model),
        "dense_flops": model_flops(model, batch_size, active_only=False),
        "theoretical_sparse_flops": model_flops(model, batch_size, active_only=True),
        "note": HONEST_COST_NOTE,
    }
