"""Active-parameter and FLOP counting -- and the honest limit of what that
number means.

These counts are what a SPARSE kernel could theoretically achieve. Our
actual forward pass computes x @ (weight * mask): a dense matmul where
masked entries are multiplied by zero, not skipped. A multiply-by-zero is
still a multiply, so wall-clock time does NOT improve with sparsity here
-- see test_dense_matmul_speed_unaffected_by_sparsity for a measured
proof, not just this claim. A real speedup needs a sparse matmul kernel or
structured (e.g. block/channel) sparsity, neither of which exists in this
repo (out of scope, see project brief).
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
    """Fraction of weight connections masked off -- matches prune/'s own
    definition exactly (bias excluded, same as the schedule targets).
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
    """Theoretical multiply-add FLOPs for one forward pass (2 FLOPs per
    connection: one multiply, one add -- the standard MAC convention),
    plus one add per output element for the bias.

    active_only=True: what a sparse kernel could theoretically skip to.
    active_only=False (default): what our dense implementation actually
    does -- the same number regardless of sparsity, since it never skips
    a masked connection's multiply.
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
    `model(Tensor(x))` (the existing dense x @ (weight*mask) path) vs.
    `compressed(x)` (prune.compress's sliced-matrix path, see
    prune/compress.py). Unlike unstructured masking, the compressed path
    genuinely does less work -- contrast with
    test_dense_matmul_speed_unaffected_by_sparsity's no-speedup result.

    CAVEAT, found in review: `model(Tensor(x))` pays for the autodiff
    engine's own per-call bookkeeping (Tensor construction, `_prev`
    set-building, `_backward` closure allocation) even though nothing
    here ever calls `.backward()`; `compressed(x)` is plain NumPy with
    none of that overhead. So the ratio from THIS function conflates two
    separate effects: the genuine FLOP reduction from a smaller matrix,
    and the autodiff engine's overhead vanishing because the compressed
    path happens to not use Tensor at all. Measured: about 28% of the
    gap is pure autodiff overhead, unrelated to compression. Use this
    function to answer "is swapping in the compressed model during
    inference faster than the current dense forward path" (a real,
    fair question about this codebase's actual code paths) -- use
    time_dense_numpy_vs_compressed_forward below for the FLOP-isolated,
    apples-to-apples number.
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
    """Forward through model's CURRENT masked weights in plain NumPy --
    no Tensor, no autodiff bookkeeping at all. Exists only so
    time_dense_numpy_vs_compressed_forward can compare apples-to-apples
    against compress_model's output (also plain NumPy): both arms then
    differ ONLY in matrix size, isolating the genuine FLOP-reduction
    effect from the autodiff engine's unrelated per-call overhead (see
    the caveat on time_dense_vs_compressed_forward above). Mirrors
    prune.compress.compress_model's own layer-walking, just without any
    slicing.
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
    """The apples-to-apples version of time_dense_vs_compressed_forward:
    both arms are plain NumPy (_dense_numpy_forward vs. compressed(x)),
    so the measured ratio reflects ONLY the FLOP reduction from a
    genuinely smaller matrix, with none of the autodiff engine's own
    bookkeeping overhead mixed in. This is the number to cite as "real
    speedup from structured compression" -- time_dense_vs_compressed_forward's
    ratio additionally (and misleadingly, if uncaveated) includes ~28%
    of unrelated autodiff-overhead removal on top of that.
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
