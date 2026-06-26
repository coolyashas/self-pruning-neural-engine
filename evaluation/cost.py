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


def cost_report(model: Sequential, batch_size: int) -> dict:
    return {
        "weight_sparsity": weight_sparsity(model),
        "active_params": active_param_count(model),
        "total_params": total_param_count(model),
        "dense_flops": model_flops(model, batch_size, active_only=False),
        "theoretical_sparse_flops": model_flops(model, batch_size, active_only=True),
        "note": HONEST_COST_NOTE,
    }
