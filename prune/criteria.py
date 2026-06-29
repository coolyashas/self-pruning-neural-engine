"""Pruning importance criteria: score every connection so the
lowest-scoring fraction can be removed. Magnitude is the trivial
baseline -- a connection's own size, with no information about how much
the loss actually depends on it. |w*g| saliency (commit 20) is what we
compare it against.
"""

from __future__ import annotations

import numpy as np

from engine.loss import softmax_cross_entropy
from engine.tensor import Tensor
from nn.linear import Linear


def magnitude_scores(layer: Linear) -> np.ndarray:
    return np.abs(layer.weight.data)


def saliency_scores(layer: Linear) -> np.ndarray:
    """First-order Taylor importance: zeroing connection i changes the
    loss by about w_i * dL/dw_i. Unlike magnitude, this reflects how much
    the loss actually depends on a connection, not just how big it is.
    Needs weight.grad already populated -- see accumulate_gradients,
    since a single mini-batch's gradient is too noisy on its own.
    """
    assert layer.weight.grad is not None, "need accumulated gradients before scoring saliency"
    return np.abs(layer.weight.data * layer.weight.grad)


def neuron_magnitude_scores(layer: Linear) -> np.ndarray:
    """Structured analogue of magnitude_scores: one score per output
    neuron (weight.shape == (in_features, out_features), so a neuron is
    a column -- axis=0 sums over everything feeding into it). Coarser by
    design: a neuron with one huge weight and many tiny ones can outrank
    a neuron with uniformly moderate weights, which a per-weight score
    would never conflate.
    """
    return np.abs(layer.weight.data).sum(axis=0)


def neuron_saliency_scores(layer: Linear) -> np.ndarray:
    """First-order loss increase from removing a whole output neuron
    (zeroing its incoming column at once): |sum_i w_i*g_i|. Sum the SIGNED
    saliencies over the column, THEN abs -- not abs-then-sum, which is the
    L1 norm and only an upper bound (signed terms can cancel, and that
    cancellation is real). Same accumulated-gradient contract as
    saliency_scores. See DESIGN.md section 1.
    """
    assert layer.weight.grad is not None, "need accumulated gradients before scoring saliency"
    return np.abs((layer.weight.data * layer.weight.grad).sum(axis=0))


def accumulate_gradients(model, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
    """Sweep (X, y) once in mini-batches, leaving every parameter's .grad
    equal to the TRUE full-dataset-mean gradient -- not a single
    optimizer step, and not simply "sum of each batch's gradient" either.
    Purely to get a stable gradient signal for saliency scoring; a single
    small batch is too noisy. Leaves .grad populated on exit -- callers
    must zero_grad before any subsequent real training step, or that
    step's gradient adds onto this.

    Each backward() call computes the gradient of THAT BATCH's mean loss
    (softmax_cross_entropy always divides by the batch's own size). If
    batch_size doesn't evenly divide N -- the common case, e.g. N=900,
    batch_size=64 leaves a final batch of size 4 -- naively summing those
    per-batch-mean gradients via repeated backward() calls (the previous
    behavior here) silently overweights the smaller batch's examples: a
    4-example batch would get the same TOTAL influence on the sum as a
    full 64-example batch, a 16x per-example overweighting. Confirmed by
    direct execution: the old result matched "sum of per-batch means",
    not the true dataset-mean gradient, by a measurable margin.

    Fixed by explicitly weighting each batch's contribution by
    n_batch / n before accumulating: sum_k (n_k/n) * batch_mean_grad_k
    = (1/n) * sum_k sum_{i in batch k} grad_i, which IS the true
    full-dataset mean gradient, regardless of how batch_size divides n.
    """
    params = model.parameters()
    n = X.shape[0]
    totals = [np.zeros_like(p.data) for p in params]
    for start in range(0, n, batch_size):
        idx = slice(start, start + batch_size)
        n_batch = X[idx].shape[0]
        for p in params:
            p.grad = None
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        batch_weight = n_batch / n
        for p, total in zip(params, totals):
            # p.grad can be None for a parameter this particular batch's
            # graph never touched (e.g. a conditionally-used branch in a
            # future, non-fully-connected architecture -- every
            # parameter in the current Sequential MLP gets a gradient
            # on every batch, but this is a generic model helper, not
            # hardcoded to that). Skip deliberately, the same way
            # optim/adam.py's step() does, instead of crashing on
            # `float * None`.
            if p.grad is not None:
                total += batch_weight * p.grad
    for p, total in zip(params, totals):
        p.grad = total


def accumulate_dense_gradients(model, X: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    """Sweep (X, y) once, accumulating each prunable Linear layer's
    w_eff.grad (the dense/unmasked gradient signal -- see nn/linear.py
    and engine/ops.py's mul() backward) into a persistent per-layer
    total equal to the TRUE full-dataset-mean gradient. Returns
    {layer: accumulated_w_eff_grad}.

    A parallel function to accumulate_gradients, not a modification of
    it: weight is the same persistent Tensor every forward call, so
    weight.grad naturally sums across a sweep via accumulate_grad's +=.
    w_eff is a FRESH Tensor every call (nn/linear.py rebuilds it each
    __call__), so its .grad does NOT accumulate on its own -- this
    function does that summing explicitly, reading layer.w_eff.grad
    immediately after each batch's backward() and adding it into the
    running total before the next forward call creates a new w_eff and
    the old one (and its .grad) becomes garbage. Getting that ordering
    backwards (read after the next forward instead of before) would hit
    a fresh w_eff with grad=None, not silently wrong numbers -- but it
    still must be done by construction, not by luck.

    Same per-batch weighting fix as accumulate_gradients (see its
    docstring for the full derivation): each backward() call computes a
    BATCH-MEAN gradient, so naively summing un-weighted contributions
    over an uneven final batch overweights its examples. Both the
    returned dense totals AND the side-effect weight.grad left on every
    parameter (dst_step relies on this matching a standalone
    accumulate_gradients call exactly, to justify skipping a second
    sweep) are weighted by n_batch/n here, so that side-effect contract
    still holds with the TRUE dataset-mean gradient, not the old "sum of
    batch means".
    """
    prunable = [layer for layer in model.layers if hasattr(layer, "mask")]
    params = model.parameters()
    n = X.shape[0]
    dense_totals = {layer: np.zeros_like(layer.weight.data) for layer in prunable}
    param_totals = [np.zeros_like(p.data) for p in params]

    for start in range(0, n, batch_size):
        idx = slice(start, start + batch_size)
        n_batch = X[idx].shape[0]
        for p in params:
            p.grad = None
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        batch_weight = n_batch / n
        for layer in prunable:
            dense_totals[layer] += batch_weight * layer.w_eff.grad
        for p, total in zip(params, param_totals):
            # same guard as accumulate_gradients above: skip a parameter
            # this batch's graph never touched instead of crashing on
            # `float * None`.
            if p.grad is not None:
                total += batch_weight * p.grad

    for p, total in zip(params, param_totals):
        p.grad = total
    return dense_totals
