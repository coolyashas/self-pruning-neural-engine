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
    """Structured analogue of saliency_scores: one score per output
    neuron, summing |w*g| over its incoming column. Same accumulated-
    gradient contract as saliency_scores -- needs weight.grad populated
    via accumulate_gradients first.
    """
    assert layer.weight.grad is not None, "need accumulated gradients before scoring saliency"
    return np.abs(layer.weight.data * layer.weight.grad).sum(axis=0)


def accumulate_gradients(model, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
    """Sweep (X, y) once in mini-batches, accumulating into every
    parameter's .grad via repeated backward() calls -- no zero_grad
    between batches, no optimizer step. Purely to get a stable gradient
    signal for saliency scoring; a single small batch is too noisy.
    Leaves .grad populated on exit -- callers must zero_grad before any
    subsequent real training step, or that step's gradient adds onto this.
    """
    for p in model.parameters():
        p.grad = None
    n = X.shape[0]
    for start in range(0, n, batch_size):
        idx = slice(start, start + batch_size)
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()


def accumulate_dense_gradients(model, X: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    """Sweep (X, y) once, accumulating each prunable Linear layer's
    w_eff.grad (the dense/unmasked gradient signal -- see nn/linear.py
    and engine/ops.py's mul() backward) into a persistent per-layer
    total. Returns {layer: accumulated_w_eff_grad}.

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

    Leaves weight.grad dirty on exit, same contract as
    accumulate_gradients -- callers must zero_grad before any subsequent
    real training step.
    """
    prunable = [layer for layer in model.layers if hasattr(layer, "mask")]
    totals = {layer: np.zeros_like(layer.weight.data) for layer in prunable}

    for p in model.parameters():
        p.grad = None
    n = X.shape[0]
    for start in range(0, n, batch_size):
        idx = slice(start, start + batch_size)
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        for layer in prunable:
            totals[layer] += layer.w_eff.grad
    return totals
