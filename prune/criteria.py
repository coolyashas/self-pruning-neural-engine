"""Pruning importance criteria: score every connection so the lowest-scoring
fraction can be removed. Magnitude (connection size) is the baseline; |w*g|
saliency is what we compare it against.
"""

from __future__ import annotations

import numpy as np

from engine.loss import softmax_cross_entropy
from engine.tensor import Tensor
from nn.linear import Linear


def magnitude_scores(layer: Linear) -> np.ndarray:
    return np.abs(layer.weight.data)


def saliency_scores(layer: Linear) -> np.ndarray:
    """First-order Taylor importance: zeroing connection i changes the loss
    by about w_i * dL/dw_i. Needs weight.grad populated (see
    accumulate_gradients -- a single mini-batch's gradient is too noisy).
    """
    assert layer.weight.grad is not None, "need accumulated gradients before scoring saliency"
    return np.abs(layer.weight.data * layer.weight.grad)


def neuron_magnitude_scores(layer: Linear) -> np.ndarray:
    """Structured analogue of magnitude_scores: one score per output neuron
    (a column; axis=0 sums over its incoming weights). Coarser by design -- a
    neuron with one huge weight can outrank a uniformly-moderate one.
    """
    return np.abs(layer.weight.data).sum(axis=0)


def neuron_saliency_scores(layer: Linear) -> np.ndarray:
    """First-order loss increase from removing a whole output neuron:
    |sum_i w_i*g_i|. Sum the SIGNED saliencies THEN abs -- not abs-then-sum
    (the L1 norm), since the real cancellation between signed terms matters.
    Same accumulated-gradient contract as saliency_scores. See DESIGN.md s.1.
    """
    assert layer.weight.grad is not None, "need accumulated gradients before scoring saliency"
    return np.abs((layer.weight.data * layer.weight.grad).sum(axis=0))


def accumulate_gradients(model, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
    """Sweep (X, y) once in mini-batches, leaving every parameter's .grad
    equal to the TRUE full-dataset-mean gradient (a stable signal for
    saliency scoring). Leaves .grad populated on exit -- callers must
    zero_grad before any real training step.

    Each backward() computes that batch's MEAN loss gradient, so when
    batch_size doesn't divide n, contributions are weighted by n_batch/n:
    sum_k (n_k/n) * batch_mean_grad_k = the true dataset-mean gradient.
    Naive summing would overweight a smaller final batch's examples.
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
            # p.grad can be None for a parameter this batch's graph never
            # touched; skip it (like adam.step) instead of `float * None`.
            if p.grad is not None:
                total += batch_weight * p.grad
    for p, total in zip(params, totals):
        p.grad = total


def accumulate_dense_gradients(model, X: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    """Sweep (X, y) once, accumulating each prunable Linear layer's w_eff.grad
    (the dense/unmasked gradient signal) into a persistent per-layer total
    equal to the TRUE full-dataset-mean gradient. Returns {layer: total}.

    Parallel to accumulate_gradients. Unlike weight (a persistent Tensor whose
    .grad sums itself across the sweep), w_eff is rebuilt fresh each forward
    call, so its .grad must be read and summed here right after each batch's
    backward(), before the next forward discards it.

    Same n_batch/n per-batch weighting as accumulate_gradients, applied to
    both the returned totals and the side-effect weight.grad (dst_step relies
    on the latter matching a standalone accumulate_gradients call).
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
            # same None-grad guard as accumulate_gradients above.
            if p.grad is not None:
                total += batch_weight * p.grad

    for p, total in zip(params, param_totals):
        p.grad = total
    return dense_totals
