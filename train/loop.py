"""Mini-batch training loop: dataset-agnostic, takes (X, y) arrays."""

from __future__ import annotations

import numpy as np

from engine.loss import softmax_cross_entropy
from engine.tensor import Tensor


def train(
    model,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    on_epoch_end=None,
) -> list[float]:
    """on_epoch_end(epoch, losses_so_far), if given, runs after each epoch
    -- lets a caller log/plot per-epoch metrics without this loop needing
    to know anything about logging, CSVs, or plots.
    """
    n = X.shape[0]
    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = softmax_cross_entropy(model(Tensor(X[idx])), y[idx])
            assert np.isfinite(loss.data), f"non-finite loss: {loss.data}"
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))
        if on_epoch_end is not None:
            on_epoch_end(epoch, losses)
    return losses
