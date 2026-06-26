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
    on_step_end=None,
) -> list[float]:
    """on_epoch_end(epoch, losses_so_far) runs after each epoch.
    on_step_end(global_step, model, loss) runs after every mini-batch step
    -- e.g. to apply a pruning schedule at step granularity, finer than
    once per epoch. Both default to None and are backward compatible.
    """
    n = X.shape[0]
    losses = []
    step = 0
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
            step += 1
            if on_step_end is not None:
                on_step_end(step, model, float(loss.data))
        if on_epoch_end is not None:
            on_epoch_end(epoch, losses)
    return losses
