"""Mini-batch training loop: dataset-agnostic, takes (X, y) arrays."""

from __future__ import annotations

import numpy as np

from engine.loss import softmax_cross_entropy
from engine.tensor import Tensor


def _clip_grad_norm(parameters, max_norm: float) -> None:
    """Scale all gradients in place so their combined global L2 norm is at
    most `max_norm`. Caps the rare oversized gradient that Adam would
    otherwise amplify into a loss spike, without changing the update
    direction. Parameters not touched by backward() (grad is None) are
    skipped.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = np.sqrt(sum(float(np.sum(g**2)) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale


def train(
    model,
    optimizer,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    on_epoch_end=None,
    on_step_end=None,
    grad_clip: float | None = None,
) -> list[float]:
    """on_epoch_end(epoch, losses_so_far) runs after each epoch.
    on_step_end(global_step, model, loss) runs after every mini-batch step
    -- e.g. to apply a pruning schedule at step granularity, finer than
    once per epoch. Both default to None and are backward compatible.

    grad_clip: if set, clip the global gradient norm to this value before
    each optimizer step (see _clip_grad_norm). None disables clipping.
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
            # plain `if: raise`, not `assert` -- this is meant to halt
            # training the moment it diverges, which must hold even
            # under `python -O` (which strips asserts entirely; a
            # silently-stripped check here would let training continue
            # computing garbage updates from a NaN/inf loss instead of
            # stopping immediately).
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss: {loss.data}")
            loss.backward()
            if grad_clip is not None:
                _clip_grad_norm(optimizer.parameters, grad_clip)
            optimizer.step()
            losses.append(float(loss.data))
            step += 1
            if on_step_end is not None:
                on_step_end(step, model, float(loss.data))
        if on_epoch_end is not None:
            on_epoch_end(epoch, losses)
    return losses
