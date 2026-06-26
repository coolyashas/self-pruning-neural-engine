"""Pruning importance criteria: score every connection so the
lowest-scoring fraction can be removed. Magnitude is the trivial
baseline -- a connection's own size, with no information about how much
the loss actually depends on it. |w*g| saliency (commit 20) is what we
compare it against.
"""

from __future__ import annotations

import numpy as np

from nn.linear import Linear


def magnitude_scores(layer: Linear) -> np.ndarray:
    return np.abs(layer.weight.data)
