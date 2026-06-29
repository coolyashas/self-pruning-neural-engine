"""2D synthetic spirals (no download). A loader just returns (X, y) arrays,
keeping the training loop dataset-agnostic."""

from __future__ import annotations

import numpy as np


def make_spirals(n_per_class: int = 300, n_classes: int = 3, noise: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """n_classes interleaved spiral arms, one label per arm."""
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        idx = slice(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + np.random.randn(n_per_class) * noise
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = c
    return X, y
