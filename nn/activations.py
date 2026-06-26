"""Activations as modules: same __call__/parameters() shape as Linear, so
Sequential can chain them without special-casing parameter-free layers."""

from __future__ import annotations

from engine.tensor import Tensor


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> list[Tensor]:
        return []
