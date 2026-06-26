"""Sequential: chains modules (anything with __call__ and parameters())."""

from __future__ import annotations

from engine.tensor import Tensor


class Sequential:
    def __init__(self, *layers) -> None:
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
