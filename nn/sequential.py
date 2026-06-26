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

    def masked_parameters(self) -> list[tuple[Tensor, Tensor | None]]:
        pairs = []
        for layer in self.layers:
            if hasattr(layer, "masked_parameters"):
                pairs.extend(layer.masked_parameters())
            else:
                pairs.extend((p, None) for p in layer.parameters())
        return pairs

    def parameters(self) -> list[Tensor]:
        return [p for p, _ in self.masked_parameters()]
