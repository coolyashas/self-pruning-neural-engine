"""Tensor: ndarray + autodiff bookkeeping. No math here, just the node type."""

from __future__ import annotations

import numpy as np


class Tensor:
    """A node in the computation graph: data, grad, and links to parents.

    `_prev` holds the Tensors this one was built from; `_backward` is the
    closure (set by an op) that pushes grad from this node to those parents.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        # float64: finite-difference gradcheck needs the precision headroom.
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None  # allocated lazily

        # set, not tuple: dedupes a parent that appears twice in one op (x + x).
        self._prev: set["Tensor"] = set(_children)
        self._backward = lambda: None  # no-op for leaves; ops overwrite this
        self._op = _op  # debug label only

    def accumulate_grad(self, grad: np.ndarray) -> None:
        """grad += grad, never overwrite — a tensor can feed multiple ops."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Run reverse-mode autodiff back through the graph from this node.

        No-arg form requires a scalar (e.g. a loss) and seeds grad=1. Builds a
        topo order by DFS (parents before children), then walks it in reverse
        so each node's _backward() fires only after all its consumers have
        accumulated into its grad. O(V+E) in graph size.
        """
        if grad is None:
            # Hard API contract (not an assert, which `python -O` would strip):
            # the no-arg form is only meaningful for a scalar root. Without this,
            # backward() on a non-scalar would silently seed every grad as 1.0.
            if self.data.size != 1:
                raise ValueError(
                    f"backward() with no grad arg needs a scalar output, got shape {self.data.shape}"
                )
            grad = np.ones_like(self.data)
        self.accumulate_grad(np.asarray(grad, dtype=np.float64))

        topo: list["Tensor"] = []
        visited: set["Tensor"] = set()

        def build(node: "Tensor") -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        for node in reversed(topo):
            if node.grad is not None:  # skip branches that never needed grad
                node._backward()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, "
            f"op={self._op!r})"
        )
