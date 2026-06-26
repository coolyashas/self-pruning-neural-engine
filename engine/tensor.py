"""Tensor: ndarray + autodiff bookkeeping. No math here, just the node type."""

from __future__ import annotations

import numpy as np


class Tensor:
    """A node in the computation graph: data, grad, and links to parents.

    `_prev` holds the Tensors this one was built from, `_backward` is the
    closure (set by an op) that pushes grad from this node to those parents.
    backward() itself — the traversal that calls these in order — is later.
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
        self.grad: np.ndarray | None = None  # allocated lazily, see below

        # set, not tuple: a parent can show up twice in one op (x + x), and
        # we only want one graph edge for it.
        self._prev: set["Tensor"] = set(_children)
        self._backward = lambda: None  # no-op for leaves; ops overwrite this
        self._op = _op  # debug label only, e.g. "add"

    def accumulate_grad(self, grad: np.ndarray) -> None:
        """grad += grad, never overwrite — a tensor can feed multiple ops."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Run reverse-mode autodiff back through the graph from this node.

        No-arg form requires a scalar (e.g. a loss) and seeds grad=1.
        Builds a topo order by DFS (parents before the node they produced),
        then walks it in reverse so every node's _backward() only fires
        once all of its consumers have already accumulated into its grad.
        """
        if grad is None:
            assert self.data.size == 1, "backward() with no grad arg needs a scalar output"
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
