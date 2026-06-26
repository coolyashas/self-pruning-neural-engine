"""Shared finite-difference gradient checking helpers, reused by every
gradcheck test file in this repo (commit 9, commit 18, ...)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from engine.tensor import Tensor


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar-valued f at x, elementwise."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        f_plus = f(x)
        x[idx] = orig - h
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def assert_grad_matches(tensor_op, numpy_op, inputs: list[np.ndarray], atol: float = 1e-5) -> None:
    """Check Tensor autodiff grad against finite differences, for every input.

    `tensor_op(*Tensors) -> Tensor` and `numpy_op(*ndarrays) -> ndarray` must
    compute the same function; one as our autodiff graph, one as plain NumPy
    for the numerical reference. Output is summed first if not already
    scalar, since backward() (no-arg form) requires a scalar root.
    """
    tensors = [Tensor(x.copy(), requires_grad=True) for x in inputs]
    out = tensor_op(*tensors)
    loss = out if out.shape == () else out.sum()
    loss.backward()

    for i, x in enumerate(inputs):
        def f(perturbed, i=i):
            args = [a.copy() for a in inputs]
            args[i] = perturbed
            return numpy_op(*args).sum()

        numeric = numerical_gradient(f, x.copy())
        analytic = tensors[i].grad
        assert analytic is not None, f"input {i} got no gradient at all"
        assert np.allclose(analytic, numeric, atol=atol), (
            f"grad mismatch for input {i}:\nanalytic={analytic}\nnumeric={numeric}"
        )
