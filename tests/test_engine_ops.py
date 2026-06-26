"""Finite-difference gradient checks for every op built so far:
add, sub, mul, div (plain + broadcasting), matmul, sum, mean.
Also covers the two graph-correctness requirements from the pitfalls doc:
shared-node gradient accumulation, and requires_grad=False being skipped.
"""

import numpy as np
import pytest

from engine.tensor import Tensor
from tests.gradcheck_utils import assert_grad_matches
from utils.seed import set_seed

set_seed(0)


@pytest.mark.parametrize(
    "op_name, tensor_op, numpy_op",
    [
        ("add", lambda a, b: a + b, lambda a, b: a + b),
        ("sub", lambda a, b: a - b, lambda a, b: a - b),
        ("mul", lambda a, b: a * b, lambda a, b: a * b),
        ("div", lambda a, b: a / b, lambda a, b: a / b),
    ],
)
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((3, 4), (3, 4)),  # matching shapes, no broadcasting
        ((3, 4), (4,)),  # bias-add style broadcast
        ((3, 4), (1, 4)),  # explicit size-1 broadcast axis
    ],
)
def test_elementwise_grad(op_name, tensor_op, numpy_op, shape_a, shape_b):
    a = np.random.randn(*shape_a)
    b = np.random.randn(*shape_b) + (2.0 if op_name == "div" else 0.0)  # avoid /0
    assert_grad_matches(tensor_op, numpy_op, [a, b])


def test_matmul_grad():
    X = np.random.randn(5, 3)
    W = np.random.randn(3, 2)
    assert_grad_matches(lambda x, w: x @ w, lambda x, w: x @ w, [X, W])


@pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, False), (1, True)])
def test_sum_grad(axis, keepdims):
    A = np.random.randn(3, 4)
    assert_grad_matches(
        lambda t: t.sum(axis=axis, keepdims=keepdims),
        lambda a: a.sum(axis=axis, keepdims=keepdims),
        [A],
    )


@pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, True)])
def test_mean_grad(axis, keepdims):
    A = np.random.randn(3, 4)
    assert_grad_matches(
        lambda t: t.mean(axis=axis, keepdims=keepdims),
        lambda a: a.mean(axis=axis, keepdims=keepdims),
        [A],
    )


def test_shared_node_accumulation():
    # straight from the pitfalls doc: a = x*y; b = a+a; loss = b.sum()
    # x and y's gradient must accumulate contributions from both branches.
    x = np.random.randn(4)
    y = np.random.randn(4)

    def tensor_op(xt, yt):
        a = xt * yt
        return a + a

    def numpy_op(xa, ya):
        a = xa * ya
        return a + a

    assert_grad_matches(tensor_op, numpy_op, [x, y])


def test_requires_grad_false_is_skipped():
    data = Tensor(np.random.randn(5, 3), requires_grad=False)
    w = Tensor(np.random.randn(3, 2), requires_grad=True)
    out = (data @ w).sum()
    out.backward()
    assert data.grad is None
    assert w.grad is not None


def test_backward_accumulates_across_calls():
    p = Tensor([1.0, 2.0], requires_grad=True)
    (p * 2).sum().backward()
    first = p.grad.copy()
    (p * 2).sum().backward()
    assert np.allclose(p.grad, 2 * first)
