"""Finite-difference gradient checks for the core ops, plus shared-node accumulation and requires_grad=False skipping."""

from pathlib import Path

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
        ((3, 4), (3, 4)),
        ((3, 4), (4,)),
        ((3, 4), (1, 4)),
        ((3, 4), (3, 1)),
        ((3, 4), ()),
        ((), (3, 4)),
        ((1, 1), (3, 4)),
        ((2, 3, 4), (4,)),
        ((2, 3, 4), (3, 4)),
        ((2, 1, 4), (4,)),
    ],
)
def test_elementwise_grad(op_name, tensor_op, numpy_op, shape_a, shape_b):
    a = np.asarray(np.random.randn(*shape_a))
    b = np.asarray(np.random.randn(*shape_b) + (2.0 if op_name == "div" else 0.0))
    assert_grad_matches(tensor_op, numpy_op, [a, b])


def test_broadcast_then_reduce():
    """Broadcast feeding a reduction: exercises _unbroadcast then _grad_to_input_shape."""
    a = np.random.randn(5, 3)
    b = np.random.randn(3)
    assert_grad_matches(
        lambda at, bt: (at + bt).sum(axis=0),
        lambda aa, ba: (aa + ba).sum(axis=0),
        [a, b],
    )


def test_broadcast_then_matmul():
    """Broadcast add feeding matmul: chains _unbroadcast with matmul backward."""
    x = np.random.randn(5, 3)
    bias = np.random.randn(3)
    w = np.random.randn(3, 2)
    assert_grad_matches(
        lambda xt, bt, wt: (xt + bt) @ wt,
        lambda xa, ba, wa: (xa + ba) @ wa,
        [x, bias, w],
    )


def test_composite_broadcast_chain_three_different_shapes():
    """Three shapes broadcasting in turn: each _unbroadcast must reduce to its own input shape."""
    a = np.random.randn(4, 3)
    b = np.random.randn(3)
    c = np.random.randn(4, 1)
    assert_grad_matches(
        lambda at, bt, ct: (at * bt) + ct,
        lambda aa, ba, ca: (aa * ba) + ca,
        [a, b, c],
    )


def test_matmul_grad():
    X = np.random.randn(5, 3)
    W = np.random.randn(3, 2)
    assert_grad_matches(lambda x, w: x @ w, lambda x, w: x @ w, [X, W])


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((6, 5, 3), (6, 3, 4)),
        ((6, 5, 3), (3, 4)),
        ((5, 3), (6, 3, 4)),
        ((6, 5, 3), (1, 3, 4)),
        ((1, 5, 3), (6, 3, 4)),
        ((2, 6, 5, 3), (6, 3, 4)),
    ],
)
def test_matmul_batched_grad(shape_a, shape_b):
    """Batched matmul: _unbroadcast_batch reduces each grad to its own batch shape, leaving matrix axes alone."""
    a = np.random.randn(*shape_a)
    b = np.random.randn(*shape_b)
    assert_grad_matches(lambda at, bt: at @ bt, lambda aa, ba: aa @ ba, [a, b])


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((3,), (3, 4)),
        ((3, 4), (4,)),
        ((), (3, 4)),
    ],
)
def test_matmul_rejects_1d_or_0d_inputs(shape_a, shape_b):
    """ndim<2 on either side is out of scope and must raise, not silently compute."""
    a = Tensor(np.random.randn(*shape_a))
    b = Tensor(np.random.randn(*shape_b))
    with pytest.raises(ValueError, match="ndim"):
        a @ b


def test_matmul_ndim_guard_survives_python_dash_O():
    """The ndim<2 guard is `if: raise`, not `assert`, so it survives `python -O`; checked via subprocess."""
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from engine.tensor import Tensor\n"
        "from engine.matmul import matmul\n"
        "a = Tensor(np.random.randn(3,))\n"
        "b = Tensor(np.random.randn(3, 4))\n"
        "matmul(a, b)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "expected a ValueError under -O, but the call succeeded silently"
    assert "ValueError" in result.stderr


def test_backward_no_arg_rejects_non_scalar_output():
    t = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    with pytest.raises(ValueError, match="scalar"):
        t.backward()


def test_backward_no_arg_scalar_guard_survives_python_dash_O():
    """The non-scalar backward() guard is `if: raise`, so it survives `python -O`; checked via subprocess."""
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from engine.tensor import Tensor\n"
        "t = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)\n"
        "t.backward()\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "expected a ValueError under -O, but the call succeeded silently"
    assert "ValueError" in result.stderr


@pytest.mark.parametrize(
    "axis, keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_sum_grad(axis, keepdims):
    A = np.random.randn(3, 4)
    assert_grad_matches(
        lambda t: t.sum(axis=axis, keepdims=keepdims),
        lambda a: a.sum(axis=axis, keepdims=keepdims),
        [A],
    )


@pytest.mark.parametrize(
    "axis, keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_mean_grad(axis, keepdims):
    A = np.random.randn(3, 4)
    assert_grad_matches(
        lambda t: t.mean(axis=axis, keepdims=keepdims),
        lambda a: a.mean(axis=axis, keepdims=keepdims),
        [A],
    )


@pytest.mark.parametrize(
    "axis, keepdims",
    [
        ((0, 2), False),
        ((1, 2), True),
        (None, False),
    ],
)
def test_sum_grad_three_dimensional(axis, keepdims):
    """3D multi-axis reduction over a non-adjacent pair (0,2) forces correctly skipping axis 1."""
    A = np.random.randn(2, 3, 4)
    assert_grad_matches(
        lambda t: t.sum(axis=axis, keepdims=keepdims),
        lambda a: a.sum(axis=axis, keepdims=keepdims),
        [A],
    )


def test_shared_node_accumulation():
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


def test_requires_grad_false_is_skipped_in_batched_matmul():
    """requires_grad=False skipping through the batched matmul path."""
    data = Tensor(np.random.randn(6, 5, 3), requires_grad=False)
    w = Tensor(np.random.randn(3, 2), requires_grad=True)
    out = (data @ w).sum()
    out.backward()
    assert data.grad is None
    assert w.grad is not None
    assert w.grad.shape == w.shape


def test_backward_accumulates_across_calls():
    p = Tensor([1.0, 2.0], requires_grad=True)
    (p * 2).sum().backward()
    first = p.grad.copy()
    (p * 2).sum().backward()
    assert np.allclose(p.grad, 2 * first)
