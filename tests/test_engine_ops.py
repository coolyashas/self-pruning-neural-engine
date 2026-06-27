"""Finite-difference gradient checks for every op built so far:
add, sub, mul, div (plain + broadcasting), matmul, sum, mean.
Also covers the two graph-correctness requirements from the pitfalls doc:
shared-node gradient accumulation, and requires_grad=False being skipped.
"""

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
        ((3, 4), (3, 4)),  # matching shapes, no broadcasting
        ((3, 4), (4,)),  # bias-add style broadcast: (N,D)+(D,)
        ((3, 4), (1, 4)),  # explicit size-1 broadcast axis: (N,D)+(1,D)
        ((3, 4), (3, 1)),  # the OTHER size-1 axis: (N,D)+(N,1)
        ((3, 4), ()),  # tensor + scalar
        ((), (3, 4)),  # scalar + tensor (the other operand order)
        ((1, 1), (3, 4)),  # fully-degenerate singleton broadcast both axes
        ((2, 3, 4), (4,)),  # higher-rank: (N,M,D)+(D,), two leading axes added
        ((2, 3, 4), (3, 4)),  # higher-rank: leading axis added, no singleton involved
        ((2, 1, 4), (4,)),  # higher-rank with an explicit singleton axis: (N,1,D)+(D,)
    ],
)
def test_elementwise_grad(op_name, tensor_op, numpy_op, shape_a, shape_b):
    # np.asarray, applied AFTER the arithmetic: np.random.randn() with no
    # size args (the scalar/shape-() case) returns a plain Python float,
    # and under NumPy 2.x a 0-d ndarray + a Python float collapses BACK
    # to a bare numpy scalar (not a 0-d ndarray) due to NEP 50's scalar
    # promotion rules -- confirmed by direct execution. gradcheck_utils
    # needs a real ndarray throughout (it indexes/assigns into a .copy()
    # of every input), so wrap once more after the "+2.0 to avoid /0"
    # offset, not before it.
    a = np.asarray(np.random.randn(*shape_a))
    b = np.asarray(np.random.randn(*shape_b) + (2.0 if op_name == "div" else 0.0))  # avoid /0
    assert_grad_matches(tensor_op, numpy_op, [a, b])


def test_broadcast_then_reduce():
    """Composite chain from the pitfalls checklist: a broadcast op
    feeding directly into a reduction, not tested in isolation by
    test_elementwise_grad/test_sum_grad separately -- (N,D)+(D,)'s
    result reduced over axis=0 exercises _unbroadcast and
    _grad_to_input_shape's reduced-axis reinsertion back to back in
    one backward() call.
    """
    a = np.random.randn(5, 3)
    b = np.random.randn(3)
    assert_grad_matches(
        lambda at, bt: (at + bt).sum(axis=0),
        lambda aa, ba: (aa + ba).sum(axis=0),
        [a, b],
    )


def test_broadcast_then_matmul():
    """Composite chain from the pitfalls checklist: a broadcast add
    feeding into matmul -- chains _unbroadcast's backward with
    matmul's dA = dY@B.T / dB = A.T@dY in one graph, not exercised by
    either op's isolated test.
    """
    x = np.random.randn(5, 3)
    bias = np.random.randn(3)
    w = np.random.randn(3, 2)
    assert_grad_matches(
        lambda xt, bt, wt: (xt + bt) @ wt,
        lambda xa, ba, wa: (xa + ba) @ wa,
        [x, bias, w],
    )


def test_composite_broadcast_chain_three_different_shapes():
    """A chain of THREE different shapes broadcasting at each step in
    turn ((N,D) * (D,) then + (N,1)), not just one broadcast op in
    isolation -- each op's _unbroadcast must correctly reduce back to
    ITS OWN inputs' shapes despite the intermediate result's shape
    having already grown via the previous broadcast.
    """
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
        ((2, 3, 4), (4, 5)),  # batched-looking input (3D), not supported
        ((3,), (3, 4)),  # vector-matrix, not supported
        ((3, 4), (4,)),  # matrix-vector, not supported
    ],
)
def test_matmul_rejects_non_2d_inputs(shape_a, shape_b):
    """matmul is a hard 2D-only scope boundary (no batched matmul, no
    vector-matrix/matrix-vector) -- not a debug-only sanity check, so
    it must raise a clear error rather than either silently computing
    something (NumPy's own @ supports all of these) or crashing later,
    confusingly, inside backward().
    """
    a = Tensor(np.random.randn(*shape_a))
    b = Tensor(np.random.randn(*shape_b))
    with pytest.raises(ValueError, match="2D"):
        a @ b


def test_matmul_2d_guard_survives_python_dash_O():
    """`assert` is stripped entirely under `python -O` -- confirmed by
    direct execution that this silently re-enables exactly the failure
    mode described in matmul.py's docstring: a 3D input's forward
    "succeeds" with no error (NumPy's @ batches it), and backward then
    either crashes confusingly far from the real cause or, if backward
    is never called (e.g. inference-only use), never errors at all.
    Implemented as `if: raise`, not `assert`, specifically so it
    survives -O -- an in-process pytest.raises check can't tell the
    difference, since pytest itself never runs under -O.
    """
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from engine.tensor import Tensor\n"
        "from engine.matmul import matmul\n"
        "a = Tensor(np.random.randn(2, 3, 4))\n"
        "b = Tensor(np.random.randn(4, 5))\n"
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
    """Same -O concern as matmul's 2D guard: under -O, calling
    backward() on a non-scalar Tensor silently seeds every element's
    grad as 1.0 (as if .sum().backward() had been called without
    anyone asking for that) instead of raising -- a silent change in
    what's being differentiated, confirmed by direct execution before
    this guard was converted from `assert` to `if: raise`.
    """
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
        ((0, 1), False),  # multi-axis reduction, collapses to a scalar on a 2D input
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
        ((0, 2), False),  # non-adjacent axes on a 3D input
        ((1, 2), True),
        (None, False),  # reduce every axis of a 3D input at once
    ],
)
def test_sum_grad_three_dimensional(axis, keepdims):
    """Multi-axis reduction on a higher-rank (3D) input -- the 2D
    tests above can't distinguish "reduces the right axes" from
    "reduces all but one", since a 2D input only has two axes to begin
    with. A 3D input with a non-adjacent axis pair (0,2) forces
    _normalize_axis/_grad_to_input_shape to correctly skip axis 1.
    """
    A = np.random.randn(2, 3, 4)
    assert_grad_matches(
        lambda t: t.sum(axis=axis, keepdims=keepdims),
        lambda a: a.sum(axis=axis, keepdims=keepdims),
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
