from pathlib import Path

import numpy as np
import pytest

from engine.tensor import Tensor
from engine.loss import softmax_cross_entropy
from tests.gradcheck_utils import assert_grad_matches
from utils.seed import set_seed

set_seed(0)


def _numpy_softmax_ce(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    log_probs = shifted - np.log(exp.sum(axis=1, keepdims=True))
    n = logits.shape[0]
    return -log_probs[np.arange(n), labels].mean()


@pytest.mark.parametrize(
    "n, c, labels",
    [
        (4, 3, [0, 1, 2, 1]),
        (6, 3, [0, 1, 2, 1, 0, 2]),
        (5, 4, [3, 0, 1, 2, 3]),
    ],
)
def test_softmax_ce_grad(n, c, labels):
    labels = np.array(labels)
    logits = np.random.randn(n, c)
    assert_grad_matches(
        lambda t: softmax_cross_entropy(t, labels),
        lambda x: _numpy_softmax_ce(x, labels),
        [logits],
    )


def test_softmax_ce_large_logits_stay_finite():
    logits = np.array([[1000.0, 1.0, 0.0], [0.0, -1000.0, 5.0]])
    labels = np.array([0, 2])
    t = Tensor(logits, requires_grad=True)
    loss = softmax_cross_entropy(t, labels)
    assert np.isfinite(loss.data)
    loss.backward()
    assert np.all(np.isfinite(t.grad))


def test_non_finite_logits_are_rejected_before_numpy_emits_runtime_warnings():
    logits = Tensor([[np.inf, 0.0, 1.0], [0.0, 2.0, 3.0]], requires_grad=True)
    labels = np.array([0, 1])
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        softmax_cross_entropy(logits, labels)


def test_column_shaped_labels_are_rejected_not_silently_misindexed():
    """A label array of shape (N, 1) instead of (N,) is a common, easy
    mistake (e.g. straight out of a CSV column). NumPy's fancy indexing
    log_probs[arange(n), labels] does NOT raise for that shape -- it
    broadcasts arange(n) (shape (n,)) against labels (shape (n,1)) into
    an (n,n) index pair, silently selecting an n x n block instead of n
    single entries, and .mean() then quietly averages the wrong thing.
    Confirmed by direct execution before this guard existed: the same
    logits with (N,) vs (N,1) labels produced two DIFFERENT scalar
    losses and two different gradients, with no exception either way.
    """
    logits_data = np.random.randn(3, 4)
    labels_2d = np.array([0, 1, 2]).reshape(3, 1)
    t = Tensor(logits_data, requires_grad=True)
    with pytest.raises(ValueError):
        softmax_cross_entropy(t, labels_2d)


def test_negative_label_is_rejected_not_silently_misindexed():
    """A label of -1 is valid NumPy negative indexing, not an
    out-of-bounds error -- without a range check, fancy-indexing
    log_probs[arange(n), labels] would silently select the LAST class
    for that row instead of raising, producing a wrong loss/gradient
    rather than a crash.
    """
    logits = Tensor(np.random.randn(3, 4), requires_grad=True)
    labels = np.array([0, -1, 2])
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, labels)


def test_out_of_range_label_is_rejected():
    logits = Tensor(np.random.randn(3, 4), requires_grad=True)
    labels = np.array([0, 4, 2])  # 4 is out of range for 4 classes (valid: 0-3)
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, labels)


def test_float_labels_are_rejected_with_a_clear_message_not_a_raw_indexerror():
    """A float label array like np.array([0.0, 1.0, 2.0]) gets past
    shape and range checks (both are dtype-agnostic comparisons) and
    would otherwise die inside log_probs[arange(n), labels] with
    NumPy's own raw "arrays used as indices must be of integer (or
    boolean) type" -- not numerically wrong, but a leaky API boundary
    that should fail with a message about THIS function's contract,
    not a downstream NumPy implementation detail.
    """
    logits = Tensor(np.random.randn(3, 4), requires_grad=True)
    labels = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="integer"):
        softmax_cross_entropy(logits, labels)


def test_label_validation_guards_survive_python_dash_O():
    """The actual claim isn't "this raises ValueError" (the tests above
    already cover that) -- it's that the guard SURVIVES `python -O`,
    which strips every `assert` statement from the running process
    entirely. An in-process pytest.raises check can't tell the
    difference between `assert` and `if: raise`, since pytest itself
    never runs under -O. This spawns a real subprocess with -O and
    confirms the (3,1)-shaped-labels case still raises there -- before
    these guards were converted from `assert` to `if: raise`, this
    exact repro silently produced a finite, wrong loss under -O instead.
    """
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from engine.tensor import Tensor\n"
        "from engine.loss import softmax_cross_entropy\n"
        "logits = Tensor(np.random.randn(3, 4), requires_grad=True)\n"
        "labels = np.array([0, 1, 2]).reshape(3, 1)\n"
        "softmax_cross_entropy(logits, labels)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "expected a ValueError under -O, but the call succeeded silently"
    assert "ValueError" in result.stderr


def test_full_pipeline_matmul_bias_broadcast_softmax_ce():
    """Linear-style affine (X@W + b, b broadcast over the batch) feeding
    into the fused loss, checked end to end through one backward() call.
    Exercises matmul's backward chained with add's unbroadcast (#3 in the
    pitfalls doc) followed by the fused softmax-CE backward.
    """
    labels = np.array([0, 1, 2, 1, 0])
    X = np.random.randn(5, 4)
    W = np.random.randn(4, 3)
    b = np.random.randn(3)  # broadcasts (3,) against (5,3)

    def tensor_op(Xt, Wt, Bt):
        logits = Xt @ Wt + Bt
        return softmax_cross_entropy(logits, labels)

    def numpy_op(Xa, Wa, Ba):
        logits = Xa @ Wa + Ba
        return _numpy_softmax_ce(logits, labels)

    assert_grad_matches(tensor_op, numpy_op, [X, W, b])
