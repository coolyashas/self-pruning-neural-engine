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
