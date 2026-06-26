import numpy as np

from tests.gradcheck_utils import assert_grad_matches
from utils.seed import set_seed

set_seed(0)


def test_relu_grad():
    # push values away from the x=0 kink so finite differences stay valid
    x = np.random.randn(5, 4)
    x = x + np.sign(x) * 0.5
    assert_grad_matches(lambda t: t.relu(), lambda a: np.maximum(a, 0.0), [x])


def test_tanh_grad():
    x = np.random.randn(5, 4)
    assert_grad_matches(lambda t: t.tanh(), lambda a: np.tanh(a), [x])


def test_relu_zeros_negative_part():
    from engine.tensor import Tensor

    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = x.relu()
    assert np.allclose(out.data, [0.0, 0.0, 0.0, 1.0, 2.0])
