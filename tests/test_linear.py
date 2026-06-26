import numpy as np

from engine.tensor import Tensor
from nn.linear import Linear
from tests.gradcheck_utils import numerical_gradient
from utils.seed import set_seed

set_seed(0)


def test_linear_forward_shape():
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(5, 4))
    y = layer(x)
    assert y.shape == (5, 3)


def test_linear_forward_and_grad():
    layer = Linear(4, 3)
    x_data = np.random.randn(5, 4)
    x = Tensor(x_data, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()

    W, B = layer.weight.data, layer.bias.data
    ng_x = numerical_gradient(lambda xx: (xx @ W + B).sum(), x_data.copy())
    ng_W = numerical_gradient(lambda ww: (x_data @ ww + B).sum(), W.copy())
    ng_B = numerical_gradient(lambda bb: (x_data @ W + bb).sum(), B.copy())

    assert np.allclose(x.grad, ng_x, atol=1e-5)
    assert np.allclose(layer.weight.grad, ng_W, atol=1e-5)
    assert np.allclose(layer.bias.grad, ng_B, atol=1e-5)


def test_he_init_scale():
    # large fan_in so the sample std is a reliable estimate of the true std
    layer = Linear(2000, 50)
    expected_std = np.sqrt(2.0 / 2000)
    assert abs(layer.weight.data.std() - expected_std) / expected_std < 0.1
    assert np.allclose(layer.bias.data, 0.0)


def test_linear_parameters():
    layer = Linear(3, 2)
    params = layer.parameters()
    assert params == [layer.weight, layer.bias]
    assert all(p.requires_grad for p in params)
