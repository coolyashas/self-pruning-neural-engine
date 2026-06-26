import numpy as np

from engine.tensor import Tensor
from engine.loss import softmax_cross_entropy
from nn import Linear, ReLU, Sequential
from tests.gradcheck_utils import numerical_gradient
from utils.seed import set_seed

set_seed(0)


def test_sequential_forward_shape():
    mlp = Sequential(Linear(2, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 3))
    x = Tensor(np.random.randn(10, 2))
    out = mlp(x)
    assert out.shape == (10, 3)


def test_sequential_parameters_collects_all_and_only_linear():
    mlp = Sequential(Linear(2, 5), ReLU(), Linear(5, 3))
    params = mlp.parameters()
    assert len(params) == 4  # 2 Linear layers * (weight, bias)
    linear0, linear2 = mlp.layers[0], mlp.layers[2]
    assert params == [linear0.weight, linear0.bias, linear2.weight, linear2.bias]


def test_sequential_end_to_end_grad_through_loss():
    """Small MLP (Linear-ReLU-Linear) -> fused loss, gradchecked end to end.
    Proves the whole stack (Linear, ReLU, Sequential, softmax-CE, backward's
    topo-sort) composes correctly together, not just each piece alone.
    """
    mlp = Sequential(Linear(3, 6), ReLU(), Linear(6, 4))
    x_data = np.random.randn(7, 3)
    labels = np.array([0, 1, 2, 3, 0, 1, 2])

    x = Tensor(x_data, requires_grad=True)
    loss = softmax_cross_entropy(mlp(x), labels)
    loss.backward()

    params = mlp.parameters()
    param_data = [p.data.copy() for p in params]

    def loss_with(values):
        originals = [p.data for p in params]
        for p, v in zip(params, values):
            p.data = v
        out = softmax_cross_entropy(mlp(Tensor(x_data)), labels).data
        for p, orig in zip(params, originals):
            p.data = orig
        return out

    for i, p in enumerate(params):
        def f(perturbed, i=i):
            values = list(param_data)
            values[i] = perturbed
            return loss_with(values)

        numeric = numerical_gradient(f, param_data[i].copy())
        assert np.allclose(p.grad, numeric, atol=1e-5), f"param {i} grad mismatch"

    ng_x = numerical_gradient(
        lambda xx: softmax_cross_entropy(mlp(Tensor(xx)), labels).data, x_data.copy()
    )
    assert np.allclose(x.grad, ng_x, atol=1e-5)
