import numpy as np

from engine.tensor import Tensor
from engine.loss import softmax_cross_entropy
from nn import Linear, ReLU, Sequential
from optim.sgd import SGD
from utils.seed import set_seed

set_seed(0)


def test_sgd_zero_momentum_is_plain_gradient_descent():
    p = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    p.grad = np.array([0.1, 0.2, 0.3])
    opt = SGD([p], lr=0.5, momentum=0.0)
    opt.step()
    assert np.allclose(p.data, [1.0 - 0.05, 2.0 - 0.1, 3.0 - 0.15])


def test_sgd_velocity_recursion_matches_hand_computation():
    p = Tensor([0.0], requires_grad=True)
    opt = SGD([p], lr=1.0, momentum=0.9)
    grads = [1.0, 1.0, 1.0]
    v_ref, x_ref = 0.0, 0.0
    for g in grads:
        p.grad = np.array([g])
        opt.step()
        v_ref = 0.9 * v_ref + g
        x_ref -= 1.0 * v_ref
        assert np.allclose(p.data, [x_ref])


def test_sgd_skips_params_with_no_grad():
    p = Tensor([5.0], requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=0.9)
    opt.step()  # grad is None, should be a no-op
    assert np.allclose(p.data, [5.0])
    assert np.allclose(opt.velocity[0], [0.0])


def test_zero_grad_resets_all_params():
    p1 = Tensor([1.0], requires_grad=True)
    p2 = Tensor([2.0], requires_grad=True)
    p1.grad = np.array([1.0])
    p2.grad = np.array([1.0])
    opt = SGD([p1, p2], lr=0.1)
    opt.zero_grad()
    assert p1.grad is None and p2.grad is None


def test_sgd_training_loop_reduces_loss():
    """Small end-to-end sanity check: real backward() + SGD.step() over a
    few iterations should actually reduce the loss, not just match a
    formula in isolation.
    """
    mlp = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    opt = SGD(mlp.parameters(), lr=0.1, momentum=0.9)
    x_data = np.random.randn(20, 2)
    labels = np.random.randint(0, 3, size=20)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = softmax_cross_entropy(mlp(Tensor(x_data)), labels)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0]
