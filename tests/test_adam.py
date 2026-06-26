import numpy as np

from engine.tensor import Tensor
from engine.loss import softmax_cross_entropy
from nn import Linear, ReLU, Sequential
from optim.adam import Adam
from utils.seed import set_seed

set_seed(0)


def test_adam_first_step_matches_hand_computation():
    # t=1 is the strongest test of bias correction: (1 - beta**1) is far
    # from 1, so an unbiased-but-wrong implementation would diverge a lot.
    p = Tensor([1.0], requires_grad=True)
    p.grad = np.array([0.5])
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    opt = Adam([p], lr=lr, beta1=b1, beta2=b2, eps=eps)
    opt.step()

    m = (1 - b1) * 0.5
    v = (1 - b2) * 0.5**2
    m_hat = m / (1 - b1**1)
    v_hat = v / (1 - b2**1)
    expected = 1.0 - lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(p.data, [expected])


def test_adam_multi_step_matches_reference_loop():
    rng = np.random.RandomState(0)
    grads = [rng.randn(3) for _ in range(5)]
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

    # independent reference implementation of the same recursion
    x_ref = np.zeros(3)
    m_ref = np.zeros(3)
    v_ref = np.zeros(3)
    for t, g in enumerate(grads, start=1):
        m_ref = b1 * m_ref + (1 - b1) * g
        v_ref = b2 * v_ref + (1 - b2) * g**2
        m_hat = m_ref / (1 - b1**t)
        v_hat = v_ref / (1 - b2**t)
        x_ref -= lr * m_hat / (np.sqrt(v_hat) + eps)

    p = Tensor(np.zeros(3), requires_grad=True)
    opt = Adam([p], lr=lr, beta1=b1, beta2=b2, eps=eps)
    for g in grads:
        p.grad = g
        opt.step()

    assert np.allclose(p.data, x_ref, atol=1e-10)


def test_adam_skips_params_with_no_grad():
    p = Tensor([5.0], requires_grad=True)
    opt = Adam([p], lr=0.1)
    opt.step()
    assert np.allclose(p.data, [5.0])
    assert np.allclose(opt.m[0], [0.0])
    assert np.allclose(opt.v[0], [0.0])


def test_zero_grad_resets_all_params():
    p1 = Tensor([1.0], requires_grad=True)
    p2 = Tensor([2.0], requires_grad=True)
    p1.grad = np.array([1.0])
    p2.grad = np.array([1.0])
    opt = Adam([p1, p2], lr=0.1)
    opt.zero_grad()
    assert p1.grad is None and p2.grad is None


def test_adam_training_loop_reduces_loss():
    mlp = Sequential(Linear(2, 8), ReLU(), Linear(8, 3))
    opt = Adam(mlp.parameters(), lr=0.01)
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
