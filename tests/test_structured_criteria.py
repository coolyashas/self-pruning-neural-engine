import numpy as np

from nn.linear import Linear
from prune.criteria import neuron_magnitude_scores, neuron_saliency_scores
from utils.seed import set_seed

set_seed(0)


def test_neuron_magnitude_scores_shape_and_formula():
    # non-square on purpose: an axis mixup would give shape (10,) not (20,)
    layer = Linear(10, 20)
    layer.weight.data = np.random.randn(10, 20)

    scores = neuron_magnitude_scores(layer)
    assert scores.shape == (20,)
    assert np.allclose(scores, np.abs(layer.weight.data).sum(axis=0))


def test_neuron_saliency_scores_shape_and_formula():
    layer = Linear(10, 20)
    layer.weight.data = np.random.randn(10, 20)
    layer.weight.grad = np.random.randn(10, 20)

    scores = neuron_saliency_scores(layer)
    assert scores.shape == (20,)
    assert np.allclose(scores, np.abs((layer.weight.data * layer.weight.grad).sum(axis=0)))


def test_neuron_saliency_sums_signed_then_abs_not_abs_then_sum():
    """The Taylor estimate of removing a neuron is |sum_i w_i*g_i|, NOT
    sum_i |w_i*g_i|: a column whose contributions cancel must score near zero.
    """
    layer = Linear(2, 2)
    # column 0: contributions +6 and -6 -> signed sum 0, abs-then-sum 12.
    # column 1: contributions +1 and +1 -> signed sum 2, abs-then-sum 2.
    layer.weight.data = np.array([[3.0, 1.0], [-3.0, 1.0]])
    layer.weight.grad = np.array([[2.0, 1.0], [2.0, 1.0]])

    sal = neuron_saliency_scores(layer)
    assert np.allclose(sal, [0.0, 2.0])
    # the cancelling column is LEAST important, the opposite of abs-then-sum
    assert np.argmin(sal) == 0


def test_neuron_saliency_requires_a_gradient():
    layer = Linear(10, 20)
    try:
        neuron_saliency_scores(layer)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_neuron_magnitude_and_saliency_can_rank_neurons_oppositely():
    """Structured analogue of test_saliency_and_magnitude_can_disagree: a
    neuron with large weights but near-zero gradient ranks low on saliency,
    high on magnitude, and vice versa.
    """
    layer = Linear(2, 2)
    # column 0: large weights, tiny gradient. column 1: small weights, huge gradient.
    layer.weight.data = np.array([[10.0, 0.1], [10.0, 0.1]])
    layer.weight.grad = np.array([[0.001, 5.0], [0.001, 5.0]])

    mag = neuron_magnitude_scores(layer)
    sal = neuron_saliency_scores(layer)

    assert np.argmax(mag) == 0  # column 0 wins on magnitude
    assert np.argmax(sal) == 1  # column 1 wins on saliency
