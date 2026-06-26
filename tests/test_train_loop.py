import numpy as np

from engine.tensor import Tensor
from nn import Linear, ReLU, Sequential
from optim import Adam
from train.dataset import make_spirals
from train.loop import train
from utils.seed import set_seed


def test_make_spirals_shapes_and_labels():
    set_seed(0)
    X, y = make_spirals(n_per_class=50, n_classes=3)
    assert X.shape == (150, 2)
    assert y.shape == (150,)
    assert sorted(np.unique(y)) == [0, 1, 2]
    assert (y == 0).sum() == 50 and (y == 1).sum() == 50 and (y == 2).sum() == 50


def test_train_loop_stable_no_nan_and_learns():
    """Full stack, real architecture from the locked scope (2-128-128-3,
    He init, ReLU, Adam): trains on spirals with no NaNs and actually
    learns, not just "loss goes down a little".
    """
    set_seed(0)
    X, y = make_spirals(n_per_class=300, n_classes=3, noise=0.2)
    mlp = Sequential(Linear(2, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 3))
    opt = Adam(mlp.parameters(), lr=0.01)

    losses = train(mlp, opt, X, y, epochs=200, batch_size=32)

    assert all(np.isfinite(l) for l in losses)
    early = np.mean(losses[: len(losses) // 10])
    late = np.mean(losses[-len(losses) // 10 :])
    assert late < early

    preds = np.argmax(mlp(Tensor(X)).data, axis=1)
    accuracy = (preds == y).mean()
    assert accuracy > 0.9
