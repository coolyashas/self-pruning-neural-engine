from pathlib import Path

import numpy as np
import pytest

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


def test_train_raises_on_non_finite_loss_instead_of_continuing():
    """Training must halt the moment it diverges, not keep computing
    updates from a NaN/inf loss. Forced by setting one weight to inf
    before the very first forward pass. The loss should reject
    non-finite logits directly, before NumPy's max-subtraction path
    even gets a chance to emit a RuntimeWarning.
    """
    set_seed(0)
    X, y = make_spirals(n_per_class=10, n_classes=3)
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    mlp.layers[0].weight.data[0, 0] = np.inf
    opt = Adam(mlp.parameters(), lr=0.01)

    with pytest.raises(FloatingPointError, match="non-finite logits"):
        train(mlp, opt, X, y, epochs=1, batch_size=8)


def test_non_finite_loss_guard_survives_python_dash_O():
    """This guard exists to STOP training immediately on divergence --
    `assert` is stripped entirely under `python -O`, which would
    silently let training continue computing garbage updates from a
    NaN/inf loss instead. Implemented as `if: raise`, not `assert`,
    specifically so it survives -O. An in-process pytest.raises check
    can't tell the difference (pytest never runs under -O itself), so
    this spawns a real subprocess with -O to confirm it still raises
    there.
    """
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from nn import Linear, ReLU, Sequential\n"
        "from optim import Adam\n"
        "from train.dataset import make_spirals\n"
        "from train.loop import train\n"
        "from utils.seed import set_seed\n"
        "set_seed(0)\n"
        "X, y = make_spirals(n_per_class=10, n_classes=3)\n"
        "mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))\n"
        "mlp.layers[0].weight.data[0, 0] = np.inf\n"
        "opt = Adam(mlp.parameters(), lr=0.01)\n"
        "train(mlp, opt, X, y, epochs=1, batch_size=8)\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "expected a FloatingPointError under -O, but training continued silently"
    assert "FloatingPointError" in result.stderr


def test_on_epoch_end_callback_fires_once_per_epoch():
    set_seed(0)
    mlp = Sequential(Linear(2, 4), ReLU(), Linear(4, 3))
    opt = Adam(mlp.parameters(), lr=0.01)
    X, y = make_spirals(n_per_class=10, n_classes=3)

    calls = []

    def on_epoch_end(epoch, losses_so_far):
        calls.append((epoch, len(losses_so_far)))

    train(mlp, opt, X, y, epochs=3, batch_size=8, on_epoch_end=on_epoch_end)

    assert [c[0] for c in calls] == [0, 1, 2]
    assert calls[0][1] < calls[1][1] < calls[2][1]  # losses list keeps growing
