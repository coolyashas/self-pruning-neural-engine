"""Part 2: train the dense (unpruned) MLP on spirals, log the learning
curve, save plot + CSV to results/.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless: save to file, never tries to open a window
import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from nn import Linear, ReLU, Sequential
from optim import Adam
from train.dataset import make_spirals
from train.loop import train
from utils.seed import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.01,
    n_per_class: int = 300,
    results_dir: str = RESULTS_DIR,
) -> list[tuple[int, float, float]]:
    set_seed(seed)
    X, y = make_spirals(n_per_class=n_per_class, n_classes=3, noise=0.2)
    mlp = Sequential(Linear(2, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 3))
    opt = Adam(mlp.parameters(), lr=lr)

    steps_per_epoch = -(-X.shape[0] // batch_size)  # ceil div
    history: list[tuple[int, float, float]] = []  # (epoch, mean_loss, accuracy)

    def on_epoch_end(epoch, losses_so_far):
        mean_loss = float(np.mean(losses_so_far[-steps_per_epoch:]))
        preds = np.argmax(mlp(Tensor(X)).data, axis=1)
        accuracy = float((preds == y).mean())
        history.append((epoch, mean_loss, accuracy))

    train(mlp, opt, X, y, epochs=epochs, batch_size=batch_size, on_epoch_end=on_epoch_end)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "part2_learning_curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_loss", "accuracy"])
        writer.writerows(history)

    epochs_logged, losses_logged, accs_logged = zip(*history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs_logged, losses_logged)
    ax1.set(xlabel="epoch", ylabel="mean loss", title="Training loss")
    ax2.plot(epochs_logged, accs_logged)
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Training accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "part2_learning_curve.png"))

    print(f"final loss={history[-1][1]:.4f} accuracy={history[-1][2]:.4f}")
    return history


if __name__ == "__main__":
    main()
