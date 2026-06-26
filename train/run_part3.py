"""Part 3: self-pruning run -- train while gradually pruning to a target
sparsity via the cubic schedule, using mask-aware Adam so pruned
connections actually freeze. Default criterion is saliency (|w*g|), the
main criterion per the project brief; magnitude is the baseline commit 24
compares it against.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from nn import Linear, ReLU, Sequential
from optim import Adam
from prune import accumulate_gradients, cubic_sparsity, magnitude_scores, prune_to_sparsity, saliency_scores
from train.dataset import make_spirals
from train.loop import train
from utils.seed import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(
    seed: int = 0,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    final_sparsity: float = 0.9,
    prune_start_step: int = 400,
    prune_end_frac: float = 0.8,  # pruning ramp ends at this fraction of total steps
    prune_every: int = 20,
    criterion: str = "saliency",
    n_per_class: int = 300,
    results_dir: str = RESULTS_DIR,
):
    set_seed(seed)
    X, y = make_spirals(n_per_class=n_per_class, n_classes=3, noise=0.2)
    mlp = Sequential(Linear(2, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 3))

    pairs = mlp.masked_parameters()
    params = [p for p, _ in pairs]
    masks = [m for _, m in pairs]
    opt = Adam(params, lr=lr, masks=masks)

    prunable_layers = [layer for layer in mlp.layers if hasattr(layer, "mask")]
    score_fn = saliency_scores if criterion == "saliency" else magnitude_scores

    steps_per_epoch = -(-X.shape[0] // batch_size)
    total_steps = epochs * steps_per_epoch
    prune_end_step = int(total_steps * prune_end_frac)

    def achieved_sparsity() -> float:
        active = sum(layer.mask.data.sum() for layer in prunable_layers)
        total = sum(layer.mask.data.size for layer in prunable_layers)
        return 1.0 - active / total

    def on_step_end(step, model, loss_value):
        if step < prune_start_step or step % prune_every != 0:
            return
        target = cubic_sparsity(step, prune_start_step, prune_end_step, final_sparsity)
        if criterion == "saliency":
            accumulate_gradients(model, X, y, batch_size=64)
        for layer in prunable_layers:
            prune_to_sparsity(layer, score_fn(layer), target)
        # the next mini-batch's optimizer.zero_grad() (inside train()) will
        # clear the saliency sweep's leftover .grad before it's used for a
        # real update -- nothing to do here, but worth knowing why it's safe.

    history: list[tuple[int, float, float, float]] = []  # (step, sparsity, mean_loss, accuracy)

    def on_epoch_end(epoch, losses_so_far):
        step = (epoch + 1) * steps_per_epoch
        mean_loss = float(np.mean(losses_so_far[-steps_per_epoch:]))
        preds = np.argmax(mlp(Tensor(X)).data, axis=1)
        accuracy = float((preds == y).mean())
        history.append((step, achieved_sparsity(), mean_loss, accuracy))

    train(
        mlp,
        opt,
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        on_epoch_end=on_epoch_end,
        on_step_end=on_step_end,
    )

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "part3_self_pruning.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "sparsity", "mean_loss", "accuracy"])
        writer.writerows(history)

    steps_logged, sparsity_logged, loss_logged, acc_logged = zip(*history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps_logged, loss_logged)
    axes[0].set(xlabel="step", ylabel="mean loss", title="Training loss")
    axes[1].plot(steps_logged, acc_logged)
    axes[1].set(xlabel="step", ylabel="accuracy", title="Training accuracy")
    axes[2].plot(steps_logged, sparsity_logged)
    axes[2].axhline(final_sparsity, linestyle="--", color="gray")
    axes[2].set(xlabel="step", ylabel="sparsity", title="Achieved sparsity")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "part3_self_pruning.png"))

    print(
        f"final sparsity={history[-1][1]:.4f} loss={history[-1][2]:.4f} accuracy={history[-1][3]:.4f}"
    )
    return history, mlp


if __name__ == "__main__":
    main()
