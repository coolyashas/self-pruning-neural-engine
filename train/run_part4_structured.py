"""Part 4: structured (neuron-level) self-pruning -- same cubic schedule as
Part 3, but pruning whole output neurons. Coarser, but the payoff is a real
compressed model (prune/compress.py) that's actually faster, not just
smaller-on-paper.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.tensor import Tensor
from evaluation.cost import weight_sparsity
from nn import Linear, ReLU, Sequential
from optim import Adam
from prune.compress import compress_model
from prune.criteria import accumulate_gradients, neuron_magnitude_scores, neuron_saliency_scores
from prune.mask import prune_neurons_to_count
from prune.schedule import cubic_sparsity
from train.dataset import make_spirals
from train.loop import train
from utils.seed import set_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(
    seed: int = 0,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    final_sparsity: float = 0.7,
    prune_start_step: int = 400,
    prune_end_frac: float = 0.8,
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

    # exclude the final Linear: only hidden-layer output neurons are targets
    prunable_layers = [layer for layer in mlp.layers[:-1] if hasattr(layer, "mask")]
    score_fn = neuron_saliency_scores if criterion == "saliency" else neuron_magnitude_scores

    steps_per_epoch = -(-X.shape[0] // batch_size)
    total_steps = epochs * steps_per_epoch
    prune_end_step = int(total_steps * prune_end_frac)

    def on_step_end(step, model, loss_value):
        if step < prune_start_step or step % prune_every != 0:
            return
        target = cubic_sparsity(step, prune_start_step, prune_end_step, final_sparsity)
        if criterion == "saliency":
            accumulate_gradients(model, X, y, batch_size=64)
        for layer in prunable_layers:
            n_out = layer.weight.shape[1]
            target_active = round((1 - target) * n_out)
            prune_neurons_to_count(layer, score_fn(layer), target_active)

    history: list[tuple[int, float, float, float]] = []  # (step, sparsity, mean_loss, accuracy)

    def on_epoch_end(epoch, losses_so_far):
        step = (epoch + 1) * steps_per_epoch
        mean_loss = float(np.mean(losses_so_far[-steps_per_epoch:]))
        preds = np.argmax(mlp(Tensor(X)).data, axis=1)
        accuracy = float((preds == y).mean())
        # weight_sparsity is connection-fraction; since neurons have equal
        # incoming counts, this stays directly comparable to run_part3's.
        history.append((step, weight_sparsity(mlp), mean_loss, accuracy))

    train(
        mlp,
        opt,
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        on_epoch_end=on_epoch_end,
        on_step_end=on_step_end,
        grad_clip=1.0,
    )

    # sanity check: the compressed model must reproduce the dense forward exactly.
    compressed = compress_model(mlp)
    sanity_x = np.random.randn(8, 2)
    dense_out = mlp(Tensor(sanity_x)).data
    compressed_out = compressed(sanity_x)
    assert np.allclose(dense_out, compressed_out, atol=1e-10), "compressed model disagrees with dense forward"

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "part4_structured_self_pruning.csv"), "w", newline="") as f:
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
    axes[2].set(xlabel="step", ylabel="sparsity", title="Achieved sparsity (structured)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "part4_structured_self_pruning.png"))
    plt.close(fig)

    active_neurons = [int(layer.mask.data.any(axis=0).sum()) for layer in prunable_layers]
    print(
        f"final sparsity={history[-1][1]:.4f} loss={history[-1][2]:.4f} "
        f"accuracy={history[-1][3]:.4f} active_neurons={active_neurons}"
    )
    return history, mlp, compressed


if __name__ == "__main__":
    main()
