"""Gradual cubic sparsity schedule (Zhu & Gupta 2017): prune fast early, then
taper near the target so the network gets fine-tuning time at high sparsity.
"""

from __future__ import annotations


def cubic_sparsity(
    step: int,
    start_step: int,
    end_step: int,
    final_sparsity: float,
    initial_sparsity: float = 0.0,
) -> float:
    """Target sparsity at `step`. Clamped flat before start_step and after
    end_step; the cubic ramp only applies in between.
    """
    if step <= start_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity
    progress = (step - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
