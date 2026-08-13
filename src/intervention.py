from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .activations import get_gpt2_layer
from .model import sentiment_score


def normalize_direction(direction: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy vector into a unit-length torch vector.
    """
    tensor = torch.tensor(
        direction,
        dtype=torch.float32
    )

    return tensor / (
        tensor.norm() + 1e-12
    )


@torch.no_grad()
def run_intervention(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    layer_idx: int,
    direction: np.ndarray,
    strength: float,
    scale: float
):
    """
    Add a directional perturbation to the final-token representation
    at a chosen transformer layer.

    Positive strength pushes the activation along the learned
    sentiment direction.

    Negative strength pushes it in the opposite direction.

    Returns:
        baseline score
        intervened score
        delta
    """

    direction_tensor = normalize_direction(
        direction
    ).to(device)

    baseline = sentiment_score(
        text,
        tokenizer,
        model,
        device
    )

    layer = get_gpt2_layer(
        model,
        layer_idx
    )

    def intervention_hook(module, inputs, output):

        hidden_states = output[0].clone()

        # Apply intervention only to the final token.
        delta = (
            strength
            * scale
            * direction_tensor
        )

        hidden_states[:, -1, :] += delta

        return (
            hidden_states,
            *output[1:]
        )

    handle = layer.register_forward_hook(
        intervention_hook
    )

    try:
        intervened = sentiment_score(
            text,
            tokenizer,
            model,
            device
        )
    finally:
        handle.remove()

    return {
        "baseline_score": baseline,
        "intervened_score": intervened,
        "delta": intervened - baseline,
        "strength": strength
    }


def run_intervention_sweep(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    layer_idx: int,
    direction: np.ndarray,
    strengths: Sequence[float],
    scale: float
):
    """
    Run multiple intervention strengths.
    """

    results = []

    for strength in strengths:

        result = run_intervention(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            layer_idx=layer_idx,
            direction=direction,
            strength=strength,
            scale=scale
        )

        results.append(result)

    return results
