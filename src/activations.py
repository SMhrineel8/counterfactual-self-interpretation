from __future__ import annotations

from typing import List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_gpt2_layer(model, layer_idx: int):
    """
    Return a GPT-2 transformer block.
    """
    return model.transformer.h[layer_idx]


@torch.no_grad()
def collect_last_token_activation(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    layer_idx: int,
    batch_size: int = 16
) -> np.ndarray:
    """
    Collect the final-token hidden representation from a GPT-2 layer.

    Uses batched inference to make activation extraction much faster.

    Returns:
        NumPy array with shape:
        [number_of_examples, hidden_size]
    """

    activations = []

    layer = get_gpt2_layer(model, layer_idx)

    for start in range(0, len(texts), batch_size):

        batch_texts = texts[start:start + batch_size]

        prompts = [
            f"Review: {text}\nSentiment:"
            for text in batch_texts
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        captured = {}

        def hook(module, hook_inputs, output):
            """
            Capture the final non-padding token representation.
            """

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if hidden_states.ndim != 3:
                raise RuntimeError(
                    "Expected activation tensor with shape "
                    "[batch, sequence, hidden], but got "
                    f"{tuple(hidden_states.shape)}"
                )

            # Find the actual final token for every item in the batch.
            attention_mask = inputs["attention_mask"]

            last_positions = (
                attention_mask.sum(dim=1) - 1
            )

            batch_indices = torch.arange(
                hidden_states.size(0),
                device=hidden_states.device
            )

            final_tokens = hidden_states[
                batch_indices,
                last_positions,
                :
            ]

            captured["activation"] = (
                final_tokens
                .detach()
                .cpu()
            )

        handle = layer.register_forward_hook(hook)

        try:
            model(**inputs)
        finally:
            handle.remove()

        if "activation" not in captured:
            raise RuntimeError(
                "The activation hook did not capture any output."
            )

        activations.append(
            captured["activation"].numpy()
        )

    return np.concatenate(
        activations,
        axis=0
    )


def train_linear_probe(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    test_activations: np.ndarray,
    test_labels: np.ndarray
):
    """
    Train a linear classifier over internal activations.

    Returns:
        classifier
        test_accuracy
        normalized_direction
    """

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    classifier.fit(
        train_activations,
        train_labels
    )

    predictions = classifier.predict(
        test_activations
    )

    accuracy = accuracy_score(
        test_labels,
        predictions
    )

    direction = classifier.coef_[0]

    direction = direction / (
        np.linalg.norm(direction) + 1e-12
    )

    return (
        classifier,
        accuracy,
        direction
    )


def find_best_layer(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    tokenizer,
    model,
    device,
    batch_size: int = 16
):
    """
    Evaluate GPT-2 layers and find the layer where
    sentiment is most linearly recoverable.

    Activation extraction is batched for efficiency.
    """

    num_layers = model.config.n_layer

    results = []

    cached_train = {}
    cached_test = {}

    for layer_idx in range(num_layers):

        print(f"Testing layer {layer_idx}...")

        train_acts = collect_last_token_activation(
            train_texts,
            tokenizer,
            model,
            device,
            layer_idx,
            batch_size=batch_size
        )

        test_acts = collect_last_token_activation(
            test_texts,
            tokenizer,
            model,
            device,
            layer_idx,
            batch_size=batch_size
        )

        (
            classifier,
            accuracy,
            direction
        ) = train_linear_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels
        )

        results.append({
            "layer": layer_idx,
            "accuracy": float(accuracy)
        })

        cached_train[layer_idx] = train_acts
        cached_test[layer_idx] = test_acts

        print(
            f"Layer {layer_idx}: "
            f"probe accuracy = {accuracy:.3f}"
        )

    best_result = max(
        results,
        key=lambda x: x["accuracy"]
    )

    best_layer = best_result["layer"]

    return (
        best_layer,
        results,
        cached_train,
        cached_test
    )
