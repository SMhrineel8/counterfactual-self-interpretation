from __future__ import annotations

from typing import List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def get_gpt2_layer(model, layer_idx: int):
    """
    Return a GPT-2 transformer block.

    GPT-2 exposes transformer blocks through:
        model.transformer.h[layer_idx]
    """
    return model.transformer.h[layer_idx]


@torch.no_grad()
def collect_last_token_activation(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    layer_idx: int
) -> np.ndarray:
    """
    Collect the final-token hidden representation from one GPT-2 layer
    for each input text.

    Returns:
        shape = [n_examples, hidden_size]
    """

    activations = []

    layer = get_gpt2_layer(model, layer_idx)

    for text in texts:

        prompt = (
            f"Review: {text}\n"
            f"Sentiment:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)

        captured = {}

        def hook(module, inputs, output):
            # GPT-2 blocks return tuples.
            hidden_states = output[0]

            captured["activation"] = (
                hidden_states[:, -1, :]
                .detach()
                .cpu()
            )

        handle = layer.register_forward_hook(hook)

        try:
            model(**inputs)
        finally:
            handle.remove()

        activation = captured["activation"][0].numpy()
        activations.append(activation)

    return np.stack(activations)


def train_linear_probe(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    test_activations: np.ndarray,
    test_labels: np.ndarray
):
    """
    Train a linear probe on hidden representations.

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

    return classifier, accuracy, direction


def find_best_layer(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    tokenizer,
    model,
    device
):
    """
    Evaluate each transformer layer and identify the layer
    whose representations best linearly predict sentiment.
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
            layer_idx
        )

        test_acts = collect_last_token_activation(
            test_texts,
            tokenizer,
            model,
            device,
            layer_idx
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
            "accuracy": accuracy
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
