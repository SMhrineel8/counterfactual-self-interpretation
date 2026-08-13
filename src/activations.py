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
    layer_idx: int
) -> np.ndarray:
    """
    Collect the final-token hidden representation from a GPT-2 layer.

    Returns:
        NumPy array with shape:
        [number_of_examples, hidden_size]
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
            """
            GPT-2 transformer blocks return:
                hidden_states
            or, depending on Transformers version/configuration,
                (hidden_states, ...)
            """

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # We expect:
            # [batch, sequence_length, hidden_size]
            #
            # But some model/config combinations can give a
            # different shape, so explicitly check it.
            if hidden_states.ndim == 3:
                final_token = hidden_states[:, -1, :]

            elif hidden_states.ndim == 2:
                # Fallback:
                # [sequence_length, hidden_size]
                final_token = hidden_states[-1, :].unsqueeze(0)

            else:
                raise RuntimeError(
                    f"Unexpected activation shape: "
                    f"{tuple(hidden_states.shape)}"
                )

            captured["activation"] = (
                final_token
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
    Train a linear classifier over internal activations.
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
    device
):
    """
    Evaluate all GPT-2 layers and find the layer
    where sentiment is most linearly recoverable.
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
