from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


POSITIVE_TEMPLATES = [
    "I really enjoyed this experience.",
    "I would happily do this again.",
    "This was far better than I expected.",
    "The overall experience was excellent.",
    "I found this genuinely satisfying.",
    "I would recommend this to my friends.",
    "The result was surprisingly impressive.",
    "This exceeded what I had hoped for.",
    "I came away feeling very pleased.",
    "The experience was rewarding."
]


NEGATIVE_TEMPLATES = [
    "I really disliked this experience.",
    "I would not do this again.",
    "This was far worse than I expected.",
    "The overall experience was terrible.",
    "I found this genuinely frustrating.",
    "I would not recommend this to my friends.",
    "The result was surprisingly disappointing.",
    "This fell far below what I had hoped for.",
    "I came away feeling very dissatisfied.",
    "The experience was frustrating."
]


def build_dataset(
    repetitions: int = 20
) -> Tuple[List[str], np.ndarray]:

    texts = []
    labels = []

    for _ in range(repetitions):

        for text in POSITIVE_TEMPLATES:
            texts.append(text)
            labels.append(1)

        for text in NEGATIVE_TEMPLATES:
            texts.append(text)
            labels.append(0)

    return texts, np.array(labels)


def split_dataset(
    texts,
    labels,
    random_state: int = 42
):
    """
    Split into train / validation / test sets.

    60% train
    20% validation
    20% test
    """

    (
        train_texts,
        temp_texts,
        train_labels,
        temp_labels
    ) = train_test_split(
        texts,
        labels,
        test_size=0.4,
        random_state=random_state,
        stratify=labels
    )

    (
        validation_texts,
        test_texts,
        validation_labels,
        test_labels
    ) = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_labels
    )

    return (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels
    )
