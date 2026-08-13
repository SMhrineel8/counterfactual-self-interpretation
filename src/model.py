from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"


def get_device() -> torch.device:
    """
    Select GPU when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Load GPT-2 tokenizer and causal language model.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # GPT-2 does not define a pad token by default.
    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    return tokenizer, model, device


def get_sentiment_token_ids(tokenizer):
    """
    Find the token IDs corresponding to ' positive' and ' negative'.

    We deliberately include the leading space because GPT-style
    tokenizers distinguish between 'positive' and ' positive'.
    """
    positive_ids = tokenizer.encode(
        " positive",
        add_special_tokens=False
    )

    negative_ids = tokenizer.encode(
        " negative",
        add_special_tokens=False
    )

    if len(positive_ids) != 1:
        raise ValueError(
            f"' positive' was tokenized into multiple tokens: {positive_ids}"
        )

    if len(negative_ids) != 1:
        raise ValueError(
            f"' negative' was tokenized into multiple tokens: {negative_ids}"
        )

    return positive_ids[0], negative_ids[0]


@torch.no_grad()
def sentiment_score(
    text: str,
    tokenizer,
    model,
    device: torch.device
):
    """
    Compute a simple sentiment logit difference.

    Positive score:
        logit(' positive') - logit(' negative')

    Positive values mean the model assigns greater logit to
    'positive' than 'negative'.
    """
    prompt = (
        f"Review: {text}\n"
        f"Sentiment:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)

    logits = outputs.logits[0, -1]

    positive_id, negative_id = get_sentiment_token_ids(tokenizer)

    positive_logit = logits[positive_id]
    negative_logit = logits[negative_id]

    score = (positive_logit - negative_logit).item()

    return score


@torch.no_grad()
def generate_self_explanation(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 60
) -> str:
    """
    Generate a qualitative self-explanation from the same model.

    Important:
    This is only a self-report proxy. It does not establish
    that the explanation reflects the model's true causal computation.
    """
    prompt = (
        f"Review: {text}\n"
        f"Sentiment:\n"
        f"Explanation:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return generated
