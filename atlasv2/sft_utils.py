"""
Utilities for Supervised Fine-Tuning (SFT) with instruction datasets.

This module contains helper functions used in:
- Alpaca-style instruction formatting
- Dataset preparation for TRL's SFTTrainer
"""

from tqdm import tqdm


def prepare_sample_text (example):
    """
    Convert a single Alpaca-style example into a formatted training string.

    The output format follows the classic Alpaca prompt:

        ### Question:
        <instruction>

        ### Input:
        <optional input>

        ### Response:
        <expected output>

    Args:
        example (dict): Dictionary with keys:
            - instruction
            - input
            - output

    Returns:
        str: Formatted text used for SFT training.
    """
    if example.get ("input", "") == "":
        return (
            f"### Question:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}\n\n"
        )

    return (
        f"### Question:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}\n\n"
    )


def chars_token_ratio (dataset, tokenizer, nb_examples: int = 200):
    """
    Estimate the average number of characters per token in a dataset.

    This value is required by TRL's ConstantLengthDataset to efficiently
    pack text samples into fixed-length token sequences.

    Args:
        dataset: Hugging Face dataset split (iterable of examples).
        tokenizer: Tokenizer used by the model.
        nb_examples (int): Number of samples used for estimation.

    Returns:
        float: Estimated characters-per-token ratio.
    """
    total_characters = 0
    total_tokens = 0

    for _, example in tqdm (
        zip (range (nb_examples), iter (dataset)),
        total = nb_examples,
        desc = "Estimating chars/token ratio"
    ):
        text = prepare_sample_text (example)
        total_characters += len (text)

        if tokenizer.is_fast:
            total_tokens += len (tokenizer (text).tokens ())
        else:
            total_tokens += len (tokenizer.tokenize (text))

    return total_characters / max (total_tokens, 1)
