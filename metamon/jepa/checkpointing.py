"""Checkpoint helpers for paired JEPA training and inference."""

from __future__ import annotations

from typing import Any

from metamon.tokenizer import PokemonTokenizer


def require_tokenizer_from_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: str,
) -> PokemonTokenizer:
    """Load the tokenizer embedded in a JEPA checkpoint.

    Online play must use the exact vocabulary used during training.  Falling
    back to a tokenizer path can silently change token IDs, so missing tokenizer
    metadata is treated as an incompatible checkpoint.
    """
    tokenizer_state = checkpoint.get("tokenizer_state")
    if tokenizer_state is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain tokenizer_state. "
            "Paired JEPA play requires checkpoints saved by train_paired.py "
            "after tokenizer metadata was added; retrain or resave the "
            "checkpoint with tokenizer_state embedded."
        )
    return PokemonTokenizer.from_state(tokenizer_state)


def save_paired_jepa_checkpoint(
    model: Any,
    path: str,
    *,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    vocab_size: int,
    max_history_blocks: int,
    tokenizer: PokemonTokenizer,
) -> None:
    """Save a paired JEPA checkpoint with all inference-critical metadata."""
    model.save_checkpoint(
        path,
        epoch=epoch,
        global_step=global_step,
        config=config,
        vocab_size=vocab_size,
        max_history_blocks=max_history_blocks,
        tokenizer_state=tokenizer.to_state(),
    )
