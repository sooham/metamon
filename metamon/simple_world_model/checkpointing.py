"""Checkpoint helpers for simple-world-model training."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Collection
from typing import Any

import torch

from metamon.simple_world_model.data import MODEL_VERSION
from metamon.tokenizer import PokemonTokenizer


def _strip_compile_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        key = key.replace("._orig_mod.", ".")
        cleaned[key] = value
    return cleaned


def model_signature(
    *,
    model_config: dict[str, Any],
    vocab_size: int,
    pad_id: int,
) -> str:
    payload = {
        "model_config": model_config,
        "vocab_size": int(vocab_size),
        "pad_id": int(pad_id),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_simple_world_model_checkpoint(
    path: str,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    model_config: dict[str, Any],
    vocab_size: int,
    pad_id: int,
    tokenizer: PokemonTokenizer,
    components: str | None = None,
    max_history_blocks: int | None = None,
    stage: str | None = None,
    action_vocabulary: dict[str, Any] | None = None,
    dataset_manifest_hash: str | None = None,
    latent_cache_manifest_hash: str | None = None,
    max_context_transitions: int | None = None,
    best_val_loss: float | None = None,
    best_val_epoch: int | None = None,
    best_val_global_step: int | None = None,
    best_val_metrics: dict[str, float] | None = None,
    last_val_metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": _strip_compile_prefixes(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_config": model_config,
            "vocab_size": int(vocab_size),
            "pad_id": int(pad_id),
            "tokenizer_state": tokenizer.to_state(),
            # `components` / `max_history_blocks` remain for explicit
            # incompatibility with pre-V/M/C checkpoints.  New callers use
            # stage and max_context_transitions instead.
            "components": components,
            "max_history_blocks": None if max_history_blocks is None else int(max_history_blocks),
            "model_version": MODEL_VERSION,
            "stage": stage or components,
            "action_vocabulary": action_vocabulary,
            "dataset_manifest_hash": dataset_manifest_hash,
            "latent_cache_manifest_hash": latent_cache_manifest_hash,
            "max_context_transitions": None if max_context_transitions is None else int(max_context_transitions),
            "model_signature": model_signature(
                model_config=model_config,
                vocab_size=vocab_size,
                pad_id=pad_id,
            ),
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "best_val_global_step": best_val_global_step,
            "best_val_metrics": best_val_metrics,
            "last_val_metrics": last_val_metrics,
            # Informational only: unlike model_config, this is not part of the
            # compatibility signature because a deliberate resume may change
            # LR, objective weights, validation cadence, or update budget.
            "training_config": training_config,
        },
        path,
    )


def load_stage_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    """Load a V/M/C checkpoint and reject old simple-world-model artifacts."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a {MODEL_VERSION} checkpoint. "
            "Old simple-world-model checkpoints are intentionally incompatible."
        )
    if expected_stage is not None and checkpoint.get("stage") != expected_stage:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has stage={checkpoint.get('stage')!r}, "
            f"expected {expected_stage!r}"
        )
    return checkpoint


def load_stage_weights(
    model: Any,
    checkpoint: dict[str, Any],
    *,
    prefixes: tuple[str, ...] | None = None,
    strict_prefixes: bool = True,
    allowed_missing_keys: Collection[str] = (),
) -> None:
    """Load selected stage modules while allowing future-stage random heads.

    ``allowed_missing_keys`` is deliberately exact rather than prefix based.
    It is used for architecture warm starts whose newly introduced tensors
    have a known zero-initialized, function-preserving state.  Ordinary stage
    loading remains strict for every selected-prefix tensor.
    """
    state = _strip_compile_prefixes(checkpoint["model_state_dict"])
    if prefixes is not None:
        state = {key: value for key, value in state.items() if key.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint tensors: {unexpected[:8]}")
    if strict_prefixes and prefixes is not None:
        allowed_missing = set(allowed_missing_keys)
        missing_required = [
            key for key in missing
            if key.startswith(prefixes) and key not in allowed_missing
        ]
        if missing_required:
            raise ValueError(f"Missing required stage tensors: {missing_required[:8]}")


def load_matching_weights(model: Any, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = _strip_compile_prefixes(ckpt.get("model_state_dict", ckpt))
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing = [key for key in current if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded warm start from {checkpoint_path}: "
        f"{len(compatible)} tensors, skipped {len(skipped)}, initialized {len(missing)}"
    )
    if skipped:
        for key in skipped[:12]:
            print(f"  skipped incompatible tensor: {key}")
        if len(skipped) > 12:
            print(f"  ... {len(skipped) - 12} more")
    return ckpt


def resume_training_state(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
    model_config: dict[str, Any],
    vocab_size: int,
    pad_id: int,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    expected = model_signature(model_config=model_config, vocab_size=vocab_size, pad_id=pad_id)
    found = ckpt.get("model_signature")
    if found != expected:
        raise ValueError(
            "Cannot --resume because the checkpoint model/config signature does not match. "
            "Use the stage's explicit warm-start option for a compatible architecture transition, "
            "or resume with the original config."
        )
    model.load_state_dict(_strip_compile_prefixes(ckpt["model_state_dict"]), strict=True)
    optimizer_state = ckpt.get("optimizer_state_dict")
    if optimizer_state is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain optimizer state for --resume")
    optimizer.load_state_dict(optimizer_state)
    print(
        f"Resumed training from {checkpoint_path}: "
        f"epoch={ckpt.get('epoch', 0)} global_step={ckpt.get('global_step', 0)}"
    )
    return ckpt
