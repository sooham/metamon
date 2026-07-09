"""Build frozen V posterior sidecars for simple-world-model M/C training.

Run directly with ``python -m metamon.simple_world_model.cache_latents`` or
through ``python -m metamon.simple_world_model.train --stage cache``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from metamon.simple_world_model.action_vocab import ActionVocabulary, canonicalize_action_ids
from metamon.simple_world_model.checkpointing import _strip_compile_prefixes
from metamon.simple_world_model.data import (
    MODEL_VERSION,
    build_action_vocabulary,
    discover_source_shards,
    expected_cache_manifest,
    format_id_to_name,
    load_dataset_metadata,
    sha256_file,
)
from metamon.simple_world_model.model import SimpleWorldModel
from metamon.tokenizer import PokemonTokenizer


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cuda_bf16(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write a complete sidecar or leave no partially-visible final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _load_v(checkpoint_path: str, device: torch.device) -> tuple[SimpleWorldModel, PokemonTokenizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Posterior caching requires a current V/M/C --stage v checkpoint; old artifacts are invalid")
    if checkpoint.get("stage") != "v":
        raise ValueError("Posterior caching requires a --stage v checkpoint")
    tokenizer_state = checkpoint.get("tokenizer_state")
    if tokenizer_state is None:
        raise ValueError("V checkpoint is missing tokenizer_state")
    tokenizer = PokemonTokenizer.from_state(tokenizer_state)
    model_cfg = checkpoint.get("model_config")
    if not model_cfg:
        raise ValueError("V checkpoint is missing model_config")
    model = SimpleWorldModel(
        vocab_size=int(checkpoint.get("vocab_size", len(tokenizer))),
        pad_id=int(checkpoint.get("pad_id", tokenizer.pad_token_id)),
        action_vocab_size=3,
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        v_cfg=model_cfg.get("v", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        m_cfg=model_cfg.get("m", {}),
        controller_cfg=model_cfg.get("controller", {}),
    ).to(device)
    state = _strip_compile_prefixes(checkpoint["model_state_dict"])
    v_state = {key: value for key, value in state.items() if key.startswith("v.")}
    missing, unexpected = model.load_state_dict(v_state, strict=False)
    missing_v = [key for key in missing if key.startswith("v.")]
    if missing_v or unexpected:
        raise ValueError(f"V checkpoint does not match requested V architecture; missing={missing_v[:4]} unexpected={unexpected[:4]}")
    model.v.eval()
    return model, tokenizer, checkpoint


def _slice(data: Any, side: str, index: int, kind: str) -> np.ndarray:
    singular = "state" if kind == "state" else "action"
    flat = data[f"{side}_{kind}s"]
    offsets = data[f"{side}_{singular}_offsets"]
    lengths = data[f"{side}_{singular}_lengths"]
    start = int(offsets[index])
    return np.asarray(flat[start : start + int(lengths[index])])


def _batch_tokens(rows: Sequence[np.ndarray], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(row) for row in rows), default=1)
    tokens = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        count = len(row)
        tokens[index, :count] = torch.as_tensor(row, dtype=torch.long, device=device)
        mask[index, :count] = True
    return tokens, mask


@torch.inference_mode()
def _encode_side(
    model: SimpleWorldModel,
    data: Any,
    *,
    side: str,
    pad_id: int,
    device: torch.device,
    batch_token_budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode every header/state block, with headers encoded *alone*."""
    lengths = np.asarray(data[f"{side}_state_lengths"])
    starts = np.asarray(data[f"{side}_battle_start"])
    latent_dim = model.latent_dim
    mu = np.empty((len(lengths), latent_dim), dtype=np.float16)
    logvar = np.empty_like(mu)
    headers = [int(value) for value in starts[:-1]]
    if headers:
        header_rows = [_slice(data, side, index, "state") for index in headers]
        header_batch_size = max(1, min(256, batch_token_budget // max(max(map(len, header_rows)), 1)))
        for start in range(0, len(headers), header_batch_size):
            stop = min(len(headers), start + header_batch_size)
            tokens, mask = _batch_tokens(header_rows[start:stop], pad_id, device)
            with _cuda_bf16(device):
                batch_mu, batch_logvar = model.v.encode(tokens, header_valid_mask=mask)
            mu[np.asarray(headers[start:stop])] = batch_mu.float().cpu().numpy().astype(np.float16)
            logvar[np.asarray(headers[start:stop])] = batch_logvar.float().cpu().numpy().astype(np.float16)

    # Map each non-header state back to its battle's permanent header.
    header_for_state = np.empty(len(lengths), dtype=np.int64)
    for battle_id, header in enumerate(headers):
        end = int(starts[battle_id + 1])
        header_for_state[header:end] = header
    pending: list[int] = []
    pending_tokens = 0

    def flush() -> None:
        nonlocal pending, pending_tokens
        if not pending:
            return
        header_rows = [_slice(data, side, int(header_for_state[index]), "state") for index in pending]
        # Apply exactly V's independent state cap here too.  The header stays
        # context; only the current-state reconstruction/input block is capped.
        state_rows = [_slice(data, side, index, "state")[: model.v.max_state_tokens] for index in pending]
        headers_t, header_mask = _batch_tokens(header_rows, pad_id, device)
        states_t, state_mask = _batch_tokens(state_rows, pad_id, device)
        with _cuda_bf16(device):
            batch_mu, batch_logvar = model.v.encode(
                headers_t, states_t, header_valid_mask=header_mask, state_valid_mask=state_mask
            )
        mu[np.asarray(pending)] = batch_mu.float().cpu().numpy().astype(np.float16)
        logvar[np.asarray(pending)] = batch_logvar.float().cpu().numpy().astype(np.float16)
        pending = []
        pending_tokens = 0

    header_set = set(headers)
    for index, length in enumerate(lengths):
        if index in header_set:
            continue
        token_count = int(length) + int(lengths[header_for_state[index]])
        if pending and pending_tokens + token_count > batch_token_budget:
            flush()
        pending.append(index)
        pending_tokens += token_count
    flush()
    return mu, logvar


def _action_ids(data: Any, *, side: str, suffix: str, tokenizer: PokemonTokenizer, vocabulary: ActionVocabulary, pad_id: int) -> np.ndarray:
    flat = data[f"{side}_{suffix}"]
    offsets = data[f"{side}_{suffix[:-1]}_offsets"]
    lengths = data[f"{side}_{suffix[:-1]}_lengths"]
    out = np.empty(len(offsets), dtype=np.int32)
    for index, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        canonical = canonicalize_action_ids(
            flat[int(offset) : int(offset) + int(length)], tokenizer=tokenizer, pad_id=pad_id
        )
        out[index] = vocabulary.encode(canonical)
    return out


def _legal_ids(data: Any, *, side: str, tokenizer: PokemonTokenizer, vocabulary: ActionVocabulary, pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = f"{side}_legal_actions"
    if key not in data.files:
        own = _action_ids(data, side=side, suffix="actions", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=pad_id)
        return own[:, None], np.ones((len(own), 1), dtype=np.bool_), np.zeros(len(own), dtype=np.int16)
    raw = np.asarray(data[key])
    mask_key = f"{side}_legal_action_mask"
    mask = np.asarray(
        data[mask_key] if mask_key in data.files else np.any(raw != pad_id, axis=-1),
        dtype=np.bool_,
    )
    ids = np.zeros(raw.shape[:2], dtype=np.int32)
    for row in range(raw.shape[0]):
        for candidate in range(raw.shape[1]):
            if mask[row, candidate]:
                ids[row, candidate] = vocabulary.encode(
                    canonicalize_action_ids(raw[row, candidate], tokenizer=tokenizer, pad_id=pad_id)
                )
    chosen_key = f"{side}_chosen_legal_action_idx"
    chosen = np.asarray(
        data[chosen_key] if chosen_key in data.files else np.zeros(raw.shape[0], dtype=np.int16), dtype=np.int16
    )
    return ids, mask, chosen


def _invert_terminal(classes: np.ndarray) -> np.ndarray:
    result = np.asarray(classes, dtype=np.int16).copy()
    result[classes == 1] = 2
    result[classes == 2] = 1
    result[classes == 3] = 4
    result[classes == 4] = 3
    return result


def _outcomes(data: Any, p1_terminal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    battle_ids = np.asarray(data["battle_id"])
    num_battles = len(np.asarray(data["p1_battle_start"])) - 1
    p1 = np.full(num_battles, 2, dtype=np.int8)  # tie/unknown-safe fallback
    matrix = p1_terminal[:, None] if p1_terminal.ndim == 1 else p1_terminal
    for battle_id in range(num_battles):
        terminal = matrix[battle_ids == battle_id].reshape(-1)
        terminal = terminal[terminal != 0]
        if terminal.size:
            last = int(terminal[-1])
            if last in {1, 3}:
                p1[battle_id] = 0
            elif last in {2, 4}:
                p1[battle_id] = 1
            else:
                p1[battle_id] = 2
    p2 = p1.copy()
    p2[p1 == 0] = 1
    p2[p1 == 1] = 0
    return p1, p2


def _estimate_cache_bytes(shard_paths: Sequence[str], latent_dim: int) -> int:
    total_states = 0
    total_actions = 0
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            total_states += len(data["p1_state_lengths"]) + len(data["p2_state_lengths"])
            total_actions += len(data["p1_action_lengths"]) + len(data["p2_action_lengths"])
    # mu + logvar fp16, plus a deliberately conservative 20% index/action
    # overhead.  This check happens before a potentially 100+ GB write.
    return int((total_states * latent_dim * 2 * 2 + total_actions * 8) * 1.2)


def _check_storage(cache_root: str | Path, required_bytes: int, *, allow_low_disk: bool) -> None:
    usage = shutil.disk_usage(Path(cache_root).resolve().parent)
    print(
        f"Latent cache estimate: {required_bytes / 1024**3:.2f} GiB; "
        f"free: {usage.free / 1024**3:.2f} GiB"
    )
    if required_bytes > usage.free * 0.9 and not allow_low_disk:
        raise RuntimeError(
            "Insufficient free storage for the posterior cache (leaving a 10% safety margin). "
            "Free space or pass --allow-low-disk only if you have external safeguards."
        )


def build_cache(args: argparse.Namespace) -> None:
    device = _device()
    model, tokenizer, _ = _load_v(args.v_checkpoint, device)
    if args.tokenizer_path:
        supplied = PokemonTokenizer().load_tokens_from_disk(args.tokenizer_path)
        if supplied.to_state() != tokenizer.to_state():
            raise ValueError("--tokenizer_path does not match tokenizer_state embedded in the V checkpoint")
    shard_paths = [
        *discover_source_shards(args.data_root, "train", args.formats),
        *discover_source_shards(args.data_root, "val", args.formats),
    ]
    if not shard_paths:
        raise FileNotFoundError("No paired shards found for posterior caching")
    vocabulary = build_action_vocabulary(
        args.data_root, tokenizer=tokenizer, pad_id=tokenizer.pad_token_id, formats=args.formats
    )
    manifest = expected_cache_manifest(
        data_root=args.data_root,
        tokenizer=tokenizer,
        v_checkpoint_path=args.v_checkpoint,
        latent_dim=model.latent_dim,
        action_vocabulary=vocabulary,
    )
    cache_root = Path(args.latent_cache_root)
    _check_storage(cache_root, _estimate_cache_bytes(shard_paths, model.latent_dim), allow_low_disk=args.allow_low_disk)
    coverage: list[dict[str, Any]] = []
    metadata = load_dataset_metadata(args.data_root)
    id_map = format_id_to_name(metadata.get("format_id_map", {}))
    for split in ("train", "val"):
        for source_path in discover_source_shards(args.data_root, split, args.formats):
            source = Path(source_path)
            destination = cache_root / split / source.name
            source_hash = sha256_file(source)
            source_relpath = str(source.relative_to(Path(args.data_root)))
            if destination.exists() and not args.overwrite:
                print(f"Reusing existing sidecar: {destination}")
                coverage.append({"split": split, "source": source.name, "source_relpath": source_relpath, "source_sha256": source_hash,
                                 "cache": str(destination.relative_to(cache_root))})
                continue
            print(f"Caching {source}")
            with np.load(source, allow_pickle=False) as data:
                p1_mu, p1_logvar = _encode_side(
                    model, data, side="p1", pad_id=tokenizer.pad_token_id, device=device,
                    batch_token_budget=args.batch_token_budget,
                )
                p2_mu, p2_logvar = _encode_side(
                    model, data, side="p2", pad_id=tokenizer.pad_token_id, device=device,
                    batch_token_budget=args.batch_token_budget,
                )
                p1_action_ids = _action_ids(data, side="p1", suffix="actions", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=tokenizer.pad_token_id)
                p1_opp_action_ids = _action_ids(data, side="p1", suffix="opponent_actions", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=tokenizer.pad_token_id)
                p2_action_ids = _action_ids(data, side="p2", suffix="actions", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=tokenizer.pad_token_id)
                p2_opp_action_ids = _action_ids(data, side="p2", suffix="opponent_actions", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=tokenizer.pad_token_id)
                p1_legal_ids, p1_legal_mask, p1_chosen = _legal_ids(data, side="p1", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=tokenizer.pad_token_id)
                p2_legal_ids, p2_legal_mask, p2_chosen = _legal_ids(data, side="p2", tokenizer=tokenizer, vocabulary=vocabulary, pad_id=tokenizer.pad_token_id)
                p1_terminal = np.asarray(
                    data["p1_next_terminal_class"]
                    if "p1_next_terminal_class" in data.files
                    else np.zeros_like(data["p1_target_state_idx"], dtype=np.int16),
                    dtype=np.int16,
                )
                if p1_terminal.ndim == 1:
                    p1_terminal = p1_terminal[:, None]
                p2_terminal = _invert_terminal(p1_terminal)
                p1_outcome, p2_outcome = _outcomes(data, p1_terminal)
                arrays = {
                    "p1_mu": p1_mu, "p1_logvar": p1_logvar,
                    "p2_mu": p2_mu, "p2_logvar": p2_logvar,
                    "p1_action_ids": p1_action_ids, "p1_opponent_action_ids": p1_opp_action_ids,
                    "p2_action_ids": p2_action_ids, "p2_opponent_action_ids": p2_opp_action_ids,
                    "p1_legal_action_ids": p1_legal_ids, "p1_legal_action_mask": p1_legal_mask,
                    "p1_chosen_legal_action_idx": p1_chosen,
                    "p2_legal_action_ids": p2_legal_ids, "p2_legal_action_mask": p2_legal_mask,
                    "p2_chosen_legal_action_idx": p2_chosen,
                    "p1_state_idx": np.asarray(data["p1_target_state_idx"], dtype=np.int32),
                    "p1_next_state_idx": np.asarray(data["p1_next_state_idx"], dtype=np.int32),
                    "p1_action_idx": np.asarray(data["p1_action_idx"], dtype=np.int32),
                    "p2_state_idx": np.asarray(data["p2_target_state_idx"], dtype=np.int32),
                    "p2_next_state_idx": np.asarray(data["p2_next_state_idx"], dtype=np.int32),
                    "p2_action_idx": np.asarray(data["p2_action_idx"], dtype=np.int32),
                    "p1_next_terminal_class": p1_terminal,
                    "p2_next_terminal_class": p2_terminal,
                    "p1_outcome": p1_outcome, "p2_outcome": p2_outcome,
                    "battle_id": np.asarray(data["battle_id"], dtype=np.int32),
                    "format_id": np.asarray(data["format_id"], dtype=np.int16),
                    "p1_battle_start": np.asarray(data["p1_battle_start"], dtype=np.int64),
                    "p2_battle_start": np.asarray(data["p2_battle_start"], dtype=np.int64),
                    "p1_battle_action_start": np.asarray(data["p1_battle_action_start"], dtype=np.int64),
                    "p2_battle_action_start": np.asarray(data["p2_battle_action_start"], dtype=np.int64),
                }
            atomic_savez(destination, **arrays)
            coverage.append({"split": split, "source": source.name, "source_relpath": source_relpath, "source_sha256": source_hash,
                             "cache": str(destination.relative_to(cache_root))})
    manifest.update({
        "source_data_root": str(Path(args.data_root).resolve()),
        "coverage": coverage,
        "cache_shard_coverage": coverage,
        "format_id_map": {str(key): value for key, value in id_map.items()},
        "v_checkpoint": str(Path(args.v_checkpoint).resolve()),
    })
    _atomic_json(cache_root / "manifest.json", manifest)
    print(f"Wrote matching latent-cache manifest: {cache_root / 'manifest.json'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V posterior latent sidecars for M/C.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--formats", nargs="+", default=None)
    parser.add_argument("--v_checkpoint", required=True)
    parser.add_argument("--latent_cache_root", required=True)
    parser.add_argument("--tokenizer_path", default=None, help="Optional assertion against the checkpoint tokenizer.")
    parser.add_argument("--batch_token_budget", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-low-disk", action="store_true")
    return parser


def main() -> None:
    build_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
