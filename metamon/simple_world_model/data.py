"""Data, cache-manifest, and batching helpers for the V/M/C pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from metamon.simple_world_model.action_vocab import ActionVocabulary, canonicalize_action_ids


CACHE_SCHEMA_VERSION = "simple_world_model_latent_cache_v1"
MODEL_VERSION = "simple_world_model_vmc_v2"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_metadata(data_root: str | Path) -> dict[str, Any]:
    path = Path(data_root) / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_manifest_hash(data_root: str | Path) -> str:
    """Hash source metadata, not paths/mtimes, so moved datasets still match."""
    metadata = load_dataset_metadata(data_root)
    if metadata:
        return _stable_hash(metadata)
    # Tiny test fixtures often omit metadata.  A deterministic directory
    # inventory is sufficient there and avoids pretending that it is a normal
    # generated production dataset.
    root = Path(data_root)
    inventory = [
        (str(path.relative_to(root)), path.stat().st_size)
        for path in sorted(root.glob("**/paired_shard_*.npz"))
    ]
    return _stable_hash({"metadata_missing": True, "inventory": inventory})


def tokenizer_state_hash(tokenizer: object) -> str:
    return _stable_hash(tokenizer.to_state())  # type: ignore[attr-defined]


def discover_source_shards(data_root: str | Path, split: str, formats: Sequence[str] | None = None) -> list[str]:
    """Find current flat shards first, with the legacy per-format fallback."""
    root = Path(data_root)
    flat = sorted((root / split).glob("paired_shard_*.npz"))
    if flat:
        return [str(path) for path in flat]
    paths: list[Path] = []
    for fmt in formats or ():
        paths.extend(sorted((root / str(fmt) / split).glob("paired_shard_*.npz")))
    return [str(path) for path in paths]


def format_id_to_name(raw: Mapping[object, object] | None) -> dict[int, str]:
    """Normalize either generator spelling of ``format_id_map``.

    Current generated metadata uses ``{"gen1ou": 0}``, while earlier
    manifests used ``{"0": "gen1ou"}``.  Cache and training loaders accept
    both so a valid paired dataset does not make the launcher exit instantly.
    """
    result: dict[int, str] = {}
    for key, value in dict(raw or {}).items():
        try:
            result[int(key)] = str(value)
        except (TypeError, ValueError):
            try:
                result[int(value)] = str(key)
            except (TypeError, ValueError):
                continue
    return result


def _format_id_map(data_root: str | Path) -> dict[int, str]:
    return format_id_to_name(load_dataset_metadata(data_root).get("format_id_map", {}))


def _normalise_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return value[:, None] if value.ndim == 1 else value


def _shard_formats(data: Mapping[str, np.ndarray], id_map: Mapping[int, str]) -> list[str]:
    if "format_id" in data:
        ids = {int(v) for v in np.asarray(data["format_id"]).reshape(-1)}
        names = [id_map.get(value, str(value)) for value in sorted(ids)]
        if names:
            return names
    if "format_name" in data:
        raw = str(np.asarray(data["format_name"]).item())
        names = [item.strip() for item in raw.split(",") if item.strip()]
        if names:
            return names
    return ["unknown"]


def _slice_block(data: Mapping[str, np.ndarray], side: str, index: int, kind: str = "state") -> np.ndarray:
    singular = "state" if kind == "state" else "action"
    flat_name = f"{side}_{kind}s"
    offsets = np.asarray(data[f"{side}_{singular}_offsets"])
    lengths = np.asarray(data[f"{side}_{singular}_lengths"])
    start = int(offsets[index])
    return np.asarray(data[flat_name][start : start + int(lengths[index])])


def build_action_vocabulary(
    data_root: str | Path,
    *,
    tokenizer: object,
    pad_id: int,
    formats: Sequence[str] | None = None,
) -> ActionVocabulary:
    """Build from every observed and legal candidate action, not clicks alone."""
    id_map = _format_id_map(data_root)
    observed: list[tuple[str, str | None]] = []
    for split in ("train", "val"):
        for path in discover_source_shards(data_root, split, formats):
            with np.load(path, allow_pickle=False) as source:
                data = {key: source[key] for key in source.files}
            shard_formats = _shard_formats(data, id_map)
            # Assigning a multi-format shard's vocabulary to every included
            # format is conservative: it cannot mask an actual target out.
            for side in ("p1", "p2"):
                for suffix in ("actions", "opponent_actions"):
                    flat = data[f"{side}_{suffix}"]
                    offsets = data[f"{side}_{suffix[:-1]}_offsets"]
                    lengths = data[f"{side}_{suffix[:-1]}_lengths"]
                    for offset, length in zip(offsets, lengths, strict=True):
                        action = canonicalize_action_ids(
                            flat[int(offset) : int(offset) + int(length)], tokenizer=tokenizer, pad_id=pad_id
                        )
                        observed.extend((action, fmt) for fmt in shard_formats)
                legal_key = f"{side}_legal_actions"
                if legal_key in data:
                    mask = np.asarray(data.get(f"{side}_legal_action_mask", np.ones(data[legal_key].shape[:2], dtype=bool)))
                    for row, row_mask in zip(data[legal_key], mask, strict=True):
                        for candidate, valid in zip(row, row_mask, strict=True):
                            if bool(valid):
                                action = canonicalize_action_ids(candidate, tokenizer=tokenizer, pad_id=pad_id)
                                observed.extend((action, fmt) for fmt in shard_formats)
    return ActionVocabulary.build(observed)


def expected_cache_manifest(
    *,
    data_root: str | Path,
    tokenizer: object,
    v_checkpoint_path: str | Path,
    latent_dim: int,
    action_vocabulary: ActionVocabulary,
    dtype: str = "float16",
) -> dict[str, Any]:
    data_hash = dataset_manifest_hash(data_root)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "dataset_manifest_hash": data_hash,
        # Explicit spelling retained in the manifest because it is useful to
        # humans inspecting cache provenance.
        "dataset_metadata_hash": data_hash,
        "tokenizer_state_hash": tokenizer_state_hash(tokenizer),
        "v_checkpoint_hash": sha256_file(v_checkpoint_path),
        "latent_dim": int(latent_dim),
        "dtype": str(dtype),
        "action_vocabulary": action_vocabulary.to_state(),
    }


def cache_manifest_path(cache_root: str | Path) -> Path:
    return Path(cache_root) / "manifest.json"


def load_cache_manifest(cache_root: str | Path) -> dict[str, Any]:
    path = cache_manifest_path(cache_root)
    if not path.exists():
        raise FileNotFoundError(f"No latent-cache manifest at {path}; run --stage cache first")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_matching_cache(
    cache_root: str | Path,
    *,
    data_root: str | Path,
    tokenizer: object,
    v_checkpoint_path: str | Path,
    latent_dim: int | None = None,
) -> dict[str, Any]:
    """Reject stale cache artifacts before M/C can train on them."""
    manifest = load_cache_manifest(cache_root)
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "dataset_manifest_hash": dataset_manifest_hash(data_root),
        "dataset_metadata_hash": dataset_manifest_hash(data_root),
        "tokenizer_state_hash": tokenizer_state_hash(tokenizer),
        "v_checkpoint_hash": sha256_file(v_checkpoint_path),
    }
    if latent_dim is not None:
        expected["latent_dim"] = int(latent_dim)
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        summary = "; ".join(f"{key}: cache={actual!r}, expected={wanted!r}" for key, (actual, wanted) in mismatches.items())
        raise ValueError(f"Latent cache manifest mismatch; rebuild cache. {summary}")
    coverage = manifest.get("cache_shard_coverage", manifest.get("coverage", []))
    coverage_mismatches: list[str] = []
    for item in coverage:
        if not isinstance(item, Mapping) or "source_sha256" not in item:
            continue
        source = Path(data_root) / str(item.get(
            "source_relpath", Path(str(item.get("split", ""))) / str(item.get("source", ""))
        ))
        cache_sidecar = Path(cache_root) / str(item.get("cache", ""))
        if not source.exists() or sha256_file(source) != item["source_sha256"] or not cache_sidecar.is_file():
            coverage_mismatches.append(f"{source} -> {cache_sidecar}")
    if coverage_mismatches:
        raise ValueError(
            "Latent cache shard coverage differs from the requested dataset; rebuild cache. "
            f"Examples: {coverage_mismatches[:3]}"
        )
    return manifest


@dataclass(frozen=True)
class VStateRef:
    shard_index: int
    side: str
    state_index: int
    header_index: int
    battle_id: int
    fmt: str
    length: int


class _NpzHandleCache:
    """Lazy per-process npz handles; avoids eagerly constructing p2 tensors."""

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self.handles: dict[int, Any] = {}

    def get(self, index: int) -> Any:
        if index not in self.handles:
            self.handles[index] = np.load(self.paths[index], allow_pickle=False)
        return self.handles[index]

    def __getstate__(self) -> dict[str, Any]:
        return {"paths": self.paths, "handles": {}}


class VStateDataset(Dataset[dict[str, Any]]):
    """One header/current-visible-state V item, from both player POVs."""

    def __init__(
        self, shard_paths: Sequence[str], *, data_root: str | Path,
        max_state_tokens: int | None = None,
    ):
        if not shard_paths:
            raise ValueError("No paired shards for V stage")
        self.paths = list(shard_paths)
        self.max_state_tokens = None if max_state_tokens is None else int(max_state_tokens)
        self.refs: list[VStateRef] = []
        id_map = _format_id_map(data_root)
        for shard_index, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as source:
                state_lengths = {side: np.asarray(source[f"{side}_state_lengths"]) for side in ("p1", "p2")}
                starts = {side: np.asarray(source[f"{side}_battle_start"]) for side in ("p1", "p2")}
                battle_ids = np.asarray(source.get("battle_id", np.empty(0, dtype=np.int32)))
                fmt_ids = _normalise_matrix(np.asarray(source.get("format_id", np.empty((0, 1), dtype=np.int16))))
                battle_fmt: dict[int, str] = {}
                for row, battle_id in enumerate(battle_ids):
                    if row < len(fmt_ids):
                        battle_fmt.setdefault(int(battle_id), id_map.get(int(fmt_ids[row, 0]), str(int(fmt_ids[row, 0]))))
                default_fmt = _shard_formats({key: source[key] for key in source.files}, id_map)[0]
                for side in ("p1", "p2"):
                    for battle_id in range(len(starts[side]) - 1):
                        header = int(starts[side][battle_id])
                        end = int(starts[side][battle_id + 1])
                        fmt = battle_fmt.get(battle_id, default_fmt)
                        # Header alone is cached but is not a V reconstruction
                        # training sample; only visible state blocks are.
                        for state_index in range(header + 1, end):
                            self.refs.append(
                                VStateRef(
                                    shard_index, side, state_index, header, battle_id, fmt,
                                    min(int(state_lengths[side][state_index]), self.max_state_tokens or int(state_lengths[side][state_index])),
                                )
                            )
        self._handles = _NpzHandleCache(self.paths)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        data = self._handles.get(ref.shard_index)
        state = _slice_block(data, ref.side, ref.state_index, "state").astype(np.int64, copy=False)
        if self.max_state_tokens is not None:
            state = state[: self.max_state_tokens]
        return {
            "header": _slice_block(data, ref.side, ref.header_index, "state").astype(np.int64, copy=False),
            "state": state,
            "fmt": ref.fmt,
            "battle_id": ref.battle_id,
            "length": ref.length,
        }


def _pad_1d(rows: Sequence[np.ndarray], pad_value: int, *, dtype: torch.dtype = torch.long) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(row) for row in rows), default=1)
    output = torch.full((len(rows), width), int(pad_value), dtype=dtype)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row)
        output[index, :count] = torch.as_tensor(row, dtype=dtype)
        mask[index, :count] = True
    return output, mask


def collate_v(samples: Sequence[dict[str, Any]], *, pad_id: int) -> dict[str, Any]:
    header, header_mask = _pad_1d([sample["header"] for sample in samples], pad_id)
    state, state_mask = _pad_1d([sample["state"] for sample in samples], pad_id)
    return {
        "header_tokens": header,
        "header_mask": header_mask,
        "state_tokens": state,
        "state_mask": state_mask,
        "formats": [str(sample["fmt"]) for sample in samples],
        "battle_ids": [int(sample["battle_id"]) for sample in samples],
    }


class BalancedFormatBatchSampler(Sampler[list[int]]):
    """Length-bucketed sampler with exact equal mass for two formats.

    The shorter source is sampled cyclically.  That is preferable to silently
    letting the dominant generation determine the representation geometry.
    """

    def __init__(
        self,
        formats: Sequence[str],
        lengths: Sequence[int],
        *,
        batch_size: int,
        balanced: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if len(formats) != len(lengths):
            raise ValueError("formats and lengths must have equal size")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.balanced = bool(balanced)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.by_format: dict[str, list[int]] = {}
        for index, fmt in enumerate(formats):
            self.by_format.setdefault(str(fmt), []).append(index)
        self.lengths = list(map(int, lengths))
        self.active_formats = sorted(fmt for fmt, indices in self.by_format.items() if indices)
        if self.balanced and len(self.active_formats) == 2 and self.batch_size % 2:
            raise ValueError("balanced Gen1/Gen9 batches require an even --batch_size")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.balanced and len(self.active_formats) == 2:
            half = self.batch_size // 2
            return max(math.ceil(len(self.by_format[fmt]) / half) for fmt in self.active_formats)
        return math.ceil(sum(len(value) for value in self.by_format.values()) / self.batch_size)

    def _ordered(self, indices: Sequence[int], rng: random.Random) -> list[int]:
        # Sort inside local buckets first; randomize bucket order, preserving
        # low padding without producing a deterministic curriculum.
        ordered = sorted(indices, key=lambda index: self.lengths[index])
        bucket = max(self.batch_size * 16, 1)
        groups = [ordered[start : start + bucket] for start in range(0, len(ordered), bucket)]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)
        return [item for group in groups for item in group]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self.balanced and len(self.active_formats) == 2:
            per_format = self.batch_size // 2
            queues = {fmt: self._ordered(self.by_format[fmt], rng) for fmt in self.active_formats}
            for batch_index in range(len(self)):
                batch: list[int] = []
                for fmt in self.active_formats:
                    queue = queues[fmt]
                    start = batch_index * per_format
                    batch.extend(queue[(start + offset) % len(queue)] for offset in range(per_format))
                if self.shuffle:
                    rng.shuffle(batch)
                yield batch
            return
        all_indices = self._ordered([index for values in self.by_format.values() for index in values], rng)
        for start in range(0, len(all_indices), self.batch_size):
            yield all_indices[start : start + self.batch_size]


@dataclass(frozen=True)
class TransitionRef:
    shard_index: int
    row: int
    rollout_step: int
    fmt: str
    length: int


class LatentTransitionDataset(Dataset[dict[str, Any]]):
    """M/C data from sidecars; constructs only the chosen p1 *or* p2 POV."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str,
        max_context_transitions: int,
        format_id_map: Mapping[int, str] | None = None,
    ):
        root = Path(cache_root) / split
        self.paths = [str(path) for path in sorted(root.glob("paired_shard_*.npz"))]
        if not self.paths:
            raise FileNotFoundError(f"No cached latent sidecars under {root}")
        self.max_context_transitions = int(max_context_transitions)
        self.format_id_map = {int(key): str(value) for key, value in (format_id_map or {}).items()}
        self.refs: list[TransitionRef] = []
        for shard_index, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as data:
                state_idx = _normalise_matrix(data["p1_state_idx"])
                format_ids = _normalise_matrix(data["format_id"])
                battle_ids = np.asarray(data["battle_id"])
                battle_starts = np.asarray(data["p1_battle_start"])
                for row in range(state_idx.shape[0]):
                    for step in range(state_idx.shape[1]):
                        fmt_id = int(format_ids[row, step]) if row < len(format_ids) else -1
                        fmt = self.format_id_map.get(fmt_id, str(fmt_id))
                        battle_start = int(battle_starts[int(battle_ids[row])])
                        self.refs.append(
                            TransitionRef(
                                shard_index, row, step, fmt,
                                min(self.max_context_transitions + 1, int(state_idx[row, step]) - battle_start),
                            )
                        )
        self._handles = _NpzHandleCache(self.paths)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.refs)

    @staticmethod
    def _side_keys(side: str) -> tuple[str, str]:
        return (f"{side}_action_ids", f"{side}_opponent_action_ids")

    def _sample(self, ref: TransitionRef, *, side: str) -> dict[str, Any]:
        data = self._handles.get(ref.shard_index)
        prefix = side
        state_idx = _normalise_matrix(np.asarray(data[f"{prefix}_state_idx"]))
        next_state_idx = _normalise_matrix(np.asarray(data[f"{prefix}_next_state_idx"]))
        action_idx = _normalise_matrix(np.asarray(data[f"{prefix}_action_idx"]))
        row, step = ref.row, ref.rollout_step
        battle_id = int(np.asarray(data["battle_id"])[row])
        state_start = int(np.asarray(data[f"{prefix}_battle_start"])[battle_id])
        state_end = int(np.asarray(data[f"{prefix}_battle_start"])[battle_id + 1])
        action_start = int(np.asarray(data[f"{prefix}_battle_action_start"])[battle_id])
        action_end = int(np.asarray(data[f"{prefix}_battle_action_start"])[battle_id + 1])
        target_state = int(state_idx[row, step])
        first_state = max(state_start + 1, target_state - self.max_context_transitions)
        state_indices = np.arange(first_state, target_state + 1, dtype=np.int64)
        first_action = action_start + (first_state - (state_start + 1))
        last_action = action_start + (target_state - (state_start + 1))
        own_ids_name, opponent_ids_name = self._side_keys(prefix)
        own_hist = np.asarray(data[own_ids_name][first_action:last_action], dtype=np.int64)
        opponent_hist = np.asarray(data[opponent_ids_name][first_action:last_action], dtype=np.int64)
        if len(own_hist) != len(state_indices) - 1:
            raise ValueError("Cached action/state timeline is misaligned")
        action = int(action_idx[row, step])
        terminal_name = f"{prefix}_next_terminal_class"
        terminal = int(_normalise_matrix(np.asarray(data[terminal_name]))[row, step])
        legal_ids = np.asarray(data[f"{prefix}_legal_action_ids"][action], dtype=np.int64)
        legal_mask = np.asarray(data[f"{prefix}_legal_action_mask"][action], dtype=np.bool_)
        chosen = int(np.asarray(data[f"{prefix}_chosen_legal_action_idx"])[action])
        next_state = int(next_state_idx[row, step])
        future_steps = max(0, min(10, state_end - next_state, action_end - action))
        return {
            "team_mu": np.asarray(data[f"{prefix}_mu"][state_start], dtype=np.float32),
            "team_logvar": np.asarray(data[f"{prefix}_logvar"][state_start], dtype=np.float32),
            "state_mu": np.asarray(data[f"{prefix}_mu"][state_indices], dtype=np.float32),
            "state_logvar": np.asarray(data[f"{prefix}_logvar"][state_indices], dtype=np.float32),
            "own_history_action_ids": own_hist,
            "opponent_history_action_ids": opponent_hist,
            "current_own_action_id": int(np.asarray(data[own_ids_name])[action]),
            "current_opponent_action_id": int(np.asarray(data[opponent_ids_name])[action]),
            "next_mu": np.asarray(data[f"{prefix}_mu"][next_state], dtype=np.float32),
            "next_logvar": np.asarray(data[f"{prefix}_logvar"][next_state], dtype=np.float32),
            # Teacher-forced action labels and V means for rollout-drift
            # validation.  Training only consumes the first next target.
            "future_mu": np.asarray(data[f"{prefix}_mu"][next_state : next_state + future_steps], dtype=np.float32),
            "future_own_action_ids": np.asarray(data[own_ids_name][action : action + future_steps], dtype=np.int64),
            "future_opponent_action_ids": np.asarray(data[opponent_ids_name][action : action + future_steps], dtype=np.int64),
            "done": terminal != 0,
            "outcome": int(np.asarray(data[f"{prefix}_outcome"])[battle_id]),
            "legal_action_ids": legal_ids,
            "legal_action_mask": legal_mask,
            "chosen_legal_action_idx": min(max(chosen, 0), max(len(legal_ids) - 1, 0)),
            "fmt": ref.fmt,
            "side": side,
        }

    def sample_with_perspective(self, index: int, side: str) -> dict[str, Any]:
        if side not in {"p1", "p2"}:
            raise ValueError("side must be p1 or p2")
        return self._sample(self.refs[index], side=side)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Shuffled transition order plus alternating perspective yields exact
        # long-run p1/p2 symmetry while avoiding a second perspective tensor.
        side = "p1" if (index + self.epoch) % 2 == 0 else "p2"
        return self._sample(self.refs[index], side=side)


def collate_latent(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch = len(samples)
    max_states = max(len(sample["state_mu"]) for sample in samples)
    max_actions = max(max_states - 1, 0)
    latent_dim = int(samples[0]["team_mu"].shape[-1])
    max_legal = max(len(sample["legal_action_ids"]) for sample in samples)
    max_future = max((len(sample["future_mu"]) for sample in samples), default=1)
    out = {
        "team_mu": torch.empty(batch, latent_dim),
        "team_logvar": torch.empty(batch, latent_dim),
        "state_mu": torch.zeros(batch, max_states, latent_dim),
        "state_logvar": torch.zeros(batch, max_states, latent_dim),
        "state_mask": torch.zeros(batch, max_states, dtype=torch.bool),
        "own_history_action_ids": torch.zeros(batch, max_actions, dtype=torch.long),
        "opponent_history_action_ids": torch.zeros(batch, max_actions, dtype=torch.long),
        "current_own_action_ids": torch.empty(batch, dtype=torch.long),
        "current_opponent_action_ids": torch.empty(batch, dtype=torch.long),
        "next_mu": torch.empty(batch, latent_dim),
        "next_logvar": torch.empty(batch, latent_dim),
        "future_mu": torch.zeros(batch, max_future, latent_dim),
        "future_mask": torch.zeros(batch, max_future, dtype=torch.bool),
        "future_own_action_ids": torch.zeros(batch, max_future, dtype=torch.long),
        "future_opponent_action_ids": torch.zeros(batch, max_future, dtype=torch.long),
        "done": torch.empty(batch, dtype=torch.float),
        "outcome": torch.empty(batch, dtype=torch.long),
        "legal_action_ids": torch.zeros(batch, max_legal, dtype=torch.long),
        "legal_action_mask": torch.zeros(batch, max_legal, dtype=torch.bool),
        "chosen_legal_action_idx": torch.empty(batch, dtype=torch.long),
        "formats": [str(sample["fmt"]) for sample in samples],
        "sides": [str(sample["side"]) for sample in samples],
    }
    for row, sample in enumerate(samples):
        n_states = len(sample["state_mu"])
        n_actions = n_states - 1
        n_legal = len(sample["legal_action_ids"])
        n_future = len(sample["future_mu"])
        out["team_mu"][row] = torch.as_tensor(sample["team_mu"])
        out["team_logvar"][row] = torch.as_tensor(sample["team_logvar"])
        out["state_mu"][row, :n_states] = torch.as_tensor(sample["state_mu"])
        out["state_logvar"][row, :n_states] = torch.as_tensor(sample["state_logvar"])
        out["state_mask"][row, :n_states] = True
        if n_actions:
            out["own_history_action_ids"][row, :n_actions] = torch.as_tensor(sample["own_history_action_ids"])
            out["opponent_history_action_ids"][row, :n_actions] = torch.as_tensor(sample["opponent_history_action_ids"])
        out["current_own_action_ids"][row] = int(sample["current_own_action_id"])
        out["current_opponent_action_ids"][row] = int(sample["current_opponent_action_id"])
        out["next_mu"][row] = torch.as_tensor(sample["next_mu"])
        out["next_logvar"][row] = torch.as_tensor(sample["next_logvar"])
        if n_future:
            out["future_mu"][row, :n_future] = torch.as_tensor(sample["future_mu"])
            out["future_mask"][row, :n_future] = True
            out["future_own_action_ids"][row, :n_future] = torch.as_tensor(sample["future_own_action_ids"])
            out["future_opponent_action_ids"][row, :n_future] = torch.as_tensor(sample["future_opponent_action_ids"])
        out["done"][row] = float(sample["done"])
        out["outcome"][row] = int(sample["outcome"])
        out["legal_action_ids"][row, :n_legal] = torch.as_tensor(sample["legal_action_ids"])
        out["legal_action_mask"][row, :n_legal] = torch.as_tensor(sample["legal_action_mask"])
        out["chosen_legal_action_idx"][row] = int(sample["chosen_legal_action_idx"])
    return out


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
