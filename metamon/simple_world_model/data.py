"""Data, cache-manifest, and batching helpers for the V/M/C pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import OrderedDict
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


def _shard_formats(data: Mapping[str, np.ndarray] | Any, id_map: Mapping[int, str]) -> list[str]:
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


def _battle_format_names(
    data: Mapping[str, np.ndarray] | Any,
    id_map: Mapping[int, str],
    *,
    num_battles: int,
) -> np.ndarray:
    """Return one format label per local battle without touching state payloads."""
    fallback = _shard_formats(data, id_map)
    default = fallback[0] if len(fallback) == 1 else "unknown"
    names = np.full(int(num_battles), default, dtype=object)
    if "battle_id" not in data or "format_id" not in data:
        return names
    battle_ids = np.asarray(data["battle_id"]).reshape(-1)
    format_ids = _normalise_matrix(np.asarray(data["format_id"])).reshape(-1)
    count = min(len(battle_ids), len(format_ids))
    if not count:
        return names
    battle_ids = battle_ids[:count].astype(np.int64, copy=False)
    format_ids = format_ids[:count].astype(np.int64, copy=False)
    valid = (battle_ids >= 0) & (battle_ids < len(names))
    for format_id in np.unique(format_ids[valid]):
        names[battle_ids[valid & (format_ids == format_id)]] = id_map.get(int(format_id), str(int(format_id)))
    return names


def _slice_block(data: Mapping[str, np.ndarray], side: str, index: int, kind: str = "state") -> np.ndarray:
    singular = "state" if kind == "state" else "action"
    flat_name = f"{side}_{kind}s"
    offsets = np.asarray(data[f"{side}_{singular}_offsets"])
    lengths = np.asarray(data[f"{side}_{singular}_lengths"])
    start = int(offsets[index])
    return np.asarray(data[flat_name][start : start + int(lengths[index])])


def _canonical_action_from_ids(
    token_ids: np.ndarray,
    *,
    tokenizer: object,
    pad_id: int,
    cache: dict[tuple[int, ...], str],
) -> str:
    ids = tuple(int(token) for token in np.asarray(token_ids).reshape(-1) if int(token) != pad_id)
    action = cache.get(ids)
    if action is None:
        action = canonicalize_action_ids(np.asarray(ids, dtype=np.int64), tokenizer=tokenizer, pad_id=pad_id)
        cache[ids] = action
    return action


def _record_action_blocks(
    blocks: np.ndarray,
    *,
    fmt: str,
    tokenizer: object,
    pad_id: int,
    canonical_cache: dict[tuple[int, ...], str],
    observed: dict[str, set[str]],
) -> None:
    """Canonicalize only distinct action token rows from one format/shard."""
    rows = np.asarray(blocks)
    if rows.size == 0:
        return
    if rows.ndim == 1:
        rows = rows[:, None]
    for row in np.unique(rows, axis=0):
        action = _canonical_action_from_ids(
            row, tokenizer=tokenizer, pad_id=pad_id, cache=canonical_cache,
        )
        observed.setdefault(action, set()).add(str(fmt))


def _record_flat_actions(
    flat: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    action_formats: np.ndarray,
    *,
    allowed_formats: set[str] | None,
    tokenizer: object,
    pad_id: int,
    canonical_cache: dict[tuple[int, ...], str],
    observed: dict[str, set[str]],
) -> None:
    """Vectorized action-block scan; avoids a Python loop over 70M clicks."""
    flat = np.asarray(flat)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    limit = min(len(offsets), len(lengths), len(action_formats))
    if not limit:
        return
    offsets, lengths, action_formats = offsets[:limit], lengths[:limit], action_formats[:limit]
    for fmt in sorted(set(str(value) for value in action_formats)):
        if allowed_formats is not None and fmt not in allowed_formats:
            continue
        format_indices = np.flatnonzero(action_formats == fmt)
        for length in np.unique(lengths[format_indices]):
            if length <= 0:
                _record_action_blocks(
                    np.zeros((1, 0), dtype=flat.dtype), fmt=fmt, tokenizer=tokenizer, pad_id=pad_id,
                    canonical_cache=canonical_cache, observed=observed,
                )
                continue
            indices = format_indices[lengths[format_indices] == length]
            starts = offsets[indices]
            valid = (starts >= 0) & (starts + int(length) <= len(flat))
            if not np.any(valid):
                continue
            starts = starts[valid]
            rows = flat[starts[:, None] + np.arange(int(length), dtype=np.int64)]
            _record_action_blocks(
                rows, fmt=fmt, tokenizer=tokenizer, pad_id=pad_id,
                canonical_cache=canonical_cache, observed=observed,
            )


def _record_legal_actions(
    raw: np.ndarray,
    mask: np.ndarray,
    action_formats: np.ndarray,
    *,
    allowed_formats: set[str] | None,
    tokenizer: object,
    pad_id: int,
    canonical_cache: dict[tuple[int, ...], str],
    observed: dict[str, set[str]],
) -> None:
    raw = np.asarray(raw)
    mask = np.asarray(mask, dtype=np.bool_)
    rows = min(len(raw), len(mask), len(action_formats))
    for fmt in sorted(set(str(value) for value in action_formats[:rows])):
        if allowed_formats is not None and fmt not in allowed_formats:
            continue
        selected = np.flatnonzero(action_formats[:rows] == fmt)
        if not len(selected):
            continue
        candidates = raw[selected]
        candidate_mask = mask[selected]
        if not np.any(candidate_mask):
            continue
        _record_action_blocks(
            candidates[candidate_mask], fmt=fmt, tokenizer=tokenizer, pad_id=pad_id,
            canonical_cache=canonical_cache, observed=observed,
        )


def build_action_vocabulary(
    data_root: str | Path,
    *,
    tokenizer: object,
    pad_id: int,
    formats: Sequence[str] | None = None,
) -> ActionVocabulary:
    """Build from every observed and legal candidate action, not clicks alone."""
    id_map = _format_id_map(data_root)
    requested = {str(fmt) for fmt in formats} if formats else None
    paths = [
        (split, path)
        for split in ("train", "val")
        for path in discover_source_shards(data_root, split, formats)
    ]
    print(f"[action-vocab] scanning {len(paths)} paired shards for observed and legal actions...", flush=True)
    observed: dict[str, set[str]] = {}
    canonical_cache: dict[tuple[int, ...], str] = {}
    for shard_number, (_, path) in enumerate(paths, start=1):
        if shard_number == 1 or shard_number % 100 == 0 or shard_number == len(paths):
            print(
                f"[action-vocab] {shard_number}/{len(paths)} shards; unique actions={len(observed)}",
                flush=True,
            )
        with np.load(path, allow_pickle=False) as source:
            # Only compact action/index arrays are read here.  In particular,
            # never materialize p1/p2_states while building the vocabulary.
            starts = np.asarray(source["p1_battle_action_start"])
            battle_formats = _battle_format_names(source, id_map, num_battles=max(len(starts) - 1, 0))
            for side in ("p1", "p2"):
                side_starts = np.asarray(source[f"{side}_battle_action_start"])
                counts = np.diff(side_starts).astype(np.int64, copy=False)
                for suffix in ("actions", "opponent_actions"):
                    stem = suffix[:-1]
                    offsets = np.asarray(source[f"{side}_{stem}_offsets"])
                    lengths = np.asarray(source[f"{side}_{stem}_lengths"])
                    action_count = len(offsets)
                    action_formats = np.repeat(battle_formats[:len(counts)], np.maximum(counts, 0))
                    if len(action_formats) < action_count:
                        fallback = _shard_formats(source, id_map)[0]
                        action_formats = np.concatenate((
                            action_formats,
                            np.full(action_count - len(action_formats), fallback, dtype=object),
                        ))
                    action_formats = action_formats[:action_count]
                    _record_flat_actions(
                        source[f"{side}_{suffix}"], offsets, lengths, action_formats,
                        allowed_formats=requested, tokenizer=tokenizer, pad_id=pad_id,
                        canonical_cache=canonical_cache, observed=observed,
                    )
                legal_key = f"{side}_legal_actions"
                if legal_key in source.files:
                    action_count = len(np.asarray(source[f"{side}_action_offsets"]))
                    action_formats = np.repeat(battle_formats[:len(counts)], np.maximum(counts, 0))
                    if len(action_formats) < action_count:
                        fallback = _shard_formats(source, id_map)[0]
                        action_formats = np.concatenate((
                            action_formats,
                            np.full(action_count - len(action_formats), fallback, dtype=object),
                        ))
                    action_formats = action_formats[:action_count]
                    raw = np.asarray(source[legal_key])
                    mask_key = f"{side}_legal_action_mask"
                    mask = np.asarray(
                        source[mask_key] if mask_key in source.files else np.any(raw != pad_id, axis=-1),
                        dtype=np.bool_,
                    )
                    _record_legal_actions(
                        raw, mask, action_formats, allowed_formats=requested,
                        tokenizer=tokenizer, pad_id=pad_id,
                        canonical_cache=canonical_cache, observed=observed,
                    )
    vocabulary = ActionVocabulary()
    for action in sorted(observed):
        for fmt in sorted(observed[action]):
            vocabulary.add(action, fmt=fmt)
    print(f"[action-vocab] complete: {len(vocabulary)} canonical actions", flush=True)
    return vocabulary


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


@dataclass(frozen=True)
class _VShardPopulation:
    """All non-header states for one shard/POV/format, kept compactly."""

    shard_index: int
    side: str
    fmt: str
    battle_indices: np.ndarray
    state_prefix: np.ndarray

    @property
    def total_states(self) -> int:
        return int(self.state_prefix[-1]) if len(self.state_prefix) else 0


@dataclass
class _VShardInfo:
    p1_starts: np.ndarray
    p2_starts: np.ndarray
    p1_lengths: np.ndarray
    p2_lengths: np.ndarray


class _CachedNpz:
    """Small per-shard ndarray cache around an NPZ handle.

    ``NpzFile`` does not cache members itself.  Without this wrapper, slicing
    one state repeatedly decompresses/copies the whole flat state array for
    every sample.  Keeping only a few active shards is both faster and safely
    below the production file-descriptor limit.
    """

    def __init__(self, path: str):
        self.source = np.load(path, allow_pickle=False)
        self.arrays: dict[str, np.ndarray] = {}

    @property
    def files(self) -> list[str]:
        return self.source.files

    def __contains__(self, key: object) -> bool:
        return str(key) in self.source.files

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.arrays:
            self.arrays[key] = np.asarray(self.source[key])
        return self.arrays[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default

    def close(self) -> None:
        self.arrays.clear()
        self.source.close()


class _NpzHandleCache:
    """Lazy per-process npz handles; avoids eagerly constructing p2 tensors."""

    def __init__(self, paths: Sequence[str], *, max_open: int = 4):
        self.paths = list(paths)
        self.max_open = max(1, int(max_open))
        self.handles: OrderedDict[int, _CachedNpz] = OrderedDict()

    def get(self, index: int) -> _CachedNpz:
        handle = self.handles.pop(index, None)
        if handle is None:
            handle = _CachedNpz(self.paths[index])
        self.handles[index] = handle
        while len(self.handles) > self.max_open:
            _, stale = self.handles.popitem(last=False)
            stale.close()
        return handle

    def close(self) -> None:
        while self.handles:
            _, handle = self.handles.popitem(last=False)
            handle.close()

    def __getstate__(self) -> dict[str, Any]:
        return {"paths": self.paths, "max_open": self.max_open, "handles": OrderedDict()}

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown timing
        try:
            self.close()
        except Exception:
            pass


def _format_quotas(counts: Mapping[str, int], max_samples: int) -> dict[str, int]:
    active = [fmt for fmt in sorted(counts) if counts[fmt] > 0]
    if not active:
        return {}
    if max_samples <= 0:
        return {fmt: int(counts[fmt]) for fmt in active}
    if len(active) == 2:
        per_format = max_samples // 2
        quotas = {fmt: min(int(counts[fmt]), per_format) for fmt in active}
        # Preserve a useful validation size if a tiny fixture has only one
        # format/battle available.
        remainder = max_samples - sum(quotas.values())
        for fmt in active:
            if remainder <= 0:
                break
            available = int(counts[fmt]) - quotas[fmt]
            take = min(available, remainder)
            quotas[fmt] += take
            remainder -= take
        return quotas
    total = sum(int(counts[fmt]) for fmt in active)
    quotas = {fmt: min(int(counts[fmt]), max(1, int(max_samples * counts[fmt] / max(total, 1)))) for fmt in active}
    while sum(quotas.values()) < min(max_samples, total):
        for fmt in active:
            if quotas[fmt] < counts[fmt]:
                quotas[fmt] += 1
                if sum(quotas.values()) >= min(max_samples, total):
                    break
    return quotas


class FixedRefDataset(Dataset[dict[str, Any]]):
    """A tiny fixed validation view over a compact training dataset."""

    def __init__(self, dataset: Dataset[dict[str, Any]], refs: Sequence[Any]):
        self.dataset = dataset
        self.refs = list(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.refs[index]]

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)  # type: ignore[attr-defined]


class VStateDataset(Dataset[dict[str, Any]]):
    """One header/current-visible-state V item, indexed by compact shard maps.

    The production corpus has roughly 76M state blocks.  A Python ``Ref`` per
    block costs many GiB and makes the launcher look frozen.  This class keeps
    only battle boundaries plus int32 state lengths (~300 MiB) and samples
    exact state locations on demand.
    """

    def __init__(
        self, shard_paths: Sequence[str], *, data_root: str | Path,
        max_state_tokens: int | None = None,
        formats: Sequence[str] | None = None,
    ):
        if not shard_paths:
            raise ValueError("No paired shards for V stage")
        self.paths = list(shard_paths)
        self.max_state_tokens = None if max_state_tokens is None else int(max_state_tokens)
        self._infos: list[_VShardInfo] = []
        self._populations_by_format_side: dict[tuple[str, str], list[_VShardPopulation]] = {}
        self._populations_by_format: dict[str, list[_VShardPopulation]] = {}
        self._population_prefix_by_format_side: dict[tuple[str, str], np.ndarray] = {}
        self._population_prefix_by_format: dict[str, np.ndarray] = {}
        self._battle_populations_by_format: dict[str, list[_VShardPopulation]] = {}
        self._battle_prefix_by_format: dict[str, np.ndarray] = {}
        self.total_by_format: dict[str, int] = {}
        self._all_populations: list[_VShardPopulation] = []
        self._all_prefix = np.empty(0, dtype=np.int64)
        requested = {str(fmt) for fmt in formats} if formats else None
        id_map = _format_id_map(data_root)
        for shard_index, path in enumerate(self.paths):
            if shard_index == 0 or (shard_index + 1) % 100 == 0 or shard_index + 1 == len(self.paths):
                print(f"[v-loader] indexing shard {shard_index + 1}/{len(self.paths)}", flush=True)
            with np.load(path, allow_pickle=False) as source:
                p1_starts = np.asarray(source["p1_battle_start"], dtype=np.int64)
                p2_starts = np.asarray(source["p2_battle_start"], dtype=np.int64)
                p1_lengths = np.asarray(source["p1_state_lengths"], dtype=np.int32)
                p2_lengths = np.asarray(source["p2_state_lengths"], dtype=np.int32)
                self._infos.append(_VShardInfo(p1_starts, p2_starts, p1_lengths, p2_lengths))
                num_battles = min(len(p1_starts), len(p2_starts)) - 1
                battle_formats = _battle_format_names(source, id_map, num_battles=max(num_battles, 0))
                for side in ("p1", "p2"):
                    starts = p1_starts if side == "p1" else p2_starts
                    counts = np.maximum(np.diff(starts[: num_battles + 1]) - 1, 0).astype(np.int64, copy=False)
                    for fmt in sorted(set(str(value) for value in battle_formats)):
                        if requested is not None and fmt not in requested:
                            continue
                        battle_indices = np.flatnonzero((battle_formats == fmt) & (counts > 0)).astype(np.int32, copy=False)
                        if not len(battle_indices):
                            continue
                        population = _VShardPopulation(
                            shard_index=shard_index,
                            side=side,
                            fmt=fmt,
                            battle_indices=battle_indices,
                            state_prefix=np.cumsum(counts[battle_indices], dtype=np.int64),
                        )
                        self._populations_by_format_side.setdefault((fmt, side), []).append(population)
                        self._populations_by_format.setdefault(fmt, []).append(population)
                        if side == "p1":
                            self._battle_populations_by_format.setdefault(fmt, []).append(population)
        for key, populations in self._populations_by_format_side.items():
            self._population_prefix_by_format_side[key] = np.cumsum(
                [population.total_states for population in populations], dtype=np.int64
            )
        for fmt, populations in self._populations_by_format.items():
            self._population_prefix_by_format[fmt] = np.cumsum(
                [population.total_states for population in populations], dtype=np.int64
            )
            self.total_by_format[fmt] = int(self._population_prefix_by_format[fmt][-1])
        for fmt, populations in self._battle_populations_by_format.items():
            self._battle_prefix_by_format[fmt] = np.cumsum(
                [len(population.battle_indices) for population in populations], dtype=np.int64
            )
        self._all_populations = [population for populations in self._populations_by_format.values() for population in populations]
        self._all_prefix = np.cumsum([population.total_states for population in self._all_populations], dtype=np.int64)
        if not len(self._all_prefix):
            requested_text = ", ".join(sorted(requested or ())) or "all formats"
            raise ValueError(f"No V state samples found for {requested_text}")
        print(
            f"[v-loader] indexed {len(self.paths)} shards, {len(self):,} current-state samples "
            f"({', '.join(f'{fmt}={count:,}' for fmt, count in sorted(self.total_by_format.items()))})",
            flush=True,
        )
        self._handles = _NpzHandleCache(self.paths)

    def __len__(self) -> int:
        return int(self._all_prefix[-1])

    @property
    def formats(self) -> list[str]:
        return sorted(self.total_by_format)

    def _ref_from_population(self, population: _VShardPopulation, rank: int) -> VStateRef:
        battle_position = int(np.searchsorted(population.state_prefix, int(rank), side="right"))
        before = int(population.state_prefix[battle_position - 1]) if battle_position else 0
        battle_id = int(population.battle_indices[battle_position])
        info = self._infos[population.shard_index]
        starts = info.p1_starts if population.side == "p1" else info.p2_starts
        lengths = info.p1_lengths if population.side == "p1" else info.p2_lengths
        header = int(starts[battle_id])
        state_index = header + 1 + (int(rank) - before)
        length = int(lengths[state_index])
        if self.max_state_tokens is not None:
            length = min(length, self.max_state_tokens)
        return VStateRef(population.shard_index, population.side, state_index, header, battle_id, population.fmt, length)

    @staticmethod
    def _choose_population(
        populations: Sequence[_VShardPopulation], prefix: np.ndarray, rng: random.Random,
    ) -> _VShardPopulation:
        target = rng.randrange(int(prefix[-1]))
        population_index = int(np.searchsorted(prefix, target, side="right"))
        return populations[population_index]

    def draw_population(self, fmt: str, rng: random.Random, *, side: str | None = None) -> _VShardPopulation:
        if side in {"p1", "p2"} and (fmt, side) in self._populations_by_format_side:
            key = (fmt, side)
            return self._choose_population(
                self._populations_by_format_side[key], self._population_prefix_by_format_side[key], rng,
            )
        return self._choose_population(self._populations_by_format[fmt], self._population_prefix_by_format[fmt], rng)

    def draw_ref_from_population(
        self, population: _VShardPopulation, rng: random.Random, *, side: str | None = None,
    ) -> VStateRef:
        del side
        return self._ref_from_population(population, rng.randrange(population.total_states))

    def draw_ref(self, fmt: str, rng: random.Random, *, side: str | None = None) -> VStateRef:
        return self.draw_ref_from_population(self.draw_population(fmt, rng, side=side), rng, side=side)

    def _ref_from_global(self, index: int, *, side: str | None = None) -> VStateRef:
        target = int(index) % len(self)
        population_index = int(np.searchsorted(self._all_prefix, target, side="right"))
        before = int(self._all_prefix[population_index - 1]) if population_index else 0
        ref = self._ref_from_population(self._all_populations[population_index], target - before)
        if side is None or side == ref.side:
            return ref
        # Direct integer access is only used in tests/debugging.  Preserve the
        # requested POV where that shard has the same format population.
        return self.draw_ref(ref.fmt, random.Random(index), side=side)

    def fixed_subset(self, max_samples: int) -> FixedRefDataset:
        quotas = _format_quotas(
            {fmt: int(prefix[-1]) for fmt, prefix in self._battle_prefix_by_format.items()}, max_samples,
        )
        refs: list[VStateRef] = []
        for fmt, quota in quotas.items():
            populations = self._battle_populations_by_format[fmt]
            prefix = self._battle_prefix_by_format[fmt]
            total_battles = int(prefix[-1])
            for offset in range(quota):
                rank = min((offset * total_battles) // max(quota, 1), total_battles - 1)
                population_index = int(np.searchsorted(prefix, rank, side="right"))
                before = int(prefix[population_index - 1]) if population_index else 0
                population = populations[population_index]
                battle_position = rank - before
                battle_id = int(population.battle_indices[battle_position])
                side = "p1" if offset % 2 == 0 else "p2"
                info = self._infos[population.shard_index]
                starts = info.p1_starts if side == "p1" else info.p2_starts
                lengths = info.p1_lengths if side == "p1" else info.p2_lengths
                header = int(starts[battle_id])
                state_count = max(int(starts[battle_id + 1]) - header - 1, 1)
                # Stable hash-like offset, one state from each distinct raw
                # battle while retaining both POVs across the full set.
                state_index = header + 1 + ((offset * 1103515245 + population.shard_index) % state_count)
                length = int(lengths[state_index])
                if self.max_state_tokens is not None:
                    length = min(length, self.max_state_tokens)
                refs.append(VStateRef(population.shard_index, side, state_index, header, battle_id, fmt, length))
        return FixedRefDataset(self, refs)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Spawn-based DataLoaders (macOS) only need ``paths`` and the incoming
        # concrete refs; retaining the 300 MiB training index per worker is
        # unnecessary.
        state["_infos"] = []
        state["_populations_by_format_side"] = {}
        state["_populations_by_format"] = {}
        state["_population_prefix_by_format_side"] = {}
        state["_population_prefix_by_format"] = {}
        state["_battle_populations_by_format"] = {}
        state["_battle_prefix_by_format"] = {}
        state["_all_populations"] = []
        state["_all_prefix"] = np.empty(0, dtype=np.int64)
        state["_handles"] = _NpzHandleCache(self.paths)
        return state

    def __getitem__(self, index: int | VStateRef) -> dict[str, Any]:
        ref = index if isinstance(index, VStateRef) else self._ref_from_global(int(index))
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


class CompactFormatBatchSampler(Sampler[list[Any]]):
    """Streaming, length-bucketed sampler for datasets too large for refs.

    It samples concrete shard references with replacement, so a 50K-update
    pilot does not first allocate or shuffle tens of millions of Python
    objects.  Each local pool is sorted by token/context length before being
    split into batches, retaining the padding benefit of conventional length
    bucketing.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        balanced: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        pool_batches: int = 32,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.balanced = bool(balanced)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.pool_batches = max(1, int(pool_batches))
        self.epoch = 0
        self.active_formats = [
            fmt for fmt in sorted(dataset.total_by_format)
            if int(dataset.total_by_format[fmt]) > 0
        ]
        if self.balanced and len(self.active_formats) == 2 and self.batch_size % 2:
            raise ValueError("balanced Gen1/Gen9 batches require an even --batch_size")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.balanced and len(self.active_formats) == 2:
            half = self.batch_size // 2
            return max(math.ceil(int(self.dataset.total_by_format[fmt]) / half) for fmt in self.active_formats)
        return math.ceil(sum(int(self.dataset.total_by_format[fmt]) for fmt in self.active_formats) / self.batch_size)

    @staticmethod
    def _chunks(values: list[Any], width: int) -> list[list[Any]]:
        return [values[index : index + width] for index in range(0, len(values), width) if len(values[index : index + width]) == width]

    def _stream(self, fmt: str, width: int, rng: random.Random, side_counter: list[int]) -> Iterator[list[Any]]:
        while True:
            pool_size = width * self.pool_batches
            refs: list[Any] = []
            # Keep each local pool on at most one active shard per POV.  NPZ
            # members are whole arrays, so this locality is what makes the
            # bounded worker cache effective instead of reloading a multi-MiB
            # state/latent member for every individual example.
            populations: dict[str, Any] = {}
            for _ in range(pool_size):
                side = "p1" if side_counter[0] % 2 == 0 else "p2"
                side_counter[0] += 1
                if side not in populations:
                    populations[side] = self.dataset.draw_population(fmt, rng, side=side)
                refs.append(self.dataset.draw_ref_from_population(populations[side], rng, side=side))
            refs.sort(key=lambda ref: int(ref.length))
            chunks = self._chunks(refs, width)
            if self.shuffle:
                rng.shuffle(chunks)
                for chunk in chunks:
                    rng.shuffle(chunk)
            yield from chunks

    def __iter__(self) -> Iterator[list[Any]]:
        if not self.active_formats:
            return
        rng = random.Random(self.seed + self.epoch)
        side_counter = [0]
        if self.balanced and len(self.active_formats) == 2:
            width = self.batch_size // 2
            streams = {fmt: self._stream(fmt, width, rng, side_counter) for fmt in self.active_formats}
            for _ in range(len(self)):
                batch = [ref for fmt in self.active_formats for ref in next(streams[fmt])]
                if self.shuffle:
                    rng.shuffle(batch)
                yield batch
            return
        # Single-format and compatibility paths still receive length buckets;
        # with more than two formats, sample in corpus-proportionate order.
        total = sum(int(self.dataset.total_by_format[fmt]) for fmt in self.active_formats)
        weighted = [
            (fmt, int(self.dataset.total_by_format[fmt]))
            for fmt in self.active_formats
        ]
        for _ in range(len(self)):
            refs: list[Any] = []
            populations: dict[tuple[str, str], Any] = {}
            for _ in range(self.batch_size):
                choice = rng.randrange(max(total, 1))
                cumulative = 0
                fmt = self.active_formats[-1]
                for candidate, weight in weighted:
                    cumulative += weight
                    if choice < cumulative:
                        fmt = candidate
                        break
                side = "p1" if side_counter[0] % 2 == 0 else "p2"
                side_counter[0] += 1
                key = (fmt, side)
                if key not in populations:
                    populations[key] = self.dataset.draw_population(fmt, rng, side=side)
                refs.append(self.dataset.draw_ref_from_population(populations[key], rng, side=side))
            refs.sort(key=lambda ref: int(ref.length))
            if self.shuffle:
                rng.shuffle(refs)
            yield refs


@dataclass(frozen=True)
class TransitionRef:
    shard_index: int
    row: int
    rollout_step: int
    fmt: str
    length: int
    side: str | None = None


@dataclass(frozen=True)
class _TransitionShardPopulation:
    shard_index: int
    fmt: str
    flat_indices: np.ndarray
    fixed_flat_indices: np.ndarray
    rollout_steps: int

    @property
    def total(self) -> int:
        return len(self.flat_indices)

    @property
    def fixed_total(self) -> int:
        return len(self.fixed_flat_indices)


class LatentTransitionDataset(Dataset[dict[str, Any]]):
    """M/C data from sidecars; constructs only the chosen p1 *or* p2 POV."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str,
        max_context_transitions: int,
        format_id_map: Mapping[int, str] | None = None,
        formats: Sequence[str] | None = None,
    ):
        root = Path(cache_root) / split
        self.paths = [str(path) for path in sorted(root.glob("paired_shard_*.npz"))]
        if not self.paths:
            raise FileNotFoundError(f"No cached latent sidecars under {root}")
        self.max_context_transitions = int(max_context_transitions)
        self.format_id_map = {int(key): str(value) for key, value in (format_id_map or {}).items()}
        requested = {str(fmt) for fmt in formats} if formats else None
        self._populations_by_format: dict[str, list[_TransitionShardPopulation]] = {}
        self._population_prefix_by_format: dict[str, np.ndarray] = {}
        self._fixed_prefix_by_format: dict[str, np.ndarray] = {}
        self.total_by_format: dict[str, int] = {}
        self._all_populations: list[_TransitionShardPopulation] = []
        self._all_prefix = np.empty(0, dtype=np.int64)
        for shard_index, path in enumerate(self.paths):
            if shard_index == 0 or (shard_index + 1) % 100 == 0 or shard_index + 1 == len(self.paths):
                print(f"[latent-loader] indexing sidecar {shard_index + 1}/{len(self.paths)}", flush=True)
            with np.load(path, allow_pickle=False) as data:
                state_idx = _normalise_matrix(np.asarray(data["p1_state_idx"]))
                rows, rollout_steps = state_idx.shape
                raw_format_ids = _normalise_matrix(np.asarray(data["format_id"]))
                format_ids = np.full((rows, rollout_steps), -1, dtype=np.int64)
                copy_rows = min(rows, raw_format_ids.shape[0])
                copy_steps = min(rollout_steps, raw_format_ids.shape[1])
                format_ids[:copy_rows, :copy_steps] = raw_format_ids[:copy_rows, :copy_steps]
                flat_format_ids = format_ids.reshape(-1)
                battle_ids = np.asarray(data["battle_id"], dtype=np.int64).reshape(-1)[:rows]
                if len(battle_ids):
                    _, first_rows = np.unique(battle_ids, return_index=True)
                    first_flat = (first_rows.astype(np.int64) * rollout_steps).astype(np.int64, copy=False)
                else:
                    first_flat = np.empty(0, dtype=np.int64)
                for raw_format_id in np.unique(flat_format_ids):
                    fmt = self.format_id_map.get(int(raw_format_id), str(int(raw_format_id)))
                    if requested is not None and fmt not in requested:
                        continue
                    flat_indices = np.flatnonzero(flat_format_ids == raw_format_id).astype(np.int32, copy=False)
                    if not len(flat_indices):
                        continue
                    fixed = first_flat[flat_format_ids[first_flat] == raw_format_id].astype(np.int32, copy=False)
                    population = _TransitionShardPopulation(
                        shard_index, fmt, flat_indices, fixed, int(rollout_steps),
                    )
                    self._populations_by_format.setdefault(fmt, []).append(population)
        for fmt, populations in self._populations_by_format.items():
            prefix = np.cumsum([population.total for population in populations], dtype=np.int64)
            self._population_prefix_by_format[fmt] = prefix
            self._fixed_prefix_by_format[fmt] = np.cumsum(
                [population.fixed_total for population in populations], dtype=np.int64
            )
            self.total_by_format[fmt] = int(prefix[-1])
        self._all_populations = [population for populations in self._populations_by_format.values() for population in populations]
        self._all_prefix = np.cumsum([population.total for population in self._all_populations], dtype=np.int64)
        if not len(self._all_prefix):
            requested_text = ", ".join(sorted(requested or ())) or "all formats"
            raise ValueError(f"No cached latent transitions found for {requested_text}")
        print(
            f"[latent-loader] indexed {len(self.paths)} sidecars, {len(self):,} transitions "
            f"({', '.join(f'{fmt}={count:,}' for fmt, count in sorted(self.total_by_format.items()))})",
            flush=True,
        )
        self._handles = _NpzHandleCache(self.paths)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self._all_prefix[-1])

    @property
    def formats(self) -> list[str]:
        return sorted(self.total_by_format)

    def _ref_from_population(
        self,
        population: _TransitionShardPopulation,
        local_index: int,
        *,
        side: str | None,
    ) -> TransitionRef:
        flat = int(population.flat_indices[int(local_index)])
        return TransitionRef(
            population.shard_index,
            flat // population.rollout_steps,
            flat % population.rollout_steps,
            population.fmt,
            self.max_context_transitions + 1,
            side,
        )

    @staticmethod
    def _choose_population(
        populations: Sequence[_TransitionShardPopulation],
        prefix: np.ndarray,
        rng: random.Random,
    ) -> _TransitionShardPopulation:
        target = rng.randrange(int(prefix[-1]))
        population_index = int(np.searchsorted(prefix, target, side="right"))
        return populations[population_index]

    def draw_population(
        self, fmt: str, rng: random.Random, *, side: str | None = None,
    ) -> _TransitionShardPopulation:
        del side
        return self._choose_population(
            self._populations_by_format[fmt], self._population_prefix_by_format[fmt], rng,
        )

    def draw_ref_from_population(
        self, population: _TransitionShardPopulation, rng: random.Random, *, side: str | None = None,
    ) -> TransitionRef:
        return self._ref_from_population(population, rng.randrange(population.total), side=side)

    def draw_ref(self, fmt: str, rng: random.Random, *, side: str | None = None) -> TransitionRef:
        return self.draw_ref_from_population(self.draw_population(fmt, rng, side=side), rng, side=side)

    def _ref_from_global(self, index: int, *, side: str | None = None) -> TransitionRef:
        target = int(index) % len(self)
        population_index = int(np.searchsorted(self._all_prefix, target, side="right"))
        before = int(self._all_prefix[population_index - 1]) if population_index else 0
        return self._ref_from_population(self._all_populations[population_index], target - before, side=side)

    def fixed_subset(self, max_samples: int) -> FixedRefDataset:
        fixed_counts = {
            fmt: int(prefix[-1]) if len(prefix) else 0
            for fmt, prefix in self._fixed_prefix_by_format.items()
        }
        quotas = _format_quotas(fixed_counts, max_samples)
        refs: list[TransitionRef] = []
        for fmt, quota in quotas.items():
            populations = self._populations_by_format[fmt]
            prefix = self._fixed_prefix_by_format[fmt]
            total = int(prefix[-1])
            for offset in range(quota):
                rank = min((offset * total) // max(quota, 1), total - 1)
                population_index = int(np.searchsorted(prefix, rank, side="right"))
                before = int(prefix[population_index - 1]) if population_index else 0
                population = populations[population_index]
                flat = int(population.fixed_flat_indices[rank - before])
                refs.append(TransitionRef(
                    population.shard_index,
                    flat // population.rollout_steps,
                    flat % population.rollout_steps,
                    fmt,
                    self.max_context_transitions + 1,
                    "p1" if offset % 2 == 0 else "p2",
                ))
        return FixedRefDataset(self, refs)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_populations_by_format"] = {}
        state["_population_prefix_by_format"] = {}
        state["_fixed_prefix_by_format"] = {}
        state["_all_populations"] = []
        state["_all_prefix"] = np.empty(0, dtype=np.int64)
        state["_handles"] = _NpzHandleCache(self.paths)
        return state

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
        return self._sample(self._ref_from_global(index, side=side), side=side)

    def __getitem__(self, index: int | TransitionRef) -> dict[str, Any]:
        # Shuffled transition order plus alternating perspective yields exact
        # long-run p1/p2 symmetry while avoiding a second perspective tensor.
        if isinstance(index, TransitionRef):
            side = index.side or "p1"
            return self._sample(index, side=side)
        side = "p1" if (int(index) + self.epoch) % 2 == 0 else "p2"
        return self._sample(self._ref_from_global(int(index), side=side), side=side)


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
