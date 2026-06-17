#!/usr/bin/env python3
"""Generate tokenized world-model training data from parsed replays.

The output is a set of uncompressed NumPy ``.npz`` shards.  Each shard stores
tokenized states once, then an explicit transition table used by SL and JEPA
training:

    states          (total_tokens,) int16  -- all state tokens concatenated
    state_lengths   (num_states,)   int32  -- token count per state
    state_offsets   (num_states,)   int64  -- start index per state in states[]

    prev_state_idx  (num_transitions,) int32 -- local state index for state[t]
    next_state_idx  (num_transitions,) int32 -- local state index for state[t+1]
    actions         (num_transitions,) int16 -- action index for the transition
    battle_id       (num_transitions,) int32 -- local battle index in this shard
    turn_idx        (num_transitions,) int32 -- transition index within battle
    format_id       (num_transitions,) int16 -- integer id from metadata.json

For audit and compatibility, shards also include:

    battle_start    (num_battles+1,) int64 -- cumulative local state index
    won             (num_battles,) bool     -- whether POV won
    format_name     scalar unicode          -- battle format for this shard
    format_id_value scalar int16            -- id for format_name

States are unpadded.  By default every stored state is wrapped as
``<bos> ... <eos>``; pass ``--exclude-bos-eos`` to store only raw
``WorldModelObservationSpace`` text tokens.  Padding happens in each training
collate function.

Usage:
    uv run python scripts/generate_world_model_data.py \\
        --parsed_replay_root /path/to/parsed-data \\
        --tokenizer_path /path/to/tokenizer.json \\
        --output_dir /path/to/world-model-samples \\
        --formats gen1ou gen9ou \\
        --battles_per_shard 1000 \\
        --processes 8
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import orjson
import tqdm

from metamon.interface import UniversalState, WorldModelObservationSpace
from metamon.tokenizer.tokenizer import PokemonTokenizer


_TOKENIZER: PokemonTokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(tokenizer_path)
    _TOKENIZER = tokenizer


def _get_tokenizer(tokenizer_path: str) -> PokemonTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _init_worker(tokenizer_path)
    assert _TOKENIZER is not None
    return _TOKENIZER


def tokenize_battle(args: tuple[str, str] | tuple[str, str, bool]) -> tuple[list[np.ndarray], np.ndarray, bool, int] | None:
    """Tokenize one parsed battle.

    Returns ``(token_ids_list, actions_arr, won, max_state_len)``.  State token
    arrays are variable-length.  By default they include ``<bos>`` as the first
    token and ``<eos>`` as the last token.  Passing ``include_bos_eos=False``
    preserves the legacy raw-state storage format.
    """
    if len(args) == 2:
        filepath, tokenizer_path = args
        include_bos_eos = True
    else:
        filepath, tokenizer_path, include_bos_eos = args
    tokenizer = _get_tokenizer(tokenizer_path)
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]

    try:
        with open(filepath, "rb") as f:
            data = orjson.loads(f.read())
    except Exception:
        return None

    obs_space = WorldModelObservationSpace()
    obs_space.reset()

    all_states = data.get("states", [])
    actions_raw = data.get("actions", [])
    if len(all_states) < 2:
        return None

    token_ids_list: list[np.ndarray] = []
    max_state_len = 0
    for state_dict in all_states:
        us = UniversalState.from_dict(copy.deepcopy(state_dict))
        obs = obs_space.state_to_obs(us)
        ids = tokenizer.tokenize(obs["text"].tolist()).astype(np.int16)
        if include_bos_eos:
            ids = np.concatenate(
                [
                    np.array([bos_id], dtype=np.int16),
                    ids,
                    np.array([eos_id], dtype=np.int16),
                ]
            )
        token_ids_list.append(ids)
        max_state_len = max(max_state_len, len(ids))

    n_transitions = len(token_ids_list) - 1
    actions_arr = np.array(actions_raw[:n_transitions], dtype=np.int16)
    if len(actions_arr) != n_transitions:
        return None

    final_us = UniversalState.from_dict(copy.deepcopy(all_states[-1]))
    won = bool(final_us.battle_won)

    return token_ids_list, actions_arr, won, max_state_len


@dataclass
class LengthStats:
    """Online tokenized-state length statistics."""

    count: int = 0
    total: int = 0
    min_len: int | None = None
    max_len: int = 0

    def update_many(self, lengths: Iterable[int]) -> None:
        for length in lengths:
            length = int(length)
            self.count += 1
            self.total += length
            self.max_len = max(self.max_len, length)
            self.min_len = length if self.min_len is None else min(self.min_len, length)

    def merge(self, other: "LengthStats") -> None:
        self.count += other.count
        self.total += other.total
        self.max_len = max(self.max_len, other.max_len)
        if other.min_len is not None:
            self.min_len = other.min_len if self.min_len is None else min(self.min_len, other.min_len)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def as_metadata(self) -> dict[str, int | float | None]:
        return {
            "state_len_count": self.count,
            "state_len_min": self.min_len,
            "state_len_max": self.max_len,
            "state_len_avg": self.avg,
        }


@dataclass
class ShardAccumulator:
    """Incrementally packs tokenized battles into one transition-indexed shard."""

    fmt: str
    fmt_id: int
    states_flat: list[np.ndarray] = field(default_factory=list)
    state_lengths: list[int] = field(default_factory=list)
    state_offsets: list[int] = field(default_factory=lambda: [0])
    prev_state_idx: list[int] = field(default_factory=list)
    next_state_idx: list[int] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    battle_ids: list[np.ndarray] = field(default_factory=list)
    turn_idx: list[np.ndarray] = field(default_factory=list)
    format_ids: list[np.ndarray] = field(default_factory=list)
    won: list[bool] = field(default_factory=list)
    raw_battle_keys: list[str] = field(default_factory=list)
    battle_start: list[int] = field(default_factory=lambda: [0])

    def __len__(self) -> int:
        return len(self.won)

    @property
    def num_states(self) -> int:
        return len(self.state_lengths)

    @property
    def num_transitions(self) -> int:
        return len(self.prev_state_idx)

    def append(
        self,
        token_ids_list: list[np.ndarray],
        actions_arr: np.ndarray,
        won: bool,
        raw_battle_key: str = "",
    ) -> None:
        battle_id = len(self.won)
        state_base = self.num_states

        for tok_arr in token_ids_list:
            self.states_flat.append(tok_arr)
            self.state_lengths.append(len(tok_arr))
            self.state_offsets.append(self.state_offsets[-1] + len(tok_arr))

        n_transitions = len(actions_arr)
        self.prev_state_idx.extend(state_base + t for t in range(n_transitions))
        self.next_state_idx.extend(state_base + t + 1 for t in range(n_transitions))
        self.actions.append(actions_arr.astype(np.int16, copy=False))
        self.battle_ids.append(np.full(n_transitions, battle_id, dtype=np.int32))
        self.turn_idx.append(np.arange(n_transitions, dtype=np.int32))
        self.format_ids.append(np.full(n_transitions, self.fmt_id, dtype=np.int16))
        self.won.append(won)
        self.raw_battle_keys.append(raw_battle_key)
        self.battle_start.append(self.battle_start[-1] + len(token_ids_list))

    def write(
        self,
        out_dir: str,
        shard_idx: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, int | float]:
        states_cat = np.concatenate(self.states_flat, axis=0).astype(np.int16)
        state_lengths_arr = np.array(self.state_lengths, dtype=np.int32)
        state_offsets_arr = np.array(self.state_offsets[:-1], dtype=np.int64)
        prev_state_idx_arr = np.array(self.prev_state_idx, dtype=np.int32)
        next_state_idx_arr = np.array(self.next_state_idx, dtype=np.int32)
        actions_arr = np.concatenate(self.actions, axis=0).astype(np.int16)
        battle_id_arr = np.concatenate(self.battle_ids, axis=0).astype(np.int32)
        turn_idx_arr = np.concatenate(self.turn_idx, axis=0).astype(np.int32)
        format_id_arr = np.concatenate(self.format_ids, axis=0).astype(np.int16)
        won_arr = np.array(self.won, dtype=bool)
        raw_battle_key_arr = np.array(self.raw_battle_keys)
        battle_start_arr = np.array(self.battle_start, dtype=np.int64)

        if rng is not None and len(actions_arr) > 1:
            order = rng.permutation(len(actions_arr))
            prev_state_idx_arr = prev_state_idx_arr[order]
            next_state_idx_arr = next_state_idx_arr[order]
            actions_arr = actions_arr[order]
            battle_id_arr = battle_id_arr[order]
            turn_idx_arr = turn_idx_arr[order]
            format_id_arr = format_id_arr[order]

        shard_name = f"seq_shard_{shard_idx:04d}.npz"
        shard_path = os.path.join(out_dir, shard_name)
        np.savez(
            shard_path,
            states=states_cat,
            state_lengths=state_lengths_arr,
            state_offsets=state_offsets_arr,
            prev_state_idx=prev_state_idx_arr,
            next_state_idx=next_state_idx_arr,
            actions=actions_arr,
            battle_id=battle_id_arr,
            turn_idx=turn_idx_arr,
            format_id=format_id_arr,
            won=won_arr,
            raw_battle_key=raw_battle_key_arr,
            battle_start=battle_start_arr,
            format_name=np.array(self.fmt),
            format_id_value=np.array(self.fmt_id, dtype=np.int16),
        )

        bytes_uncompressed = (
            states_cat.nbytes
            + state_lengths_arr.nbytes
            + state_offsets_arr.nbytes
            + prev_state_idx_arr.nbytes
            + next_state_idx_arr.nbytes
            + actions_arr.nbytes
            + battle_id_arr.nbytes
            + turn_idx_arr.nbytes
            + format_id_arr.nbytes
            + won_arr.nbytes
            + raw_battle_key_arr.nbytes
            + battle_start_arr.nbytes
        )
        avg_len = float(state_lengths_arr.mean()) if len(state_lengths_arr) else 0.0
        min_len = int(state_lengths_arr.min()) if len(state_lengths_arr) else 0
        max_len = int(state_lengths_arr.max()) if len(state_lengths_arr) else 0
        return {
            "battles": len(self),
            "states": len(state_lengths_arr),
            "transitions": len(actions_arr),
            "avg_len": avg_len,
            "min_len": min_len,
            "max_len": max_len,
            "uncompressed_mb": bytes_uncompressed / (1024 * 1024),
        }


def iter_json_files(fmt_dir: str) -> list[str]:
    json_files: list[str] = []
    for root, _, files in os.walk(fmt_dir):
        for f in files:
            if f.endswith(".json") and not f.endswith(".json.lz4"):
                json_files.append(os.path.join(root, f))
    json_files.sort()
    return json_files


def raw_battle_key(path: str) -> str:
    stem = Path(path).stem
    for suffix in ("_WIN", "_LOSS"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def group_json_files(json_files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in json_files:
        groups.setdefault(raw_battle_key(path), []).append(path)
    return {key: sorted(paths) for key, paths in sorted(groups.items())}


def split_groups(
    groups: dict[str, list[str]],
    val_split: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str], list[str]]:
    keys = np.array(list(groups.keys()), dtype=object)
    rng.shuffle(keys)
    n_groups = len(keys)
    if n_groups == 0:
        return [], [], [], []
    if val_split <= 0:
        n_val = 0
    elif val_split >= 1:
        n_val = n_groups
    else:
        n_val = max(1, int(round(n_groups * val_split)))
        if n_groups > 1:
            n_val = min(n_val, n_groups - 1)

    val_keys = sorted(str(key) for key in keys[:n_val])
    train_keys = sorted(str(key) for key in keys[n_val:])
    train_files = [path for key in train_keys for path in groups[key]]
    val_files = [path for key in val_keys for path in groups[key]]
    return train_keys, val_keys, train_files, val_files


def iter_tokenized_battles(
    json_files: list[str],
    tokenizer_path: str,
    processes: int,
    desc: str,
    include_bos_eos: bool = True,
) -> Iterable[tuple[list[np.ndarray], np.ndarray, bool, int] | None]:
    work = [(f, tokenizer_path, include_bos_eos) for f in json_files]
    if processes > 1:
        with Pool(processes, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            yield from tqdm.tqdm(
                pool.imap(tokenize_battle, work, chunksize=100),
                total=len(work),
                desc=desc,
            )
    else:
        _init_worker(tokenizer_path)
        for item in tqdm.tqdm(work, desc=desc):
            yield tokenize_battle(item)


def write_split_shards(
    *,
    fmt: str,
    fmt_id: int,
    split_name: str,
    json_files: list[str],
    tokenizer_path: str,
    processes: int,
    include_bos_eos: bool,
    battles_per_shard: int,
    out_dir: str,
    rng: np.random.Generator,
) -> tuple[dict[str, int | float], LengthStats, int]:
    os.makedirs(out_dir, exist_ok=True)
    shard_idx = 0
    totals: dict[str, int | float] = {
        "num_parsed_files": len(json_files),
        "num_battles": 0,
        "num_shards": 0,
        "total_states": 0,
        "total_transitions": 0,
        "failed": 0,
    }
    len_stats = LengthStats()
    max_state_len = 0
    acc = ShardAccumulator(fmt=fmt, fmt_id=fmt_id)

    for path, result in zip(
        json_files,
        iter_tokenized_battles(
            json_files,
            tokenizer_path,
            processes,
            desc=f"  Tokenizing {fmt}/{split_name}",
            include_bos_eos=include_bos_eos,
        ),
    ):
        if result is None:
            totals["failed"] = int(totals["failed"]) + 1
            continue

        token_ids_list, actions_arr, won, battle_max_state_len = result
        acc.append(token_ids_list, actions_arr, won, raw_battle_key(path))
        len_stats.update_many(len(tokens) for tokens in token_ids_list)
        max_state_len = max(max_state_len, battle_max_state_len)

        if len(acc) >= battles_per_shard:
            stats = acc.write(out_dir, shard_idx, rng=rng)
            totals["num_battles"] = int(totals["num_battles"]) + int(stats["battles"])
            totals["total_states"] = int(totals["total_states"]) + int(stats["states"])
            totals["total_transitions"] = int(totals["total_transitions"]) + int(stats["transitions"])
            print(
                f"  {split_name} shard {shard_idx:04d}: {stats['battles']} battles, "
                f"{stats['states']} states "
                f"(avg {stats['avg_len']:.1f}, min {stats['min_len']}, "
                f"max {stats['max_len']} tok/state), "
                f"{stats['transitions']} transitions, "
                f"{stats['uncompressed_mb']:.0f} MB uncompressed"
            )
            shard_idx += 1
            acc = ShardAccumulator(fmt=fmt, fmt_id=fmt_id)

    if len(acc) > 0:
        stats = acc.write(out_dir, shard_idx, rng=rng)
        totals["num_battles"] = int(totals["num_battles"]) + int(stats["battles"])
        totals["total_states"] = int(totals["total_states"]) + int(stats["states"])
        totals["total_transitions"] = int(totals["total_transitions"]) + int(stats["transitions"])
        print(
            f"  {split_name} shard {shard_idx:04d}: {stats['battles']} battles, "
            f"{stats['states']} states "
            f"(avg {stats['avg_len']:.1f}, min {stats['min_len']}, "
            f"max {stats['max_len']} tok/state), "
            f"{stats['transitions']} transitions, "
            f"{stats['uncompressed_mb']:.0f} MB uncompressed"
        )
        shard_idx += 1

    totals["num_shards"] = shard_idx
    return totals, len_stats, max_state_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate transition-indexed world-model data from parsed replays."
    )
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--battles_per_shard", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-bos-eos",
        action="store_true",
        help="Store raw state tokens without wrapping each state in <bos> ... <eos>.",
    )
    args = parser.parse_args()

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens")
    include_bos_eos = not args.exclude_bos_eos
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]
    if include_bos_eos and (bos_id == tokenizer.unknown_token_id or eos_id == tokenizer.unknown_token_id):
        raise ValueError("Tokenizer must contain <bos> and <eos> unless --exclude-bos-eos is set")
    print(f"State boundary tokens: {'included' if include_bos_eos else 'excluded'}")

    format_id_map = {fmt: i for i, fmt in enumerate(args.formats)}
    max_state_len_overall = 0
    overall_len_stats = LengthStats()

    for fmt in args.formats:
        fmt_dir = os.path.join(args.parsed_replay_root, fmt)
        if not os.path.isdir(fmt_dir):
            print(f"Skipping {fmt}: directory not found at {fmt_dir}")
            continue

        json_files = iter_json_files(fmt_dir)
        if not json_files:
            print(f"No JSON files found in {fmt_dir}")
            continue

        groups = group_json_files(json_files)
        split_rng = np.random.default_rng(args.seed + format_id_map[fmt])
        train_keys, val_keys, train_files, val_files = split_groups(
            groups, args.val_split, split_rng
        )

        print(
            f"\nProcessing {fmt}: {len(json_files)} parsed POV files, "
            f"{len(groups)} raw battle groups"
        )
        print(
            f"  Split: {len(train_keys)} train groups / {len(val_keys)} val groups "
            f"({len(train_files)} train files / {len(val_files)} val files)"
        )
        out_dir = os.path.join(args.output_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)

        shard_rng = np.random.default_rng(args.seed + 1009 * (format_id_map[fmt] + 1))
        train_totals, train_len_stats, train_max_state_len = write_split_shards(
            fmt=fmt,
            fmt_id=format_id_map[fmt],
            split_name="train",
            json_files=train_files,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            include_bos_eos=include_bos_eos,
            battles_per_shard=args.battles_per_shard,
            out_dir=os.path.join(out_dir, "train"),
            rng=shard_rng,
        )
        val_totals, val_len_stats, val_max_state_len = write_split_shards(
            fmt=fmt,
            fmt_id=format_id_map[fmt],
            split_name="val",
            json_files=val_files,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            include_bos_eos=include_bos_eos,
            battles_per_shard=args.battles_per_shard,
            out_dir=os.path.join(out_dir, "val"),
            rng=shard_rng,
        )

        fmt_len_stats = LengthStats()
        fmt_len_stats.merge(train_len_stats)
        fmt_len_stats.merge(val_len_stats)
        fmt_max_state_len = max(train_max_state_len, val_max_state_len)
        failed = int(train_totals["failed"]) + int(val_totals["failed"])
        total_battles = int(train_totals["num_battles"]) + int(val_totals["num_battles"])
        total_states = int(train_totals["total_states"]) + int(val_totals["total_states"])
        total_transitions = int(train_totals["total_transitions"]) + int(val_totals["total_transitions"])
        total_shards = int(train_totals["num_shards"]) + int(val_totals["num_shards"])

        if failed:
            print(f"  {failed} parsed POV files failed to tokenize, skipping")
        if fmt_max_state_len:
            print(f"  Max stored state length: {fmt_max_state_len}")
            max_state_len_overall = max(max_state_len_overall, fmt_max_state_len)
        if fmt_len_stats.count:
            print(
                f"  State length stats (stored tokens): "
                f"avg {fmt_len_stats.avg:.2f}, min {fmt_len_stats.min_len}, "
                f"max {fmt_len_stats.max_len}, n={fmt_len_stats.count}"
            )
            overall_len_stats.merge(fmt_len_stats)

        tokenizer_version = os.path.splitext(os.path.basename(args.tokenizer_path))[0]
        metadata = {
            "schema_version": "transition_table_v1",
            "tokenizer_version": tokenizer_version,
            "format": fmt,
            "format_id": format_id_map[fmt],
            "format_id_map": format_id_map,
            "split_mode": "raw_battle_group",
            "seed": args.seed,
            "val_split": args.val_split,
            "num_raw_battle_groups": len(groups),
            "train_raw_battle_groups": len(train_keys),
            "val_raw_battle_groups": len(val_keys),
            "num_parsed_files": len(json_files),
            "train_parsed_files": len(train_files),
            "val_parsed_files": len(val_files),
            "num_battles": total_battles,
            "num_shards": total_shards,
            "train_num_battles": int(train_totals["num_battles"]),
            "val_num_battles": int(val_totals["num_battles"]),
            "train_num_shards": int(train_totals["num_shards"]),
            "val_num_shards": int(val_totals["num_shards"]),
            "battles_per_shard": args.battles_per_shard,
            "total_states": total_states,
            "total_transitions": total_transitions,
            "train_total_states": int(train_totals["total_states"]),
            "val_total_states": int(val_totals["total_states"]),
            "train_total_transitions": int(train_totals["total_transitions"]),
            "val_total_transitions": int(val_totals["total_transitions"]),
            "storage": "transition_indexed_variable_length",
            "compressed": False,
            "state_includes_bos_eos": include_bos_eos,
            **fmt_len_stats.as_metadata(),
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Wrote metadata to {meta_path}")
        print(
            f"  Total: {total_battles} battles, "
            f"{total_states} states, {total_transitions} transitions"
        )

    print("\nAll formats complete.")
    if max_state_len_overall > 0:
        print(
            f"Maximum stored state length across all formats: {max_state_len_overall}"
        )
    if overall_len_stats.count:
        print(
            f"Overall state length stats (stored tokens): "
            f"avg {overall_len_stats.avg:.2f}, min {overall_len_stats.min_len}, "
            f"max {overall_len_stats.max_len}, n={overall_len_stats.count}"
        )


if __name__ == "__main__":
    main()
