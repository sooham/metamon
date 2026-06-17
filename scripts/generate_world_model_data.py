#!/usr/bin/env python3
"""Generate tokenized world-model training data from new-format parsed replay .txt files.

Parses each ``.txt`` file (``docs/new_parser_format_spec.md``), tokenizes the team
header, each state block, and each action block, then packs everything into
uncompressed NumPy ``.npz`` shards.

**Storage layout per shard** (``seq_shard_0000.npz``)::

    states                  (total_tokens,)    int16  — all state + header tokens concatenated
    state_offsets           (num_states,)      int64  — start index of each state in states[]
    state_lengths           (num_states,)      int32  — token count per state

    player_actions          (total_pa_tokens,) int16  — all player-action content tokens
    player_action_offsets   (num_actions,)     int64  — start index per action
    player_action_lengths   (num_actions,)     int32  — token count per action

    opponent_actions        (total_oa_tokens,) int16  — all opponent-action content tokens
    opponent_action_offsets (num_actions,)     int64  — start index per action
    opponent_action_lengths (num_actions,)     int32  — token count per action

    prev_state_idx          (num_transitions,) int32  — local state index for state[t]
    next_state_idx          (num_transitions,) int32  — local state index for state[t+1]
    battle_id               (num_transitions,) int32  — local battle index in this shard
    turn_idx                (num_transitions,) int32  — transition index within battle
    format_id               (num_transitions,) int16  — integer id from metadata

    battle_start            (num_battles+1,)   int64  — cumulative local state index
    battle_action_start     (num_battles+1,)   int64  — cumulative local action index
    won                     (num_battles,)     bool   — whether POV won
    raw_battle_key          (num_battles,)     object — battle ID strings

    format_name             scalar unicode            — battle format string
    format_id_value         scalar int16              — id for format_name

**Transition indexing:**
    prev_state_idx[t] points to the state *before* the action pair.
    next_state_idx[t] points to the state *after* the action pair.
    These are local state indices (0..N-1 within the battle).

    For a battle with N states and N-1 action pairs, transitions 0..N-2
    represent the natural state→action→state progression.  Subturns add
    extra states/actions but are indexed the same way — the training code
    can filter or skip them using the turn_idx / battle metadata.

**Validation split** is by raw battle key (both WIN and LOSS files always in
the same split), so no battle leaks between train and val.

**Usage:**:

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
import json
import os
import re
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import tqdm

from metamon.tokenizer.tokenizer import PokemonTokenizer

# ---------------------------------------------------------------------------
# Multiprocessing worker helpers
# ---------------------------------------------------------------------------

_TOKENIZER: PokemonTokenizer | None = None
_BOS_ID: int = -1
_EOS_ID: int = -1
_CHOSEN_MOVE_START_ID: int = -1
_END_CHOSEN_MOVE_ID: int = -1
_OPPONENT_CHOSEN_MOVE_START_ID: int = -1
_END_OPPONENT_CHOSEN_MOVE_ID: int = -1


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID
    global _CHOSEN_MOVE_START_ID, _END_CHOSEN_MOVE_ID
    global _OPPONENT_CHOSEN_MOVE_START_ID, _END_OPPONENT_CHOSEN_MOVE_ID
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(tokenizer_path)
    _TOKENIZER = tokenizer
    _BOS_ID = tokenizer["<bos>"]
    _EOS_ID = tokenizer["<eos>"]
    _CHOSEN_MOVE_START_ID = tokenizer["<chosen_move>"]
    _END_CHOSEN_MOVE_ID = tokenizer["<end_chosen_move>"]
    _OPPONENT_CHOSEN_MOVE_START_ID = tokenizer["<opponent_chosen_move>"]
    _END_OPPONENT_CHOSEN_MOVE_ID = tokenizer["<end_opponent_chosen_move>"]


# Regexes for parsing the new text format -- anchored by structural tokens.
# The tokenizer already handles them, but we need to split the raw text into
# header / states / actions *before* tokenization.

_BOS_RE = re.compile(r"<bos>")
_EOS_RE = re.compile(r"<eos>")
_BOA_RE = re.compile(r"<boa>")
_EOA_RE = re.compile(r"<eoa>")

# Extract the content between <chosen_move ...> and <end_chosen_move>
_CHOSEN_MOVE_RE = re.compile(
    r"<chosen_move[^>]*>(.*?)<end_chosen_move>", re.DOTALL
)
_OPPONENT_CHOSEN_MOVE_RE = re.compile(
    r"<opponent_chosen_move>(.*?)<end_opponent_chosen_move>", re.DOTALL
)


def _tokenize_action_text(text: str) -> np.ndarray:
    """Tokenize the *content* of an action tag (without the surrounding tags).

    E.g. ``"switch alakazam"`` → ``[token("switch"), token("alakazam")]``
         ``"unknown"``          → ``[token("unknown")]``
    """
    text = text.strip()
    if not text:
        return np.array([], dtype=np.int16)
    return np.array(
        [_TOKENIZER[word] for word in text.split()], dtype=np.int16
    )


def _tokenize_text_block(text: str) -> np.ndarray:
    """Tokenize a full text block (state, header, etc.)."""
    words = text.split()
    if not words:
        return np.array([], dtype=np.int16)
    return np.array([_TOKENIZER[word] for word in words], dtype=np.int16)


def _parse_single_battle_file(filepath: str) -> tuple | None:
    """Parse one new-format .txt file into tokenized arrays.

    Returns ``(state_token_arrays, player_action_arrays, opponent_action_arrays, won)``
    or ``None`` on failure.

    *state_token_arrays* includes the team header as the first element.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None

    # ── split into blocks ──────────────────────────────────────────
    # States: everything between <bos> and the next <eos>
    # Actions: everything between <boa> and the next <eoa>

    bos_positions = [m.start() for m in _BOS_RE.finditer(text)]
    eos_positions = [m.end() for m in _EOS_RE.finditer(text)]
    boa_positions = [m.start() for m in _BOA_RE.finditer(text)]
    eoa_positions = [m.end() for m in _EOA_RE.finditer(text)]

    if len(bos_positions) != len(eos_positions):
        return None
    if len(boa_positions) != len(eoa_positions):
        return None
    if len(bos_positions) < 2:
        return None  # need at least 2 states for 1 transition

    # ── team header: everything before the first <bos> ──────────────
    header_text = text[: bos_positions[0]].strip()

    # ── tokenize header + states ──────────────────────────────────
    state_token_arrays: list[np.ndarray] = []

    # Header is stored as "state -1" (always the first element)
    header_tokens = _tokenize_text_block(header_text)
    state_token_arrays.append(header_tokens)

    for start, end in zip(bos_positions, eos_positions):
        state_text = text[start:end]
        # Include the <bos> and <eos> tokens themselves
        tokens = _tokenize_text_block(state_text)
        state_token_arrays.append(tokens)

    # ── tokenize actions ───────────────────────────────────────────
    player_action_arrays: list[np.ndarray] = []
    opponent_action_arrays: list[np.ndarray] = []

    for start, end in zip(boa_positions, eoa_positions):
        action_block = text[start:end]

        # Player action
        cm_match = _CHOSEN_MOVE_RE.search(action_block)
        if cm_match:
            player_action_arrays.append(_tokenize_action_text(cm_match.group(1)))
        else:
            player_action_arrays.append(np.array([], dtype=np.int16))

        # Opponent action
        om_match = _OPPONENT_CHOSEN_MOVE_RE.search(action_block)
        if om_match:
            opponent_action_arrays.append(_tokenize_action_text(om_match.group(1)))
        else:
            opponent_action_arrays.append(np.array([], dtype=np.int16))

    # ── extract won/lost ───────────────────────────────────────────
    won = False
    if "<terminal>won<end_terminal>" in text or "<terminal>forfeit<end_terminal>" in text:
        won = True

    return state_token_arrays, player_action_arrays, opponent_action_arrays, won


def tokenize_battle(args: tuple[str, str]) -> tuple | None:
    """Multiprocessing worker entry point.

    Returns the same tuple as :func:`_parse_single_battle_file` or ``None``.
    """
    filepath, tokenizer_path = args
    global _TOKENIZER
    if _TOKENIZER is None:
        _init_worker(tokenizer_path)
    return _parse_single_battle_file(filepath)


# ---------------------------------------------------------------------------
# Shard accumulator (packing tokenized battles into .npz)
# ---------------------------------------------------------------------------


@dataclass
class ShardAccumulator:
    """Packs tokenized battles into one transition-indexed shard."""

    fmt: str
    fmt_id: int

    # Per-battle accumulators
    states_flat: list[np.ndarray] = field(default_factory=list)
    state_offsets: list[int] = field(default_factory=lambda: [0])  # cumulative
    state_lengths: list[int] = field(default_factory=list)

    player_actions_flat: list[np.ndarray] = field(default_factory=list)
    player_action_offsets: list[int] = field(default_factory=lambda: [0])
    player_action_lengths: list[int] = field(default_factory=list)

    opponent_actions_flat: list[np.ndarray] = field(default_factory=list)
    opponent_action_offsets: list[int] = field(default_factory=lambda: [0])
    opponent_action_lengths: list[int] = field(default_factory=list)

    # Transition table
    prev_state_idx: list[int] = field(default_factory=list)
    next_state_idx: list[int] = field(default_factory=list)
    battle_ids: list[np.ndarray] = field(default_factory=list)
    turn_idx: list[np.ndarray] = field(default_factory=list)
    format_ids: list[np.ndarray] = field(default_factory=list)

    # Per-battle metadata
    won: list[bool] = field(default_factory=list)
    raw_battle_keys: list[str] = field(default_factory=list)
    battle_start: list[int] = field(default_factory=lambda: [0])  # cumulative states
    battle_action_start: list[int] = field(default_factory=lambda: [0])  # cumulative actions

    # Tracking
    max_battle_len: int = 0  # longest battle in tokens (header + states + actions)

    def __len__(self) -> int:
        return len(self.won)

    @property
    def num_states(self) -> int:
        return len(self.state_lengths)

    @property
    def num_actions(self) -> int:
        return len(self.player_action_lengths)

    def append(
        self,
        state_token_arrays: list[np.ndarray],
        player_action_arrays: list[np.ndarray],
        opponent_action_arrays: list[np.ndarray],
        won: bool,
        raw_battle_key: str = "",
    ) -> None:
        """Append one fully tokenized battle.

        Args:
            state_token_arrays: Header + states.  **Header is index 0**;
                state 0 is index 1, state 1 is index 2, …
            player_action_arrays: One per transition (length = num_states - 1).
            opponent_action_arrays: One per transition (same length).
            won: Whether the POV player won.
            raw_battle_key: Unique battle identifier.
        """
        battle_id = len(self.won)
        state_base = self.num_states

        # Track total token count for this battle (header + states + all actions)
        battle_token_count = sum(len(a) for a in state_token_arrays)
        battle_token_count += sum(len(a) for a in player_action_arrays)
        battle_token_count += sum(len(a) for a in opponent_action_arrays)
        if battle_token_count > self.max_battle_len:
            self.max_battle_len = battle_token_count

        # States (header + N states, so N+1 arrays, but only N transitions)
        for tok_arr in state_token_arrays:
            self.states_flat.append(tok_arr)
            self.state_lengths.append(len(tok_arr))
            self.state_offsets.append(self.state_offsets[-1] + len(tok_arr))

        # Player actions
        for tok_arr in player_action_arrays:
            self.player_actions_flat.append(tok_arr)
            self.player_action_lengths.append(len(tok_arr))
            self.player_action_offsets.append(
                self.player_action_offsets[-1] + len(tok_arr)
            )

        # Opponent actions
        for tok_arr in opponent_action_arrays:
            self.opponent_actions_flat.append(tok_arr)
            self.opponent_action_lengths.append(len(tok_arr))
            self.opponent_action_offsets.append(
                self.opponent_action_offsets[-1] + len(tok_arr)
            )

        # Transition table: state indices 1-based within the battle
        # (index 0 = header, index 1 = state_0, index 2 = state_1, ...)
        # Transition t goes from state_t to state_{t+1} using action[t].
        n_transitions = len(player_action_arrays)
        if n_transitions > 0:
            # state 0 (index 1) → state 1 (index 2): transition 0
            # ...
            # state N-1 (index N) → state N (index N+1): transition N-1
            self.prev_state_idx.extend(
                state_base + 1 + t for t in range(n_transitions)
            )
            self.next_state_idx.extend(
                state_base + 2 + t for t in range(n_transitions)
            )
            self.battle_ids.append(np.full(n_transitions, battle_id, dtype=np.int32))
            self.turn_idx.append(np.arange(n_transitions, dtype=np.int32))
            self.format_ids.append(
                np.full(n_transitions, self.fmt_id, dtype=np.int16)
            )

        self.won.append(won)
        self.raw_battle_keys.append(raw_battle_key)
        self.battle_start.append(self.battle_start[-1] + len(state_token_arrays))
        self.battle_action_start.append(
            self.battle_action_start[-1] + n_transitions
        )

    def write(
        self,
        out_dir: str,
        shard_idx: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, int | float]:
        """Write the accumulated data to a ``.npz`` shard file.

        Transitions are optionally shuffled (stable shuffle — battle-level
        grouping is preserved by the physical layout but the transition
        row order is randomised).
        """
        # Concatenate
        states_cat = np.concatenate(self.states_flat, axis=0).astype(np.int16)
        state_offsets_arr = np.array(self.state_offsets[:-1], dtype=np.int64)
        state_lengths_arr = np.array(self.state_lengths, dtype=np.int32)

        pa_cat = np.concatenate(self.player_actions_flat, axis=0).astype(np.int16)
        pa_offsets_arr = np.array(self.player_action_offsets[:-1], dtype=np.int64)
        pa_lengths_arr = np.array(self.player_action_lengths, dtype=np.int32)

        oa_cat = np.concatenate(self.opponent_actions_flat, axis=0).astype(np.int16)
        oa_offsets_arr = np.array(self.opponent_action_offsets[:-1], dtype=np.int64)
        oa_lengths_arr = np.array(self.opponent_action_lengths, dtype=np.int32)

        prev_arr = np.array(self.prev_state_idx, dtype=np.int32)
        next_arr = np.array(self.next_state_idx, dtype=np.int32)
        battle_id_arr = np.concatenate(self.battle_ids, axis=0).astype(np.int32)
        turn_idx_arr = np.concatenate(self.turn_idx, axis=0).astype(np.int32)
        format_id_arr = np.concatenate(self.format_ids, axis=0).astype(np.int16)
        won_arr = np.array(self.won, dtype=bool)
        raw_key_arr = np.array(self.raw_battle_keys)
        battle_start_arr = np.array(self.battle_start, dtype=np.int64)
        battle_action_start_arr = np.array(self.battle_action_start, dtype=np.int64)

        # Optional shuffle of transition rows
        if rng is not None and len(prev_arr) > 1:
            order = rng.permutation(len(prev_arr))
            prev_arr = prev_arr[order]
            next_arr = next_arr[order]
            battle_id_arr = battle_id_arr[order]
            turn_idx_arr = turn_idx_arr[order]
            format_id_arr = format_id_arr[order]

        shard_name = f"seq_shard_{shard_idx:04d}.npz"
        shard_path = os.path.join(out_dir, shard_name)
        np.savez(
            shard_path,
            states=states_cat,
            state_offsets=state_offsets_arr,
            state_lengths=state_lengths_arr,
            player_actions=pa_cat,
            player_action_offsets=pa_offsets_arr,
            player_action_lengths=pa_lengths_arr,
            opponent_actions=oa_cat,
            opponent_action_offsets=oa_offsets_arr,
            opponent_action_lengths=oa_lengths_arr,
            prev_state_idx=prev_arr,
            next_state_idx=next_arr,
            battle_id=battle_id_arr,
            turn_idx=turn_idx_arr,
            format_id=format_id_arr,
            won=won_arr,
            raw_battle_key=raw_key_arr,
            battle_start=battle_start_arr,
            battle_action_start=battle_action_start_arr,
            format_name=np.array(self.fmt),
            format_id_value=np.array(self.fmt_id, dtype=np.int16),
        )

        bytes_uncompressed = (
            states_cat.nbytes
            + state_offsets_arr.nbytes
            + state_lengths_arr.nbytes
            + pa_cat.nbytes
            + pa_offsets_arr.nbytes
            + pa_lengths_arr.nbytes
            + oa_cat.nbytes
            + oa_offsets_arr.nbytes
            + oa_lengths_arr.nbytes
            + prev_arr.nbytes
            + next_arr.nbytes
            + battle_id_arr.nbytes
            + turn_idx_arr.nbytes
            + format_id_arr.nbytes
            + won_arr.nbytes
            + raw_key_arr.nbytes
            + battle_start_arr.nbytes
            + battle_action_start_arr.nbytes
        )
        avg_len = float(state_lengths_arr.mean()) if len(state_lengths_arr) else 0.0
        min_len = int(state_lengths_arr.min()) if len(state_lengths_arr) else 0
        max_len = int(state_lengths_arr.max()) if len(state_lengths_arr) else 0
        avg_pa = float(pa_lengths_arr.mean()) if len(pa_lengths_arr) else 0.0
        avg_oa = float(oa_lengths_arr.mean()) if len(oa_lengths_arr) else 0.0

        return {
            "battles": len(self),
            "states": len(state_lengths_arr),
            "actions": len(pa_lengths_arr),
            "transitions": len(prev_arr),
            "avg_state_len": avg_len,
            "min_state_len": min_len,
            "max_state_len": max_len,
            "avg_pa_len": avg_pa,
            "avg_oa_len": avg_oa,
            "max_battle_len": self.max_battle_len,
            "uncompressed_mb": bytes_uncompressed / (1024 * 1024),
        }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def iter_txt_files(fmt_dir: str) -> list[str]:
    """Walk *fmt_dir* and return sorted ``.txt`` file paths."""
    txt_files: list[str] = []
    for root, _, files in os.walk(fmt_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    txt_files.sort()
    return txt_files


def raw_battle_key(path: str) -> str:
    """Extract the raw-battle key from a POV file path.

    Strips the ``_WIN`` / ``_LOSS`` suffix so both POVs map to the same key.
    """
    stem = Path(path).stem
    for suffix in ("_WIN", "_LOSS"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def group_txt_files(txt_files: list[str]) -> dict[str, list[str]]:
    """Group POV ``.txt`` files by raw battle key.

    Returns ``{key: [win_or_loss_pov_path, ...]}``.
    """
    groups: dict[str, list[str]] = {}
    for path in txt_files:
        groups.setdefault(raw_battle_key(path), []).append(path)
    return {key: sorted(paths) for key, paths in sorted(groups.items())}


def split_groups(
    groups: dict[str, list[str]],
    val_split: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split by *raw battle key* so both POVs stay together.

    Returns ``(train_keys, val_keys, train_files, val_files)``.
    """
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
        n_val = min(n_val, n_groups - 1) if n_groups > 1 else n_val

    val_keys = sorted(str(key) for key in keys[:n_val])
    train_keys = sorted(str(key) for key in keys[n_val:])
    train_files = [path for key in train_keys for path in groups[key]]
    val_files = [path for key in val_keys for path in groups[key]]
    return train_keys, val_keys, train_files, val_files


def iter_tokenized_battles(
    txt_files: list[str],
    tokenizer_path: str,
    processes: int,
    desc: str,
) -> Iterable[tuple | None]:
    """Multiprocess tokenize a list of ``.txt`` files."""
    work = [(f, tokenizer_path) for f in txt_files]
    if processes > 1:
        with Pool(
            processes, initializer=_init_worker, initargs=(tokenizer_path,)
        ) as pool:
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
    txt_files: list[str],
    tokenizer_path: str,
    processes: int,
    battles_per_shard: int,
    out_dir: str,
    rng: np.random.Generator,
) -> tuple[dict, int, int]:
    """Tokenize and write shards for one split (train or val)."""
    os.makedirs(out_dir, exist_ok=True)

    totals: dict[str, int] = {
        "num_parsed_files": len(txt_files),
        "num_battles": 0,
        "num_shards": 0,
        "total_states": 0,
        "total_actions": 0,
        "total_transitions": 0,
        "failed": 0,
    }
    shard_idx = 0
    max_battle_len = 0
    acc = ShardAccumulator(fmt=fmt, fmt_id=fmt_id)

    for path, result in zip(
        txt_files,
        iter_tokenized_battles(
            txt_files, tokenizer_path, processes, desc=f"  Tokenizing {fmt}/{split_name}"
        ),
    ):
        if result is None:
            totals["failed"] += 1
            continue

        state_tokens, pa_tokens, oa_tokens, won = result
        acc.append(state_tokens, pa_tokens, oa_tokens, won, raw_battle_key(path))

        if len(acc) >= battles_per_shard:
            stats = acc.write(out_dir, shard_idx, rng=rng)
            totals["num_battles"] += stats["battles"]
            totals["total_states"] += stats["states"]
            totals["total_actions"] += stats["actions"]
            totals["total_transitions"] += stats["transitions"]
            max_battle_len = max(max_battle_len, stats["max_battle_len"])
            print(
                f"  {split_name} shard {shard_idx:04d}: {stats['battles']} battles, "
                f"{stats['states']} states "
                f"(avg {stats['avg_state_len']:.1f} tok/state), "
                f"{stats['transitions']} transitions, "
                f"action lens avg pa={stats['avg_pa_len']:.1f} "
                f"oa={stats['avg_oa_len']:.1f}, "
                f"max battle {stats['max_battle_len']} tok, "
                f"{stats['uncompressed_mb']:.0f} MB"
            )
            shard_idx += 1
            acc = ShardAccumulator(fmt=fmt, fmt_id=fmt_id)

    # Final partial shard
    if len(acc) > 0:
        stats = acc.write(out_dir, shard_idx, rng=rng)
        totals["num_battles"] += stats["battles"]
        totals["total_states"] += stats["states"]
        totals["total_actions"] += stats["actions"]
        totals["total_transitions"] += stats["transitions"]
        max_battle_len = max(max_battle_len, stats["max_battle_len"])
        print(
            f"  {split_name} shard {shard_idx:04d}: {stats['battles']} battles, "
            f"{stats['states']} states "
            f"(avg {stats['avg_state_len']:.1f} tok/state), "
            f"{stats['transitions']} transitions, "
            f"action lens avg pa={stats['avg_pa_len']:.1f} "
            f"oa={stats['avg_oa_len']:.1f}, "
            f"max battle {stats['max_battle_len']} tok, "
            f"{stats['uncompressed_mb']:.0f} MB"
        )
        shard_idx += 1

    totals["num_shards"] = shard_idx
    return totals, max_battle_len, shard_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate world-model .npz shards from new-format parsed replay .txt files."
    )
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--battles_per_shard", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load tokenizer to verify it has required tokens
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens")
    for tok in ["<bos>", "<eos>", "<chosen_move>", "<end_chosen_move>",
                "<opponent_chosen_move>", "<end_opponent_chosen_move>"]:
        tid = tokenizer[tok]
        if tid == tokenizer.unknown_token_id:
            raise ValueError(f"Tokenizer must contain '{tok}' token")
    print("Required structural tokens present ✓")

    format_id_map = {fmt: i for i, fmt in enumerate(args.formats)}

    for fmt in args.formats:
        fmt_dir = os.path.join(args.parsed_replay_root, fmt)
        if not os.path.isdir(fmt_dir):
            print(f"Skipping {fmt}: directory not found at {fmt_dir}")
            continue

        txt_files = iter_txt_files(fmt_dir)
        if not txt_files:
            print(f"No .txt files found in {fmt_dir}")
            continue

        groups = group_txt_files(txt_files)
        split_rng = np.random.default_rng(args.seed + format_id_map[fmt])
        train_keys, val_keys, train_files, val_files = split_groups(
            groups, args.val_split, split_rng
        )

        print(
            f"\nProcessing {fmt}: {len(txt_files)} POV files, "
            f"{len(groups)} battle groups"
        )
        print(
            f"  Split: {len(train_keys)} train / {len(val_keys)} val groups "
            f"({len(train_files)} train / {len(val_files)} val files)"
        )

        out_dir = os.path.join(args.output_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)

        shard_rng = np.random.default_rng(args.seed + 1009 * (format_id_map[fmt] + 1))
        train_totals, train_btl_max, _ = write_split_shards(
            fmt=fmt,
            fmt_id=format_id_map[fmt],
            split_name="train",
            txt_files=train_files,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            battles_per_shard=args.battles_per_shard,
            out_dir=os.path.join(out_dir, "train"),
            rng=shard_rng,
        )
        val_totals, val_btl_max, _ = write_split_shards(
            fmt=fmt,
            fmt_id=format_id_map[fmt],
            split_name="val",
            txt_files=val_files,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            battles_per_shard=args.battles_per_shard,
            out_dir=os.path.join(out_dir, "val"),
            rng=shard_rng,
        )

        fmt_battle_max = max(train_btl_max, val_btl_max)

        failed = train_totals["failed"] + val_totals["failed"]
        total_battles = train_totals["num_battles"] + val_totals["num_battles"]
        total_states = train_totals["total_states"] + val_totals["total_states"]
        total_transitions = train_totals["total_transitions"] + val_totals["total_transitions"]
        total_shards = train_totals["num_shards"] + val_totals["num_shards"]

        if failed:
            print(f"  {failed} files failed to tokenize")

        tokenizer_version = os.path.splitext(os.path.basename(args.tokenizer_path))[0]
        metadata = {
            "schema_version": "transition_table_v2",
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
            "num_parsed_files": len(txt_files),
            "train_parsed_files": len(train_files),
            "val_parsed_files": len(val_files),
            "num_battles": total_battles,
            "num_shards": total_shards,
            "train_num_battles": train_totals["num_battles"],
            "val_num_battles": val_totals["num_battles"],
            "train_num_shards": train_totals["num_shards"],
            "val_num_shards": val_totals["num_shards"],
            "battles_per_shard": args.battles_per_shard,
            "total_states": total_states,
            "total_transitions": total_transitions,
            "max_battle_len": fmt_battle_max,
            "storage": "transition_indexed_variable_length_v2",
            "compressed": False,
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Wrote metadata → {meta_path}")
        print(
            f"  Total: {total_battles} battles, {total_states} states, "
            f"{total_transitions} transitions, {total_shards} shards, "
            f"max battle {fmt_battle_max} tokens"
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
