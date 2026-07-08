#!/usr/bin/env python3
"""Generate tokenized world-model training data from new-format parsed replay .txt files.

Parses each ``.txt`` file (``docs/new_parser_format_spec.md``), tokenizes the team
header, each state block, and each action block, then packs everything into
uncompressed NumPy ``.npz`` shards.

**Storage layout per shard** (``paired_shard_0000.npz``)::

    p1_states / p2_states                  (total_tokens,)       int16
    p1_state_offsets / p2_state_offsets    (num_states,)         int64
    p1_state_lengths / p2_state_lengths    (num_states,)         int32

    p1_actions / p2_actions                (total_action_tokens,) int16
    p1_action_offsets / p2_action_offsets  (num_actions,)         int64
    p1_action_lengths / p2_action_lengths  (num_actions,)         int32
    p1_opponent_actions / p2_opponent_actions and offsets/lengths mirror these.
    p1_legal_actions / p2_legal_actions    (num_actions, max_legal, max_action_tokens) int16
    p1_legal_action_mask / p2_legal_action_mask (num_actions, max_legal) bool
    p1_chosen_legal_action_idx / p2_chosen_legal_action_idx (num_actions,) int16

    p1_target_state_idx / p2_target_state_idx (num_samples, K) int32
    p1_next_state_idx / p2_next_state_idx     (num_samples, K) int32
    p1_action_idx / p2_action_idx             (num_samples, K) int32
    p1_next_terminal_class                    (num_samples, K) int16

    battle_id                              (num_samples,)    int32
    turn_idx / turn_number / subturn_idx   (num_samples, K) int32/int16
    format_id                              (num_samples, K) int16
    rollout_len                            scalar int16

    battle_start            (num_battles+1,)   int64  — cumulative local state index
    battle_action_start     (num_battles+1,)   int64  — cumulative local action index
    raw_battle_key          (num_battles,)     object — battle ID strings

    format_name             scalar unicode            — battle format string
    format_id_value         scalar int16              — id for format_name

**Rollout indexing:**
    Each sample row contains K contiguous aligned transitions from one battle.
    Step j in a row points to target state[t+j], action[t+j], and
    next state[t+j+1]. Training histories are target-excluded: for target
    state[t], the model receives only the team header plus state/action blocks
    before state[t].
    Windows with skipped POV subturn gaps are not emitted.

**Validation split** is by raw battle key (both WIN and LOSS files always in
the same split), so no battle leaks between train and val.

**Usage:**:

    uv run python scripts/generate_world_model_data.py \\
        --parsed_replay_root /path/to/parsed-data \\
        --tokenizer_path /path/to/tokenizer.json \\
        --output_dir /path/to/world-model-samples \\
        --formats gen1ou gen9ou \\
        --battles_per_shard 1000 \\
        --rollout_len 1 \\
        --processes 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
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
_UNKNOWN_ID: int = -1
_PAD_ID: int = 0


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _UNKNOWN_ID, _PAD_ID
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(tokenizer_path)
    _TOKENIZER = tokenizer
    _BOS_ID = tokenizer["<bos>"]
    _EOS_ID = tokenizer["<eos>"]
    _UNKNOWN_ID = tokenizer["unknown"]
    _PAD_ID = tokenizer.pad_token_id


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
_TURN_RE = re.compile(r"<turn>\s*(\d+)\s*<end_turn>", re.DOTALL)
_TERMINAL_RE = re.compile(
    r"<terminal>\s*(won|lost|forfeit_won|forfeit_lost|tie)\s*<end_terminal>", re.DOTALL
)
_BATTLE_ID_RE = re.compile(r"^((?:smogtours-)?[A-Za-z0-9]+-\d+)")
_MOVE_ENTRY_RE = re.compile(r"<move>\s*([^\s<>]+)(?:\s+.*?)?<end_move>", re.DOTALL)
_YOU_RE = re.compile(r"<you>(.*?)<end_you>", re.DOTALL)
_BENCH_ENTRY_RE = re.compile(r"<poke\d+>\s*(.*?)\s*<end_poke\d+>", re.DOTALL)
_FAINTED_STATUSES = {"fnt"}

TERMINAL_CLASSES = (
    "ongoing",
    "won",
    "lost",
    "forfeit_won",
    "forfeit_lost",
    "tie",
)
TERMINAL_CLASS_TO_ID = {name: idx for idx, name in enumerate(TERMINAL_CLASSES)}
TERMINAL_ID_TO_CLASS = {idx: name for name, idx in TERMINAL_CLASS_TO_ID.items()}


def _terminal_class_from_state_text(state_text: str) -> int:
    match = _TERMINAL_RE.search(state_text)
    if match is None:
        return TERMINAL_CLASS_TO_ID["ongoing"]
    return TERMINAL_CLASS_TO_ID[match.group(1)]


@dataclass
class TokenizedPOV:
    state_token_arrays: list[np.ndarray]  # header + states
    player_action_arrays: list[np.ndarray]
    opponent_action_arrays: list[np.ndarray]
    turn_numbers: list[int]
    path: str
    move_counts: dict[str, int] = field(default_factory=dict)
    state_terminal_classes: list[int] = field(default_factory=list)  # header + states
    legal_action_arrays: list[list[np.ndarray]] = field(default_factory=list)
    chosen_legal_action_idx: list[int] = field(default_factory=list)


def _tokenize_action_text(text: str) -> np.ndarray:
    """Tokenize the *content* of an action tag (without the surrounding tags).

    E.g. ``"switch alakazam"`` → ``[token("switch"), token("alakazam")]``
         ``"unknown"``          → ``[token("unknown")]``
    """
    text = text.strip()
    if not text:
        return np.array([_UNKNOWN_ID], dtype=np.int16)
    return np.array(
        [_TOKENIZER[word] for word in text.split()], dtype=np.int16
    )


def _canonical_action_text(text: str) -> str:
    text = text.strip()
    if not text:
        return "unknown unknown"
    parts = text.split()
    if len(parts) >= 2 and parts[0] in {"move", "switch"}:
        return f"{parts[0]} {parts[1]}"
    if parts[0] == "unknown":
        return "unknown unknown"
    if parts[0] == "none":
        return "none"
    return f"move {parts[0]}"


def _chosen_move_name(text: str) -> str | None:
    """Return the move name from a chosen-action block, excluding switches."""
    canonical = _canonical_action_text(text)
    parts = canonical.split(maxsplit=1)
    if len(parts) == 2 and parts[0] == "move" and parts[1] != "unknown":
        return parts[1]
    return None


def _record_chosen_move(counts: Counter[str], text: str) -> None:
    move = _chosen_move_name(text)
    if move is not None:
        counts[move] += 1


def _bench_switch_candidates(state_text: str, *, forced_revival: bool) -> list[str]:
    candidates: list[str] = []
    for match in _BENCH_ENTRY_RE.finditer(state_text):
        words = match.group(1).split()
        if len(words) < 2:
            continue
        species = words[0]
        hp = words[1]
        status = words[-1] if words[-1] in _FAINTED_STATUSES else None
        is_fainted = status == "fnt" or hp == "0.00"
        if forced_revival:
            if is_fainted:
                candidates.append(f"switch {species}")
        elif not is_fainted:
            candidates.append(f"switch {species}")
    return candidates


def _legal_action_texts_from_state(state_text: str, chosen_text: str) -> tuple[list[str], int]:
    """Infer current-player legal action contents from one serialized state.

    The state text exposes available moves, POV bench, and side markers such as
    ``forceswitch`` / ``forcedrevival``.  We never inspect or require opponent
    legal actions.  If a replay choice is absent from the inferred list, it is
    appended so every training target remains representable.
    """
    chosen = _canonical_action_text(chosen_text)
    you_match = _YOU_RE.search(state_text)
    you_tokens = set(you_match.group(1).split()) if you_match else set()
    forced_switch = "forceswitch" in you_tokens
    forced_revival = "forcedrevival" in you_tokens

    texts: list[str] = []
    if not forced_switch and not forced_revival:
        for move_match in _MOVE_ENTRY_RE.finditer(state_text):
            move_name = move_match.group(1).strip()
            if move_name:
                texts.append(f"move {move_name}")

    texts.extend(_bench_switch_candidates(state_text, forced_revival=forced_revival))

    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        canonical = _canonical_action_text(text)
        if canonical not in seen:
            deduped.append(canonical)
            seen.add(canonical)

    if chosen not in seen:
        deduped.append(chosen)
        seen.add(chosen)
    if not deduped:
        deduped.append("unknown unknown")

    return deduped, deduped.index(chosen) if chosen in seen else 0


def _tokenize_text_block(text: str) -> np.ndarray:
    """Tokenize a full text block (state, header, etc.)."""
    words = text.split()
    if not words:
        return np.array([], dtype=np.int16)
    return np.array([_TOKENIZER[word] for word in words], dtype=np.int16)


def _parse_single_battle_file_detailed(filepath: str) -> TokenizedPOV | None:
    """Parse one new-format .txt file into tokenized arrays + alignment metadata."""
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
    turn_numbers: list[int] = []
    state_texts: list[str] = []
    state_terminal_classes: list[int] = []

    # Header is stored as "state -1" (always the first element)
    header_tokens = _tokenize_text_block(header_text)
    state_token_arrays.append(header_tokens)
    state_terminal_classes.append(TERMINAL_CLASS_TO_ID["ongoing"])

    for start, end in zip(bos_positions, eos_positions):
        state_text = text[start:end]
        state_texts.append(state_text)
        state_terminal_classes.append(_terminal_class_from_state_text(state_text))
        # Include the <bos> and <eos> tokens themselves
        tokens = _tokenize_text_block(state_text)
        state_token_arrays.append(tokens)
        turn_match = _TURN_RE.search(state_text)
        if turn_match:
            turn_numbers.append(int(turn_match.group(1)))
        else:
            return None

    # ── tokenize actions ───────────────────────────────────────────
    player_action_arrays: list[np.ndarray] = []
    opponent_action_arrays: list[np.ndarray] = []
    chosen_action_texts: list[str] = []
    move_counts: Counter[str] = Counter()

    for start, end in zip(boa_positions, eoa_positions):
        action_block = text[start:end]

        # Player action
        cm_match = _CHOSEN_MOVE_RE.search(action_block)
        if cm_match:
            chosen_text = _canonical_action_text(cm_match.group(1))
            player_action_arrays.append(_tokenize_action_text(chosen_text))
            _record_chosen_move(move_counts, chosen_text)
        else:
            chosen_text = "unknown unknown"
            player_action_arrays.append(_tokenize_action_text(chosen_text))
        chosen_action_texts.append(chosen_text)

        # Opponent action
        om_match = _OPPONENT_CHOSEN_MOVE_RE.search(action_block)
        if om_match:
            opponent_text = om_match.group(1)
            opponent_action_arrays.append(_tokenize_action_text(opponent_text))
            _record_chosen_move(move_counts, opponent_text)
        else:
            opponent_action_arrays.append(np.array([_UNKNOWN_ID], dtype=np.int16))

    legal_action_arrays: list[list[np.ndarray]] = []
    chosen_legal_action_idx: list[int] = []
    for action_idx, chosen_text in enumerate(chosen_action_texts):
        if action_idx >= len(state_texts):
            break
        legal_texts, chosen_idx = _legal_action_texts_from_state(
            state_texts[action_idx],
            chosen_text,
        )
        legal_action_arrays.append([
            _tokenize_action_text(text) for text in legal_texts
        ])
        chosen_legal_action_idx.append(int(chosen_idx))

    return TokenizedPOV(
        state_token_arrays=state_token_arrays,
        player_action_arrays=player_action_arrays,
        opponent_action_arrays=opponent_action_arrays,
        move_counts=dict(move_counts),
        state_terminal_classes=state_terminal_classes,
        legal_action_arrays=legal_action_arrays,
        chosen_legal_action_idx=chosen_legal_action_idx,
        turn_numbers=turn_numbers,
        path=filepath,
    )


def _is_unknown_action(tokens: np.ndarray) -> bool:
    if len(tokens) == 0:
        return True
    # Check if all tokens are the unknown token ID
    return bool(np.all(tokens == _UNKNOWN_ID))


def _actions_compatible(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right) or _is_unknown_action(left) or _is_unknown_action(right))


def _subturn_indices(turn_numbers: list[int]) -> list[int]:
    counts: dict[int, int] = {}
    out: list[int] = []
    for turn in turn_numbers:
        idx = counts.get(turn, 0)
        out.append(idx)
        counts[turn] = idx + 1
    return out


@dataclass
class PairedBattle:
    raw_battle_key: str
    p1: TokenizedPOV
    p2: TokenizedPOV
    aligned_rows: list["PairedTransitionRow"]
    rollout_windows: list[list["PairedTransitionRow"]]
    fmt_id: int = 0


@dataclass(frozen=True)
class PairedTransitionRow:
    p1_state_idx: int
    p1_next_state_idx: int
    p1_action_idx: int
    p2_state_idx: int
    p2_next_state_idx: int
    p2_action_idx: int
    turn_number: int
    subturn_idx: int


def _state_occurrence_keys(turn_numbers: list[int]) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    keys: list[tuple[int, int]] = []
    for turn in turn_numbers:
        occurrence = counts.get(turn, 0)
        keys.append((turn, occurrence))
        counts[turn] = occurrence + 1
    return keys


def _paired_transition_rows(
    p1: TokenizedPOV,
    p2: TokenizedPOV,
) -> list[PairedTransitionRow]:
    p1_keys = _state_occurrence_keys(p1.turn_numbers)
    p2_keys = _state_occurrence_keys(p2.turn_numbers)
    p2_by_key = {key: idx + 1 for idx, key in enumerate(p2_keys)}

    rows: list[PairedTransitionRow] = []
    for idx in range(len(p1_keys) - 1):
        cur_key = p1_keys[idx]
        next_key = p1_keys[idx + 1]
        if cur_key not in p2_by_key or next_key not in p2_by_key:
            continue

        p1_state_idx = idx + 1
        p1_next_state_idx = idx + 2
        p2_state_idx = p2_by_key[cur_key]
        p2_next_state_idx = p2_by_key[next_key]

        # This model predicts one action pair and one next state. If either POV
        # has extra forced-switch/subturn states between these common states,
        # skip that interval rather than folding multiple decisions into one row.
        if p2_next_state_idx != p2_state_idx + 1:
            continue

        p1_action_idx = p1_state_idx - 1
        p2_action_idx = p2_state_idx - 1
        if (
            p1_action_idx >= len(p1.player_action_arrays)
            or p2_action_idx >= len(p2.player_action_arrays)
        ):
            continue
        if not _actions_compatible(
            p1.player_action_arrays[p1_action_idx],
            p2.opponent_action_arrays[p2_action_idx],
        ):
            continue
        if not _actions_compatible(
            p2.player_action_arrays[p2_action_idx],
            p1.opponent_action_arrays[p1_action_idx],
        ):
            continue

        rows.append(PairedTransitionRow(
            p1_state_idx=p1_state_idx,
            p1_next_state_idx=p1_next_state_idx,
            p1_action_idx=p1_action_idx,
            p2_state_idx=p2_state_idx,
            p2_next_state_idx=p2_next_state_idx,
            p2_action_idx=p2_action_idx,
            turn_number=cur_key[0],
            subturn_idx=cur_key[1],
        ))
    return rows


def _rows_are_contiguous(left: PairedTransitionRow, right: PairedTransitionRow) -> bool:
    return (
        left.p1_next_state_idx == right.p1_state_idx
        and left.p2_next_state_idx == right.p2_state_idx
        and left.p1_action_idx + 1 == right.p1_action_idx
        and left.p2_action_idx + 1 == right.p2_action_idx
    )


def _contiguous_rollout_windows(
    rows: list[PairedTransitionRow],
    rollout_len: int,
) -> list[list[PairedTransitionRow]]:
    """Return all K-step windows with no skipped transition gaps."""
    if rollout_len < 1:
        raise ValueError("rollout_len must be >= 1")
    if len(rows) < rollout_len:
        return []

    windows: list[list[PairedTransitionRow]] = []
    for start in range(0, len(rows) - rollout_len + 1):
        window = rows[start : start + rollout_len]
        if all(_rows_are_contiguous(a, b) for a, b in zip(window, window[1:])):
            windows.append(window)
    return windows


def _validate_paired_battle(
    key: str,
    p1: TokenizedPOV,
    p2: TokenizedPOV,
    rollout_len: int,
) -> tuple[str | None, list[PairedTransitionRow], list[list[PairedTransitionRow]]]:
    n_states_1 = len(p1.state_token_arrays)
    n_states_2 = len(p2.state_token_arrays)
    n_actions_1 = len(p1.player_action_arrays)
    n_actions_2 = len(p2.player_action_arrays)
    if len(p1.opponent_action_arrays) != n_actions_1 or len(p2.opponent_action_arrays) != n_actions_2:
        return "opponent_action_count_mismatch", [], []
    expected_actions = n_states_1 - 2  # header + (T+1 states) -> T transitions
    if n_actions_1 != expected_actions:
        return f"p1_state_action_mismatch:{n_states_1}_states/{n_actions_1}_actions", [], []
    expected_actions = n_states_2 - 2
    if n_actions_2 != expected_actions:
        return f"p2_state_action_mismatch:{n_states_2}_states/{n_actions_2}_actions", [], []

    if min(n_actions_1, n_actions_2) < rollout_len:
        return f"too_short_for_rollout_len_{rollout_len}", [], []

    rows = _paired_transition_rows(p1, p2)
    if not rows:
        detail = f"p1:{n_states_1-1}st/{n_actions_1}act p2:{n_states_2-1}st/{n_actions_2}act"
        return f"no_aligned_transitions::{detail}", [], []

    windows = _contiguous_rollout_windows(rows, rollout_len)
    if not windows:
        # Count gaps in row alignment
        gaps = sum(
            1 for a, b in zip(rows, rows[1:])
            if not _rows_are_contiguous(a, b)
        )
        detail = f"{len(rows)} rows, {gaps} gaps"
        return f"no_contiguous_rollout_windows_{rollout_len}::{detail}", rows, []

    return None, rows, windows


def tokenize_battle_pair(args: tuple[str, list[str], str, int, int]) -> tuple[PairedBattle | None, str | None]:
    """Tokenize and validate one raw-battle group containing both POV files."""
    key, paths, tokenizer_path, rollout_len, fmt_id = args
    global _TOKENIZER
    if _TOKENIZER is None:
        _init_worker(tokenizer_path)

    if len(paths) != 2:
        return None, f"expected_2_povs_got_{len(paths)}"

    p1 = _parse_single_battle_file_detailed(paths[0])
    p2 = _parse_single_battle_file_detailed(paths[1])
    if p1 is None or p2 is None:
        return None, "parse_failed"

    reason, rows, windows = _validate_paired_battle(key, p1, p2, rollout_len)
    if reason is not None:
        return None, reason
    return PairedBattle(
        raw_battle_key=key,
        p1=p1,
        p2=p2,
        aligned_rows=rows,
        rollout_windows=windows,
        fmt_id=fmt_id,
    ), None


@dataclass
class PairedShardAccumulator:
    """Packs paired-POV battles into K-step rollout-window shards."""

    format_names: dict[int, str] = field(default_factory=dict)
    rollout_len: int = 1
    _fmt_ids: set[int] = field(default_factory=set)

    p1_states_flat: list[np.ndarray] = field(default_factory=list)
    p1_state_offsets: list[int] = field(default_factory=lambda: [0])
    p1_state_lengths: list[int] = field(default_factory=list)
    p2_states_flat: list[np.ndarray] = field(default_factory=list)
    p2_state_offsets: list[int] = field(default_factory=lambda: [0])
    p2_state_lengths: list[int] = field(default_factory=list)

    p1_actions_flat: list[np.ndarray] = field(default_factory=list)
    p1_action_offsets: list[int] = field(default_factory=lambda: [0])
    p1_action_lengths: list[int] = field(default_factory=list)
    p1_opponent_actions_flat: list[np.ndarray] = field(default_factory=list)
    p1_opponent_action_offsets: list[int] = field(default_factory=lambda: [0])
    p1_opponent_action_lengths: list[int] = field(default_factory=list)
    p2_actions_flat: list[np.ndarray] = field(default_factory=list)
    p2_action_offsets: list[int] = field(default_factory=lambda: [0])
    p2_action_lengths: list[int] = field(default_factory=list)
    p2_opponent_actions_flat: list[np.ndarray] = field(default_factory=list)
    p2_opponent_action_offsets: list[int] = field(default_factory=lambda: [0])
    p2_opponent_action_lengths: list[int] = field(default_factory=list)
    p1_legal_action_candidates: list[list[np.ndarray]] = field(default_factory=list)
    p1_chosen_legal_action_idx_flat: list[int] = field(default_factory=list)
    p2_legal_action_candidates: list[list[np.ndarray]] = field(default_factory=list)
    p2_chosen_legal_action_idx_flat: list[int] = field(default_factory=list)

    p1_state_idx: list[np.ndarray] = field(default_factory=list)
    p1_next_state_idx: list[np.ndarray] = field(default_factory=list)
    p1_action_idx: list[np.ndarray] = field(default_factory=list)
    p1_next_terminal_class: list[np.ndarray] = field(default_factory=list)
    p2_state_idx: list[np.ndarray] = field(default_factory=list)
    p2_next_state_idx: list[np.ndarray] = field(default_factory=list)
    p2_action_idx: list[np.ndarray] = field(default_factory=list)
    battle_ids: list[np.ndarray] = field(default_factory=list)
    turn_idx: list[np.ndarray] = field(default_factory=list)
    turn_number: list[np.ndarray] = field(default_factory=list)
    subturn_idx: list[np.ndarray] = field(default_factory=list)
    format_ids: list[np.ndarray] = field(default_factory=list)

    raw_battle_keys: list[str] = field(default_factory=list)
    p1_battle_start: list[int] = field(default_factory=lambda: [0])
    p2_battle_start: list[int] = field(default_factory=lambda: [0])
    p1_battle_action_start: list[int] = field(default_factory=lambda: [0])
    p2_battle_action_start: list[int] = field(default_factory=lambda: [0])
    max_battle_len: int = 0

    # Per-battle sequence-length statistics (accumulated over shards)
    _max_state_block_len: int = 0
    _min_state_block_len: int = 2**31
    _sum_state_block_len: int = 0
    _count_state_blocks: int = 0
    _max_temporal_seq: int = 0
    _min_temporal_seq: int = 2**31
    _sum_temporal_seq: int = 0
    _count_temporal_battles: int = 0

    def __len__(self) -> int:
        return len(self.raw_battle_keys)

    @property
    def num_states(self) -> int:
        return len(self.p1_state_lengths)

    @property
    def num_actions(self) -> int:
        return len(self.p1_action_lengths)

    @staticmethod
    def _legal_actions_or_fallback(pov: TokenizedPOV) -> tuple[list[list[np.ndarray]], list[int]]:
        if (
            len(pov.legal_action_arrays) == len(pov.player_action_arrays)
            and len(pov.chosen_legal_action_idx) == len(pov.player_action_arrays)
        ):
            return pov.legal_action_arrays, pov.chosen_legal_action_idx
        return [[action] for action in pov.player_action_arrays], [0] * len(pov.player_action_arrays)

    @staticmethod
    def _terminal_classes_or_ongoing(pov: TokenizedPOV) -> list[int]:
        if len(pov.state_terminal_classes) == len(pov.state_token_arrays):
            return pov.state_terminal_classes
        return [TERMINAL_CLASS_TO_ID["ongoing"]] * len(pov.state_token_arrays)

    @staticmethod
    def _pack_legal_actions(
        candidates: list[list[np.ndarray]],
        chosen_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_legal = max((len(row) for row in candidates), default=1)
        max_tokens = max(
            (len(action) for row in candidates for action in row),
            default=1,
        )
        max_legal = max(max_legal, 1)
        max_tokens = max(max_tokens, 1)
        packed = np.full(
            (len(candidates), max_legal, max_tokens),
            _PAD_ID,
            dtype=np.int16,
        )
        mask = np.zeros((len(candidates), max_legal), dtype=np.bool_)
        chosen = np.zeros(len(candidates), dtype=np.int16)
        for action_idx, row in enumerate(candidates):
            if not row:
                row = [np.array([_UNKNOWN_ID], dtype=np.int16)]
            chosen[action_idx] = int(chosen_indices[action_idx]) if action_idx < len(chosen_indices) else 0
            chosen[action_idx] = min(max(int(chosen[action_idx]), 0), len(row) - 1)
            for legal_idx, action in enumerate(row[:max_legal]):
                tokens = np.asarray(action, dtype=np.int16)
                packed[action_idx, legal_idx, :len(tokens)] = tokens[:max_tokens]
                mask[action_idx, legal_idx] = True
        return packed, mask, chosen

    def append(self, battle: PairedBattle) -> None:
        if any(len(window) != self.rollout_len for window in battle.rollout_windows):
            raise ValueError("rollout window length does not match accumulator rollout_len")
        battle_id = len(self.raw_battle_keys)
        p1_state_base = len(self.p1_state_lengths)
        p2_state_base = len(self.p2_state_lengths)
        p1_action_base = len(self.p1_action_lengths)
        p2_action_base = len(self.p2_action_lengths)

        n_rollout_samples = len(battle.rollout_windows)

        battle_token_count = (
            sum(len(a) for a in battle.p1.state_token_arrays)
            + sum(len(a) for a in battle.p2.state_token_arrays)
            + sum(len(a) for a in battle.p1.player_action_arrays)
            + sum(len(a) for a in battle.p1.opponent_action_arrays)
            + sum(len(a) for a in battle.p2.player_action_arrays)
            + sum(len(a) for a in battle.p2.opponent_action_arrays)
        )
        self.max_battle_len = max(self.max_battle_len, battle_token_count)

        # ── per-battle sequence-length stats ───────────────────────
        for pov_state_arrays in (battle.p1.state_token_arrays, battle.p2.state_token_arrays):
            num_state_blocks = len(pov_state_arrays)  # header + all game states
            # Max state block token length across all individual blocks
            max_sl = max(len(a) for a in pov_state_arrays)
            min_sl = min(len(a) for a in pov_state_arrays)
            total_sl = sum(len(a) for a in pov_state_arrays)
            self._max_state_block_len = max(self._max_state_block_len, max_sl)
            self._min_state_block_len = min(self._min_state_block_len, min_sl)
            self._sum_state_block_len += total_sl
            self._count_state_blocks += num_state_blocks

            # Temporal interleaved sequence length:  header, state_0,
            # p_action_0, o_action_0, state_1, ... → 3*N - 2 (see
            # JEPATemporalEncoder.forward max_seq calculation).
            temporal_seq = 3 * num_state_blocks - 2
            self._max_temporal_seq = max(self._max_temporal_seq, temporal_seq)
            self._min_temporal_seq = min(self._min_temporal_seq, temporal_seq)
            self._sum_temporal_seq += temporal_seq
            self._count_temporal_battles += 1

        for tok_arr in battle.p1.state_token_arrays:
            self.p1_states_flat.append(tok_arr)
            self.p1_state_lengths.append(len(tok_arr))
            self.p1_state_offsets.append(self.p1_state_offsets[-1] + len(tok_arr))
        for tok_arr in battle.p2.state_token_arrays:
            self.p2_states_flat.append(tok_arr)
            self.p2_state_lengths.append(len(tok_arr))
            self.p2_state_offsets.append(self.p2_state_offsets[-1] + len(tok_arr))

        for tok_arr in battle.p1.player_action_arrays:
            self.p1_actions_flat.append(tok_arr)
            self.p1_action_lengths.append(len(tok_arr))
            self.p1_action_offsets.append(self.p1_action_offsets[-1] + len(tok_arr))
        for tok_arr in battle.p1.opponent_action_arrays:
            self.p1_opponent_actions_flat.append(tok_arr)
            self.p1_opponent_action_lengths.append(len(tok_arr))
            self.p1_opponent_action_offsets.append(
                self.p1_opponent_action_offsets[-1] + len(tok_arr)
            )
        for tok_arr in battle.p2.player_action_arrays:
            self.p2_actions_flat.append(tok_arr)
            self.p2_action_lengths.append(len(tok_arr))
            self.p2_action_offsets.append(self.p2_action_offsets[-1] + len(tok_arr))
        for tok_arr in battle.p2.opponent_action_arrays:
            self.p2_opponent_actions_flat.append(tok_arr)
            self.p2_opponent_action_lengths.append(len(tok_arr))
            self.p2_opponent_action_offsets.append(
                self.p2_opponent_action_offsets[-1] + len(tok_arr)
            )
        p1_legal, p1_chosen = self._legal_actions_or_fallback(battle.p1)
        p2_legal, p2_chosen = self._legal_actions_or_fallback(battle.p2)
        self.p1_legal_action_candidates.extend(p1_legal)
        self.p1_chosen_legal_action_idx_flat.extend(int(v) for v in p1_chosen)
        self.p2_legal_action_candidates.extend(p2_legal)
        self.p2_chosen_legal_action_idx_flat.extend(int(v) for v in p2_chosen)

        if n_rollout_samples > 0:
            row_ordinal = {id(row): idx for idx, row in enumerate(battle.aligned_rows)}
            p1_terminal_classes = self._terminal_classes_or_ongoing(battle.p1)

            self.p1_state_idx.append(np.array(
                [[p1_state_base + row.p1_state_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.p1_next_state_idx.append(np.array(
                [[p1_state_base + row.p1_next_state_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.p1_action_idx.append(np.array(
                [[p1_action_base + row.p1_action_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.p1_next_terminal_class.append(np.array(
                [[p1_terminal_classes[row.p1_next_state_idx] for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int16,
            ))
            self.p2_state_idx.append(np.array(
                [[p2_state_base + row.p2_state_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.p2_next_state_idx.append(np.array(
                [[p2_state_base + row.p2_next_state_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.p2_action_idx.append(np.array(
                [[p2_action_base + row.p2_action_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.battle_ids.append(np.full(n_rollout_samples, battle_id, dtype=np.int32))
            self.turn_idx.append(np.array(
                [[row_ordinal[id(row)] for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.turn_number.append(np.array(
                [[row.turn_number for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int32,
            ))
            self.subturn_idx.append(np.array(
                [[row.subturn_idx for row in window]
                 for window in battle.rollout_windows],
                dtype=np.int16,
            ))
            self.format_ids.append(np.full(
                (n_rollout_samples, self.rollout_len),
                battle.fmt_id,
                dtype=np.int16,
            ))
            self._fmt_ids.add(battle.fmt_id)

        self.raw_battle_keys.append(battle.raw_battle_key)
        self.p1_battle_start.append(
            self.p1_battle_start[-1] + len(battle.p1.state_token_arrays)
        )
        self.p2_battle_start.append(
            self.p2_battle_start[-1] + len(battle.p2.state_token_arrays)
        )
        self.p1_battle_action_start.append(
            self.p1_battle_action_start[-1] + len(battle.p1.player_action_arrays)
        )
        self.p2_battle_action_start.append(
            self.p2_battle_action_start[-1] + len(battle.p2.player_action_arrays)
        )

    def write(self, out_dir: str, shard_idx: int, rng: np.random.Generator | None = None) -> dict[str, int | float]:
        p1_states = np.concatenate(self.p1_states_flat, axis=0).astype(np.int16)
        p2_states = np.concatenate(self.p2_states_flat, axis=0).astype(np.int16)
        p1_actions = np.concatenate(self.p1_actions_flat, axis=0).astype(np.int16)
        p1_opponent_actions = np.concatenate(self.p1_opponent_actions_flat, axis=0).astype(np.int16)
        p2_actions = np.concatenate(self.p2_actions_flat, axis=0).astype(np.int16)
        p2_opponent_actions = np.concatenate(self.p2_opponent_actions_flat, axis=0).astype(np.int16)

        p1_state_idx_arr = np.concatenate(self.p1_state_idx, axis=0).astype(np.int32)
        p1_next_state_idx_arr = np.concatenate(self.p1_next_state_idx, axis=0).astype(np.int32)
        p1_action_idx_arr = np.concatenate(self.p1_action_idx, axis=0).astype(np.int32)
        p1_next_terminal_class_arr = np.concatenate(self.p1_next_terminal_class, axis=0).astype(np.int16)
        p2_state_idx_arr = np.concatenate(self.p2_state_idx, axis=0).astype(np.int32)
        p2_next_state_idx_arr = np.concatenate(self.p2_next_state_idx, axis=0).astype(np.int32)
        p2_action_idx_arr = np.concatenate(self.p2_action_idx, axis=0).astype(np.int32)
        battle_id_arr = np.concatenate(self.battle_ids, axis=0).astype(np.int32)
        turn_idx_arr = np.concatenate(self.turn_idx, axis=0).astype(np.int32)
        turn_number_arr = np.concatenate(self.turn_number, axis=0).astype(np.int32)
        subturn_idx_arr = np.concatenate(self.subturn_idx, axis=0).astype(np.int16)
        format_id_arr = np.concatenate(self.format_ids, axis=0).astype(np.int16)
        p1_legal_actions_arr, p1_legal_mask_arr, p1_chosen_legal_idx_arr = self._pack_legal_actions(
            self.p1_legal_action_candidates,
            self.p1_chosen_legal_action_idx_flat,
        )
        p2_legal_actions_arr, p2_legal_mask_arr, p2_chosen_legal_idx_arr = self._pack_legal_actions(
            self.p2_legal_action_candidates,
            self.p2_chosen_legal_action_idx_flat,
        )

        if rng is not None and len(p1_state_idx_arr) > 1:
            order = rng.permutation(len(p1_state_idx_arr))
            p1_state_idx_arr = p1_state_idx_arr[order]
            p1_next_state_idx_arr = p1_next_state_idx_arr[order]
            p1_action_idx_arr = p1_action_idx_arr[order]
            p1_next_terminal_class_arr = p1_next_terminal_class_arr[order]
            p2_state_idx_arr = p2_state_idx_arr[order]
            p2_next_state_idx_arr = p2_next_state_idx_arr[order]
            p2_action_idx_arr = p2_action_idx_arr[order]
            battle_id_arr = battle_id_arr[order]
            turn_idx_arr = turn_idx_arr[order]
            turn_number_arr = turn_number_arr[order]
            subturn_idx_arr = subturn_idx_arr[order]
            format_id_arr = format_id_arr[order]

        shard_name = f"paired_shard_{shard_idx:04d}.npz"
        shard_path = os.path.join(out_dir, shard_name)
        np.savez(
            shard_path,
            p1_states=p1_states,
            p1_state_offsets=np.array(self.p1_state_offsets[:-1], dtype=np.int64),
            p1_state_lengths=np.array(self.p1_state_lengths, dtype=np.int32),
            p2_states=p2_states,
            p2_state_offsets=np.array(self.p2_state_offsets[:-1], dtype=np.int64),
            p2_state_lengths=np.array(self.p2_state_lengths, dtype=np.int32),
            p1_actions=p1_actions,
            p1_action_offsets=np.array(self.p1_action_offsets[:-1], dtype=np.int64),
            p1_action_lengths=np.array(self.p1_action_lengths, dtype=np.int32),
            p1_opponent_actions=p1_opponent_actions,
            p1_opponent_action_offsets=np.array(self.p1_opponent_action_offsets[:-1], dtype=np.int64),
            p1_opponent_action_lengths=np.array(self.p1_opponent_action_lengths, dtype=np.int32),
            p2_actions=p2_actions,
            p2_action_offsets=np.array(self.p2_action_offsets[:-1], dtype=np.int64),
            p2_action_lengths=np.array(self.p2_action_lengths, dtype=np.int32),
            p2_opponent_actions=p2_opponent_actions,
            p2_opponent_action_offsets=np.array(self.p2_opponent_action_offsets[:-1], dtype=np.int64),
            p2_opponent_action_lengths=np.array(self.p2_opponent_action_lengths, dtype=np.int32),
            p1_target_state_idx=p1_state_idx_arr,
            p1_next_state_idx=p1_next_state_idx_arr,
            p1_action_idx=p1_action_idx_arr,
            p1_next_terminal_class=p1_next_terminal_class_arr,
            p2_target_state_idx=p2_state_idx_arr,
            p2_next_state_idx=p2_next_state_idx_arr,
            p2_action_idx=p2_action_idx_arr,
            p1_legal_actions=p1_legal_actions_arr,
            p1_legal_action_mask=p1_legal_mask_arr,
            p1_chosen_legal_action_idx=p1_chosen_legal_idx_arr,
            p2_legal_actions=p2_legal_actions_arr,
            p2_legal_action_mask=p2_legal_mask_arr,
            p2_chosen_legal_action_idx=p2_chosen_legal_idx_arr,
            battle_id=battle_id_arr,
            turn_idx=turn_idx_arr,
            turn_number=turn_number_arr,
            subturn_idx=subturn_idx_arr,
            format_id=format_id_arr,
            raw_battle_key=np.array(self.raw_battle_keys),
            p1_battle_start=np.array(self.p1_battle_start, dtype=np.int64),
            p2_battle_start=np.array(self.p2_battle_start, dtype=np.int64),
            p1_battle_action_start=np.array(self.p1_battle_action_start, dtype=np.int64),
            p2_battle_action_start=np.array(self.p2_battle_action_start, dtype=np.int64),
            format_name=np.array(
                ",".join(sorted(self.format_names.get(i, str(i)) for i in self._fmt_ids))
                if self._fmt_ids else "unknown"
            ),
            format_id_value=np.array(
                -1 if len(self._fmt_ids) != 1 else next(iter(self._fmt_ids)),
                dtype=np.int16,
            ),
            rollout_len=np.array(self.rollout_len, dtype=np.int16),
        )

        rollout_samples = len(p1_state_idx_arr)
        transition_steps = rollout_samples * self.rollout_len
        return {
            "battles": len(self),
            "states": len(self.p1_state_lengths),
            "actions": len(self.p1_action_lengths),
            "rollout_samples": rollout_samples,
            "transition_steps": transition_steps,
            "transitions": transition_steps,
            "rollout_len": self.rollout_len,
            "max_battle_len": self.max_battle_len,
            "uncompressed_mb": (
                p1_states.nbytes
                + p2_states.nbytes
                + p1_actions.nbytes
                + p1_opponent_actions.nbytes
                + p2_actions.nbytes
                + p2_opponent_actions.nbytes
            ) / (1024 * 1024),
            # Sequence-length statistics for this shard (to be aggregated)
            "max_state_block_len": self._max_state_block_len,
            "min_state_block_len": self._min_state_block_len,
            "sum_state_block_len": self._sum_state_block_len,
            "count_state_blocks": self._count_state_blocks,
            "max_temporal_seq": self._max_temporal_seq,
            "min_temporal_seq": self._min_temporal_seq,
            "sum_temporal_seq": self._sum_temporal_seq,
            "count_temporal_battles": self._count_temporal_battles,
        }


# ---------------------------------------------------------------------------
# Move histogram helpers
# ---------------------------------------------------------------------------


def _merge_move_histograms(*histograms: dict[str, int]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for histogram in histograms:
        for move, count in histogram.items():
            merged[move] += int(count)
    return dict(merged)


def _adjusted_move_count(raw_double_count: int) -> float:
    return raw_double_count / 2.0


def _format_histogram_count(count: float) -> str:
    return str(int(count)) if count.is_integer() else f"{count:.1f}"


def _move_histogram_rows(
    raw_double_counts: dict[str, int],
) -> list[dict[str, int | float | str]]:
    total = sum(_adjusted_move_count(c) for c in raw_double_counts.values())
    sorted_items = sorted(raw_double_counts.items(), key=lambda item: (-item[1], item[0]))
    rows: list[dict[str, int | float | str]] = []
    for rank, (move, raw_count) in enumerate(sorted_items, start=1):
        count = _adjusted_move_count(int(raw_count))
        rows.append(
            {
                "rank": rank,
                "move": move,
                "count": count,
                "percent": (100.0 * count / total) if total else 0.0,
                "raw_double_count": int(raw_count),
            }
        )
    return rows


def _write_move_histogram_plot(
    rows: list[dict[str, int | float | str]],
    png_path: str,
    title: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  WARNING: matplotlib is not installed; wrote move histogram "
            "tables but skipped the PNG plot."
        )
        return False

    n_rows = max(len(rows), 1)
    fig_height = max(4.5, min(120.0, 0.22 * n_rows + 1.8))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    if rows:
        moves = [str(row["move"]) for row in rows]
        counts = [float(row["count"]) for row in rows]
        y_pos = np.arange(len(rows))
        ax.barh(y_pos, counts, color="#4c78a8")
        ax.set_yticks(y_pos)
        font_size = 8 if len(rows) <= 80 else (6 if len(rows) <= 250 else 4)
        ax.set_yticklabels(moves, fontsize=font_size)
        ax.invert_yaxis()
        ax.set_xlabel("Chosen move count (two-view raw count / 2)")
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No chosen move actions found", ha="center", va="center")
        ax.set_axis_off()

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return True


def _write_move_histogram_outputs(
    output_dir: str,
    raw_double_counts: dict[str, int],
    *,
    title: str,
) -> dict[str, int | float | str | None]:
    """Write adjusted chosen-move histogram tables and a matplotlib PNG.

    The raw counter intentionally includes both ``<chosen_move>`` and
    ``<opponent_chosen_move>`` from both POV files.  That makes each actual
    move click appear twice, so the public ``count`` column is divided by 2.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = _move_histogram_rows(raw_double_counts)
    csv_path = os.path.join(output_dir, "move_histogram.csv")
    md_path = os.path.join(output_dir, "move_histogram.md")
    png_path = os.path.join(output_dir, "move_histogram.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "move", "count", "percent", "raw_double_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "move": row["move"],
                    "count": _format_histogram_count(float(row["count"])),
                    "percent": f"{float(row['percent']):.4f}",
                    "raw_double_count": row["raw_double_count"],
                }
            )

    md_lines = [
        "| rank | move | count | percent | raw_double_count |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['rank']} | {row['move']} | "
            f"{_format_histogram_count(float(row['count']))} | "
            f"{float(row['percent']):.4f}% | {row['raw_double_count']} |"
        )
    if not rows:
        md_lines.append("|  |  | 0 | 0.0000% | 0 |")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    plot_written = _write_move_histogram_plot(rows, png_path, title)
    total_adjusted = sum(_adjusted_move_count(c) for c in raw_double_counts.values())
    print(f"  Move histogram table → {md_path}")
    print(f"  Move histogram CSV   → {csv_path}")
    if plot_written:
        print(f"  Move histogram plot  → {png_path}")
    if rows:
        print("  Top chosen moves (count = raw two-view count / 2):")
        for row in rows[:20]:
            print(
                f"    {row['rank']:>3}. {row['move']}: "
                f"{_format_histogram_count(float(row['count']))}"
            )
    return {
        "table_path": md_path,
        "csv_path": csv_path,
        "plot_path": png_path if plot_written else None,
        "num_unique_moves": len(rows),
        "total_move_actions": total_adjusted,
        "raw_double_count_total": int(sum(raw_double_counts.values())),
        "count_note": "count = raw two-view chosen-move count / 2",
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

    Uses the replay id prefix (e.g. ``gen1ou-123`` or
    ``smogtours-gen1ou-123``), so reversed POV filenames map to one battle.
    """
    stem = Path(path).stem
    match = _BATTLE_ID_RE.match(stem)
    if match:
        return match.group(1)
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


def iter_tokenized_battle_pairs(
    groups: dict[str, list[str]],
    keys: list[str],
    tokenizer_path: str,
    rollout_len: int,
    processes: int,
    desc: str,
    fmt_ids: dict[str, int] | None = None,
) -> Iterable[tuple[str, PairedBattle | None, str | None]]:
    """Multiprocess tokenize paired POV groups.

    Yields ``(raw_battle_key, paired_battle_or_none, skip_reason_or_none)``.
    """
    _fmt_ids = fmt_ids or {}
    work = [(key, groups[key], tokenizer_path, rollout_len, _fmt_ids.get(key, 0)) for key in keys]
    if processes > 1:
        with Pool(
            processes, initializer=_init_worker, initargs=(tokenizer_path,)
        ) as pool:
            for key, result in zip(
                keys,
                tqdm.tqdm(
                    pool.imap(tokenize_battle_pair, work, chunksize=50),
                    total=len(work),
                    desc=desc,
                ),
            ):
                battle, reason = result
                yield key, battle, reason
    else:
        _init_worker(tokenizer_path)
        for item in tqdm.tqdm(work, desc=desc):
            battle, reason = tokenize_battle_pair(item)
            yield item[0], battle, reason


def write_paired_split_shards(
    *,
    split_name: str,
    groups: dict[str, list[str]],
    group_keys: list[str],
    tokenizer_path: str,
    processes: int,
    battles_per_shard: int,
    rollout_len: int,
    out_dir: str,
    rng: np.random.Generator,
    format_names: dict[int, str],
    fmt_ids: dict[str, int],
) -> tuple[dict, int, int]:
    """Tokenize and write paired-POV shards for one split."""
    os.makedirs(out_dir, exist_ok=True)

    totals: dict[str, int | dict[str, int] | dict[str, list[str]]] = {
        "num_raw_battle_groups": len(group_keys),
        "num_battles": 0,
        "num_shards": 0,
        "total_states": 0,
        "total_actions": 0,
        "total_transitions": 0,
        "total_rollout_samples": 0,
        "failed": 0,
        "skip_reasons": {},
        "skip_keys": {},  # reason → list of raw_battle_key
    }
    # Aggregate sequence-length stats across shards
    seq_agg = {
        "max_state_block_len": 0,
        "min_state_block_len": 2**31,
        "sum_state_block_len": 0,
        "count_state_blocks": 0,
        "max_temporal_seq": 0,
        "min_temporal_seq": 2**31,
        "sum_temporal_seq": 0,
        "count_temporal_battles": 0,
    }
    move_counts: Counter[str] = Counter()

    def _aggregate_seq(stats: dict, agg: dict) -> None:
        for k in seq_agg:
            if k.startswith("max_"):
                agg[k] = max(agg[k], int(stats[k]))
            elif k.startswith("min_"):
                agg[k] = min(agg[k], int(stats[k]))
            elif k.startswith("sum_") or k.startswith("count_"):
                agg[k] += int(stats[k])

    shard_idx = 0
    max_battle_len = 0
    acc = PairedShardAccumulator(format_names=format_names, rollout_len=rollout_len)

    for key, battle, reason in iter_tokenized_battle_pairs(
        groups,
        group_keys,
        tokenizer_path,
        rollout_len,
        processes,
        desc=f"  Pair-tokenizing {split_name}",
        fmt_ids=fmt_ids,
    ):
        if battle is None:
            totals["failed"] = int(totals["failed"]) + 1
            skip_reasons = totals["skip_reasons"]
            assert isinstance(skip_reasons, dict)
            full_reason = reason or "unknown"
            # Split reason::detail for aggregation
            if "::" in full_reason:
                simple_reason, detail = full_reason.split("::", 1)
            else:
                simple_reason, detail = full_reason, ""
            skip_reasons[simple_reason] = skip_reasons.get(simple_reason, 0) + 1
            skip_keys = totals["skip_keys"]
            assert isinstance(skip_keys, dict)
            skip_keys.setdefault(simple_reason, []).append((key, detail))
            continue

        move_counts.update(battle.p1.move_counts)
        move_counts.update(battle.p2.move_counts)
        acc.append(battle)
        if len(acc) >= battles_per_shard:
            stats = acc.write(out_dir, shard_idx, rng=rng)
            totals["num_battles"] = int(totals["num_battles"]) + int(stats["battles"])
            totals["total_states"] = int(totals["total_states"]) + int(stats["states"])
            totals["total_actions"] = int(totals["total_actions"]) + int(stats["actions"])
            totals["total_transitions"] = int(totals["total_transitions"]) + int(stats["transitions"])
            totals["total_rollout_samples"] = int(totals["total_rollout_samples"]) + int(stats["rollout_samples"])
            max_battle_len = max(max_battle_len, int(stats["max_battle_len"]))
            _aggregate_seq(stats, seq_agg)
            print(
                f"  {split_name} paired shard {shard_idx:04d}: "
                f"{stats['battles']} battle pairs, {stats['states']} states/side, "
                f"{stats['rollout_samples']} rollout samples x K={rollout_len} "
                f"({stats['transitions']} transition steps), "
                f"max battle {stats['max_battle_len']} tok, "
                f"{stats['uncompressed_mb']:.0f} MB"
            )
            shard_idx += 1
            acc = PairedShardAccumulator(format_names=format_names, rollout_len=rollout_len)

    if len(acc) > 0:
        stats = acc.write(out_dir, shard_idx, rng=rng)
        totals["num_battles"] = int(totals["num_battles"]) + int(stats["battles"])
        totals["total_states"] = int(totals["total_states"]) + int(stats["states"])
        totals["total_actions"] = int(totals["total_actions"]) + int(stats["actions"])
        totals["total_transitions"] = int(totals["total_transitions"]) + int(stats["transitions"])
        totals["total_rollout_samples"] = int(totals["total_rollout_samples"]) + int(stats["rollout_samples"])
        max_battle_len = max(max_battle_len, int(stats["max_battle_len"]))
        _aggregate_seq(stats, seq_agg)
        print(
            f"  {split_name} paired shard {shard_idx:04d}: "
            f"{stats['battles']} battle pairs, {stats['states']} states/side, "
            f"{stats['rollout_samples']} rollout samples x K={rollout_len} "
            f"({stats['transitions']} transition steps), "
            f"max battle {stats['max_battle_len']} tok, "
            f"{stats['uncompressed_mb']:.0f} MB"
        )
        shard_idx += 1

    totals["num_shards"] = shard_idx
    totals["seq_agg"] = seq_agg
    totals["move_histogram_raw_double_count"] = dict(sorted(move_counts.items()))
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
    parser.add_argument(
        "--rollout_len",
        type=int,
        default=1,
        help="Number of contiguous aligned transitions per experience-replay sample.",
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_groups",
        type=int,
        default=0,
        help="Optional cap on raw battle groups per format, useful for smoke data generation.",
    )
    args = parser.parse_args()
    if args.rollout_len < 1:
        raise ValueError("--rollout_len must be >= 1")

    # Load tokenizer to verify it has required tokens
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens")
    for tok in ["<bos>", "<eos>", "unknown"]:
        tid = tokenizer[tok]
        if tid == tokenizer.unknown_token_id:
            raise ValueError(f"Tokenizer must contain '{tok}' token")
    print("Required world-model tokens present ✓")

    format_id_map = {fmt: i for i, fmt in enumerate(args.formats)}
    format_names = {v: k for k, v in format_id_map.items()}

    # ───────────────────────────────────────────────────────────────
    # Multi-format interleaved mode: gather battles from all formats
    # into a single pool, shuffle, split train/val once, then write
    # shards that mix formats together so each .npz contains battles
    # from every format.
    # ───────────────────────────────────────────────────────────────
    all_battle_groups: list[tuple[str, str, list[str], int]] = (
        []
    )  # (fmt_name, key, paths, fmt_id)
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
        incomplete = {k for k, v in groups.items() if len(v) < 2}
        if incomplete:
            for k in incomplete:
                del groups[k]
            print(
                f"  [{fmt}] Dropped {len(incomplete)} battles"
                f" with only one POV file"
            )
        for key, paths in groups.items():
            all_battle_groups.append(
                (fmt, key, paths, format_id_map[fmt])
            )

    if not all_battle_groups:
        print("No valid battle pairs found across any format. Exiting.")
        return

    print(
        f"\nCombined pool: {len(all_battle_groups)} battle pairs"
        f" from {len(args.formats)} formats"
    )

    # ── shuffle combined pool ──────────────────────────────────
    pool_rng = np.random.default_rng(args.seed)
    indices = pool_rng.permutation(len(all_battle_groups))
    all_battle_groups = [all_battle_groups[i] for i in indices]

    # ── train / val split ──────────────────────────────────────
    n_val = 0
    if args.val_split > 0:
        n_val = max(1, int(round(len(all_battle_groups) * args.val_split)))
        if len(all_battle_groups) > 1:
            n_val = min(n_val, len(all_battle_groups) - 1)

    # ── apply max_groups cap ───────────────────────────────────
    if args.max_groups > 0:
        n_total = min(len(all_battle_groups), args.max_groups)
        if args.val_split > 0 and n_total > 1:
            n_val = max(1, int(round(n_total * args.val_split)))
            n_val = min(n_val, n_total - 1)
        else:
            n_val = 0
        all_battle_groups = all_battle_groups[:n_total]

    val_groups_list = all_battle_groups[:n_val]
    train_groups_list = all_battle_groups[n_val:]

    print(
        f"  Split: {len(train_groups_list)} train /"
        f" {len(val_groups_list)} val battle pairs"
    )

    def _build_split(battle_list):
        groups_dict = {g[1]: g[2] for g in battle_list}
        fmt_ids_dict = {g[1]: g[3] for g in battle_list}
        keys = [g[1] for g in battle_list]
        return groups_dict, fmt_ids_dict, keys

    # ── shuffle keys again per split (decorrelates within shard) ─
    shard_rng = np.random.default_rng(args.seed + 1009)

    train_groups_dict, train_fmt_ids, train_keys = _build_split(
        train_groups_list
    )
    shard_rng.shuffle(train_keys)
    val_groups_dict, val_fmt_ids, val_keys = _build_split(val_groups_list)
    shard_rng.shuffle(val_keys)

    # ── write train shards ─────────────────────────────────────
    train_totals, train_btl_max, train_shard_count = (
        write_paired_split_shards(
            split_name="train",
            groups=train_groups_dict,
            group_keys=train_keys,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            battles_per_shard=args.battles_per_shard,
            rollout_len=args.rollout_len,
            out_dir=os.path.join(args.output_dir, "train"),
            rng=shard_rng,
            format_names=format_names,
            fmt_ids=train_fmt_ids,
        )
    )

    # ── write val shards ───────────────────────────────────────
    val_totals, val_btl_max, val_shard_count = (
        write_paired_split_shards(
            split_name="val",
            groups=val_groups_dict,
            group_keys=val_keys,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            battles_per_shard=args.battles_per_shard,
            rollout_len=args.rollout_len,
            out_dir=os.path.join(args.output_dir, "val"),
            rng=shard_rng,
            format_names=format_names,
            fmt_ids=val_fmt_ids,
        )
    )

    fmt_battle_max = max(train_btl_max, val_btl_max)

    failed = train_totals["failed"] + val_totals["failed"]
    total_battles = (
        train_totals["num_battles"] + val_totals["num_battles"]
    )
    total_states = (
        train_totals["total_states"] + val_totals["total_states"]
    )
    total_transitions = (
        train_totals["total_transitions"]
        + val_totals["total_transitions"]
    )
    total_rollout_samples = (
        train_totals["total_rollout_samples"]
        + val_totals["total_rollout_samples"]
    )
    total_shards = train_shard_count + val_shard_count

    if failed:
        print(f"  {failed} files failed to tokenize")
        for split_label, split_totals in (
            ("train", train_totals),
            ("val", val_totals),
        ):
            skip_reasons = split_totals.get("skip_reasons", {})
            skip_keys = split_totals.get("skip_keys", {})
            if not skip_reasons:
                continue
            print(
                f"  WARNING: {split_label} skipped battle"
                f" groups:"
            )
            for reason, count in sorted(skip_reasons.items()):
                entries = (skip_keys or {}).get(reason, [])
                # entries is list of (key, detail) tuples
                example_keys = [str(e[0]) for e in entries[:5]]
                preview = ", ".join(example_keys)
                more = f" (+{len(entries) - 5} more)" if len(entries) > 5 else ""
                detail_note = ""
                if entries and entries[0][1]:
                    detail_note = f"  [{entries[0][1]}]"
                print(f"    {reason}: {count}  e.g. [{preview}{more}]{detail_note}")

        # ── write full skip log ─────────────────────────────────
        skip_log_path = os.path.join(args.output_dir, "skip_log.txt")
        with open(skip_log_path, "w") as slf:
            slf.write("Skipped battle groups by reason\n")
            slf.write("=" * 60 + "\n\n")
            for split_label, split_totals in (
                ("train", train_totals),
                ("val", val_totals),
            ):
                skip_keys = split_totals.get("skip_keys", {})
                if not skip_keys:
                    continue
                slf.write(f"[{split_label}]\n")
                for reason, entries in sorted(skip_keys.items()):
                    count = len(entries)
                    slf.write(f"  {reason}: {count} battles\n")
                    for key, detail in entries:
                        detail_str = f"  [{detail}]" if detail else ""
                        slf.write(f"    {key}{detail_str}\n")
                    slf.write("\n")
        print(f"  Full skip log → {skip_log_path}")

    move_histogram = _write_move_histogram_outputs(
        args.output_dir,
        _merge_move_histograms(
            train_totals.get("move_histogram_raw_double_count", {}),
            val_totals.get("move_histogram_raw_double_count", {}),
        ),
        title=f"Chosen Move Histogram ({', '.join(sorted(args.formats))})",
    )

    # ── metadata ───────────────────────────────────────────────
    tokenizer_version = os.path.splitext(
        os.path.basename(args.tokenizer_path)
    )[0]
    metadata = {
        "schema_version": "paired_pov_rollout_v3",
        "tokenizer_version": tokenizer_version,
        "format": ",".join(sorted(args.formats)),
        "formats": sorted(args.formats),
        "format_id_map": format_id_map,
        "split_mode": "raw_battle_group_interleaved",
        "paired_pov": True,
        "rollout_len": args.rollout_len,
        "seed": args.seed,
        "val_split": args.val_split,
        "max_groups": args.max_groups,
        "num_raw_battle_groups": len(all_battle_groups),
        "train_raw_battle_groups": len(train_groups_list),
        "val_raw_battle_groups": len(val_groups_list),
        "num_battles": total_battles,
        "num_shards": total_shards,
        "train_num_battles": train_totals["num_battles"],
        "val_num_battles": val_totals["num_battles"],
        "train_num_shards": train_shard_count,
        "val_num_shards": val_shard_count,
        "train_rollout_samples": train_totals[
            "total_rollout_samples"
        ],
        "val_rollout_samples": val_totals["total_rollout_samples"],
        "battles_per_shard": args.battles_per_shard,
        "total_states": total_states,
        "total_transitions": total_transitions,
        "total_transition_steps": total_transitions,
        "total_rollout_samples": total_rollout_samples,
        "max_battle_len": fmt_battle_max,
        "storage": "paired_pov_rollout_indexed_variable_length_v3",
        "compressed": False,
        "train_skip_reasons": train_totals.get("skip_reasons", {}),
        "val_skip_reasons": val_totals.get("skip_reasons", {}),
        "move_histogram": move_histogram,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote metadata → {meta_path}")

    # ── aggregate sequence-length stats across train + val ─────
    def _agg(a: dict, b: dict, key: str, how: str) -> int:
        va = int(a.get(key, 0))
        vb = int(b.get(key, 0))
        if how == "max":
            return max(va, vb)
        if how == "min":
            return (
                min(va, vb) if va and vb else (va or vb)
            )
        return va + vb  # sum / count

    train_seq = train_totals.get("seq_agg", {})
    val_seq = val_totals.get("seq_agg", {})
    combined_seq: dict[str, int] = {}
    for k in [
        "max_state_block_len",
        "min_state_block_len",
        "sum_state_block_len",
        "count_state_blocks",
        "max_temporal_seq",
        "min_temporal_seq",
        "sum_temporal_seq",
        "count_temporal_battles",
    ]:
        how = (
            "max"
            if k.startswith("max_")
            else ("min" if k.startswith("min_") else "sum")
        )
        combined_seq[k] = _agg(train_seq, val_seq, k, how)

    count_blocks = combined_seq["count_state_blocks"]
    avg_state_block_len = (
        combined_seq["sum_state_block_len"] / count_blocks
        if count_blocks
        else 0.0
    )
    count_temporal = combined_seq["count_temporal_battles"]
    avg_temporal_seq = (
        combined_seq["sum_temporal_seq"] / count_temporal
        if count_temporal
        else 0.0
    )

    SAFETY_MULTIPLIER = 1.2
    state_block_raw_max = combined_seq["max_state_block_len"]
    temporal_raw_max = combined_seq["max_temporal_seq"]
    seq_stats = {
        "safety_multiplier": SAFETY_MULTIPLIER,
        "state_block_len": {
            "max_raw": state_block_raw_max,
            "max": int(
                math.floor(state_block_raw_max * SAFETY_MULTIPLIER)
            ),
            "min": combined_seq["min_state_block_len"],
            "avg": round(avg_state_block_len, 2),
            "count": count_blocks,
        },
        "temporal_sequence_len": {
            "max_raw": temporal_raw_max,
            "max": int(
                math.floor(temporal_raw_max * SAFETY_MULTIPLIER)
            ),
            "min": combined_seq["min_temporal_seq"],
            "avg": round(avg_temporal_seq, 2),
            "count": count_temporal,
            "note": (
                "Interleaved blocks per POV history: 1 (header) +"
                " 3*(T) + 1 (current state). "
                "max = 3*num_state_blocks - 2 (matches"
                " JEPATemporalEncoder.forward max_seq)."
            ),
        },
    }
    seq_path = os.path.join(args.output_dir, "sequence_stats.json")
    with open(seq_path, "w") as f:
        json.dump(seq_stats, f, indent=2)
    print(f"  Sequence stats → {seq_path}")
    print(
        f"    State block len (tokens): "
        f"min={seq_stats['state_block_len']['min']}, "
        f"avg={seq_stats['state_block_len']['avg']:.1f}, "
        f"max={seq_stats['state_block_len']['max']}"
    )
    print(
        f"    Temporal seq (blocks):   "
        f"min={seq_stats['temporal_sequence_len']['min']}, "
        f"avg={seq_stats['temporal_sequence_len']['avg']:.1f}, "
        f"max={seq_stats['temporal_sequence_len']['max']}"
    )

    print(
        f"  Total: {total_battles} battles, {total_states} states, "
        f"{total_rollout_samples} rollout samples"
        f" x K={args.rollout_len} "
        f"({total_transitions} transition steps),"
        f" {total_shards} shards, "
        f"max battle {fmt_battle_max} tokens"
    )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
