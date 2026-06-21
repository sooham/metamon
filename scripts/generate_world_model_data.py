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
    p1_legal_actions / p2_legal_actions    (num_actions, C, T)    int16
    p1_legal_action_mask / p2_legal_action_mask (num_actions, C)  bool
    p1_chosen_legal_action_idx / p2_chosen_legal_action_idx       int16

    p1_state_idx / p2_state_idx            (num_samples, K) int32
    p1_next_state_idx / p2_next_state_idx  (num_samples, K) int32
    p1_action_idx / p2_action_idx          (num_samples, K) int32

    battle_id                              (num_samples,)    int32
    turn_idx / turn_number / subturn_idx   (num_samples, K) int32/int16
    format_id                              (num_samples, K) int16
    rollout_len                            scalar int16

    battle_start            (num_battles+1,)   int64  — cumulative local state index
    battle_action_start     (num_battles+1,)   int64  — cumulative local action index
    p1_won / p2_won         (num_battles,)     bool
    raw_battle_key          (num_battles,)     object — battle ID strings

    format_name             scalar unicode            — battle format string
    format_id_value         scalar int16              — id for format_name

**Rollout indexing:**
    Each sample row contains K contiguous aligned transitions from one battle.
    Step j in a row points to state[t+j], action[t+j], and state[t+j+1].
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
import json
import math
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
    r"<terminal>\s*(won|lost|forfeit|tie)\s*<end_terminal>", re.DOTALL
)
_BATTLE_ID_RE = re.compile(r"^((?:smogtours-)?[A-Za-z0-9]+-\d+)")
_MOVE_ENTRY_RE = re.compile(r"<move>\s*([^\s<>]+)(?:\s+.*?)?<end_move>", re.DOTALL)
_YOU_RE = re.compile(r"<you>(.*?)<end_you>", re.DOTALL)
_BENCH_ENTRY_RE = re.compile(r"<poke\d+>\s*(.*?)\s*<end_poke\d+>", re.DOTALL)
_FAINTED_STATUSES = {"fnt"}


@dataclass
class TokenizedPOV:
    state_token_arrays: list[np.ndarray]  # header + states
    player_action_arrays: list[np.ndarray]
    opponent_action_arrays: list[np.ndarray]
    turn_numbers: list[int]
    won: bool
    path: str
    rank_valid: bool = True
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
    return f"move {parts[0]}"


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

    # Header is stored as "state -1" (always the first element)
    header_tokens = _tokenize_text_block(header_text)
    state_token_arrays.append(header_tokens)

    for start, end in zip(bos_positions, eos_positions):
        state_text = text[start:end]
        state_texts.append(state_text)
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
    player_action_texts: list[str] = []

    for start, end in zip(boa_positions, eoa_positions):
        action_block = text[start:end]

        # Player action
        cm_match = _CHOSEN_MOVE_RE.search(action_block)
        if cm_match:
            chosen_text = _canonical_action_text(cm_match.group(1))
            player_action_arrays.append(_tokenize_action_text(chosen_text))
            player_action_texts.append(chosen_text)
        else:
            chosen_text = "unknown unknown"
            player_action_arrays.append(_tokenize_action_text(chosen_text))
            player_action_texts.append(chosen_text)

        # Opponent action
        om_match = _OPPONENT_CHOSEN_MOVE_RE.search(action_block)
        if om_match:
            opponent_action_arrays.append(_tokenize_action_text(om_match.group(1)))
        else:
            opponent_action_arrays.append(np.array([_UNKNOWN_ID], dtype=np.int16))

    legal_action_arrays: list[list[np.ndarray]] = []
    chosen_legal_action_idx: list[int] = []
    for state_text, chosen_text in zip(state_texts, player_action_texts):
        legal_texts, chosen_idx = _legal_action_texts_from_state(state_text, chosen_text)
        legal_action_arrays.append([_tokenize_action_text(action) for action in legal_texts])
        chosen_legal_action_idx.append(chosen_idx)

    # ── extract terminal outcome ───────────────────────────────────
    terminal_match = _TERMINAL_RE.search(text)
    terminal = terminal_match.group(1) if terminal_match is not None else None
    won = terminal in {"won", "forfeit"}
    rank_valid = terminal in {"won", "lost", "forfeit"}

    return TokenizedPOV(
        state_token_arrays=state_token_arrays,
        player_action_arrays=player_action_arrays,
        opponent_action_arrays=opponent_action_arrays,
        turn_numbers=turn_numbers,
        won=won,
        path=filepath,
        rank_valid=rank_valid,
        legal_action_arrays=legal_action_arrays,
        chosen_legal_action_idx=chosen_legal_action_idx,
    )


def _is_unknown_action(tokens: np.ndarray) -> bool:
    return len(tokens) == 0 or (len(tokens) == 1 and int(tokens[0]) == _UNKNOWN_ID)


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
    if not p1.legal_action_arrays:
        p1.legal_action_arrays = [[action] for action in p1.player_action_arrays]
        p1.chosen_legal_action_idx = [0] * n_actions_1
    if not p2.legal_action_arrays:
        p2.legal_action_arrays = [[action] for action in p2.player_action_arrays]
        p2.chosen_legal_action_idx = [0] * n_actions_2
    if len(p1.legal_action_arrays) != n_actions_1 or len(p1.chosen_legal_action_idx) != n_actions_1:
        return "p1_legal_action_count_mismatch", [], []
    if len(p2.legal_action_arrays) != n_actions_2 or len(p2.chosen_legal_action_idx) != n_actions_2:
        return "p2_legal_action_count_mismatch", [], []

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
        return "no_aligned_transitions", [], []

    windows = _contiguous_rollout_windows(rows, rollout_len)
    if not windows:
        return f"no_contiguous_rollout_windows_{rollout_len}", rows, []

    return None, rows, windows


def tokenize_battle_pair(args: tuple[str, list[str], str, int]) -> tuple[PairedBattle | None, str | None]:
    """Tokenize and validate one raw-battle group containing both POV files."""
    key, paths, tokenizer_path, rollout_len = args
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
    ), None


@dataclass
class PairedShardAccumulator:
    """Packs paired-POV battles into K-step rollout-window shards."""

    fmt: str
    fmt_id: int
    rollout_len: int = 1

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
    p1_legal_action_arrays: list[list[np.ndarray]] = field(default_factory=list)
    p1_chosen_legal_action_idx_all: list[int] = field(default_factory=list)
    p2_actions_flat: list[np.ndarray] = field(default_factory=list)
    p2_action_offsets: list[int] = field(default_factory=lambda: [0])
    p2_action_lengths: list[int] = field(default_factory=list)
    p2_opponent_actions_flat: list[np.ndarray] = field(default_factory=list)
    p2_opponent_action_offsets: list[int] = field(default_factory=lambda: [0])
    p2_opponent_action_lengths: list[int] = field(default_factory=list)
    p2_legal_action_arrays: list[list[np.ndarray]] = field(default_factory=list)
    p2_chosen_legal_action_idx_all: list[int] = field(default_factory=list)

    p1_state_idx: list[np.ndarray] = field(default_factory=list)
    p1_next_state_idx: list[np.ndarray] = field(default_factory=list)
    p1_action_idx: list[np.ndarray] = field(default_factory=list)
    p2_state_idx: list[np.ndarray] = field(default_factory=list)
    p2_next_state_idx: list[np.ndarray] = field(default_factory=list)
    p2_action_idx: list[np.ndarray] = field(default_factory=list)
    battle_ids: list[np.ndarray] = field(default_factory=list)
    turn_idx: list[np.ndarray] = field(default_factory=list)
    turn_number: list[np.ndarray] = field(default_factory=list)
    subturn_idx: list[np.ndarray] = field(default_factory=list)
    format_ids: list[np.ndarray] = field(default_factory=list)

    p1_won: list[bool] = field(default_factory=list)
    p2_won: list[bool] = field(default_factory=list)
    rank_valid: list[bool] = field(default_factory=list)
    raw_battle_keys: list[str] = field(default_factory=list)
    battle_start: list[int] = field(default_factory=lambda: [0])
    battle_action_start: list[int] = field(default_factory=lambda: [0])
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
        p1_legal = battle.p1.legal_action_arrays or [[a] for a in battle.p1.player_action_arrays]
        p1_chosen = battle.p1.chosen_legal_action_idx or [0] * len(p1_legal)
        self.p1_legal_action_arrays.extend(p1_legal)
        self.p1_chosen_legal_action_idx_all.extend(int(idx) for idx in p1_chosen)
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
        p2_legal = battle.p2.legal_action_arrays or [[a] for a in battle.p2.player_action_arrays]
        p2_chosen = battle.p2.chosen_legal_action_idx or [0] * len(p2_legal)
        self.p2_legal_action_arrays.extend(p2_legal)
        self.p2_chosen_legal_action_idx_all.extend(int(idx) for idx in p2_chosen)

        if n_rollout_samples > 0:
            row_ordinal = {id(row): idx for idx, row in enumerate(battle.aligned_rows)}

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
                self.fmt_id,
                dtype=np.int16,
            ))

        self.p1_won.append(battle.p1.won)
        self.p2_won.append(battle.p2.won)
        self.rank_valid.append(
            battle.p1.rank_valid and battle.p2.rank_valid and (battle.p1.won != battle.p2.won)
        )
        self.raw_battle_keys.append(battle.raw_battle_key)
        self.battle_start.append(self.battle_start[-1] + len(battle.p1.state_token_arrays))
        self.battle_action_start.append(
            self.battle_action_start[-1] + len(battle.p1.player_action_arrays)
        )
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

        def pack_legal(
            legal_actions: list[list[np.ndarray]],
            chosen_indices: list[int],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            max_candidates = max((len(actions) for actions in legal_actions), default=1)
            max_tokens = max(
                (len(action) for actions in legal_actions for action in actions),
                default=1,
            )
            dense = np.full(
                (len(legal_actions), max_candidates, max_tokens),
                _PAD_ID,
                dtype=np.int16,
            )
            mask = np.zeros((len(legal_actions), max_candidates), dtype=bool)
            chosen = np.zeros(len(legal_actions), dtype=np.int16)
            for action_idx, actions in enumerate(legal_actions):
                for candidate_idx, action in enumerate(actions):
                    dense[action_idx, candidate_idx, :len(action)] = action.astype(np.int16, copy=False)
                    mask[action_idx, candidate_idx] = True
                if actions:
                    raw_idx = chosen_indices[action_idx] if action_idx < len(chosen_indices) else 0
                    chosen[action_idx] = np.int16(min(max(int(raw_idx), 0), len(actions) - 1))
            return dense, mask, chosen

        p1_legal_actions, p1_legal_action_mask, p1_chosen_legal_action_idx = pack_legal(
            self.p1_legal_action_arrays,
            self.p1_chosen_legal_action_idx_all,
        )
        p2_legal_actions, p2_legal_action_mask, p2_chosen_legal_action_idx = pack_legal(
            self.p2_legal_action_arrays,
            self.p2_chosen_legal_action_idx_all,
        )

        p1_state_idx_arr = np.concatenate(self.p1_state_idx, axis=0).astype(np.int32)
        p1_next_state_idx_arr = np.concatenate(self.p1_next_state_idx, axis=0).astype(np.int32)
        p1_action_idx_arr = np.concatenate(self.p1_action_idx, axis=0).astype(np.int32)
        p2_state_idx_arr = np.concatenate(self.p2_state_idx, axis=0).astype(np.int32)
        p2_next_state_idx_arr = np.concatenate(self.p2_next_state_idx, axis=0).astype(np.int32)
        p2_action_idx_arr = np.concatenate(self.p2_action_idx, axis=0).astype(np.int32)
        battle_id_arr = np.concatenate(self.battle_ids, axis=0).astype(np.int32)
        turn_idx_arr = np.concatenate(self.turn_idx, axis=0).astype(np.int32)
        turn_number_arr = np.concatenate(self.turn_number, axis=0).astype(np.int32)
        subturn_idx_arr = np.concatenate(self.subturn_idx, axis=0).astype(np.int16)
        format_id_arr = np.concatenate(self.format_ids, axis=0).astype(np.int16)

        if rng is not None and len(p1_state_idx_arr) > 1:
            order = rng.permutation(len(p1_state_idx_arr))
            p1_state_idx_arr = p1_state_idx_arr[order]
            p1_next_state_idx_arr = p1_next_state_idx_arr[order]
            p1_action_idx_arr = p1_action_idx_arr[order]
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
            p1_legal_actions=p1_legal_actions,
            p1_legal_action_mask=p1_legal_action_mask,
            p1_chosen_legal_action_idx=p1_chosen_legal_action_idx,
            p2_actions=p2_actions,
            p2_action_offsets=np.array(self.p2_action_offsets[:-1], dtype=np.int64),
            p2_action_lengths=np.array(self.p2_action_lengths, dtype=np.int32),
            p2_opponent_actions=p2_opponent_actions,
            p2_opponent_action_offsets=np.array(self.p2_opponent_action_offsets[:-1], dtype=np.int64),
            p2_opponent_action_lengths=np.array(self.p2_opponent_action_lengths, dtype=np.int32),
            p2_legal_actions=p2_legal_actions,
            p2_legal_action_mask=p2_legal_action_mask,
            p2_chosen_legal_action_idx=p2_chosen_legal_action_idx,
            p1_state_idx=p1_state_idx_arr,
            p1_next_state_idx=p1_next_state_idx_arr,
            p1_action_idx=p1_action_idx_arr,
            p2_state_idx=p2_state_idx_arr,
            p2_next_state_idx=p2_next_state_idx_arr,
            p2_action_idx=p2_action_idx_arr,
            state_idx=p1_state_idx_arr,
            next_state_idx=p1_next_state_idx_arr,
            action_idx=p1_action_idx_arr,
            battle_id=battle_id_arr,
            turn_idx=turn_idx_arr,
            turn_number=turn_number_arr,
            subturn_idx=subturn_idx_arr,
            format_id=format_id_arr,
            p1_won=np.array(self.p1_won, dtype=bool),
            p2_won=np.array(self.p2_won, dtype=bool),
            rank_valid=np.array(self.rank_valid, dtype=bool),
            raw_battle_key=np.array(self.raw_battle_keys),
            battle_start=np.array(self.battle_start, dtype=np.int64),
            battle_action_start=np.array(self.battle_action_start, dtype=np.int64),
            p1_battle_start=np.array(self.p1_battle_start, dtype=np.int64),
            p2_battle_start=np.array(self.p2_battle_start, dtype=np.int64),
            p1_battle_action_start=np.array(self.p1_battle_action_start, dtype=np.int64),
            p2_battle_action_start=np.array(self.p2_battle_action_start, dtype=np.int64),
            format_name=np.array(self.fmt),
            format_id_value=np.array(self.fmt_id, dtype=np.int16),
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
) -> Iterable[tuple[str, PairedBattle | None, str | None]]:
    """Multiprocess tokenize paired POV groups.

    Yields ``(raw_battle_key, paired_battle_or_none, skip_reason_or_none)``.
    """
    work = [(key, groups[key], tokenizer_path, rollout_len) for key in keys]
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
    fmt: str,
    fmt_id: int,
    split_name: str,
    groups: dict[str, list[str]],
    group_keys: list[str],
    tokenizer_path: str,
    processes: int,
    battles_per_shard: int,
    rollout_len: int,
    out_dir: str,
    rng: np.random.Generator,
) -> tuple[dict, int, int]:
    """Tokenize and write paired-POV shards for one split."""
    os.makedirs(out_dir, exist_ok=True)

    totals: dict[str, int | dict[str, int]] = {
        "num_raw_battle_groups": len(group_keys),
        "num_battles": 0,
        "num_shards": 0,
        "total_states": 0,
        "total_actions": 0,
        "total_transitions": 0,
        "total_rollout_samples": 0,
        "failed": 0,
        "skip_reasons": {},
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
    acc = PairedShardAccumulator(fmt=fmt, fmt_id=fmt_id, rollout_len=rollout_len)

    for _, battle, reason in iter_tokenized_battle_pairs(
        groups,
        group_keys,
        tokenizer_path,
        rollout_len,
        processes,
        desc=f"  Pair-tokenizing {fmt}/{split_name}",
    ):
        if battle is None:
            totals["failed"] = int(totals["failed"]) + 1
            skip_reasons = totals["skip_reasons"]
            assert isinstance(skip_reasons, dict)
            reason = reason or "unknown"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

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
            acc = PairedShardAccumulator(fmt=fmt, fmt_id=fmt_id, rollout_len=rollout_len)

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
        # Drop battles with only one POV — paired training needs both sides.
        incomplete = {k for k, v in groups.items() if len(v) < 2}
        if incomplete:
            for k in incomplete:
                del groups[k]
            print(f"  Dropped {len(incomplete)} battles with only one POV file")

        split_rng = np.random.default_rng(args.seed + format_id_map[fmt])
        train_keys, val_keys, train_files, val_files = split_groups(
            groups, args.val_split, split_rng
        )
        if args.max_groups > 0:
            if args.val_split > 0 and val_keys and args.max_groups > 1:
                val_limit = max(1, int(round(args.max_groups * args.val_split)))
                val_limit = min(val_limit, len(val_keys), args.max_groups - 1)
            else:
                val_limit = 0
            train_limit = min(len(train_keys), args.max_groups - val_limit)
            train_keys = train_keys[:train_limit]
            val_keys = val_keys[:val_limit]
            if not train_keys and val_keys:
                train_keys = val_keys[:1]
                val_keys = val_keys[1:]
            train_files = [path for key in train_keys for path in groups[key]]
            val_files = [path for key in val_keys for path in groups[key]]

        pairs_total = len(groups) * 2
        dropped_povs = len(txt_files) - pairs_total
        print(
            f"\nProcessing {fmt}: {pairs_total} POV files ({len(groups)} complete battle pairs), "
            f"{dropped_povs} dropped (single-POV)"
        )
        print(
            f"  Split: {len(train_keys)} train / {len(val_keys)} val groups "
            f"({len(train_files)} train / {len(val_files)} val files)"
        )

        out_dir = os.path.join(args.output_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)

        shard_rng = np.random.default_rng(args.seed + 1009 * (format_id_map[fmt] + 1))
        # Shuffle battle keys within each split so consecutive battles
        # don't cluster by name similarity in the same shard, making batches
        # more independent.
        shard_rng.shuffle(train_keys)
        shard_rng.shuffle(val_keys)
        # Rebuild file lists to match shuffled key order.
        train_files = [path for key in train_keys for path in groups[key]]
        val_files = [path for key in val_keys for path in groups[key]]

        train_totals, train_btl_max, _ = write_paired_split_shards(
            fmt=fmt,
            fmt_id=format_id_map[fmt],
            split_name="train",
            groups=groups,
            group_keys=train_keys,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            battles_per_shard=args.battles_per_shard,
            rollout_len=args.rollout_len,
            out_dir=os.path.join(out_dir, "train"),
            rng=shard_rng,
        )
        val_totals, val_btl_max, _ = write_paired_split_shards(
            fmt=fmt,
            fmt_id=format_id_map[fmt],
            split_name="val",
            groups=groups,
            group_keys=val_keys,
            tokenizer_path=args.tokenizer_path,
            processes=args.processes,
            battles_per_shard=args.battles_per_shard,
            rollout_len=args.rollout_len,
            out_dir=os.path.join(out_dir, "val"),
            rng=shard_rng,
        )

        fmt_battle_max = max(train_btl_max, val_btl_max)

        failed = train_totals["failed"] + val_totals["failed"]
        total_battles = train_totals["num_battles"] + val_totals["num_battles"]
        total_states = train_totals["total_states"] + val_totals["total_states"]
        total_transitions = train_totals["total_transitions"] + val_totals["total_transitions"]
        total_rollout_samples = train_totals["total_rollout_samples"] + val_totals["total_rollout_samples"]
        total_shards = train_totals["num_shards"] + val_totals["num_shards"]

        if failed:
            print(f"  {failed} files failed to tokenize")
            for split_name, split_totals in (("train", train_totals), ("val", val_totals)):
                skip_reasons = split_totals.get("skip_reasons", {})
                if skip_reasons:
                    print(f"  WARNING: {fmt}/{split_name} skipped battle groups:")
                    for reason, count in sorted(skip_reasons.items()):
                        print(f"    {reason}: {count}")

        tokenizer_version = os.path.splitext(os.path.basename(args.tokenizer_path))[0]
        metadata = {
            "schema_version": "paired_pov_rollout_v2",
            "tokenizer_version": tokenizer_version,
            "format": fmt,
            "format_id": format_id_map[fmt],
            "format_id_map": format_id_map,
            "split_mode": "raw_battle_group",
            "paired_pov": True,
            "rollout_len": args.rollout_len,
            "seed": args.seed,
            "val_split": args.val_split,
            "max_groups": args.max_groups,
            "num_raw_battle_groups": len(groups),
            "train_raw_battle_groups": len(train_keys),
            "val_raw_battle_groups": len(val_keys),
            "num_parsed_files": pairs_total,
            "train_parsed_files": len(train_files),
            "val_parsed_files": len(val_files),
            "num_battles": total_battles,
            "num_shards": total_shards,
            "train_num_battles": train_totals["num_battles"],
            "val_num_battles": val_totals["num_battles"],
            "train_num_shards": train_totals["num_shards"],
            "val_num_shards": val_totals["num_shards"],
            "train_rollout_samples": train_totals["total_rollout_samples"],
            "val_rollout_samples": val_totals["total_rollout_samples"],
            "battles_per_shard": args.battles_per_shard,
            "total_states": total_states,
            "total_transitions": total_transitions,
            "total_transition_steps": total_transitions,
            "total_rollout_samples": total_rollout_samples,
            "max_battle_len": fmt_battle_max,
            "storage": "paired_pov_rollout_indexed_variable_length_v2",
            "compressed": False,
            "train_skip_reasons": train_totals.get("skip_reasons", {}),
            "val_skip_reasons": val_totals.get("skip_reasons", {}),
        }
        meta_path = os.path.join(out_dir, "metadata.json")
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
                return min(va, vb) if va and vb else (va or vb)
            return va + vb  # sum / count

        train_seq = train_totals.get("seq_agg", {})
        val_seq = val_totals.get("seq_agg", {})
        # Combine train+val: max of maxes, min of mins, sum of sums/counts
        combined_seq: dict[str, int] = {}
        for k in ["max_state_block_len", "min_state_block_len",
                   "sum_state_block_len", "count_state_blocks",
                   "max_temporal_seq", "min_temporal_seq",
                   "sum_temporal_seq", "count_temporal_battles"]:
            how = "max" if k.startswith("max_") else ("min" if k.startswith("min_") else "sum")
            combined_seq[k] = _agg(train_seq, val_seq, k, how)

        # Derived stats
        count_blocks = combined_seq["count_state_blocks"]
        avg_state_block_len = (combined_seq["sum_state_block_len"] / count_blocks
                               if count_blocks else 0.0)
        count_temporal = combined_seq["count_temporal_battles"]
        avg_temporal_seq = (combined_seq["sum_temporal_seq"] / count_temporal
                            if count_temporal else 0.0)

        SAFETY_MULTIPLIER = 1.2
        state_block_raw_max = combined_seq["max_state_block_len"]
        temporal_raw_max = combined_seq["max_temporal_seq"]
        seq_stats = {
            "safety_multiplier": SAFETY_MULTIPLIER,
            "state_block_len": {
                "max_raw": state_block_raw_max,
                "max": int(math.floor(state_block_raw_max * SAFETY_MULTIPLIER)),
                "min": combined_seq["min_state_block_len"],
                "avg": round(avg_state_block_len, 2),
                "count": count_blocks,
            },
            "temporal_sequence_len": {
                "max_raw": temporal_raw_max,
                "max": int(math.floor(temporal_raw_max * SAFETY_MULTIPLIER)),
                "min": combined_seq["min_temporal_seq"],
                "avg": round(avg_temporal_seq, 2),
                "count": count_temporal,
                "note": "Interleaved blocks per POV history: 1 (header) + 3*(T) + 1 (current state). "
                        "max = 3*num_state_blocks - 2 (matches JEPATemporalEncoder.forward max_seq).",
            },
        }
        seq_path = os.path.join(out_dir, "sequence_stats.json")
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
            f"{total_rollout_samples} rollout samples x K={args.rollout_len} "
            f"({total_transitions} transition steps), {total_shards} shards, "
            f"max battle {fmt_battle_max} tokens"
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
