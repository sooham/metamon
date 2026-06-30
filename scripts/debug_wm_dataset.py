#!/usr/bin/env python3
"""Inspect a world-model paired shard: decode token IDs to text and print sample
rollouts for sanity checking.

Usage:

    uv run python scripts/debug_wm_dataset.py --shard $METAMON_CACHE_DIR/world-model-samples/gen1ou/train/paired_shard_0000.npz --tokenizer $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json --samples 3 --max_blocks 5
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

from metamon.tokenizer.tokenizer import PokemonTokenizer


def load_tokenizer(path: str) -> PokemonTokenizer:
    tok = PokemonTokenizer()
    tok.load_tokens_from_disk(path)
    return tok


def build_reverse(tok: PokemonTokenizer) -> dict[int, str]:
    """Build an ID → token-string reverse map."""
    rev: dict[int, str] = {}
    for token, tid in tok._initial_ids.items():
        rev[tid] = token
    for token, tid in tok._new_ids.items():
        rev[tid] = token
    return rev


def detokenize_block(block: np.ndarray, rev: dict[int, str]) -> str:
    """Convert an int16 token-array view back to text."""
    parts = [rev.get(int(t), f"<UNK:{t}>") for t in block]
    return " ".join(parts)


def _dtype_label(arr: np.ndarray) -> str:
    return f"{arr.dtype}{list(arr.shape)}"


_FIELD_DESCRIPTIONS = {
    # ── P1 token data: states ──
    "p1_states":                        "Flattened token IDs for all P1 state/header blocks",
    "p1_state_offsets":                 "Start index (tokens) of each P1 state block within p1_states",
    "p1_state_lengths":                 "Token count of each P1 state block",
    # ── P2 token data: states ──
    "p2_states":                        "Flattened token IDs for all P2 state/header blocks",
    "p2_state_offsets":                 "Start index (tokens) of each P2 state block within p2_states",
    "p2_state_lengths":                 "Token count of each P2 state block",
    # ── P1 token data: own actions ──
    "p1_actions":                       "Flattened token IDs for P1's own chosen actions (2 tokens each)",
    "p1_action_offsets":                "Start index of each P1 action within p1_actions",
    "p1_action_lengths":                "Token count of each P1 action (almost always 2)",
    # ── P1 token data: opponent actions (from P1's perspective) ──
    "p1_opponent_actions":              "Flattened token IDs for opponent actions as seen by P1",
    "p1_opponent_action_offsets":       "Start index of each opponent action from P1's view",
    "p1_opponent_action_lengths":       "Token count of each opponent action from P1's view",
    # ── P2 token data: own actions ──
    "p2_actions":                       "Flattened token IDs for P2's own chosen actions (2 tokens each)",
    "p2_action_offsets":                "Start index of each P2 action within p2_actions",
    "p2_action_lengths":                "Token count of each P2 action (almost always 2)",
    # ── P2 token data: opponent actions (from P2's perspective) ──
    "p2_opponent_actions":              "Flattened token IDs for opponent actions as seen by P2",
    "p2_opponent_action_offsets":       "Start index of each opponent action from P2's view",
    "p2_opponent_action_lengths":       "Token count of each opponent action from P2's view",
    # ── Legal action candidates ──
    "p1_legal_actions":                 "Dense 3D: (num_actions, max_candidates, max_tokens) — P1 legal candidates",
    "p1_legal_action_mask":             "Boolean mask: True for real candidates, False for padding slots",
    "p1_chosen_legal_action_idx":       "Index of replay-chosen action within P1's legal candidates",
    "p2_legal_actions":                 "Dense 3D: (num_actions, max_candidates, max_tokens) — P2 legal candidates",
    "p2_legal_action_mask":             "Boolean mask: True for real candidates, False for padding slots",
    "p2_chosen_legal_action_idx":       "Index of replay-chosen action within P2's legal candidates",
    # ── Rollout index matrices (the core sample structure) ──
    "p1_state_idx":                     "Rollout index: P1 current state per (sample, step)",
    "p1_next_state_idx":                "Rollout index: P1 next state per (sample, step)",
    "p1_action_idx":                    "Rollout index: P1 chosen action per (sample, step)",
    "p2_state_idx":                     "Rollout index: P2 current state per (sample, step)",
    "p2_next_state_idx":                "Rollout index: P2 next state per (sample, step)",
    "p2_action_idx":                    "Rollout index: P2 chosen action per (sample, step)",
    # ── Per-sample metadata ──
    "battle_id":                        "Which battle (0..num_battles-1) each rollout row belongs to",
    "turn_idx":                         "Ordinal index of this aligned transition row within the battle",
    "turn_number":                      "Actual game turn number (1 = pre-battle lead-selection state)",
    "subturn_idx":                      "Sub-turn counter (0 = first state of turn, >0 = forced-switch subturn)",
    "format_id":                        "Integer token ID for the battle format (e.g. gen1ou)",
    # ── Per-battle labels ──
    "p1_won":                           "True when P1's POV file ended in won or forfeit",
    "p2_won":                           "True when P2's POV file ended in won or forfeit",
    "rank_valid":                       "True when outcome is definitive (exactly one winner, not a tie)",
    "raw_battle_key":                   "String battle ID (e.g. gen1ou-749168)",
    # ── Per-battle boundary arrays (prefix-sums, len = num_battles+1) ──
    "p1_battle_start":                  "Prefix-sum: cumulative P1 state blocks across battles",
    "p2_battle_start":                  "Prefix-sum: cumulative P2 state blocks across battles",
    "p1_battle_action_start":           "Prefix-sum: cumulative P1 action blocks across battles",
    "p2_battle_action_start":           "Prefix-sum: cumulative P2 action blocks across battles",
    # ── Scalars ──
    "rollout_len":                      "K — number of contiguous transitions per rollout sample",
    "format_name":                      "Human-readable format string (e.g. gen1ou)",
    "format_id_value":                  "Integer token ID corresponding to format_name",
}


def print_shard_summary(data: dict) -> None:
    """Print an overview of all arrays in the shard with descriptions."""
    print("=" * 110)
    print(f"{'SHARD SUMMARY':^110}")
    print("=" * 110)
    print(f"  {'Key':<35s}  {'Dtype':<8s}  {'Shape':<24s}  Description")
    print(f"  {'─'*35}  {'─'*8}  {'─'*24}  {'─'*40}")
    for key in sorted(data.keys()):
        arr = data[key]
        desc = _FIELD_DESCRIPTIONS.get(key, "")
        if isinstance(arr, np.ndarray):
            shape = "×".join(str(s) for s in arr.shape)
            print(f"  {key:<35s}  {arr.dtype!s:<8s}  {shape:<24s}  {desc}")
        elif isinstance(arr, (np.integer, np.floating)):
            print(f"  {key:<35s}  {'scalar':<8s}  {str(arr):<24s}  {desc}")
        elif isinstance(arr, str):
            print(f"  {key:<35s}  {'str':<8s}  {arr:<24s}  {desc}")
        else:
            print(f"  {key:<35s}  {type(arr).__name__:<8s}  {'':<24s}  {desc}")
    print("=" * 110)
    print()


def print_sample(
    sample: dict,
    rev: dict[int, str],
    *,
    max_blocks: int = 0,
) -> None:
    """Pretty-print one rollout-step from a flat sample dict."""
    for pov in ("p1", "p2"):
        state_t = sample[f"{pov}_state_T"]
        state_t1 = sample[f"{pov}_state_T1"]
        n_new = len(state_t1) - len(state_t)

        print(f"\n  ▸ {pov.upper()} STATE HISTORY (state_T)  ({len(state_t)} blocks):")
        _print_blocks(state_t, rev, max_blocks)

        print(f"\n  ▸ {pov.upper()} NEXT STATE HISTORY (state_T1)  ({len(state_t1)} blocks, +{n_new} new):")
        # Print shared blocks, then mark new ones with ***
        limit = max_blocks if max_blocks > 0 else len(state_t1)
        for bi in range(min(len(state_t), limit)):
            text = detokenize_block(state_t1[bi], rev)
            print(f"    [{bi}] ({len(state_t1[bi])} tokens) {text}")
        for bi in range(len(state_t), limit):
            text = detokenize_block(state_t1[bi], rev)
            print(f"    [{bi}] ({len(state_t1[bi])} tokens) *** NEW *** {text}")

        p_actions = sample[f"{pov}_player_hist_T"]
        o_actions = sample[f"{pov}_opponent_hist_T"]
        # Every block returned by _slice_view is a real, non-empty action.
        # Validity would be all-True here; in padded training batches some
        # slots are padding (valid=False) to reach uniform max_blocks.
        print(f"\n  ▸ {pov.upper()} PLAYER ACTION HISTORY  ({len(p_actions)} actions, all valid)")
        _print_action_list(p_actions, rev)

        print(f"\n  ▸ {pov.upper()} OPPONENT ACTION HISTORY  ({len(o_actions)} actions, all valid)")
        _print_action_list(o_actions, rev)

        print(f"\n  ▸ {pov.upper()} CHOSEN ACTION:")
        text = detokenize_block(sample[f"{pov}_action"], rev)
        print(f"    {text}")

        opp_action_key = f"actual_p{'2' if pov == 'p1' else '1'}_action_from_{'p1' if pov == 'p1' else 'p2'}_perspective"
        if opp_action_key in sample:
            print(f"\n  ▸ {pov.upper()} SEEN OPPONENT ACTION:")
            print(f"    {detokenize_block(sample[opp_action_key], rev)}")

        legal_key = f"{pov}_legal_actions"
        mask_key = f"{pov}_legal_action_mask"
        chosen_key = f"{pov}_chosen_legal_action_idx"
        if legal_key in sample:
            legal = sample[legal_key]
            mask = sample[mask_key]
            chosen = sample[chosen_key]
            print(f"\n  ▸ {pov.upper()} LEGAL ACTIONS (chosen={chosen}):")
            _print_legal_actions(legal, mask, chosen, rev)

        won_key = f"{pov}_won"
        if won_key in sample:
            print(f"\n  ▸ {pov.upper()} WON: {sample[won_key]}")

    rank_valid = sample.get("rank_valid")
    if rank_valid is not None:
        print(f"\n  ▸ rank_valid: {rank_valid}")
    print()


def _print_blocks(blocks: list[np.ndarray], rev: dict[int, str], max_blocks: int) -> None:
    limit = max_blocks if max_blocks > 0 else len(blocks)
    for bi, block in enumerate(blocks):
        if bi >= limit:
            break
        text = detokenize_block(block, rev)
        print(f"    [{bi}] ({len(block)} tokens) {text}")


def _print_action_list(
    actions: list[np.ndarray], rev: dict[int, str]
) -> None:
    for ai, action in enumerate(actions):
        text = detokenize_block(action, rev)
        print(f"    [{ai}] {text}")


def _print_legal_actions(
    legal: np.ndarray, mask: np.ndarray, chosen: int, rev: dict[int, str]
) -> None:
    for ci in range(len(legal)):
        if not mask[ci]:
            continue
        text = detokenize_block(legal[ci], rev)
        marker = " ← CHOSEN" if ci == chosen else ""
        print(f"    [{ci}] {text}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect world-model paired shards with tokenizer decoding"
    )
    parser.add_argument("--shard", required=True, help="Path to paired_shard_*.npz")
    parser.add_argument(
        "--tokenizer", required=True,
        help="Path to tokenizer JSON (e.g. WorldModelObservationSpace-v1.json)",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of rollout samples to display (default: 3)",
    )
    parser.add_argument(
        "--max_blocks", type=int, default=0,
        help="Max state blocks to print per history; 0 = all (default: 0 = all)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only show the summary, no per-sample decode",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.shard):
        sys.exit(f"Shard not found: {args.shard}")
    if not os.path.isfile(args.tokenizer):
        sys.exit(f"Tokenizer not found: {args.tokenizer}")

    # Load
    print(f"Loading shard: {args.shard}")
    data = dict(np.load(args.shard))

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)
    rev = build_reverse(tok)

    print_shard_summary(data)

    # Extract key metadata
    n_samples = len(data["battle_id"])
    rollout_len = int(data["rollout_len"]) if "rollout_len" in data else 1
    print(f"Rollout samples: {n_samples}  ×  rollout_len: {rollout_len}")
    print()

    if args.quiet:
        return

    # ── Inlined helpers (avoid importing train_paired which has import errors) ──

    def _slice_view(flat: np.ndarray, offsets: np.ndarray,
                    lengths: np.ndarray, start: int, end: int) -> list[np.ndarray]:
        end = min(end, len(lengths))
        out: list[np.ndarray] = []
        for i in range(start, end):
            off = int(offsets[i])
            length = int(lengths[i])
            out.append(flat[off : off + length])
        return out

    def _slice_state_window(
        flat: np.ndarray, offsets: np.ndarray, lengths: np.ndarray,
        battle_start: int, state_start: int, state_end: int,
    ) -> list[np.ndarray]:
        if state_start <= battle_start:
            return _slice_view(flat, offsets, lengths, battle_start, state_end)
        return (
            _slice_view(flat, offsets, lengths, battle_start, battle_start + 1)
            + _slice_view(flat, offsets, lengths, state_start, state_end)
        )

    p1_state_idx = np.asarray(data["p1_state_idx"])
    if p1_state_idx.ndim == 1:
        p1_state_idx = p1_state_idx[:, None]
    p1_action_idx = np.asarray(data["p1_action_idx"])
    if p1_action_idx.ndim == 1:
        p1_action_idx = p1_action_idx[:, None]
    p2_state_idx = np.asarray(data["p2_state_idx"])
    if p2_state_idx.ndim == 1:
        p2_state_idx = p2_state_idx[:, None]
    p2_action_idx = np.asarray(data["p2_action_idx"])
    if p2_action_idx.ndim == 1:
        p2_action_idx = p2_action_idx[:, None]
    p1_next_state_idx = np.asarray(data["p1_next_state_idx"])
    if p1_next_state_idx.ndim == 1:
        p1_next_state_idx = p1_next_state_idx[:, None]
    p2_next_state_idx = np.asarray(data["p2_next_state_idx"])
    if p2_next_state_idx.ndim == 1:
        p2_next_state_idx = p2_next_state_idx[:, None]

    for row in range(min(args.samples, n_samples)):
        battle_id = int(data["battle_id"][row])

        # Determine per-POV battle start
        p1_bs = int(data["p1_battle_start"][battle_id])
        p2_bs = int(data["p2_battle_start"][battle_id])
        p1_as = int(data["p1_battle_action_start"][battle_id])
        p2_as = int(data["p2_battle_action_start"][battle_id])

        for step in range(rollout_len):
            p1_si = int(p1_state_idx[row, step])
            p1_nsi = int(p1_next_state_idx[row, step])
            p1_ai = int(p1_action_idx[row, step])
            p2_si = int(p2_state_idx[row, step])
            p2_nsi = int(p2_next_state_idx[row, step])
            p2_ai = int(p2_action_idx[row, step])

            p1_won = bool(data["p1_won"][battle_id])
            p2_won = bool(data["p2_won"][battle_id])
            rank_valid = (
                bool(data["rank_valid"][battle_id])
                if "rank_valid" in data
                else p1_won != p2_won
            )

            # Build flat sample dict for one rollout step
            sample = {}

            # State histories (simplified — full history, no windowing)
            sample["p1_state_T"] = _slice_state_window(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_bs, p1_bs, p1_si + 1)
            sample["p1_state_T1"] = _slice_state_window(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_bs, p1_bs, p1_nsi + 1)
            sample["p2_state_T"] = _slice_state_window(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_bs, p2_bs, p2_si + 1)
            sample["p2_state_T1"] = _slice_state_window(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_bs, p2_bs, p2_nsi + 1)

            # Action histories (all actions up to current)
            sample["p1_player_hist_T"] = _slice_view(data["p1_actions"], data["p1_action_offsets"], data["p1_action_lengths"], p1_as, p1_ai)
            sample["p1_opponent_hist_T"] = _slice_view(data["p1_opponent_actions"], data["p1_opponent_action_offsets"], data["p1_opponent_action_lengths"], p1_as, p1_ai)
            sample["p2_player_hist_T"] = _slice_view(data["p2_actions"], data["p2_action_offsets"], data["p2_action_lengths"], p2_as, p2_ai)
            sample["p2_opponent_hist_T"] = _slice_view(data["p2_opponent_actions"], data["p2_opponent_action_offsets"], data["p2_opponent_action_lengths"], p2_as, p2_ai)

            # Chosen actions
            sample["p1_action"] = _slice_view(data["p1_actions"], data["p1_action_offsets"], data["p1_action_lengths"], p1_ai, p1_ai + 1)[0]
            sample["p2_action"] = _slice_view(data["p2_actions"], data["p2_action_offsets"], data["p2_action_lengths"], p2_ai, p2_ai + 1)[0]
            sample["actual_p2_action_from_p1_perspective"] = _slice_view(data["p1_opponent_actions"], data["p1_opponent_action_offsets"], data["p1_opponent_action_lengths"], p1_ai, p1_ai + 1)[0]
            sample["actual_p1_action_from_p2_perspective"] = _slice_view(data["p2_opponent_actions"], data["p2_opponent_action_offsets"], data["p2_opponent_action_lengths"], p2_ai, p2_ai + 1)[0]

            # Legal actions
            if "p1_legal_actions" in data:
                sample["p1_legal_actions"] = data["p1_legal_actions"][p1_ai]
                sample["p1_legal_action_mask"] = data["p1_legal_action_mask"][p1_ai]
                sample["p1_chosen_legal_action_idx"] = int(data["p1_chosen_legal_action_idx"][p1_ai])
            if "p2_legal_actions" in data:
                sample["p2_legal_actions"] = data["p2_legal_actions"][p2_ai]
                sample["p2_legal_action_mask"] = data["p2_legal_action_mask"][p2_ai]
                sample["p2_chosen_legal_action_idx"] = int(data["p2_chosen_legal_action_idx"][p2_ai])

            sample["p1_won"] = p1_won
            sample["p2_won"] = p2_won
            sample["rank_valid"] = rank_valid

            raw_key = str(data["raw_battle_key"][battle_id])
            turn_num = int(data["turn_number"][row, step])
            sub = int(data["subturn_idx"][row, step])
            print(f"──── sample {row}  step {step}  battle_id={battle_id} ({raw_key})  turn={turn_num}.{sub} ────")
            print_sample(sample, rev, max_blocks=args.max_blocks)


if __name__ == "__main__":
    main()
