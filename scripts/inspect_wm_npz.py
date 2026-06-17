#!/usr/bin/env python3
"""Inspect world-model training data from .npz shards (v2 format).

Randomly picks a battle from the sharded .npz files, prints token IDs and
detokenized text for selected states, with player/opponent actions.

Usage:
    uv run python scripts/inspect_wm_npz.py \\
        --wm_dir ~/Repositories/poke-datasets/world-model-samples \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v1.json \\
        --format gen1ou

    uv run python scripts/inspect_wm_npz.py \\
        --wm_dir ~/Repositories/poke-datasets/world-model-samples \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v1.json \\
        --format gen1ou --show-all --pretty --showdown \\
        --parsed_replay_root ~/Repositories/poke-datasets/parsed-replays
"""

import argparse
import json
import os
import random
import re
import sys
import webbrowser

import numpy as np

from metamon.tokenizer.tokenizer import PokemonTokenizer
from metamon.tokenizer.wm_detokenizer import format_pretty


def find_npz_shards(wm_dir: str, fmt: str) -> list[str]:
    """Return sorted .npz shard paths for a format.

    Searches in ``wm_dir/fmt/`` and also in ``wm_dir/fmt/train`` and
    ``wm_dir/fmt/val`` subdirectories.
    """
    search_roots = [
        os.path.join(wm_dir, fmt),
        os.path.join(wm_dir, fmt, "train"),
        os.path.join(wm_dir, fmt, "val"),
        wm_dir,  # allow direct path to train/val dir
    ]
    shards = []
    for root in search_roots:
        if os.path.isdir(root):
            for f in os.listdir(root):
                if f.endswith(".npz"):
                    shards.append(os.path.join(root, f))
    if not shards:
        print(f"ERROR: No .npz shards found under {wm_dir}/{fmt}")
        sys.exit(1)
    return sorted(shards)


def find_parsed_txt_files(parsed_root: str, fmt: str) -> list[str]:
    fmt_dir = os.path.join(parsed_root, fmt)
    if not os.path.isdir(fmt_dir):
        return []
    txt_files = []
    for root, _, files in os.walk(fmt_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    txt_files.sort()
    return txt_files


def _detokenize(ids: np.ndarray, tokenizer: PokemonTokenizer) -> list[str]:
    """Detokenize a 1-D array of token IDs to strings."""
    result = []
    for tid in ids:
        for token, tid2 in tokenizer._initial_ids.items():
            if tid2 == int(tid):
                result.append(token)
                break
        else:
            for token, tid2 in tokenizer._new_ids.items():
                if tid2 == int(tid):
                    result.append(token)
                    break
            else:
                result.append("<unk>")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inspect world-model training data from .npz shards (v2)."
    )
    parser.add_argument("--wm_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--state-idx", type=int, nargs="*", default=None,
                        help="State indices to display (default: 0 and last).")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--showdown", action="store_true")
    parser.add_argument("--parsed_replay_root", default=None)
    parser.add_argument("--shard", type=str, default=None)
    parser.add_argument("--battle", type=int, default=None)
    args = parser.parse_args()

    # ── Load tokenizer ──
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens\n")

    # ── Pick shard and battle ──
    shards = find_npz_shards(args.wm_dir, args.format)

    if args.shard:
        shard_path = args.shard
        if not os.path.isfile(shard_path):
            print(f"ERROR: Shard not found: {shard_path}")
            sys.exit(1)
    else:
        shard_path = random.choice(shards)

    print(f"Shard: {os.path.basename(shard_path)}")

    data = np.load(shard_path, allow_pickle=True)
    states_all = data["states"]           # (total_tokens,) int16
    state_offsets = data["state_offsets"] # (num_states,) int64
    state_lengths = data["state_lengths"] # (num_states,) int32

    pa_all = data["player_actions"]             # (total_pa_tokens,) int16
    pa_offsets = data["player_action_offsets"]  # (num_actions,) int64
    pa_lengths = data["player_action_lengths"]  # (num_actions,) int32

    oa_all = data["opponent_actions"]             # (total_oa_tokens,) int16
    oa_offsets = data["opponent_action_offsets"]  # (num_actions,) int64
    oa_lengths = data["opponent_action_lengths"]  # (num_actions,) int32

    prev_state_idx = data["prev_state_idx"]  # (num_transitions,) int32
    next_state_idx = data["next_state_idx"]  # (num_transitions,) int32
    battle_id_arr = data["battle_id"]        # (num_transitions,) int32
    turn_idx_arr = data["turn_idx"]          # (num_transitions,) int32

    won_all = data["won"]                    # (num_battles,) bool
    battle_start = data["battle_start"]      # (num_battles+1,) int64
    battle_action_start = data["battle_action_start"]  # (num_battles+1,) int64
    num_battles = len(battle_start) - 1

    # ── Select battle ──
    if args.battle is not None:
        battle_idx = args.battle
        if battle_idx < 0 or battle_idx >= num_battles:
            print(f"ERROR: Battle index {battle_idx} out of range [0, {num_battles - 1}]")
            sys.exit(1)
    else:
        battle_idx = random.randint(0, num_battles - 1)

    # ── Extract battle data ──
    state_start = battle_start[battle_idx]
    state_end = battle_start[battle_idx + 1]
    num_states = int(state_end - state_start)
    won = bool(won_all[battle_idx])

    action_start = int(battle_action_start[battle_idx])
    action_end = int(battle_action_start[battle_idx + 1])
    num_actions = action_end - action_start

    print(f"Battle {battle_idx}/{num_battles}: "
          f"{num_states} states (incl. header), {num_actions} actions, won={won}")

    # ── Showdown lookup (parse .txt files for battle ID) ──
    if args.showdown and args.parsed_replay_root:
        txt_files = find_parsed_txt_files(args.parsed_replay_root, args.format)
        if txt_files and battle_idx < len(txt_files):
            # Rough match by index
            candidate = txt_files[min(battle_idx, len(txt_files) - 1)]
            stem = os.path.basename(candidate).replace(".txt", "")
            battle_id = stem.split("_")[0]
            url = f"https://replay.pokemonshowdown.com/{battle_id}"
            print(f"Showdown: {url}")
            webbrowser.open(url)

    # ── Determine which states to show ──
    if args.show_all:
        indices = list(range(num_states))
    elif args.state_idx is not None:
        indices = [i if i >= 0 else num_states + i for i in args.state_idx]
        indices = [i for i in indices if 0 <= i < num_states]
    else:
        indices = [0, num_states - 1]

    # ── Collect transitions for this battle ──
    # Filter transition table to this battle
    battle_mask = battle_id_arr == battle_idx
    b_prev = prev_state_idx[battle_mask]
    b_next = next_state_idx[battle_mask]
    b_turns = turn_idx_arr[battle_mask]

    print()

    # ── Display states ──
    for idx in indices:
        offset = int(state_offsets[idx])
        length = int(state_lengths[idx])
        token_ids = states_all[offset:offset + length]

        print(f"=== State {idx}/{num_states} ({length} tokens) ===")
        if idx == 0:
            print("  (team header)")

        # Find the action that leads FROM this state (if any)
        # The prev_state_idx values are local to the battle
        # State 0 = header, State 1 = state_0 actual
        # Transition t goes from state_{t+1} to state_{t+2}
        local_state_idx = idx  # local state index within battle
        transition_mask = b_prev == local_state_idx
        if transition_mask.any():
            t_idx = np.where(transition_mask)[0][0]
            pa_off = int(pa_offsets[t_idx + action_start])
            pa_len = int(pa_lengths[t_idx + action_start])
            oa_off = int(oa_offsets[t_idx + action_start])
            oa_len = int(oa_lengths[t_idx + action_start])

            pa_tokens = pa_all[pa_off:pa_off + pa_len] if pa_len > 0 else np.array([], dtype=np.int16)
            oa_tokens = oa_all[oa_off:oa_off + oa_len] if oa_len > 0 else np.array([], dtype=np.int16)

            pa_text = " ".join(_detokenize(pa_tokens, tokenizer)) if pa_len > 0 else "(empty)"
            oa_text = " ".join(_detokenize(oa_tokens, tokenizer)) if oa_len > 0 else "(empty)"
            print(f"  → player action:   [{pa_text}]")
            print(f"  → opponent action: [{oa_text}]")

        # Show token IDs
        id_list = [int(t) for t in token_ids]
        if len(id_list) <= 60:
            print(f"  IDs: {id_list}")
        else:
            print(f"  IDs (first 40): {id_list[:40]}")
            print(f"  IDs (last 5):   {id_list[-5:]}")

        print()

        # Show detokenized text
        tokens = _detokenize(token_ids, tokenizer)
        if args.pretty:
            print(format_pretty(tokens))
        else:
            text = " ".join(tokens)
            if len(text) > 600:
                print(f"  Text: {text[:600]}...")
            else:
                print(f"  Text: {text}")
        print()

    print(f"Displayed {len(indices)} of {num_states} states from battle {battle_idx} "
          f"in shard {os.path.basename(shard_path)}")


if __name__ == "__main__":
    main()
