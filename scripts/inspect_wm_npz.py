#!/usr/bin/env python3
"""Inspect world-model training data from .npz shards.

Randomly picks a battle from the sharded .npz files in the world-model-samples
directory, prints the tokenized representations (integer IDs and detokenized
text), and optionally opens the original Showdown replay.

Usage:
    # Inspect random battle states (default: first and last)
    uv run python scripts/inspect_wm_npz.py \\
        --wm_dir ~/Repositories/poke-datasets/world-model-samples \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \\
        --format gen1ou

    # Show all states for the random battle, pretty-printed
    uv run python scripts/inspect_wm_npz.py \\
        --wm_dir ~/Repositories/poke-datasets/world-model-samples \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \\
        --format gen1ou --show-all --pretty

    # Specific state indices and open in Showdown
    uv run python scripts/inspect_wm_npz.py \\
        --wm_dir ~/Repositories/poke-datasets/world-model-samples \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \\
        --format gen1ou --state-idx 0 5 --pretty --showdown \\
        --parsed_replay_root ~/Repositories/poke-datasets/parsed
"""

import argparse
import os
import random
import sys
import webbrowser

import numpy as np

from metamon.tokenizer.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.tokenizer.wm_detokenizer import format_pretty, detokenize_state


def find_npz_shards(wm_dir: str, fmt: str) -> list[str]:
    """Return a sorted list of .npz shard file paths for a format."""
    fmt_dir = os.path.join(wm_dir, fmt)
    if not os.path.isdir(fmt_dir):
        print(f"ERROR: World-model directory not found: {fmt_dir}")
        sys.exit(1)
    shards = sorted(
        os.path.join(fmt_dir, f)
        for f in os.listdir(fmt_dir)
        if f.endswith(".npz")
    )
    if not shards:
        print(f"ERROR: No .npz shards found in {fmt_dir}")
        sys.exit(1)
    return shards


def find_parsed_json_files(parsed_root: str, fmt: str) -> list[str]:
    """Return a sorted list of parsed-replay JSON file paths for a format.

    Matches the sort order used by ``generate_world_model_data.py``, which
    does an ``os.walk`` + ``sorted()`` on the full paths.
    """
    fmt_dir = os.path.join(parsed_root, fmt)
    if not os.path.isdir(fmt_dir):
        return []
    json_files = []
    for root, _, files in os.walk(fmt_dir):
        for f in files:
            if f.endswith(".json") and not f.endswith(".json.lz4"):
                json_files.append(os.path.join(root, f))
    json_files.sort()
    return json_files


def extract_battle_id(json_path: str) -> str:
    """Extract the Showdown battle ID from a parsed-replay filename.

    E.g. ``smogtours-gen1ou-749168_Unrated_..._WIN.json`` → ``smogtours-gen1ou-749168``.
    """
    return os.path.basename(json_path).split("_")[0]


def match_battle_id(
    num_states: int,
    won: bool,
    json_files: list[str],
    global_index: int | None = None,
) -> str | None:
    """Try to find the Showdown battle ID matching the npz battle.

    Strategy (tried in order):
      1. If ``global_index`` is provided, use ``json_files[global_index]`` —
         this works when no battles were skipped during tokenization.
      2. Scan a limited number of files, looking for a ``len(states)`` match.

    Returns the battle ID string or None.
    """
    # Strategy 1: direct index lookup
    if global_index is not None and global_index < len(json_files):
        candidate = json_files[global_index]
        # Quick verification: check if num_states roughly matches
        try:
            import orjson
            with open(candidate, "rb") as f:
                data = orjson.loads(f.read())
            if len(data["states"]) == num_states:
                return extract_battle_id(candidate)
        except Exception:
            pass

    # Strategy 2: scan (limited) for matching state count
    import orjson
    max_scan = min(len(json_files), 5000)  # don't scan forever
    candidates = []
    for fp in json_files[:max_scan]:
        try:
            with open(fp, "rb") as f:
                data = orjson.loads(f.read())
            if len(data["states"]) == num_states:
                # Check terminal state matches
                last_state = data["states"][-1]
                battle_won = last_state.get("battle_won", False)
                if battle_won == won:
                    return extract_battle_id(fp)
                else:
                    candidates.append(extract_battle_id(fp))
        except Exception:
            continue
    # Fallback: return any match with the right length, even if won flag differs
    if candidates:
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Inspect world-model training data from .npz shards."
    )
    parser.add_argument("--wm_dir", required=True,
                        help="Path to the world-model-samples directory.")
    parser.add_argument("--tokenizer_path", required=True,
                        help="Path to the WorldModel tokenizer JSON file.")
    parser.add_argument("--format", required=True,
                        help="Battle format (e.g., gen1ou).")
    parser.add_argument("--state-idx", type=int, nargs="*", default=None,
                        help="State indices to display (default: 0 and last).")
    parser.add_argument("--pretty", action="store_true",
                        help="Show indented, structured output for readability.")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all states of the selected battle.")
    parser.add_argument("--showdown", action="store_true",
                        help="Open the original battle in Pokémon Showdown.")
    parser.add_argument("--parsed_replay_root", default=None,
                        help="Path to parsed replays (required for --showdown).")
    parser.add_argument("--shard", type=str, default=None,
                        help="Specific shard file to use (default: random).")
    parser.add_argument("--battle", type=int, default=None,
                        help="Specific battle index within shard (default: random).")
    parser.add_argument("--show-actions", action="store_true",
                        help="Also display action indices.")
    args = parser.parse_args()

    # ── Load tokenizer ──
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens\n")

    # ── Pick a shard and battle ──
    shards = find_npz_shards(args.wm_dir, args.format)

    if args.shard:
        shard_path = args.shard
        if not os.path.isfile(shard_path):
            print(f"ERROR: Shard file not found: {shard_path}")
            sys.exit(1)
        # Extract shard index from filename: seq_shard_XXXX.npz → XXXX
        shard_idx = int(os.path.basename(shard_path).replace("seq_shard_", "").replace(".npz", ""))
    else:
        shard_path = random.choice(shards)
        shard_idx = int(os.path.basename(shard_path).replace("seq_shard_", "").replace(".npz", ""))

    print(f"Shard: {os.path.basename(shard_path)} (index {shard_idx})")

    data = np.load(shard_path, allow_pickle=True)
    states_all = data["states"]
    actions_all = data["actions"]
    won_all = data["won"]
    battle_start = data["battle_start"]
    num_battles = len(battle_start) - 1

    # Read battles_per_shard from metadata if available
    metadata_path = os.path.join(os.path.dirname(shard_path), "metadata.json")
    battles_per_shard = 1000  # default
    if os.path.isfile(metadata_path):
        import json
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            battles_per_shard = meta.get("battles_per_shard", 1000)
        except Exception:
            pass

    if args.battle is not None:
        battle_idx = args.battle
        if battle_idx < 0 or battle_idx >= num_battles:
            print(f"ERROR: Battle index {battle_idx} out of range [0, {num_battles - 1}]")
            sys.exit(1)
    else:
        battle_idx = random.randint(0, num_battles - 1)

    global_index = shard_idx * battles_per_shard + battle_idx

    # ── Extract battle data ──
    start = battle_start[battle_idx]
    end = battle_start[battle_idx + 1]
    battle_states = states_all[start:end]       # shape (num_states, state_dim)
    # Actions for this battle: a_start = start - battle_idx (each preceding battle
    # adds one fewer action than state).
    a_start = start - battle_idx
    a_end = end - battle_idx - 1
    battle_actions = actions_all[a_start:a_end] if a_end > a_start else np.array([], dtype=np.int16)
    won = bool(won_all[battle_idx])

    num_states = battle_states.shape[0]
    state_dim = battle_states.shape[1]
    print(f"Battle {battle_idx} in shard (global index ~{global_index}): "
          f"{num_states} states, {len(battle_actions)} actions, won={won}")
    print(f"State dimension: {state_dim}\n")

    # ── Showdown lookup ──
    if args.showdown:
        if not args.parsed_replay_root:
            print("WARNING: --showdown requires --parsed_replay_root. Skipping Showdown link.")
        else:
            json_files = find_parsed_json_files(args.parsed_replay_root, args.format)
            if json_files:
                battle_id = match_battle_id(num_states, won, json_files, global_index)
                if battle_id:
                    url = f"https://replay.pokemonshowdown.com/{battle_id}"
                    print(f"Opening Showdown: {url}")
                    webbrowser.open(url)
                else:
                    print(f"WARNING: Could not find matching battle ID for "
                          f"(states={num_states}, won={won}).")
            else:
                print("WARNING: No parsed replays found for matching.")

    # ── Determine state indices to display ──
    if args.show_all:
        indices = list(range(num_states))
    elif args.state_idx is not None:
        indices = []
        for idx in args.state_idx:
            if idx < 0:
                idx = num_states + idx
            if 0 <= idx < num_states:
                indices.append(idx)
    else:
        # Default: first and last state
        indices = [0, num_states - 1]

    # ── Display states ──
    for idx in indices:
        token_ids = battle_states[idx]
        tokens = detokenize_state(token_ids, tokenizer, strip_padding=True)

        print(f"=== State {idx}/{num_states} ({len(tokens)} tokens) ===")

        if args.show_actions and idx < num_states - 1:
            # Action index in flat actions array = battle start offset + transition index.
            # (start - battle_idx) is where this battle's actions begin.
            action_idx = (start - battle_idx) + idx
            if action_idx < len(actions_all):
                print(f"  action: {int(actions_all[action_idx])}")

        # Show raw token IDs (compact, first 40 + last 5)
        id_list = [int(t) for t in token_ids]
        if len(id_list) <= 50:
            print(f"  IDs: {id_list}")
        else:
            print(f"  IDs (first 40): {id_list[:40]}")
            print(f"  IDs (last 5):   {id_list[-5:]}")
            print(f"  IDs (full):     {id_list}")

        print()

        if args.pretty:
            print(format_pretty(tokens))
            print()
        else:
            # Plain detokenized text
            text = " ".join(tokens)
            if len(text) > 500:
                print(f"  Text (first 500 chars): {text[:500]}...")
                print(f"  Text (full, space-joined): {text}")
            else:
                print(f"  Text: {text}")
            print()

    # ── Summary ──
    print(f"Displayed {len(indices)} of {num_states} states from battle {battle_idx} "
          f"in shard {os.path.basename(shard_path)}")


if __name__ == "__main__":
    main()
