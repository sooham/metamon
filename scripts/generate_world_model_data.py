#!/usr/bin/env python3
"""Generate world-model training data from parsed replays.

Converts each parsed replay into a sequence of tokenized
WorldModelObservationSpace states and packs them into sharded .npz files.

States are stored **unpadded** (variable-length) with a companion
``state_lengths`` array.  Padding to batch-max happens at training time.

Output per shard (seq_shard_XXXX.npz):
    states        (total_tokens,)  int16   — all state tokens concatenated (no padding)
    state_lengths (num_states,)    int32   — token count per state
    actions       (total_actions,) int16   — action index for each transition
    won           (num_battles,)   bool    — whether POV won each battle
    battle_start  (num_battles+1,) int64   — cumulative state index per battle
Plus metadata.json with aggregate stats.

Training pairs (derived on-the-fly, no duplication):
    (states[t], actions[t], states[t+1])

Usage:
    uv run python scripts/generate_world_model_data.py \\
        --parsed_replay_root /path/to/parsed-data \\
        --tokenizer_path /path/to/tokenizer.json \\
        --output_dir /path/to/world-model-samples \\
        --formats gen1ou gen9ou \\
        --battles_per_shard 1000 \\
        --processes 8
"""

import argparse
import copy
import json
import os
from multiprocessing import Pool

import numpy as np
import orjson
import tqdm

from metamon.interface import UniversalState, WorldModelObservationSpace
from metamon.tokenizer.tokenizer import PokemonTokenizer


def tokenize_battle(args: tuple) -> tuple:
    """Tokenize one battle → (token_ids_list, actions_arr, won, ok).

    Returns variable-length token lists — no padding applied here.
    """
    filepath, tokenizer = args
    try:
        with open(filepath, "rb") as f:
            data = orjson.loads(f.read())
    except Exception:
        return None

    obs_space = WorldModelObservationSpace()
    obs_space.reset()

    all_states = data["states"]
    actions_raw = data["actions"]

    token_ids_list = []  # list of 1-D numpy arrays (variable length each)
    for state_dict in all_states:
        us = UniversalState.from_dict(copy.deepcopy(state_dict))
        obs = obs_space.state_to_obs(us)
        ids_raw = tokenizer.tokenize(obs["text"].tolist())
        # Store unpadded — variable length
        token_ids_list.append(ids_raw.astype(np.int16))

    actions_arr = np.array(actions_raw[: len(token_ids_list) - 1], dtype=np.int16)

    final_us = UniversalState.from_dict(copy.deepcopy(all_states[-1]))
    won = bool(final_us.battle_won)

    return token_ids_list, actions_arr, won


def main():
    parser = argparse.ArgumentParser(
        description="Generate world-model training data from parsed replays."
    )
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--battles_per_shard", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    # Load tokenizer once in main process
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens")

    for fmt in args.formats:
        fmt_dir = os.path.join(args.parsed_replay_root, fmt)
        if not os.path.isdir(fmt_dir):
            print(f"Skipping {fmt}: directory not found at {fmt_dir}")
            continue

        # Gather all JSON files
        json_files = []
        for root, _, files in os.walk(fmt_dir):
            for f in files:
                if f.endswith(".json") and not f.endswith(".json.lz4"):
                    json_files.append(os.path.join(root, f))
        json_files.sort()

        if not json_files:
            print(f"No JSON files found in {fmt_dir}")
            continue

        print(f"\nProcessing {fmt}: {len(json_files)} battles")

        # Tokenize all battles in parallel
        work = [(f, tokenizer) for f in json_files]
        if args.processes > 1:
            with Pool(args.processes) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.imap(tokenize_battle, work, chunksize=100),
                        total=len(work),
                        desc=f"  Tokenizing {fmt}",
                    )
                )
        else:
            results = []
            for w in tqdm.tqdm(work, desc=f"  Tokenizing {fmt}"):
                results.append(tokenize_battle(w))

        # Filter failures and pack into shards
        battles = [r for r in results if r is not None]
        n_failed = len(results) - len(battles)
        if n_failed:
            print(f"  {n_failed} battles failed to tokenize, skipping")

        out_dir = os.path.join(args.output_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)

        # Split into shards
        shard_size = args.battles_per_shard
        num_shards = (len(battles) + shard_size - 1) // shard_size

        total_states = 0
        total_actions = 0

        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, len(battles))
            shard_battles = battles[start:end]

            # Accumulate concatenated arrays
            all_states_flat = []     # list of 1-D arrays
            all_state_lengths = []   # list of ints
            all_state_offsets = [0]  # cumulative token offset per state
            all_actions = []         # list of 1-D arrays
            won_list = []
            battle_start = [0]       # cumulative state index per battle

            for token_ids_list, actions_arr, won in shard_battles:
                for tok_arr in token_ids_list:
                    all_states_flat.append(tok_arr)
                    all_state_lengths.append(len(tok_arr))
                    all_state_offsets.append(all_state_offsets[-1] + len(tok_arr))
                all_actions.append(actions_arr)
                won_list.append(won)
                battle_start.append(battle_start[-1] + len(token_ids_list))

            states_cat = np.concatenate(all_states_flat, axis=0).astype(np.int16)
            state_lengths_arr = np.array(all_state_lengths, dtype=np.int32)
            # state_offsets[i] = start token index of state i in the flat states array
            state_offsets_arr = np.array(all_state_offsets[:-1], dtype=np.int64)
            actions_cat = np.concatenate(all_actions, axis=0).astype(np.int16)
            won_arr = np.array(won_list, dtype=bool)
            battle_start_arr = np.array(battle_start, dtype=np.int64)

            shard_name = f"seq_shard_{shard_idx:04d}.npz"
            shard_path = os.path.join(out_dir, shard_name)
            np.savez_compressed(
                shard_path,
                states=states_cat,
                state_lengths=state_lengths_arr,
                state_offsets=state_offsets_arr,
                actions=actions_cat,
                won=won_arr,
                battle_start=battle_start_arr,
            )

            total_states += len(all_state_lengths)
            total_actions += len(actions_cat)

            uncomp_mb = (states_cat.nbytes + state_lengths_arr.nbytes +
                         state_offsets_arr.nbytes +
                         actions_cat.nbytes + won_arr.nbytes +
                         battle_start_arr.nbytes) / (1024 * 1024)
            avg_len = state_lengths_arr.mean()
            print(
                f"  Shard {shard_idx:04d}: {len(shard_battles)} battles, "
                f"{len(all_state_lengths)} states (avg {avg_len:.0f} tok/state), "
                f"{len(actions_cat)} actions, "
                f"{uncomp_mb:.0f} MB uncompressed"
            )

        # Write metadata
        tokenizer_version = os.path.splitext(os.path.basename(args.tokenizer_path))[0]
        metadata = {
            "tokenizer_version": tokenizer_version,
            "format": fmt,
            "num_battles": len(battles),
            "num_shards": num_shards,
            "battles_per_shard": shard_size,
            "total_states": total_states,
            "total_actions": total_actions,
            "storage": "variable_length",
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Wrote metadata to {meta_path}")
        print(f"  Total: {len(battles)} battles, {total_states} states, {total_actions} actions")

    print("\nAll formats complete.")


if __name__ == "__main__":
    main()
