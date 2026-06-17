#!/usr/bin/env python3
"""Inspect a parsed replay through the WorldModelObservationSpace.

Prints the full 336-token text observation for requested states.
Supports --pretty for indented, structured output and --show-all to
display every state from first to last.

Usage:
    uv run python scripts/inspect_world_model_state.py <parsed_replay.json> [state_idx...] [--pretty] [--show-all]
"""
import sys
import copy
import orjson
from metamon.interface import UniversalState, WorldModelObservationSpace
from metamon.tokenizer.wm_detokenizer import format_pretty


def any_team_wiped(us: UniversalState) -> bool:
    """Return True if one side has no usable Pokémon left."""
    # Player: active + switches + fainted = full team of 6
    player_alive = 1  # active (assume not fainted here but check below)
    if us.player_active_pokemon.status == "fnt":
        player_alive = 0
    player_alive += len(us.available_switches)
    # Opponent: active + bench (opponent_fainted are dead, bench are alive)
    opponent_alive = 0 if us.opponent_active_pokemon.status == "fnt" else 1
    opponent_alive += len(us.opponent_bench)
    return player_alive == 0 or opponent_alive == 0


def action_label(action_idx, state, next_state=None):
    """Return a human-readable label for an action index given the state.

    When ``next_state`` is provided, action_idx 0 is disambiguated:
    if the next state's ``player_prev_move`` matches the first sorted move,
    it's labeled as a MOVE; otherwise as NO-OP.
    """
    if action_idx == -1:
        return "MISSING (unrevealed)"
    elif 0 <= action_idx <= 3:
        # Move index into sorted active moves (action_idx is 0-based)
        from metamon.interface import consistent_move_order
        sorted_moves = consistent_move_order(state.player_active_pokemon.moves)
        if action_idx < len(sorted_moves):
            move_name = sorted_moves[action_idx].name
            if action_idx == 0 and next_state is not None:
                # Disambiguate: was this actually a no-op?
                if next_state.player_prev_move.name == move_name:
                    return f"MOVE: {move_name}"
                else:
                    return "NO-OP (Recharge/Struggle/Fight)"
            return f"MOVE: {move_name}"
        else:
            return f"MOVE index {action_idx} (out of range — only {len(sorted_moves)} moves)"
    elif 4 <= action_idx <= 8:
        # Switch index into sorted available switches
        from metamon.interface import consistent_pokemon_order
        sorted_switches = consistent_pokemon_order(state.available_switches)
        switch_idx = action_idx - 4
        if switch_idx < len(sorted_switches):
            return f"SWITCH: {sorted_switches[switch_idx].name}"
        else:
            return f"SWITCH index {action_idx} (out of range — only {len(sorted_switches)} switches)"
    elif 9 <= action_idx <= 12:
        # Tera move — same move index mapping
        from metamon.interface import consistent_move_order
        sorted_moves = consistent_move_order(state.player_active_pokemon.moves)
        move_idx = action_idx - 9
        if move_idx < len(sorted_moves):
            return f"TERA MOVE: {sorted_moves[move_idx].name}"
        else:
            return f"TERA MOVE index {action_idx} (out of range — only {len(sorted_moves)} moves)"
    else:
        return f"UNKNOWN action index {action_idx}"


def inspect_file(path, indices, pretty=False, show_all=False):
    with open(path, "r") as f:
        data = orjson.loads(f.read())

    obs_space = WorldModelObservationSpace()
    obs_space.reset()

    all_states = data["states"]
    all_actions = data.get("actions", [])

    if show_all:
        indices = []
        for idx in range(len(all_states)):
            indices.append(idx)
            us = UniversalState.from_dict(copy.deepcopy(all_states[idx]))
            if any_team_wiped(us):
                break

    for i in indices:
        idx = i if i >= 0 else len(all_states) + i
        us = UniversalState.from_dict(copy.deepcopy(all_states[idx]))
        obs = obs_space.state_to_obs(us)
        tokens = obs["text"].tolist().split(" ")

        # Look up the corresponding action (use next_state to disambiguate idx 0)
        action_idx = all_actions[idx] if idx < len(all_actions) else None
        next_us = None
        if action_idx is not None and idx + 1 < len(all_states):
            next_us = UniversalState.from_dict(copy.deepcopy(all_states[idx + 1]))
        label = action_label(action_idx, us, next_us) if action_idx is not None else "N/A"

        print(f"=== State {idx} ({len(tokens)} tokens) ===")
        print(f"  action_idx={action_idx} ({label})")
        print(f"  format={us.format}")
        print(f"  bench={len(us.opponent_bench)} fainted={len(us.fainted_pokemon)} opp_fainted={len(us.opponent_fainted)}")
        if pretty:
            print(format_pretty(tokens))
        else:
            print(obs["text"].tolist())
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Inspect a parsed replay through the WorldModelObservationSpace."
    )
    parser.add_argument("file", help="Path to parsed replay JSON")
    parser.add_argument(
        "indices", nargs="*", type=int, default=None,
        help="State indices to display (default: 0 -1)",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Show indented, structured output for readability",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Show all states from first until one team is wiped",
    )
    args = parser.parse_args()

    indices = args.indices if args.indices else [0, -1]
    inspect_file(args.file, indices, pretty=args.pretty, show_all=args.show_all)
