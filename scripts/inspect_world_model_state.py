#!/usr/bin/env python3
"""Inspect a new-format parsed replay .txt file.

Displays the raw text of requested state blocks, with optional pretty-printing
via the world-model detokenizer.

Usage:
    uv run python scripts/inspect_world_model_state.py <parsed_replay.txt> [state_idx...] [--pretty] [--show-all]
"""
import sys
import re
import argparse

from metamon.tokenizer.wm_detokenizer import format_pretty


def _extract_states(text: str) -> list[str]:
    """Return the text content of each ``<bos>…<eos>`` block."""
    states = []
    for m in re.finditer(r"<bos>(.*?)<eos>", text, re.DOTALL):
        states.append(m.group(0))  # include the <bos>/<eos> tags
    return states


def _extract_actions(text: str) -> list[dict]:
    """Return parsed action info for each ``<boa>…<eoa>`` block.

    Each dict has keys: ``chosen_move``, ``opponent_chosen_move``.
    """
    actions = []
    for m in re.finditer(r"<boa>(.*?)<eoa>", text, re.DOTALL):
        block = m.group(1)
        cm = re.search(r"<chosen_move[^>]*>(.*?)<end_chosen_move>", block, re.DOTALL)
        om = re.search(r"<opponent_chosen_move>(.*?)<end_opponent_chosen_move>", block, re.DOTALL)
        actions.append({
            "chosen_move": cm.group(1).strip() if cm else "",
            "opponent_chosen_move": om.group(1).strip() if om else "",
        })
    return actions


def inspect_file(path: str, indices: list[int], pretty: bool = False, show_all: bool = False):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    states = _extract_states(text)
    actions = _extract_actions(text)

    print(f"File: {path}")
    print(f"  {len(states)} states, {len(actions)} actions")
    print()

    if show_all:
        indices = list(range(len(states)))

    for i in indices:
        idx = i if i >= 0 else len(states) + i
        if idx < 0 or idx >= len(states):
            print(f"State {idx}: out of range (0..{len(states) - 1})")
            continue

        state_text = states[idx]
        tokens = state_text.split()

        # Show the action that follows this state (if any)
        action_info = ""
        if idx < len(actions):
            a = actions[idx]
            action_info = f"  → player: {a['chosen_move']}  |  opponent: {a['opponent_chosen_move']}"

        print(f"=== State {idx}/{len(states)} ({len(tokens)} tokens) {action_info} ===")

        if pretty:
            print(format_pretty(tokens))
        else:
            # Show the raw state text compactly
            raw = state_text.strip()
            if len(raw) > 800:
                print(raw[:800] + "...")
            else:
                print(raw)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a new-format parsed replay .txt file."
    )
    parser.add_argument("file", help="Path to parsed replay .txt")
    parser.add_argument(
        "indices", nargs="*", type=int, default=None,
        help="State indices to display (default: 0 -1)",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Show indented, structured output via detokenizer",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Show all states from first to last",
    )
    args = parser.parse_args()

    indices = args.indices if args.indices else [0, -1]
    inspect_file(args.file, indices, pretty=args.pretty, show_all=args.show_all)
