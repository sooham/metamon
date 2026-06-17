#!/usr/bin/env python3
"""Validate new-format parsed replay .txt files for structural correctness.

Checks tag balancing, state/action counts, terminal markers, and HP format.

Usage:
    uv run python scripts/validate_world_model.py <parsed_replay.txt>...
"""
import sys
import re


def validate_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    errors: list[str] = []

    # ── tag balance ──────────────────────────────────────────────────
    paired = [
        (r"<begin_team>", "<end_team>"),
        (r"<bos>", "<eos>"),
        (r"<boa>", "<eoa>"),
        (r"<arena>", "<end_arena>"),
        (r"<begin_moves>", "<end_moves>"),
        (r"<bench>", "<end_bench>"),
        (r"<conditions>", "<end_conditions>"),
        (r"<you>", "<end_you>"),
        (r"<active>", "<end_active>"),
        (r"<opponent>", "<end_opponent>"),
        (r"<move>", "<end_move>"),
        (r"<format>", "<end_format>"),
        (r"<turn>", "<end_turn>"),
        (r"<chosen_move", "<end_chosen_move>"),
        (r"<opponent_chosen_move>", "<end_opponent_chosen_move>"),
        (r"<boosts>", "<end_boosts>"),
        (r"<terminal>", "<end_terminal>"),
    ]
    for open_pat, close_tag in paired:
        open_count = len(re.findall(open_pat, text))
        close_count = text.count(close_tag)
        if open_count != close_count:
            errors.append(f"{open_pat} ({open_count}) != {close_tag} ({close_count})")

    # ── state / action count ────────────────────────────────────────
    n_states = text.count("<bos>")
    n_eos = text.count("<eos>")
    n_actions = text.count("<boa>")
    n_eoa = text.count("<eoa>")
    if n_states != n_eos:
        errors.append(f"<bos> ({n_states}) != <eos> ({n_eos})")
    if n_actions != n_eoa:
        errors.append(f"<boa> ({n_actions}) != <eoa> ({n_eoa})")
    if n_states != n_actions + 1:
        errors.append(f"states ({n_states}) != actions+1 ({n_actions + 1})")

    # ── terminal ────────────────────────────────────────────────────
    if "<terminal>" not in text or "<end_terminal>" not in text:
        errors.append("missing <terminal> / <end_terminal>")
    else:
        m = re.search(r"<terminal>\s*(won|lost|tie|forfeit)\s*<end_terminal>", text)
        if not m:
            errors.append("invalid terminal value")

    # ── HP format ────────────────────────────────────────────────────
    hp_vals = re.findall(r'\b(\d\.\d{2})\b', text)
    if not hp_vals:
        errors.append("no HP values found")

    # ── no XML-style closing tags ────────────────────────────────────
    xml_close = re.findall(r'</[a-zA-Z_]+>', text)
    if xml_close:
        errors.append(f"XML-style closing tags found: {xml_close}")

    # ── no bracketed value tokens ────────────────────────────────────
    for tok in ["<noboosts>", "<noeffect>", "<nostatus>", "<noweather>"]:
        if tok in text:
            errors.append(f"bracketed value token found: {tok}")

    # ── team header ──────────────────────────────────────────────────
    if "<begin_team>" not in text or "<end_team>" not in text:
        errors.append("missing team header")

    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_world_model.py <file.txt>...")
        sys.exit(1)

    all_ok = True
    for path in sys.argv[1:]:
        errs = validate_file(path)
        if errs:
            all_ok = False
            print(f"FAIL: {path}")
            for e in errs:
                print(f"  - {e}")
        else:
            n_states = open(path).read().count("<bos>")
            print(f"  OK: {path} ({n_states} states)")

    if all_ok:
        print("All validations passed.")
    else:
        sys.exit(1)
