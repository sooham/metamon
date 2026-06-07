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


# ── token block sizes (WorldModelObservationSpace) ──────────────────
# Active pokemon: name(1) + hp(4) + item(1) + ability(1) + types(2) + effect(1) + status(1) = 11
# Inactive pokemon (player): name(1) + hp(4) + item(1) + ability(1) + moveset_tag(1) + 4×move(4) = 12
# Inactive pokemon (opponent): + status(1) + effect(1) = 14
# Opponent active extras: <opponent_moveset>(1) + 4×move(4) = 5
# Move (active): name(1) + type(1) + category(1) = 3  (+ <move> tag = 4)
# Move (inactive/pad): name(1) = 1
HP_TOKENS = 4
ACTIVE_POKEMON_TOKENS = 1 + HP_TOKENS + 1 + 1 + 2 + 1 + 1  # 11
INACTIVE_POKEMON_TOKENS = 1 + HP_TOKENS + 1 + 1 + 1 + 4     # 12 (player bench / fainted)
OPPONENT_INACTIVE_TOKENS = INACTIVE_POKEMON_TOKENS + 2      # 14 (opp bench / fainted: +status +effect)
OPPONENT_ACTIVE_MOVES_TOKENS = 1 + 4                         # 5  (<opponent_moveset> + 4 moves)


def _is_blank_slot(tokens, start, count):
    """Check if a block of tokens is entirely <blank> padding."""
    return all(t == "<blank>" for t in tokens[start : start + count])


def format_pretty(tokens: list[str]) -> str:
    """Re-parse the flat token list into an indented, human-readable tree."""
    lines = []
    i = 0

    def take(n):
        nonlocal i
        chunk = tokens[i : i + n]
        i += n
        return chunk

    def peek(n=1):
        return tokens[i : i + n]

    # ── header ──
    fmt = take(1)[0]
    choice = take(1)[0]
    lines.append(f"  format : {fmt}")
    lines.append(f"  choice : {choice}")

    # ── helper: parse a pokemon block ──
    def parse_pokemon(is_active, has_status_effect=False):
        nonlocal i
        name = take(1)[0]
        hp = " ".join(take(HP_TOKENS))
        item = take(1)[0]
        ability = take(1)[0]
        if is_active:
            types_ = " ".join(take(2))
            effect = take(1)[0]
            status = take(1)[0]
            return {
                "name": name, "hp": hp, "item": item, "ability": ability,
                "types": types_, "effect": effect, "status": status,
            }
        else:
            if has_status_effect:
                status = take(1)[0]
                effect = take(1)[0]
            moveset_tag = take(1)[0]  # <moveset> or <opponent_moveset>
            moves = [take(1)[0] for _ in range(4)]
            result = {
                "name": name, "hp": hp, "item": item, "ability": ability,
                "tag": moveset_tag, "moves": moves,
            }
            if has_status_effect:
                result["status"] = status
                result["effect"] = effect
            return result

    def pokemon_line(p, indent="  │   "):
        """Render a single pokemon dict to a compact line."""
        hp_str = p["hp"].replace(" ", "")
        if "types" in p:
            # active pokemon
            return (
                f"{indent}{p['name']:<14} HP={hp_str:>5}  {p['item']:<12} {p['ability']:<14}"
                f"  [{p['types']}]  {p['effect']}  {p['status']}"
            )
        else:
            # inactive / bench / fainted
            moves = " | ".join(p["moves"])
            extras = ""
            if "status" in p:
                extras = f"  {p['status']}  {p['effect']}"
            return (
                f"{indent}{p['name']:<14} HP={hp_str:>5}  {p['item']:<12} {p['ability']:<14}"
                f"{extras}  moves: {moves}"
            )

    def is_empty_block(tag, block_tokens):
        """Check if a padded block (tag + content) is entirely blank."""
        # The tag itself might be present even when blank (e.g. <switch> <blank>...)
        return all(t == "<blank>" for t in block_tokens)

    # ── Player ──
    if peek(1) == ["<player>"]:
        take(1)  # <player>
        player_active = parse_pokemon(is_active=True)
        lines.append("  ┌─ Player ──────────────────────────────────────────")
        lines.append(pokemon_line(player_active, "  │ ▶ "))

        # moves (4 × <move> name type category)
        lines.append("  │   Moves:")
        for m in range(4):
            if peek(1) == ["<move>"]:
                take(1)
                move_name = take(1)[0]
                move_type = take(1)[0]
                move_cat = take(1)[0]
                lines.append(f"  │     {m+1}. {move_name:<16} {move_type:<10} {move_cat}")
            else:
                break

        # switches (5 × <switch> + 12 tokens)
        lines.append("  │   Bench:")
        for s in range(5):
            if peek(1) == ["<switch>"]:
                tag = take(1)[0]
                if _is_blank_slot(tokens, i, INACTIVE_POKEMON_TOKENS):
                    take(INACTIVE_POKEMON_TOKENS)
                    lines.append(f"  │     (empty)")
                else:
                    sw = parse_pokemon(is_active=False)
                    lines.append(pokemon_line(sw, "  │     "))
            else:
                break

    # ── Opponent ──
    if peek(1) == ["<opponent>"]:
        take(1)
        opp_active = parse_pokemon(is_active=True)
        lines.append("  ┌─ Opponent ────────────────────────────────────────")
        lines.append(pokemon_line(opp_active, "  │ ▶ "))
        # opponent active moves (<opponent_moveset> + 4 moves)
        if peek(1) == ["<opponent_moveset>"]:
            take(1)
            opp_moves = [take(1)[0] for _ in range(4)]
            lines.append(f"  │     moves: {' | '.join(opp_moves)}")

        # opponent bench (5 × <opponent_switch> + 14 tokens)
        lines.append("  │   Bench:")
        for s in range(5):
            if peek(1) == ["<opponent_switch>"]:
                tag = take(1)[0]
                if _is_blank_slot(tokens, i, OPPONENT_INACTIVE_TOKENS):
                    take(OPPONENT_INACTIVE_TOKENS)
                    lines.append(f"  │     (empty)")
                else:
                    ob = parse_pokemon(is_active=False, has_status_effect=True)
                    lines.append(pokemon_line(ob, "  │     "))
            else:
                break

    # ── Player Fainted ──
    if peek(1) == ["<fainted>"]:
        lines.append("  ┌─ Player Fainted ──────────────────────────────────")
        for f in range(5):
            if peek(1) == ["<fainted>"]:
                tag = take(1)[0]
                if _is_blank_slot(tokens, i, INACTIVE_POKEMON_TOKENS):
                    take(INACTIVE_POKEMON_TOKENS)
                    lines.append(f"  │     (empty)")
                else:
                    pf = parse_pokemon(is_active=False)
                    lines.append(pokemon_line(pf, "  │     "))
            else:
                break

    # ── Opponent Fainted ──
    if peek(1) == ["<opponent_fainted>"]:
        lines.append("  ┌─ Opponent Fainted ────────────────────────────────")
        for f in range(5):
            if peek(1) == ["<opponent_fainted>"]:
                tag = take(1)[0]
                if _is_blank_slot(tokens, i, OPPONENT_INACTIVE_TOKENS):
                    take(OPPONENT_INACTIVE_TOKENS)
                    lines.append(f"  │     (empty)")
                else:
                    of = parse_pokemon(is_active=False, has_status_effect=True)
                    lines.append(pokemon_line(of, "  │     "))
            else:
                break

    # ── Conditions ──
    if peek(1) == ["<conditions>"]:
        take(1)
        weather = take(1)[0]
        player_cond = take(1)[0]
        opp_cond = take(1)[0]
        lines.append("  ┌─ Conditions ──────────────────────────────────────")
        lines.append(f"  │   weather = {weather}")
        lines.append(f"  │   player  = {player_cond}")
        lines.append(f"  │   opponent = {opp_cond}")

    # ── Previous Moves ──
    if peek(2) == ["<player_prev>", "<opp_prev>"] or peek(1) == ["<player_prev>"]:
        # Some states may only have <player_prev> <opp_prev> structure
        take(1)  # <player_prev>
        player_prev = take(1)[0]
        take(1)  # <opp_prev>
        opp_prev = take(1)[0]
        lines.append("  ┌─ Previous ────────────────────────────────────────")
        lines.append(f"  │   player   = {player_prev}")
        lines.append(f"  │   opponent = {opp_prev}")

    # ── Terminal ──
    if i < len(tokens) and tokens[i] in ("<ongoing>", "<won>", "<lost>"):
        terminal = tokens[i]
        take(1)
        label = {"<ongoing>": "ongoing", "<won>": "POV won", "<lost>": "POV lost"}[terminal]
        lines.append(f"  ┌─ Terminal ────────────────────────────────────────")
        lines.append(f"  │   {label}")

    lines.append("  └───────────────────────────────────────────────────")
    return "\n".join(lines)


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


def inspect_file(path, indices, pretty=False, show_all=False):
    with open(path, "r") as f:
        data = orjson.loads(f.read())

    obs_space = WorldModelObservationSpace()
    obs_space.reset()

    all_states = data["states"]

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
        print(f"=== State {idx} ({len(tokens)} tokens) ===")
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
