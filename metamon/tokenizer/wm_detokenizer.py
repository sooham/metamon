"""World Model detokenizer: convert token IDs → human-readable text.

Parses the new stateful text format (v2) specified in
``docs/new_parser_format_spec.md`` and renders it as an indented tree.
"""

import re
from typing import Optional


# ── Structural tokens that begin a new top-level block ──────────────────
_BLOCK_STARTERS = frozenset({
    "<begin_team>", "<end_team>",
    "<begin_opponent_team>", "<end_opponent_team>",
    "<bos>", "<eos>", "<boa>", "<eoa>",
    "<format>", "<turn>",
    "<arena>", "<end_arena>",
    "<active>", "<end_active>",
    "<opponent>", "<end_opponent>",
    "<begin_moves>", "<end_moves>", "<move>", "<end_move>",
    "<bench>", "<end_bench>",
    "<conditions>", "<end_conditions>",
    "<you>", "<end_you>",
    "<boosts>", "<end_boosts>",
    "<chosen_move>", "<end_chosen_move>",
    "<opponent_chosen_move>", "<end_opponent_chosen_move>",
    "<terminal>", "<end_terminal>",
})


def format_pretty(tokens: list[str]) -> str:
    """Parse a flat list of tokens from the new text format into an indented tree.

    The new format is stateful and block-structured.  This function walks
    through the token list and reconstructs the hierarchical display.

    Handles: team header, opponent team preview, arena (active/opponent),
    available moves, bench, conditions, action blocks, terminal states.
    """
    lines: list[str] = []
    i = 0
    n = len(tokens)

    def take(k=1):
        nonlocal i
        chunk = tokens[i : i + k]
        i += k
        return chunk if k > 1 else (chunk[0] if chunk else None)

    def peek(k=1):
        return tokens[i : i + k]

    def expect(tag):
        t = take()
        if t != tag:
            lines.append(f"  !! expected {tag}, got {t}")

    def read_until(end_tags):
        """Read tokens until we hit one of *end_tags*. Returns the accumulated list."""
        nonlocal i
        result = []
        while i < n and tokens[i] not in end_tags:
            result.append(tokens[i])
            i += 1
        return result

    def parse_boosts():
        """Parse ``noboosts`` or ``<boosts> tok1 tok2 ... <end_boosts>``."""
        if peek(1) and peek(1)[0] == "noboosts":
            take()
            return []
        if peek(1) and peek(1)[0] == "<boosts>":
            take()  # <boosts>
            boosts = []
            while peek(1) and peek(1)[0] != "<end_boosts>":
                boosts.append(take())
            if peek(1):
                take()  # <end_boosts>
            return boosts
        return []

    # ── Team header ──
    if peek(1) and peek(1)[0] == "<begin_team>":
        lines.append("┌─ TEAM ──────────────────────────────────────────────")
        take()  # <begin_team>
        while peek(1) and peek(1)[0].startswith("<poke") and peek(1)[0] != "<end_team>":
            poke_tag = take()
            species_info = read_until({"<begin_moves>"})
            species_str = " ".join(species_info) if species_info else "?"
            lines.append(f"│ {poke_tag}: {species_str}")
            if peek(1) and peek(1)[0] == "<begin_moves>":
                take()
                moves = []
                while peek(1) and peek(1)[0] == "<move>":
                    take()
                    move_name = take() or "?"
                    if peek(1) and peek(1)[0] == "<end_move>":
                        take()
                    moves.append(move_name)
                if peek(1) and peek(1)[0] == "<end_moves>":
                    take()
                lines.append(f"│   moves: {' | '.join(moves)}")
            end_tag = take()  # <end_pokeN>
        if peek(1) and peek(1)[0] == "<end_team>":
            take()
        lines.append("")

    # ── Opponent team preview ──
    if peek(1) and peek(1)[0] == "<begin_opponent_team>":
        lines.append("┌─ OPPONENT TEAM PREVIEW ────────────────────────────")
        take()
        while peek(1) and peek(1)[0].startswith("<poke"):
            poke_tag = take()
            species = take() or "?"
            lines.append(f"│ {poke_tag}: {species}")
            end_tag = take()  # <end_pokeN>
        if peek(1) and peek(1)[0] == "<end_opponent_team>":
            take()
        lines.append("")

    # ── State / Action blocks ──
    state_num = 0
    while i < n:
        tok = peek(1)
        if not tok:
            break

        if tok[0] == "<bos>":
            take()
            state_num += 1
            lines.append(f"┌─ STATE {state_num} ─" + "─" * 42)

            # format
            if peek(1) and peek(1)[0] == "<format>":
                take()
                fmt_name = take() or "?"
                if peek(1) and peek(1)[0] == "<end_format>":
                    take()
                lines.append(f"│ format: {fmt_name}")

            # turn
            if peek(1) and peek(1)[0] == "<turn>":
                take()
                turn_num = take() or "?"
                if peek(1) and peek(1)[0] == "<end_turn>":
                    take()
                lines.append(f"│ turn: {turn_num}")

            # arena
            if peek(1) and peek(1)[0] == "<arena>":
                take()
                while peek(1) and peek(1)[0] in ("<active>", "<active1>", "<active2>",
                                                   "<opponent>", "<opponent1>", "<opponent2>"):
                    slot_tag = take()
                    info = read_until({"<end_active>", "<end_active1>", "<end_active2>",
                                       "<end_opponent>", "<end_opponent1>", "<end_opponent2>"})
                    end_tag = take()
                    info_str = " ".join(info)
                    label = slot_tag.strip("<>")
                    lines.append(f"│ {label}: {info_str}")
                if peek(1) and peek(1)[0] == "<end_arena>":
                    take()

            # available moves
            if peek(1) and peek(1)[0] == "<begin_moves>":
                take()
                move_idx = 0
                while peek(1) and peek(1)[0] == "<move>":
                    take()
                    move_info = read_until({"<end_move>"})
                    if peek(1) and peek(1)[0] == "<end_move>":
                        take()
                    move_idx += 1
                    lines.append(f"│   move {move_idx}: {' '.join(move_info)}")
                if peek(1) and peek(1)[0] == "<end_moves>":
                    take()

            # bench
            if peek(1) and peek(1)[0] == "<bench>":
                take()
                lines.append("│ bench:")
                while peek(1) and peek(1)[0].startswith("<poke"):
                    poke_tag = take()
                    info = read_until({f"<end_{poke_tag.strip('<>')}>"})
                    end_poke = take()  # <end_pokeN>
                    info_str = " ".join(info)
                    lines.append(f"│   {poke_tag}: {info_str}")
                if peek(1) and peek(1)[0] == "<end_bench>":
                    take()

            # conditions
            if peek(1) and peek(1)[0] == "<conditions>":
                take()
                weather = take() or "?"
                lines.append(f"│ conditions:")
                lines.append(f"│   weather: {weather}")

                # optional battle field
                if peek(1) and peek(1)[0] not in ("<you>",):
                    bf = take()
                    lines.append(f"│   field: {bf}")

                # you
                if peek(1) and peek(1)[0] == "<you>":
                    take()
                    you_parts = read_until({"<end_you>"})
                    if peek(1) and peek(1)[0] == "<end_you>":
                        take()
                    if you_parts:
                        lines.append(f"│   you: {' '.join(you_parts)}")

                # opponent
                if peek(1) and peek(1)[0] == "<opponent>":
                    take()
                    opp_parts = read_until({"<end_opponent>"})
                    if peek(1) and peek(1)[0] == "<end_opponent>":
                        take()
                    if opp_parts:
                        lines.append(f"│   opponent: {' '.join(opp_parts)}")

                if peek(1) and peek(1)[0] == "<end_conditions>":
                    take()

            # terminal
            if peek(1) and peek(1)[0] == "<terminal>":
                take()
                outcome = take() or "?"
                if peek(1) and peek(1)[0] == "<end_terminal>":
                    take()
                lines.append(f"│ TERMINAL: {outcome}")

            if peek(1) and peek(1)[0] == "<eos>":
                take()

        elif tok[0] == "<boa>":
            take()
            lines.append("├─ ACTION ─" + "─" * 42)

            if peek(1) and peek(1)[0] == "<turn>":
                take()
                turn_num = take() or "?"
                if peek(1) and peek(1)[0] == "<end_turn>":
                    take()
                lines.append(f"│ turn: {turn_num}")

            # chosen_move (multi-word value until <end_chosen_move>)
            if peek(1) and peek(1)[0] == "<chosen_move>":
                take()  # <chosen_move>
                move_parts = read_until({"<end_chosen_move>"})
                if peek(1) and peek(1)[0] == "<end_chosen_move>":
                    take()
                move_str = " ".join(move_parts) if move_parts else "?"
                lines.append(f"│ chosen: {move_str}")

            # opponent_chosen_move (multi-word value)
            if peek(1) and peek(1)[0] == "<opponent_chosen_move>":
                take()
                opp_parts = read_until({"<end_opponent_chosen_move>"})
                if peek(1) and peek(1)[0] == "<end_opponent_chosen_move>":
                    take()
                opp_str = " ".join(opp_parts) if opp_parts else "?"
                lines.append(f"│ opponent: {opp_str}")

            if peek(1) and peek(1)[0] == "<eoa>":
                take()

        else:
            # Skip unknown / non-structural tokens between blocks
            i += 1

    return "\n".join(lines)


def detokenize_state(token_ids, tokenizer, strip_padding: bool = True) -> list[str]:
    """Convert integer token IDs for one state into a list of string tokens.

    Args:
        token_ids: 1-D array-like of token IDs.
        tokenizer: A PokemonTokenizer instance.
        strip_padding: If True, strip trailing padding tokens.

    Returns:
        List of string tokens.
    """
    ids_list = [int(tid) for tid in token_ids]
    if strip_padding:
        pad_values = {0, -1, tokenizer.pad_token_id}
        while ids_list and ids_list[-1] in pad_values:
            ids_list.pop()
    return tokenizer.detokenize(ids_list)
