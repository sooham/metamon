"""World Model detokenizer: convert token IDs → human-readable text.

Provides functions to turn a flat list of WorldModelObservationSpace token
IDs (or string tokens) into a structured, indented, human-readable tree.

Text blocks are variable-length:
  - <switch>, <opponent_switch>, <fainted>, <opponent_fainted> only emit
    blocks for Pokémon that actually exist (no <blank> slot padding).
  - <opponent_moveset> only emits revealed moves (no unknownmove fillers).
"""

# ── token block sizes (WorldModelObservationSpace) ──────────────────
# Active pokemon: name(1) + hp(4) + item(1) + ability(1) + types(2) + effect(1) + status(1) = 11
# Inactive pokemon (player): name(1) + hp(4) + item(1) + ability(1) + moveset_tag(1) + 4×move(4) = 12
# Inactive pokemon (opponent): + status(1) + effect(1) = 14
# Move (active): name(1) + type(1) + category(1) = 3  (+ <move> tag = 4)
# Move (inactive/pad): name(1) = 1
HP_TOKENS = 4

# Structural tokens that begin a new top-level block
_STRUCTURAL_TOKENS = frozenset({
    "<player>", "<move>", "<switch>", "<opponent>", "<opponent_switch>",
    "<fainted>", "<opponent_fainted>", "<conditions>", "<player_prev>",
    "<opp_prev>", "<ongoing>", "<won>", "<lost>",
    "<boosts>", "<moveset>", "<opponent_moveset>",
})


def format_pretty(tokens: list[str]) -> str:
    """Re-parse the flat token list into an indented, human-readable tree.

    Handles variable-length opponent movesets and variable-count fainted blocks.
    """
    lines = []
    i = 0

    def take(n):
        nonlocal i
        chunk = tokens[i : i + n]
        i += n
        return chunk

    def peek(n=1):
        return tokens[i : i + n]

    def _tokens_until_structural() -> list[str]:
        """Read tokens until the next structural token (or end of list)."""
        nonlocal i
        result = []
        while i < len(tokens) and tokens[i] not in _STRUCTURAL_TOKENS:
            result.append(tokens[i])
            i += 1
        return result

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
        boosts = []
        if is_active:
            types_ = " ".join(take(2))
            effect = take(1)[0]
            status = take(1)[0]
            # consume <boosts> block
            if peek(1) == ["<boosts>"]:
                take(1)
                boosts = _tokens_until_structural()
            return {
                "name": name, "hp": hp, "item": item, "ability": ability,
                "types": types_, "effect": effect, "status": status,
                "boosts": boosts,
            }
        else:
            if has_status_effect:
                status = take(1)[0]
                effect = take(1)[0]
            # consume <boosts> block (before <moveset> or <opponent_moveset>)
            if peek(1) == ["<boosts>"]:
                take(1)
                boosts = _tokens_until_structural()
            moveset_tag = take(1)[0]  # <moveset> or <opponent_moveset>
            # Read moves until next structural token (opponent movesets
            # are variable-length, player movesets have 4 <blank>-padded moves).
            moves = _tokens_until_structural()
            result = {
                "name": name, "hp": hp, "item": item, "ability": ability,
                "tag": moveset_tag, "moves": moves, "boosts": boosts,
            }
            if has_status_effect:
                result["status"] = status
                result["effect"] = effect
            return result

    def pokemon_line(p, indent="  │   "):
        """Render a single pokemon dict to a compact line."""
        hp_str = p["hp"].replace(" ", "")
        boost_str = ""
        if p.get("boosts") and p["boosts"] != ["none"]:
            boost_str = f"  [{' '.join(p['boosts'])}]"
        if "types" in p:
            # active pokemon
            return (
                f"{indent}{p['name']:<14} HP={hp_str:>5}  {p['item']:<12} {p['ability']:<14}"
                f"  [{p['types']}]  {p['effect']}  {p['status']}{boost_str}"
            )
        else:
            # inactive / bench / fainted
            moves = " | ".join(p["moves"]) if p["moves"] else "(none)"
            extras = ""
            if "status" in p:
                extras = f"  {p['status']}  {p['effect']}"
            return (
                f"{indent}{p['name']:<14} HP={hp_str:>5}  {p['item']:<12} {p['ability']:<14}"
                f"{extras}  moves: {moves}{boost_str}"
            )

    # ── Player ──
    if peek(1) == ["<player>"]:
        take(1)  # <player>
        player_active = parse_pokemon(is_active=True)
        lines.append("  ┌─ Player ──────────────────────────────────────────")
        lines.append(pokemon_line(player_active, "  │ ▶ "))

        # moves (4 × <move> name type category) — always 4, padded with <blank>
        lines.append("  │   Moves:")
        for m in range(4):
            if peek(1) == ["<move>"]:
                take(1)
                move_name = take(1)[0]
                move_type = take(1)[0]
                move_cat = take(1)[0]
                if move_name != "<blank>":
                    lines.append(f"  │     {m+1}. {move_name:<16} {move_type:<10} {move_cat}")
                else:
                    lines.append(f"  │     {m+1}. (empty)")
            else:
                break

        # switches (0–5 blocks, no text padding)
        if peek(1) == ["<switch>"]:
            lines.append("  │   Bench:")
        while peek(1) == ["<switch>"]:
            take(1)  # <switch>
            sw = parse_pokemon(is_active=False)
            lines.append(pokemon_line(sw, "  │     "))

    # ── Opponent ──
    if peek(1) == ["<opponent>"]:
        take(1)
        opp_active = parse_pokemon(is_active=True)
        lines.append("  ┌─ Opponent ────────────────────────────────────────")
        lines.append(pokemon_line(opp_active, "  │ ▶ "))
        # opponent active moves (<opponent_moveset> + variable count of revealed moves)
        if peek(1) == ["<opponent_moveset>"]:
            take(1)
            opp_moves = _tokens_until_structural()
            if opp_moves:
                lines.append(f"  │     moves: {' | '.join(opp_moves)}")
            else:
                lines.append(f"  │     moves: (none revealed)")

        # opponent bench (0–5 blocks, no text padding, variable moves per mon)
        if peek(1) == ["<opponent_switch>"]:
            lines.append("  │   Bench:")
        while peek(1) == ["<opponent_switch>"]:
            take(1)
            ob = parse_pokemon(is_active=False, has_status_effect=True)
            lines.append(pokemon_line(ob, "  │     "))

    # ── Player Fainted ──
    if peek(1) == ["<fainted>"]:
        lines.append("  ┌─ Player Fainted ──────────────────────────────────")
        while peek(1) == ["<fainted>"]:
            take(1)
            pf = parse_pokemon(is_active=False)
            lines.append(pokemon_line(pf, "  │     "))

    # ── Opponent Fainted ──
    if peek(1) == ["<opponent_fainted>"]:
        lines.append("  ┌─ Opponent Fainted ────────────────────────────────")
        while peek(1) == ["<opponent_fainted>"]:
            take(1)
            of = parse_pokemon(is_active=False, has_status_effect=True)
            lines.append(pokemon_line(of, "  │     "))

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
    if peek(1) == ["<player_prev>"]:
        take(1)  # <player_prev>
        player_prev = take(1)[0]
        if peek(1) == ["<opp_prev>"]:
            take(1)  # <opp_prev>
            opp_prev = take(1)[0]
        else:
            opp_prev = "?"
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


def detokenize_state(token_ids, tokenizer, strip_padding: bool = True) -> list[str]:
    """Convert integer token IDs for one state into a list of string tokens.

    Args:
        token_ids: 1-D array-like of token IDs for a single state.
        tokenizer: A PokemonTokenizer instance used to map IDs → strings.
        strip_padding: If True, strip trailing padding tokens (both the
            tokenizer's ``pad_token_id`` and legacy 0 / -1 values from
            older data-generation runs).

    Returns:
        List of string tokens.
    """
    ids_list = [int(tid) for tid in token_ids]
    if strip_padding:
        pad_values = {0, -1, tokenizer.pad_token_id}
        while ids_list and ids_list[-1] in pad_values:
            ids_list.pop()
    return tokenizer.detokenize(ids_list)
