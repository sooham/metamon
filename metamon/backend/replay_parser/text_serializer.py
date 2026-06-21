"""Text serializer for the new POVReplay output format.

Converts a :class:`~metamon.backend.replay_parser.backward.POVReplay` into the
stateful text format specified in ``docs/new_parser_format_spec.md``.

Usage::

    from metamon.backend.replay_parser.text_serializer import serialize_pov_replay
    text = serialize_pov_replay(pov_replay)
    with open("output.txt", "w") as f:
        f.write(text)
"""

from __future__ import annotations

from typing import Optional, List

from metamon.backend.replay_parser.backward import POVReplay
from metamon.backend.replay_parser.replay_state import (
    Action,
    Move,
    Pokemon,
    Nothing,
    Turn,
    Winner,
)
from metamon.backend.replay_parser.pe_datatypes import (
    PEEffect,
    PEField,
    PESideCondition,
    PEStatus,
    PEWeather,
)
from metamon.backend.replay_parser.str_parsing import clean_name


# ── helpers ────────────────────────────────────────────────────────────────

def _hp_str(hp: int, max_hp: int, *, full_hp: bool = True) -> str:
    """Format HP as a "percentage current_hp max_hp" string.

    When *full_hp* is True (default), outputs all three values:
    ``"1.00 355 355"``.  When *full_hp* is False, outputs only the
    fixed-point ``X.XX`` string for backward compatibility.

    Returns ``"unknown"`` if *hp* or *max_hp* is None.
    """
    if hp is None or max_hp is None or max_hp == 0:
        return "unknown"
    pct = hp / max_hp
    if full_hp:
        return f"{pct:.2f} {hp} {max_hp}"
    return f"{pct:.2f}"


def _status_str(status) -> Optional[str]:
    """Convert a PEStatus or Nothing enum to a status token string.

    Returns None for NO_STATUS (healthy) — callers should suppress the field.
    Returns "fnt" for FNT, "par" for PAR, etc.
    """
    if status is None or status == Nothing.NO_STATUS:
        return None
    if isinstance(status, PEStatus):
        return status.name.lower()  # "FNT" → "fnt", "PAR" → "par", etc.
    # BackwardMarkers.FORCE_UNKNOWN or other sentinels
    return None


def _action_choice_text(action: Optional[Action], *, reveal_noop: bool) -> str:
    """Return canonical action-block content with a fixed two-token minimum."""
    if action is None:
        return "unknown unknown"
    if action.is_switch or action.is_revival:
        target_name = clean_name(action.target.name) if action.target else "unknown"
        return f"switch {target_name}"
    if action.is_noop:
        return "move recharge" if reveal_noop else "unknown unknown"
    if action.name is None:
        return "unknown unknown"
    action_name = clean_name(action.name)
    if not action_name or action_name == "unknown":
        return "unknown unknown"
    return f"move {action_name}"


def _effect_str(effects: dict) -> str:
    """Return the primary volatile effect token.

    Returns "noeffect" if no effects are active.
    """
    if not effects:
        return "noeffect"
    # Return the first effect (most Pokémon have at most one volatile effect at a time)
    effect = next(iter(effects))
    if isinstance(effect, PEEffect):
        return clean_name(effect.name.lower())
    return "noeffect"


def _effect_status_boosts_str(effects: dict, status, boosts) -> str:
    """Return a combined token for effect + status + boosts.

    When all three are at their defaults (noeffect + nostatus + noboosts),
    returns the single token "clean".  Otherwise returns them as
    space-separated individual tokens.
    """
    eff = _effect_str(effects)
    st = _status_str(status) or "nostatus"
    bst = _boosts_str(boosts)
    if eff == "noeffect" and st == "nostatus" and bst == "noboosts":
        return "clean"
    return f"{eff} {st} {bst}"


def _item_str(pokemon: Pokemon, gen: int, *, is_opponent: bool = False) -> Optional[str]:
    """Return item token for a Pokémon, or None if gen doesn't support items.

    For opponent Pokémon, returns "unknownitem" if the item is not yet revealed.
    """
    if gen < 2:
        return None
    if is_opponent:
        item = pokemon.active_item
        if item is None or item == Nothing.NO_ITEM:
            return "unknownitem"
        return clean_name(item) if isinstance(item, str) else "unknownitem"
    else:
        item = pokemon.had_item if pokemon.had_item is not None else pokemon.active_item
        if item is None or item == Nothing.NO_ITEM:
            return None
        if item == Nothing.NO_ITEM:
            return None
        return clean_name(item) if isinstance(item, str) else None


def _ability_str(pokemon: Pokemon, gen: int, *, is_opponent: bool = False) -> Optional[str]:
    """Return ability token for a Pokémon, or None if gen doesn't support abilities.

    For opponent Pokémon, returns "unknownability" if the ability is not yet revealed.
    """
    if gen < 3:
        return None
    if is_opponent:
        ab = pokemon.active_ability
        if ab is None or ab == Nothing.NO_ABILITY:
            return "unknownability"
        return clean_name(ab) if isinstance(ab, str) else "unknownability"
    else:
        # For POV player: use active_ability (current) since backward fill
        # already propagates had_ability → active_ability.  Using had_ability
        # would show the wrong ability after Skill Swap / Trace / etc.
        ab = pokemon.active_ability
        if ab is None or ab == Nothing.NO_ABILITY:
            ab = pokemon.had_ability
        if ab is None or ab == Nothing.NO_ABILITY:
            return None
        return clean_name(ab) if isinstance(ab, str) else None


def _gender_str(pokemon: Pokemon, gen: int) -> Optional[str]:
    """Return gender token ("M", "F", "N") for a Pokémon, or None for Gen 1."""
    if gen < 2:
        return None
    g = pokemon.gender
    if g is not None:
        return g
    return "N"


def _type_str(types) -> list[str]:
    """Return type name(s) for a Pokémon's type list.

    Single-typed Pokémon return a single-element list (no "notype" filler).
    Dual-typed Pokémon return a two-element list.
    """
    if types is None or len(types) == 0:
        return ["unknown"]
    type_names = []
    for t in types:
        if hasattr(t, 'name'):
            type_names.append(t.name.lower())
        else:
            type_names.append(str(t).lower())
    return type_names[:2]


def _move_type_str(move: Move) -> str:
    """Return the type string for a move."""
    if hasattr(move, 'type') and move.type is not None:
        if hasattr(move.type, 'name'):
            return move.type.name.lower()
        return str(move.type).lower()
    return "unknown"


def _move_category_str(move: Move) -> str:
    """Return the category string for a move."""
    if hasattr(move, 'category') and move.category is not None:
        if hasattr(move.category, 'name'):
            return move.category.name.lower()
        return str(move.category).lower()
    return "status"


def _boosts_str(boosts) -> str:
    """Return boost tokens: "noboosts" or "<boosts> atk+1 spa-2 <end_boosts>"."""
    if boosts is None:
        return "noboosts"
    stat_attrs = [
        ("atk_", "atk"), ("def_", "def"), ("spa_", "spa"),
        ("spd_", "spd"), ("spe_", "spe"),
        ("accuracy_", "accuracy"), ("evasion_", "evasion"),
    ]
    tokens = []
    for attr, name in stat_attrs:
        val = getattr(boosts, attr, 0)
        if val != 0:
            sign = "+" if val > 0 else ""
            tokens.append(f"{name}{sign}{val}")
    if not tokens:
        return "noboosts"
    return "<boosts> " + " ".join(tokens) + " <end_boosts>"


def _weather_str(weather) -> str:
    """Return weather token string."""
    if weather is None or weather == Nothing.NO_WEATHER:
        return "noweather"
    if hasattr(weather, 'name'):
        return weather.name.lower()
    return str(weather).lower()


def _battle_field_str(battle_field: dict) -> Optional[str]:
    """Return battle field token, or None if no field."""
    if not battle_field:
        return None
    # Return the first field effect (most common case: one at a time)
    field = next(iter(battle_field))
    if isinstance(field, PEField):
        return clean_name(field.name.lower())
    return None


def _side_conditions_str(conditions: dict) -> str:
    """Return space-separated side condition tokens."""
    if not conditions:
        return ""
    tokens = []
    for cond, value in conditions.items():
        if isinstance(cond, PESideCondition):
            name = clean_name(cond.name.lower())
            # For stackable conditions, append layer count
            from metamon.backend.replay_parser.pe_datatypes import PEStackableConditions
            if cond in PEStackableConditions and isinstance(value, int) and value > 0:
                name = f"{name}_{value}"
            tokens.append(name)
    return " ".join(tokens)


def _species_str(pokemon: Pokemon) -> str:
    """Return the species string for a Pokémon.

    For transformed Pokémon: "ditto snorlax" (actual transformed_species).
    """
    name = clean_name(pokemon.name) if pokemon.name else "unknown"
    if pokemon.transformed_into is not None and pokemon.transformed_into.name:
        transformed_name = clean_name(pokemon.transformed_into.name)
        return f"{name} {transformed_name}"
    return name


def _tera_str(pokemon: Pokemon, gen: int) -> Optional[str]:
    """Return tera type token, or None."""
    if gen != 9:
        return None
    if pokemon.tera_type is not None and pokemon.tera_type != Nothing.NO_TERA_TYPE:
        return f"tera:{clean_name(str(pokemon.tera_type))}"
    return None


# ── block writers ──────────────────────────────────────────────────────────

def _write_team_header(pov: POVReplay) -> list[str]:
    """Build the <begin_team> … <end_team> section."""
    lines = ["<begin_team>"]
    gen = pov.gen

    # Get the POV player's team from the backward-filled replay's last turn
    filled_turnlist = pov.filled_replay.turnlist
    if not filled_turnlist:
        lines.append("<end_team>")
        return lines

    last_turn = filled_turnlist[-1]
    team = last_turn.pokemon_1 if pov.from_p1_pov else last_turn.pokemon_2

    # Sort alphabetically by species name
    valid_pokes = [p for p in team if p is not None]
    valid_pokes.sort(key=lambda p: clean_name(p.name))

    for i, poke in enumerate(valid_pokes):
        lines.append(f"<poke{i + 1}>")

        # species
        species = clean_name(poke.name)
        line_parts = [species]

        # max HP (after species, before types — matches bench ordering)
        max_hp_val = poke.max_hp
        if max_hp_val is not None:
            line_parts.append(str(max_hp_val))

        # types
        line_parts.extend(_type_str(poke.type))

        # item (Gen 2+)
        item = _item_str(poke, gen, is_opponent=False)
        if item is not None:
            line_parts.append(item)

        # ability (Gen 3+)
        ability = _ability_str(poke, gen, is_opponent=False)
        if ability is not None:
            line_parts.append(ability)

        # gender (Gen 2+)
        gender = _gender_str(poke, gen)
        if gender is not None:
            line_parts.append(gender)

        lines.append(" " + " ".join(line_parts))

        # moves (always exactly 4; cap had_moves in case Transform/Mimic/
        # Zoroark edge cases pushed it above 4)
        lines.append(" <begin_moves>")
        had_moves = list(poke.had_moves.values()) if poke.had_moves else []
        had_moves.sort(key=lambda m: clean_name(m.name))
        for move in had_moves[:4]:
            lines.append(f" <move> {clean_name(move.name)} <end_move>")
        lines.append(" <end_moves>")

        lines.append(f"<end_poke{i + 1}>")

    lines.append("<end_team>")
    return lines


def _write_opponent_team_preview(pov: POVReplay) -> list[str]:
    """Build the <begin_opponent_team> … <end_opponent_team> section.

    Only emitted for Gen 5+ when team preview data is available.
    """
    if pov.gen < 5:
        return []

    # Get opponent team preview from the first non-empty turn
    turnlist = pov.replay.turnlist if pov.replay else []
    teampreview = None
    for turn in turnlist:
        tp = turn.teampreview_2 if pov.from_p1_pov else turn.teampreview_1
        if tp:
            teampreview = tp
            break

    if not teampreview:
        return []

    lines = ["<begin_opponent_team>"]
    # Sort alphabetically
    sorted_preview = sorted(
        teampreview, key=lambda p: clean_name(p.name) if p and p.name else ""
    )
    for i, poke in enumerate(sorted_preview):
        if poke is None:
            continue
        species = clean_name(poke.name) if poke.name else "unknown"
        lines.append(f"<poke{i + 1}>")
        lines.append(" " + species)
        lines.append(f"<end_poke{i + 1}>")
    lines.append("<end_opponent_team>")
    return lines


def _write_last_turn_results_block(
    prev_player_action: Optional[Action],
    prev_opponent_action: Optional[Action],
    is_first_state: bool = False,
) -> list[str]:
    """Build a <last_turn_results> … <end_last_turn_results> block.

    Describes the outcome of the action that transitioned from the previous
    state into this one.  The first state has an empty block.

    Outcome tokens per action:
      - ``success`` — the move executed normally (or was a switch / recharge).
      - ``fail`` — the move was attempted but failed (e.g. Sucker Punch whiff,
        Barrier at +6, Reflect used twice).
      - ``cant <reason>`` — the Pokémon couldn't execute its chosen move
        (paralysis, sleep, freeze, flinch, etc.).

    Format::

        <last_turn_results>
        <active>movename outcome [reason]<end_active>
        <opponent>movename outcome [reason]<end_opponent>
        <end_last_turn_results>
    """
    lines = ["<last_turn_results>"]

    if is_first_state:
        lines.append("<end_last_turn_results>")
        return lines

    # ── active ──
    lines.append("<active>")
    if prev_player_action is not None:
        if prev_player_action.is_switch or prev_player_action.is_revival:
            target_name = clean_name(prev_player_action.target.name) if prev_player_action.target else "unknown"
            action_name = f"switch {target_name}"
        elif prev_player_action.is_noop or prev_player_action.name is None:
            action_name = "recharge" if prev_player_action.is_noop else "unknown"
        else:
            action_name = clean_name(prev_player_action.name)

        # Determine outcome
        cant = None
        if prev_player_action.user is not None:
            cant = getattr(prev_player_action.user, "cant_reason", None)
        if cant:
            lines.append(f"{action_name} cant {cant}")
        elif prev_player_action.failed:
            lines.append(f"{action_name} fail")
        else:
            lines.append(f"{action_name} success")
    else:
        lines.append("unknown")
    lines.append("<end_active>")

    # ── opponent ──
    lines.append("<opponent>")
    if prev_opponent_action is not None:
        if prev_opponent_action.is_switch:
            target_name = clean_name(prev_opponent_action.target.name) if prev_opponent_action.target else "unknown"
            action_name = f"switch {target_name}"
        elif prev_opponent_action.is_noop or prev_opponent_action.name is None:
            action_name = "unknown"
        else:
            action_name = clean_name(prev_opponent_action.name)

        cant = None
        if prev_opponent_action.user is not None:
            cant = getattr(prev_opponent_action.user, "cant_reason", None)
        if cant:
            lines.append(f"{action_name} cant {cant}")
        elif prev_opponent_action.failed:
            lines.append(f"{action_name} fail")
        else:
            lines.append(f"{action_name} success")
    else:
        lines.append("unknown")
    lines.append("<end_opponent>")

    lines.append("<end_last_turn_results>")
    return lines


def _write_state_block(
    turn: Turn,
    pov: POVReplay,
    is_terminal: bool = False,
    display_turn: int | None = None,
    prev_player_action: Optional[Action] = None,
    prev_opponent_action: Optional[Action] = None,
) -> list[str]:
    """Build a single <bos> … <eos> state block."""
    lines = ["<bos>"]
    gen = pov.gen
    p1 = pov.from_p1_pov

    # format
    fmt = pov.format or f"gen{gen}ou"
    lines.append("<format>")
    lines.append(fmt)
    lines.append("<end_format>")

    # turn number — use passed display_turn, or fall back to raw + 1
    if display_turn is not None:
        turn_num = display_turn
    else:
        turn_num = (turn.turn_number if turn.turn_number is not None else 0) + 1
    lines.append("<turn>")
    lines.append(str(turn_num))
    lines.append("<end_turn>")

    # ── last turn results ──
    has_prev_actions = prev_player_action is not None or prev_opponent_action is not None
    lines.extend(_write_last_turn_results_block(
        prev_player_action, prev_opponent_action,
        is_first_state=(not has_prev_actions),
    ))

    # ── arena ──
    lines.append("<arena>")

    active = (turn.active_pokemon_1 if p1 else turn.active_pokemon_2)[0]
    opponent = (turn.active_pokemon_2 if p1 else turn.active_pokemon_1)[0]

    if active is not None:
        lines.append("<active>")
        parts = [
            _species_str(active),
            _hp_str(active.current_hp, active.max_hp),
            *_type_str(active.type),
        ]
        item = _item_str(active, gen, is_opponent=False)
        if item is not None:
            parts.append(item)
        ability = _ability_str(active, gen, is_opponent=False)
        if ability is not None:
            parts.append(ability)
        parts.append(_effect_status_boosts_str(active.effects, active.status, active.boosts))
        tera = _tera_str(active, gen)
        if tera is not None:
            parts.append(tera)
        lines.append(" " + " ".join(parts))
        lines.append("<end_active>")

    if opponent is not None:
        lines.append("<opponent>")
        parts = [
            _species_str(opponent),
            _hp_str(opponent.current_hp, opponent.max_hp),
            *_type_str(opponent.type),
        ]
        item = _item_str(opponent, gen, is_opponent=True)
        if item is not None:
            parts.append(item)
        ability = _ability_str(opponent, gen, is_opponent=True)
        if ability is not None:
            parts.append(ability)
        parts.append(_effect_status_boosts_str(opponent.effects, opponent.status, opponent.boosts))
        tera = _tera_str(opponent, gen)
        if tera is not None:
            parts.append(tera)
        lines.append(" " + " ".join(parts))
        lines.append("<end_opponent>")

    # ── conditions (inside arena) ──
    weather_str = _weather_str(turn.weather)
    bf = _battle_field_str(turn.battle_field)

    # player conditions
    player_conds = turn.conditions_1 if p1 else turn.conditions_2
    player_cond_str = _side_conditions_str(player_conds)
    you_parts = []
    if turn.is_force_revival:
        you_parts.append("forcedrevival")
    elif turn.is_force_switch:
        you_parts.append("forceswitch")
    if turn.can_tera_1 if p1 else turn.can_tera_2:
        if gen == 9:
            you_parts.append("cantera")
    if player_cond_str:
        you_parts.append(player_cond_str)
    you_inner = " ".join(you_parts)

    # opponent conditions
    opp_conds = turn.conditions_2 if p1 else turn.conditions_1
    opp_cond_str = _side_conditions_str(opp_conds)

    # ── collapse only when the arena is completely clear ──
    no_field = bf is None or bf == "nofield"
    if weather_str == "noweather" and no_field and not you_inner and not opp_cond_str:
        lines.append(" <empty_conditions>")
    else:
        lines.append(" <conditions>")
        lines.append("  " + weather_str)
        if bf and bf != "nofield":
            lines.append("  " + bf)
        if you_inner:
            lines.append(f"  <you> {you_inner} <end_you>")
        else:
            lines.append("  <you_empty>")
        if opp_cond_str:
            lines.append(f"  <opponent> {opp_cond_str} <end_opponent>")
        else:
            lines.append("  <opponent_empty>")
        lines.append(" <end_conditions>")

    lines.append("<end_arena>")

    # ── available moves (0–4; cap in case Transform/Mimic edge cases) ──
    lines.append("<begin_moves>")
    if active is not None:
        moves = list(active.moves.values()) if active.moves else []
        moves.sort(key=lambda m: clean_name(m.name))
        for move in moves[:4]:
            move_name = clean_name(move.name)
            move_type = _move_type_str(move)
            move_cat = _move_category_str(move)
            lines.append(f" <move> {move_name} {move_type} {move_cat} <end_move>")
    lines.append("<end_moves>")

    # ── bench ──
    lines.append("<bench>")
    team = turn.pokemon_1 if p1 else turn.pokemon_2
    if team:
        # Filter to non-active, non-None Pokémon
        bench_pokes = [
            p for p in team
            if p is not None and p != active
        ]
        bench_pokes.sort(key=lambda p: clean_name(p.name))
        for i, poke in enumerate(bench_pokes):
            lines.append(f"<poke{i + 1}>")
            parts = [
                clean_name(poke.name),
                _hp_str(poke.current_hp, poke.max_hp),
                *_type_str(poke.type),
            ]
            item = _item_str(poke, gen, is_opponent=False)
            if item is not None:
                parts.append(item)
            ability = _ability_str(poke, gen, is_opponent=False)
            if ability is not None:
                parts.append(ability)
            gender = _gender_str(poke, gen)
            if gender is not None:
                parts.append(gender)
            status = _status_str(poke.status)
            if status:
                parts.append(status)
            lines.append(" " + " ".join(parts))
            lines.append(f"<end_poke{i + 1}>")
    lines.append("<end_bench>")

    # ── terminal ──
    if is_terminal:
        winner = pov.winner
        replay_winner = pov.replay.winner if pov.replay else None
        from metamon.backend.replay_parser.replay_state import Winner
        if winner:
            lines.append("<terminal>")
            lines.append("won")
            lines.append("<end_terminal>")
        elif replay_winner == Winner.TIE:
            lines.append("<terminal>")
            lines.append("tie")
            lines.append("<end_terminal>")
        else:
            lines.append("<terminal>")
            lines.append("lost")
            lines.append("<end_terminal>")

    lines.append("<eos>")
    return lines


def _write_action_block(
    turn: Turn,
    player_action: Optional[Action],
    opponent_action: Optional[Action],
) -> list[str]:
    """Build a single <boa> … <eoa> action block."""
    lines = ["<boa>"]

    turn_num = turn.turn_number if turn.turn_number is not None else 1
    lines.append("<turn>")
    lines.append(str(turn_num))
    lines.append("<end_turn>")

    # Outcomes live in <last_turn_results> of the next state.
    lines.append("<chosen_move>")
    lines.append(_action_choice_text(player_action, reveal_noop=True))
    lines.append("<end_chosen_move>")

    lines.append("<opponent_chosen_move>")
    lines.append(_action_choice_text(opponent_action, reveal_noop=False))
    lines.append("<end_opponent_chosen_move>")

    lines.append("<eoa>")
    return lines


# ── main entry point ───────────────────────────────────────────────────────

def serialize_pov_replay(pov: POVReplay) -> str:
    """Serialize a POVReplay to the new text format.

    Returns the complete text output as a single string.
    """
    lines: list[str] = []

    # Team header (backward-filled knowledge)
    lines.extend(_write_team_header(pov))
    lines.append("")

    # Opponent team preview (Gen 5+)
    opp_preview = _write_opponent_team_preview(pov)
    if opp_preview:
        lines.extend(opp_preview)
        lines.append("")

    povturns = pov.povturnlist
    actions = pov.actionlist
    opp_actions = pov.opponent_actionlist

    n = len(povturns)
    # Turn number display logic:
    # - Spec convention: initial state is "turn 1", actions belong to the
    #   same turn as the state they follow.
    # - Parser convention: turn 0 = pre-battle, turn N = after turn N.
    # - Subturns (forced switches) share the raw turn with their parent;
    #   the entire subturn chain belongs to the same logical turn.
    # - Strategy: display = raw_turn + 1 for states before the first
    #   subturn; after any subturn is seen, display = raw_turn (no +1)
    #   so that subturns and all later turns stay aligned.
    offset = 1  # +1 for initial turns; drops to 0 after first subturn
    for i in range(n):
        turn = povturns[i]
        raw = turn.turn_number if turn.turn_number is not None else 0

        is_terminal = (i == n - 1)

        if offset == 1:
            display_turn = raw + 1
            if turn.is_force_switch or turn.is_force_revival:
                offset = 0
                display_turn = raw  # subturn keeps the action's turn
        else:
            display_turn = raw

        # Determine previous actions for <last_turn_results>
        if i == 0:
            prev_player_action = None
            prev_opponent_action = None
        else:
            prev_player_action = actions[i - 1][0] if i - 1 < len(actions) else None
            prev_opponent_action = opp_actions[i - 1] if i - 1 < len(opp_actions) else None

        # Write state block
        lines.extend(_write_state_block(
            turn, pov, is_terminal=is_terminal, display_turn=display_turn,
            prev_player_action=prev_player_action,
            prev_opponent_action=prev_opponent_action,
        ))
        lines.append("")

        # Write action block (skip for terminal state)
        if not is_terminal:
            next_turn = povturns[i + 1]
            player_action = actions[i][0] if i < len(actions) else None
            opponent_action = opp_actions[i] if i < len(opp_actions) else None
            lines.extend(_write_action_block(next_turn, player_action, opponent_action))
            lines.append("")

    return "\n".join(lines)


# ── doubles serialization ──────────────────────────────────────────────────

def _write_state_block_doubles(
    turn: Turn,
    pov: POVReplay,
    is_terminal: bool = False,
    prev_player_actions: list = None,
    prev_opponent_actions: list = None,
) -> list[str]:
    """Build a single <bos> … <eos> state block for doubles."""
    lines = ["<bos>"]
    gen = pov.gen
    p1 = pov.from_p1_pov

    fmt = pov.format or f"gen{gen}ou"
    lines.append("<format>")
    lines.append(fmt)
    lines.append("<end_format>")

    turn_num = (turn.turn_number if turn.turn_number is not None else 0) + 1
    lines.append("<turn>")
    lines.append(str(turn_num))
    lines.append("<end_turn>")

    # ── last turn results ──
    lines.append("<last_turn_results>")
    if prev_player_actions is None and prev_opponent_actions is None:
        # First state: empty block
        pass
    else:
        # Per-slot results for doubles
        for slot_idx, slot_tag in enumerate(("active1", "active2")):
            lines.append(f"<{slot_tag}>")
            pa = prev_player_actions[slot_idx] if prev_player_actions and slot_idx < len(prev_player_actions) else None
            if pa is not None:
                if pa.is_switch or pa.is_revival:
                    target_name = clean_name(pa.target.name) if pa.target else "unknown"
                    action_name = f"switch {target_name}"
                elif pa.is_noop or pa.name is None:
                    action_name = "recharge" if pa.is_noop else "unknown"
                else:
                    action_name = clean_name(pa.name)
                cant = getattr(pa.user, "cant_reason", None) if pa.user else None
                if cant:
                    lines.append(f"{action_name} cant {cant}")
                elif pa.failed:
                    lines.append(f"{action_name} fail")
                else:
                    lines.append(f"{action_name} success")
            else:
                lines.append("unknown")
            lines.append(f"<end_{slot_tag}>")

        for slot_idx, slot_tag in enumerate(("opponent1", "opponent2")):
            lines.append(f"<{slot_tag}>")
            oa = prev_opponent_actions[slot_idx] if prev_opponent_actions and slot_idx < len(prev_opponent_actions) else None
            if oa is not None:
                if oa.is_switch:
                    target_name = clean_name(oa.target.name) if oa.target else "unknown"
                    action_name = f"switch {target_name}"
                elif oa.is_noop or oa.name is None:
                    action_name = "unknown"
                else:
                    action_name = clean_name(oa.name)
                cant = getattr(oa.user, "cant_reason", None) if oa.user else None
                if cant:
                    lines.append(f"{action_name} cant {cant}")
                elif oa.failed:
                    lines.append(f"{action_name} fail")
                else:
                    lines.append(f"{action_name} success")
            else:
                lines.append("unknown")
            lines.append(f"<end_{slot_tag}>")
    lines.append("<end_last_turn_results>")

    # ── arena (doubles: active1/active2, opponent1/opponent2) ──
    lines.append("<arena>")

    player_actives = turn.active_pokemon_1 if p1 else turn.active_pokemon_2
    opponent_actives = turn.active_pokemon_2 if p1 else turn.active_pokemon_1

    for slot_idx, slot_tag in enumerate(["active1", "active2"]):
        poke = player_actives[slot_idx]
        if poke is not None:
            lines.append(f"<{slot_tag}>")
            parts = [
                _species_str(poke),
                _hp_str(poke.current_hp, poke.max_hp),
                *_type_str(poke.type),
            ]
            item = _item_str(poke, gen, is_opponent=False)
            if item is not None:
                parts.append(item)
            ability = _ability_str(poke, gen, is_opponent=False)
            if ability is not None:
                parts.append(ability)
            parts.append(_effect_status_boosts_str(poke.effects, poke.status, poke.boosts))
            tera = _tera_str(poke, gen)
            if tera is not None:
                parts.append(tera)
            lines.append(" " + " ".join(parts))
            lines.append(f"<end_{slot_tag}>")

    for slot_idx, slot_tag in enumerate(["opponent1", "opponent2"]):
        poke = opponent_actives[slot_idx]
        if poke is not None:
            lines.append(f"<{slot_tag}>")
            parts = [
                _species_str(poke),
                _hp_str(poke.current_hp, poke.max_hp),
                *_type_str(poke.type),
            ]
            item = _item_str(poke, gen, is_opponent=True)
            if item is not None:
                parts.append(item)
            ability = _ability_str(poke, gen, is_opponent=True)
            if ability is not None:
                parts.append(ability)
            parts.append(_effect_status_boosts_str(poke.effects, poke.status, poke.boosts))
            tera = _tera_str(poke, gen)
            if tera is not None:
                parts.append(tera)
            lines.append(" " + " ".join(parts))
            lines.append(f"<end_{slot_tag}>")

    # ── conditions (inside arena) ──
    weather_str = _weather_str(turn.weather)
    bf = _battle_field_str(turn.battle_field)

    player_conds = turn.conditions_1 if p1 else turn.conditions_2
    player_cond_str = _side_conditions_str(player_conds)
    you_parts = []
    if turn.is_force_revival:
        you_parts.append("forcedrevival")
    elif turn.is_force_switch:
        you_parts.append("forceswitch")
    if turn.can_tera_1 if p1 else turn.can_tera_2:
        if gen == 9:
            you_parts.append("cantera")
    if player_cond_str:
        you_parts.append(player_cond_str)
    you_inner = " ".join(you_parts)

    opp_conds = turn.conditions_2 if p1 else turn.conditions_1
    opp_cond_str = _side_conditions_str(opp_conds)

    no_field = bf is None or bf == "nofield"
    if weather_str == "noweather" and no_field and not you_inner and not opp_cond_str:
        lines.append(" <empty_conditions>")
    else:
        lines.append(" <conditions>")
        lines.append("  " + weather_str)
        if bf and bf != "nofield":
            lines.append("  " + bf)
        if you_inner:
            lines.append(f"  <you> {you_inner} <end_you>")
        else:
            lines.append("  <you_empty>")
        if opp_cond_str:
            lines.append(f"  <opponent> {opp_cond_str} <end_opponent>")
        else:
            lines.append("  <opponent_empty>")
        lines.append(" <end_conditions>")

    lines.append("<end_arena>")

    # ── available moves (per-slot in doubles) ──
    for slot_idx in (0, 1):
        poke = player_actives[slot_idx]
        if poke is None:
            continue
        slot = slot_idx + 1
        lines.append(f'<begin_moves:{slot}>')
        moves = list(poke.moves.values()) if poke.moves else []
        moves.sort(key=lambda m: clean_name(m.name))
        for move in moves[:4]:
            move_name = clean_name(move.name)
            move_type = _move_type_str(move)
            move_cat = _move_category_str(move)
            lines.append(f" <move> {move_name} {move_type} {move_cat} <end_move>")
        lines.append("<end_moves>")

    # ── bench ── (same as singles: all non-active POV Pokémon) ──
    lines.append("<bench>")
    team = turn.pokemon_1 if p1 else turn.pokemon_2
    if team:
        active_set = {id(p) for p in player_actives if p is not None}
        bench_pokes = [p for p in team if p is not None and id(p) not in active_set]
        bench_pokes.sort(key=lambda p: clean_name(p.name))
        for i, poke in enumerate(bench_pokes):
            lines.append(f"<poke{i + 1}>")
            parts = [
                clean_name(poke.name),
                _hp_str(poke.current_hp, poke.max_hp),
                *_type_str(poke.type),
            ]
            item = _item_str(poke, gen, is_opponent=False)
            if item is not None:
                parts.append(item)
            ability = _ability_str(poke, gen, is_opponent=False)
            if ability is not None:
                parts.append(ability)
            gender = _gender_str(poke, gen)
            if gender is not None:
                parts.append(gender)
            status = _status_str(poke.status)
            if status:
                parts.append(status)
            lines.append(" " + " ".join(parts))
            lines.append(f"<end_poke{i + 1}>")
    lines.append("<end_bench>")

    # ── terminal ──
    if is_terminal:
        winner = pov.winner
        replay_winner = pov.replay.winner if pov.replay else None
        from metamon.backend.replay_parser.replay_state import Winner
        if winner:
            lines.append("<terminal>")
            lines.append("won")
            lines.append("<end_terminal>")
        elif replay_winner == Winner.TIE:
            lines.append("<terminal>")
            lines.append("tie")
            lines.append("<end_terminal>")
        else:
            lines.append("<terminal>")
            lines.append("lost")
            lines.append("<end_terminal>")

    lines.append("<eos>")
    return lines


def _write_action_block_doubles(
    turn: Turn,
    player_actions: list,
    opponent_actions: list,
) -> list[str]:
    """Build a single <boa> … <eoa> action block for doubles.

    *player_actions* and *opponent_actions* are each ``[Action|None, Action|None]``.
    """
    lines = ["<boa>"]

    turn_num = turn.turn_number if turn.turn_number is not None else 0
    lines.append("<turn>")
    lines.append(str(turn_num))
    lines.append("<end_turn>")

    for slot_idx in (0, 1):
        slot = slot_idx + 1

        # player action for this slot — outcome now in <last_turn_results> of next state
        pa = player_actions[slot_idx] if slot_idx < len(player_actions) else None
        lines.append(f"<chosen_move:{slot}>")
        lines.append(_action_choice_text(pa, reveal_noop=True))
        lines.append("<end_chosen_move>")

        # opponent action for this slot — outcome now in <last_turn_results> of next state
        oa = opponent_actions[slot_idx] if slot_idx < len(opponent_actions) else None
        lines.append(f"<opponent_chosen_move:{slot}>")
        lines.append(_action_choice_text(oa, reveal_noop=False))
        lines.append("<end_opponent_chosen_move>")

    lines.append("<eoa>")
    return lines


def serialize_pov_replay_doubles(pov: POVReplay) -> str:
    """Serialize a doubles POVReplay to the new text format.

    Returns the complete text output as a single string.
    """
    lines: list[str] = []

    # Team header (backward-filled knowledge)
    lines.extend(_write_team_header(pov))
    lines.append("")

    # Opponent team preview (Gen 5+)
    opp_preview = _write_opponent_team_preview(pov)
    if opp_preview:
        lines.extend(opp_preview)
        lines.append("")

    povturns = pov.povturnlist
    actions = pov.actionlist
    opp_actions = pov.opponent_actionlist

    n = len(povturns)
    for i in range(n):
        turn = povturns[i]

        is_terminal = (i == n - 1)

        # Determine previous actions for <last_turn_results>
        if i == 0:
            prev_player_actions = None
            prev_opponent_actions = None
        else:
            prev_player_actions = actions[i - 1] if i - 1 < len(actions) else [None, None]
            prev_opponent_actions = opp_actions[i - 1] if i - 1 < len(opp_actions) else [None, None]

        # Write state block
        lines.extend(_write_state_block_doubles(
            turn, pov, is_terminal=is_terminal,
            prev_player_actions=prev_player_actions,
            prev_opponent_actions=prev_opponent_actions,
        ))
        lines.append("")

        # Write action block (skip for terminal state)
        if not is_terminal:
            next_turn = povturns[i + 1]
            player_actions = actions[i] if i < len(actions) else [None, None]
            opp_as = opp_actions[i] if i < len(opp_actions) else [None, None]
            lines.extend(_write_action_block_doubles(next_turn, player_actions, opp_as))
            lines.append("")

    return "\n".join(lines)
