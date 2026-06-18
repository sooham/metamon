"""Online JEPA serialization matching parsed-replay training text.

The paired JEPA dataset is tokenized from ``text_serializer.py`` output.  Online
play has to build equivalent header/state/action blocks from a live poke-env
``Battle``.  This module keeps those conversions out of ``play.py`` so the
wire format can be tested independently.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from metamon.backend.replay_parser.str_parsing import clean_name, move_name, pokemon_name
from metamon.tokenizer import PokemonTokenizer


def generation_from_format(fmt: str, fallback: int = 1) -> int:
    match = re.search(r"gen(\d+)", fmt or "")
    return int(match.group(1)) if match else fallback


def tokenize_words(
    tokenizer: PokemonTokenizer,
    words: list[str],
    *,
    warn_unknown: bool = True,
) -> np.ndarray:
    ids: list[int] = []
    unk_id = tokenizer["<unk>"]
    unknown_words: list[str] = []
    for word in words:
        if word in tokenizer:
            ids.append(tokenizer[word])
        else:
            ids.append(unk_id)
            unknown_words.append(word)
    if warn_unknown and unknown_words:
        print(f"WARNING: <unk> tokens for words not in tokenizer: {unknown_words}")
    return np.array(ids, dtype=np.int16)


def _enum_name(value) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", None)
    if raw is None:
        raw = getattr(value, "value", None)
    if raw is None:
        raw = str(value)
    # poke-env __str__ returns strings like "PAR (status) object"; raw names
    # such as "Stealth Rock" still need to remain a single clean token.
    raw = str(raw)
    if " (" in raw:
        raw = raw.split(" (", 1)[0]
    return clean_name(raw)


def _sort_key(obj) -> str:
    raw = (
        getattr(obj, "species", None)
        or getattr(obj, "base_species", None)
        or getattr(obj, "name", None)
        or str(obj)
    )
    return pokemon_name(str(raw))


def _move_sort_key(obj) -> str:
    raw = getattr(obj, "id", None) or getattr(obj, "name", None) or str(obj)
    return move_name(str(raw))


def _pokemon_species(pokemon) -> str:
    if pokemon is None:
        return "unknown"
    raw = (
        getattr(pokemon, "species", None)
        or getattr(pokemon, "base_species", None)
        or getattr(pokemon, "name", None)
        or "unknown"
    )
    return pokemon_name(str(raw)) or "unknown"


def _hp_token(pokemon) -> str:
    if pokemon is None:
        return "unknown"
    current_hp = getattr(pokemon, "current_hp", None)
    max_hp = getattr(pokemon, "max_hp", None)
    if current_hp is not None and max_hp:
        pct = float(current_hp) / float(max_hp)
        return f"{max(0.0, min(1.0, pct)):.2f}"
    hp_fraction = getattr(pokemon, "current_hp_fraction", None)
    if hp_fraction is None:
        return "unknown"
    return f"{max(0.0, min(1.0, float(hp_fraction))):.2f}"


def _status_token(pokemon) -> Optional[str]:
    status = getattr(pokemon, "status", None)
    if status is None:
        return None
    token = _enum_name(status)
    return token or None


def _type_tokens(pokemon) -> list[str]:
    if pokemon is None:
        return ["unknown"]
    out: list[str] = []
    for typ in getattr(pokemon, "types", []) or []:
        token = _enum_name(typ)
        if token:
            out.append(token)
    return out[:2] or ["unknown"]


def _move_token(move) -> str:
    if move is None:
        return "unknown"
    raw = getattr(move, "id", None) or getattr(move, "name", None) or str(move)
    return move_name(str(raw)) or "unknown"


def _move_type(move) -> str:
    try:
        typ = getattr(move, "type", None)
    except Exception:
        typ = None
    return _enum_name(typ) or "unknown"


def _move_category(move) -> str:
    try:
        category = getattr(move, "category", None)
    except Exception:
        category = None
    return _enum_name(category) or "status"


def _item_token(pokemon, gen: int, *, is_opponent: bool) -> Optional[str]:
    if gen < 2:
        return None
    raw = getattr(pokemon, "item", None) if pokemon is not None else None
    if raw in {None, "", "noitem", "none"}:
        return "unknownitem" if is_opponent else None
    token = clean_name(str(raw))
    if token in {"", "noitem", "none"}:
        return "unknownitem" if is_opponent else None
    if is_opponent and token in {"unknown", "unknownitem"}:
        return "unknownitem"
    return token


def _ability_token(pokemon, gen: int, *, is_opponent: bool) -> Optional[str]:
    if gen < 3:
        return None
    raw = getattr(pokemon, "ability", None) if pokemon is not None else None
    if raw in {None, "", "noability", "none"}:
        return "unknownability" if is_opponent else None
    token = clean_name(str(raw))
    if token in {"", "noability", "none"}:
        return "unknownability" if is_opponent else None
    if is_opponent and token in {"unknown", "unknownability"}:
        return "unknownability"
    return token


def _gender_token(pokemon, gen: int) -> Optional[str]:
    if gen < 2:
        return None
    gender = getattr(pokemon, "gender", None)
    if gender is None:
        return "N"
    raw = getattr(gender, "name", None) or str(gender)
    raw = str(raw).upper()
    if raw.startswith("M"):
        return "M"
    if raw.startswith("F"):
        return "F"
    return "N"


def _boost_tokens(pokemon) -> list[str]:
    boosts = getattr(pokemon, "boosts", None) or {}
    parts = ["<boosts>"]
    has_any = False
    for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]:
        amount = int(boosts.get(stat, 0))
        if amount != 0:
            parts.append(f"{stat}{amount:+d}")
            has_any = True
    if not has_any:
        return ["noboosts"]
    parts.append("<end_boosts>")
    return parts


def _effect_token(pokemon) -> str:
    effects = getattr(pokemon, "effects", None) or {}
    if not effects:
        return "noeffect"
    try:
        effect = min(effects.keys(), key=effects.get)
    except Exception:
        effect = next(iter(effects))
    return _enum_name(effect) or "noeffect"


def _condition_tokens(pokemon) -> list[str]:
    effect = _effect_token(pokemon)
    status = _status_token(pokemon) or "nostatus"
    boosts = _boost_tokens(pokemon)
    if effect == "noeffect" and status == "nostatus" and boosts == ["noboosts"]:
        return ["clean"]
    return [effect, status, *boosts]


def _tera_token(pokemon, gen: int) -> Optional[str]:
    if gen != 9 or pokemon is None:
        return None
    tera = getattr(pokemon, "tera_type", None) or getattr(pokemon, "terastallized_type", None)
    token = _enum_name(tera)
    return f"tera:{token}" if token else None


def _weather_token(weather) -> str:
    if not weather:
        return "noweather"
    if isinstance(weather, dict):
        weather = next(iter(weather.keys()), None)
    return _enum_name(weather) or "noweather"


def _field_token(fields) -> Optional[str]:
    if not fields:
        return None
    if isinstance(fields, dict):
        field = next(iter(fields.keys()), None)
    else:
        field = next(iter(fields), None)
    token = _enum_name(field)
    return token if token and token != "nofield" else None


def _side_condition_tokens(conditions) -> list[str]:
    if not conditions:
        return []
    out: list[str] = []
    items = conditions.items() if isinstance(conditions, dict) else ((c, None) for c in conditions)
    for cond, value in items:
        token = _enum_name(cond)
        if not token or token == "unknown":
            continue
        if token in {"spikes", "toxicspikes"} and isinstance(value, int) and value > 0:
            token = f"{token}_{value}"
        out.append(token)
    return out


def _team_pokemon(battle) -> list:
    return sorted([p for p in getattr(battle, "team", {}).values() if p is not None], key=_sort_key)


def _bench_pokemon(battle, active) -> list:
    out = []
    active_species = _pokemon_species(active)
    active_hp = _hp_token(active)
    for pokemon in getattr(battle, "team", {}).values():
        if pokemon is None:
            continue
        if pokemon is active:
            continue
        if active is not None and _pokemon_species(pokemon) == active_species and _hp_token(pokemon) == active_hp:
            continue
        out.append(pokemon)
    return sorted(out, key=_sort_key)


def _move_values(pokemon) -> list:
    if pokemon is None:
        return []
    moves = getattr(pokemon, "moves", {}) or {}
    if isinstance(moves, dict):
        moves = list(moves.values())
    return sorted(moves, key=_move_sort_key)


def _format_action_text(action_name: str) -> str:
    if action_name.startswith("move: "):
        return move_name(action_name.removeprefix("move: "))
    if action_name.startswith("tera+move: "):
        return move_name(action_name.removeprefix("tera+move: "))
    if action_name.startswith("switch: "):
        return f"switch {pokemon_name(action_name.removeprefix('switch: '))}"
    if action_name.startswith("switch "):
        return f"switch {pokemon_name(action_name.removeprefix('switch '))}"
    if action_name.startswith("opponent move: "):
        return move_name(action_name.removeprefix("opponent move: "))
    if action_name.startswith("opponent switch: "):
        return f"switch {pokemon_name(action_name.removeprefix('opponent switch: '))}"
    return clean_name(action_name) or "unknown"


def action_words(action_text: str, *, opponent: bool) -> list[str]:
    if opponent:
        start = "<opponent_chosen_move>"
        end = "<end_opponent_chosen_move>"
    else:
        start = "<chosen_move>"
        end = "<end_chosen_move>"
    return [start, *_format_action_text(action_text).split(), end]


def action_block(tokenizer: PokemonTokenizer, action_text: str, *, opponent: bool) -> np.ndarray:
    return tokenize_words(tokenizer, action_words(action_text, opponent=opponent))


def team_context_words(battle, fmt: str) -> list[str]:
    gen = generation_from_format(fmt, getattr(battle, "gen", 1))
    words = ["<begin_team>"]
    for idx, pokemon in enumerate(_team_pokemon(battle)[:6], start=1):
        words.append(f"<poke{idx}>")
        words.extend([_pokemon_species(pokemon), *_type_tokens(pokemon)])
        item = _item_token(pokemon, gen, is_opponent=False)
        if item is not None:
            words.append(item)
        ability = _ability_token(pokemon, gen, is_opponent=False)
        if ability is not None:
            words.append(ability)
        gender = _gender_token(pokemon, gen)
        if gender is not None:
            words.append(gender)
        words.append("<begin_moves>")
        for move in _move_values(pokemon)[:4]:
            words.extend(["<move>", _move_token(move), "<end_move>"])
        words.extend(["<end_moves>", f"<end_poke{idx}>"])
    words.append("<end_team>")

    preview = getattr(battle, "teampreview_opponent_team", None) or []
    preview = sorted([p for p in preview if p is not None], key=_sort_key)
    if gen >= 5 and preview:
        words.append("<begin_opponent_team>")
        for idx, pokemon in enumerate(preview[:6], start=1):
            words.extend([f"<poke{idx}>", _pokemon_species(pokemon), f"<end_poke{idx}>"])
        words.append("<end_opponent_team>")
    return words


def team_context_block(tokenizer: PokemonTokenizer, battle, fmt: str) -> np.ndarray:
    return tokenize_words(tokenizer, team_context_words(battle, fmt))


def _normalized_force_switch(value) -> bool:
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


def is_force_switch_state(battle) -> bool:
    if _normalized_force_switch(getattr(battle, "force_switch", False)):
        return True
    active = getattr(battle, "active_pokemon", None)
    if active is None:
        return True
    if getattr(active, "fainted", False):
        return True
    hp = getattr(active, "current_hp_fraction", None)
    if hp is not None and float(hp) <= 0.0:
        return True
    available_moves = getattr(battle, "available_moves", None) or []
    available_switches = getattr(battle, "available_switches", None) or []
    return bool(available_switches) and not available_moves


def state_words(
    battle,
    fmt: str,
    *,
    is_force_switch: bool = False,
    override_active=None,
    is_terminal: bool = False,
) -> list[str]:
    gen = generation_from_format(fmt, getattr(battle, "gen", 1))
    active = override_active if override_active is not None else getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)

    words: list[str] = [
        "<bos>",
        "<format>", fmt, "<end_format>",
        "<turn>", str(max(1, int(getattr(battle, "turn", 1) or 1))), "<end_turn>",
        "<arena>",
    ]

    if active is not None:
        words.append("<active>")
        words.extend([_pokemon_species(active), _hp_token(active), *_type_tokens(active)])
        item = _item_token(active, gen, is_opponent=False)
        if item is not None:
            words.append(item)
        ability = _ability_token(active, gen, is_opponent=False)
        if ability is not None:
            words.append(ability)
        words.extend(_condition_tokens(active))
        tera = _tera_token(active, gen)
        if tera is not None:
            words.append(tera)
        words.append("<end_active>")

    if opponent is not None:
        words.append("<opponent>")
        words.extend([_pokemon_species(opponent), _hp_token(opponent), *_type_tokens(opponent)])
        item = _item_token(opponent, gen, is_opponent=True)
        if item is not None:
            words.append(item)
        ability = _ability_token(opponent, gen, is_opponent=True)
        if ability is not None:
            words.append(ability)
        words.extend(_condition_tokens(opponent))
        tera = _tera_token(opponent, gen)
        if tera is not None:
            words.append(tera)
        words.append("<end_opponent>")

    words.extend(["<end_arena>", "<begin_moves>"])
    for move in _move_values(active)[:4]:
        words.extend(["<move>", _move_token(move), _move_type(move), _move_category(move), "<end_move>"])
    words.append("<end_moves>")

    words.append("<bench>")
    for idx, pokemon in enumerate(_bench_pokemon(battle, active), start=1):
        words.append(f"<poke{idx}>")
        words.extend([_pokemon_species(pokemon), _hp_token(pokemon), *_type_tokens(pokemon)])
        item = _item_token(pokemon, gen, is_opponent=False)
        if item is not None:
            words.append(item)
        ability = _ability_token(pokemon, gen, is_opponent=False)
        if ability is not None:
            words.append(ability)
        gender = _gender_token(pokemon, gen)
        if gender is not None:
            words.append(gender)
        status = _status_token(pokemon)
        if status is not None:
            words.append(status)
        words.append(f"<end_poke{idx}>")
    words.append("<end_bench>")

    weather = _weather_token(getattr(battle, "weather", None))
    field = _field_token(getattr(battle, "fields", None))
    you_parts: list[str] = []
    if is_force_switch:
        you_parts.append("forceswitch")
    can_tera = getattr(battle, "can_tera", None)
    if gen == 9 and can_tera is not None and can_tera is not False:
        you_parts.append("cantera")
    you_parts.extend(_side_condition_tokens(getattr(battle, "side_conditions", None)))
    opp_parts = _side_condition_tokens(getattr(battle, "opponent_side_conditions", None))

    words.append("<conditions>")
    if weather == "noweather" and field is None and not you_parts and not opp_parts:
        words.append("<conditions_empty>")
    else:
        words.append(weather)
        if field is not None:
            words.append(field)
        if you_parts:
            words.extend(["<you>", *you_parts, "<end_you>"])
        else:
            words.append("<you_empty>")
        if opp_parts:
            words.extend(["<opponent>", *opp_parts, "<end_opponent>"])
        else:
            words.append("<opponent_empty>")
        words.append("<end_conditions>")

    if is_terminal:
        if getattr(battle, "won", False):
            terminal = "won"
        elif getattr(battle, "lost", False):
            terminal = "lost"
        else:
            terminal = "tie"
        words.extend(["<terminal>", terminal, "<end_terminal>"])
    words.append("<eos>")
    return words


def state_block(
    tokenizer: PokemonTokenizer,
    battle,
    fmt: str,
    *,
    is_force_switch: bool = False,
    override_active=None,
    is_terminal: bool = False,
) -> np.ndarray:
    return tokenize_words(
        tokenizer,
        state_words(
            battle,
            fmt,
            is_force_switch=is_force_switch,
            override_active=override_active,
            is_terminal=is_terminal,
        ),
    )
