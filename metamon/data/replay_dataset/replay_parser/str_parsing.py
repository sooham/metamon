import re
from typing import Optional

from metamon.data.replay_dataset.replay_parser.exceptions import *
from poke_env.data import to_id_str


def parse_hp_fraction(raw: str) -> tuple[int, int]:
    fracs = re.findall(r"\b\d+/\d+\b", raw)
    if len(fracs) != 1 or "/" not in fracs[0]:
        raise StrParsingException("parse_hp_fraction", raw)
    num, den = [int(x) for x in fracs[0].split("/")]
    return num, den


def parse_condition(raw: str) -> str:
    condition = raw.replace("move:", "").lower().strip().replace(" ", "")
    if "ability" in raw or ":" in condition:
        raise StrParsingException("parse_condition", raw)
    return condition


def parse_move_from_extra(raw: str) -> str:
    move = re.sub(f"\[from\]\s?move:\s?", "", raw).strip()
    if "[from]" not in raw or "move" not in raw or not move:
        raise StrParsingException("parse_move_from_extra", raw)
    return move


def parse_mon_from_extra(raw: str) -> str:
    mon = re.sub(f"\[of\]\s?", "", raw).strip()
    if "[of]" not in raw or not mon:
        raise StrParsingException("parse_mon_from_extra", raw)
    return mon


def parse_ability_from_extra(raw: str) -> str:
    ability = re.sub(f"\[from\]\s?ability:\s?", "", raw).strip()
    if "[from]" not in raw or "ability" not in raw or not ability:
        raise StrParsingException("parse_ability_from_extra", raw)
    return ability


def parse_item_from_extra(raw: str) -> str:
    item = re.sub(f"\[from\]\s?item:\s?", "", raw).strip()
    if "[from]" not in raw or "item" not in raw or not item:
        raise StrParsingException("parse_item_from_extra", raw)
    return item


def parse_extra(raw: str) -> str:
    *_, info = raw.split("[from]")
    if "[from]" not in raw or not info:
        raise StrParsingException("parse_extra", raw)
    info = info.strip()
    if info.startswith("move:"):
        info = info.replace("move:", "").strip()
    return info


def parse_effect(effect: str) -> str:
    for p in ["item", "move", "ability"]:
        effect = effect.replace(p, "")
    return to_id_str(effect)


def parse_ability(raw: str) -> str:
    ability = raw.replace("ability:", "").strip()
    if not ability or ":" in ability:
        raise StrParsingException("parse_ability", raw)
    return ability


def parse_from_effect_of(message: list[str]) -> tuple[Optional[str]]:
    """
    for the misc. "[from] item/ability/move [of] id: pokemon" messages
    """
    item, ability, move, of_pokemon = None, None, None, None
    for arg in message:
        if "[from]" in arg:
            if "item" in arg:
                item = parse_item_from_extra(arg)
            if "ability" in arg:
                ability = parse_ability_from_extra(arg)
            if "move" in arg:
                move = parse_move_from_extra(arg)
        if "[of]" in arg:
            of_pokemon = parse_mon_from_extra(arg)
    return item, ability, move, of_pokemon
