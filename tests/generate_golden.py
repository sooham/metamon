import argparse
import orjson
import os
from typing import Any, Dict, List, Optional

from tests.helpers import (
    load_raw_replay,
    run_forward_fill,
)
from metamon.backend.replay_parser.backward import backward_fill, POVReplay
from metamon.backend.replay_parser.replay_state import (
    Action,
    Move,
    Nothing,
    Pokemon,
    Replacement,
    Turn,
)
from metamon.backend.replay_parser.pe_datatypes import (
    PEField,
    PESideCondition,
)
from metamon.backend.replay_parser.exceptions import WarningFlags
from metamon.backend.team_prediction.predictor import (
    NoPredictor,
)


GOLDEN_BATTLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "backend", "golden_outputs")
)


def get_golden_battle_filename(battle_id):
    return os.path.join(GOLDEN_BATTLE_DIR, battle_id, "raw.json")


def get_golden_battle_raw_replay(battle_id):
    raw_filename = os.path.join(GOLDEN_BATTLE_DIR, battle_id, "raw.json")
    return load_raw_replay(raw_filename)


def get_golden_battle_forward(battle_id):
    return os.path.join(GOLDEN_BATTLE_DIR, battle_id, "forward.json")


def get_golden_battle_win_pov(battle_id):
    return os.path.join(GOLDEN_BATTLE_DIR, battle_id, "win_pov.json")


def get_golden_battle_loss_pov(battle_id):
    return os.path.join(GOLDEN_BATTLE_DIR, battle_id, "loss_pov.json")


# ---------------------------------------------------------------------------
# Serialization helpers for POV golden files
# ---------------------------------------------------------------------------

def _serialize_parsed_replay(replay) -> Dict[str, Any]:
    """Serialize a ParsedReplay to a dict for forward.json golden files."""
    return {
        "gameid": replay.gameid,
        "time_played": replay.time_played.isoformat(),
        "format": replay.format,
        "gen": replay.gen,
        "winner": replay.winner.name if replay.winner else None,
        "players": replay.players,
        "ratings": replay.ratings,
        "rules": replay.rules,
        "check_warnings": [w.name for w in replay.check_warnings],
        "showteam_data": replay.showteam_data,
        "turns": [_serialize_turn(t) for t in replay.turnlist],
    }

def _serialize_move(move: Optional[Move]) -> Optional[Dict[str, Any]]:
    if move is None:
        return None
    return {
        "name": move.name,
        "pp": move.pp,
        "max_pp": move.maximum_pp,
    }


def _serialize_moves_dict(moves: Dict[str, Move]) -> List[Dict[str, Any]]:
    return [_serialize_move(m) for m in moves.values()]


def _serialize_pokemon(p: Optional[Pokemon]) -> Optional[Dict[str, Any]]:
    if p is None:
        return None
    return {
        "name": p.name,
        "had_name": p.had_name,
        "unique_id": p.unique_id,
        "lvl": p.lvl,
        "current_hp": p.current_hp,
        "max_hp": p.max_hp,
        "status": p.status.name,
        "type": p.type,
        "had_type": p.had_type,
        "active_item": p.active_item if not isinstance(p.active_item, Nothing) else p.active_item.name,
        "had_item": p.had_item if not isinstance(p.had_item, Nothing) else p.had_item.name,
        "active_ability": p.active_ability if not isinstance(p.active_ability, Nothing) else p.active_ability.name,
        "had_ability": p.had_ability if not isinstance(p.had_ability, Nothing) else p.had_ability.name,
        "moves": _serialize_moves_dict(p.moves),
        "had_moves": _serialize_moves_dict(p.had_moves),
        "boosts": p.boosts.to_dict(),
        "tera_type": p.tera_type if not isinstance(p.tera_type, Nothing) else p.tera_type.name,
        "transformed_into": p.transformed_into.unique_id if p.transformed_into else None,
        "effects": {e.name: v for e, v in p.effects.items()},
        "last_used_move": _serialize_move(p.last_used_move),
        "transformed_this_turn": p.transformed_this_turn,
    }


def _serialize_pokemon_list(poke_list: List[Optional[Pokemon]]) -> List[Optional[Dict[str, Any]]]:
    return [_serialize_pokemon(p) for p in poke_list]


def _serialize_action(a: Optional[Action]) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    return {
        "name": a.name,
        "is_switch": a.is_switch,
        "is_noop": a.is_noop,
        "is_tera": a.is_tera,
        "is_revival": a.is_revival,
        "user": a.user.unique_id if a.user else None,
        "target": a.target.unique_id if a.target else None,
    }


def _serialize_action_list(actions: List[Optional[Action]]) -> List[Optional[Dict[str, Any]]]:
    return [_serialize_action(a) for a in actions]


def _serialize_conditions(conds: Dict[PESideCondition, int]) -> Dict[str, int]:
    return {k.name: v for k, v in conds.items()}


def _serialize_field(field: Dict[PEField, int]) -> Dict[str, int]:
    return {k.name: v for k, v in field.items()}


def _serialize_weather(w) -> Optional[str]:
    if w is None or isinstance(w, Nothing):
        return w.name if isinstance(w, Nothing) else None
    return w.name


def _serialize_replacement(r: Replacement) -> Dict[str, Any]:
    return {
        "replaced": r.replaced.unique_id if r.replaced else None,
        "replaced_with": r.replaced_with.unique_id if r.replaced_with else None,
        "turn_range": list(r.turn_range),
    }


def _serialize_turn(t: Turn) -> Dict[str, Any]:
    return {
        "turn_number": t.turn_number,
        "pokemon_1": _serialize_pokemon_list(t.pokemon_1),
        "pokemon_2": _serialize_pokemon_list(t.pokemon_2),
        "active_pokemon_1": _serialize_pokemon_list(t.active_pokemon_1),
        "active_pokemon_2": _serialize_pokemon_list(t.active_pokemon_2),
        "moves_1": _serialize_action_list(t.moves_1),
        "moves_2": _serialize_action_list(t.moves_2),
        "choices_1": _serialize_action_list(t.choices_1),
        "choices_2": _serialize_action_list(t.choices_2),
        "weather": _serialize_weather(t.weather),
        "battle_field": _serialize_field(t.battle_field),
        "conditions_1": _serialize_conditions(t.conditions_1),
        "conditions_2": _serialize_conditions(t.conditions_2),
        "replacements_1": [_serialize_replacement(r) for r in t.replacements_1],
        "replacements_2": [_serialize_replacement(r) for r in t.replacements_2],
        "is_force_switch": t.is_force_switch,
        "can_tera_1": t.can_tera_1,
        "can_tera_2": t.can_tera_2,
        "teampreview_1": _serialize_pokemon_list(t.teampreview_1),
        "teampreview_2": _serialize_pokemon_list(t.teampreview_2),
        "subturns": [
            {
                "team": s.team,
                "slot": s.slot,
                "action": _serialize_action(s.action),
                "turn": _serialize_turn(s.turn) if s.turn else None,
            }
            for s in t.subturns
        ],
    }


def _serialize_pov_replay(pov: POVReplay) -> Dict[str, Any]:
    return {
        "gameid": pov.gameid,
        "format": pov.format,
        "gen": pov.gen,
        "winner": pov.winner,
        "from_p1_pov": pov.from_p1_pov,
        "rating": pov.rating,
        "check_warnings": [w.name for w in pov.check_warnings],
        "states": [_serialize_turn(t) for t in pov.povturnlist],
        "actions": _serialize_action_list(
            [slot[0] for slot in pov.actionlist]
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("battle_id", help="Battle ID (e.g. gen1ou-316031019)")
    parser.add_argument(
        "--predictor",
        type=str,
        choices=["none", "naive"],
        default="none",
        help='Team Predictor to use ("none" or "naive"), defaults to "none"',
    )
    args = parser.parse_args()
    battle_id = args.battle_id
    predictor = None
    if args.predictor == "none":
        predictor = NoPredictor()

    golden_raw_filename = get_golden_battle_filename(battle_id)
    parsed_replay = run_forward_fill(golden_raw_filename)

    # Serialize the parsed replay as forward.json in the battle's golden directory
    forward_output_path = get_golden_battle_forward(battle_id)
    os.makedirs(os.path.dirname(forward_output_path), exist_ok=True)
    with open(forward_output_path, "wb") as f:
        f.write(orjson.dumps(
            _serialize_parsed_replay(parsed_replay),
            option=orjson.OPT_INDENT_2,
        ))

    # Run the backward fill
    pov_p1, pov_p2 = backward_fill(parsed_replay, team_predictor=predictor)

    # Determine WIN vs LOSS and serialize
    win_pov = pov_p1 if pov_p1.winner else pov_p2
    loss_pov = pov_p2 if pov_p1.winner else pov_p1

    win_output_path = get_golden_battle_win_pov(battle_id)
    with open(win_output_path, "wb") as f:
        f.write(orjson.dumps(
            _serialize_pov_replay(win_pov),
            option=orjson.OPT_INDENT_2,
        ))

    loss_output_path = get_golden_battle_loss_pov(battle_id)
    with open(loss_output_path, "wb") as f:
        f.write(orjson.dumps(
            _serialize_pov_replay(loss_pov),
            option=orjson.OPT_INDENT_2,
        ))

    print(f"Golden files written for {battle_id}:")
    print(f"  forward:  {forward_output_path}")
    print(f"  win_pov:  {win_output_path}")
    print(f"  loss_pov: {loss_output_path}")


if __name__ == "__main__":
    main()