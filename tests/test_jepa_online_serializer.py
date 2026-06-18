from types import SimpleNamespace

from metamon.jepa.online_serializer import (
    action_words,
    is_force_switch_state,
    state_words,
    team_context_words,
)


def _move(name: str, typ: str = "Normal", category: str = "Physical"):
    return SimpleNamespace(id=name, type=typ, category=category)


def _pokemon(
    species: str,
    *,
    hp: float = 1.0,
    types: list[str] | None = None,
    moves: list | None = None,
    active: bool = False,
    fainted: bool = False,
    status=None,
    item=None,
    ability=None,
    gender=None,
):
    return SimpleNamespace(
        species=species,
        base_species=species,
        current_hp_fraction=hp,
        types=types or ["Normal"],
        moves={m.id: m for m in (moves or [])},
        boosts={},
        effects={},
        active=active,
        fainted=fainted,
        status=status,
        item=item,
        ability=ability,
        gender=gender,
        previous_move=None,
    )


def _battle(**overrides):
    active = _pokemon(
        "Jynx",
        types=["Ice", "Psychic"],
        moves=[_move("blizzard", "Ice", "Special"), _move("lovelykiss", "Normal", "Status")],
        active=True,
    )
    bench = _pokemon(
        "Alakazam",
        hp=0.75,
        types=["Psychic"],
        moves=[_move("psychic", "Psychic", "Special")],
    )
    opponent = _pokemon(
        "Starmie",
        hp=0.63,
        types=["Water", "Psychic"],
        moves=[_move("surf", "Water", "Special")],
        active=True,
    )
    values = dict(
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        team={"jynx": active, "alakazam": bench},
        teampreview_opponent_team=[opponent],
        turn=2,
        gen=1,
        force_switch=False,
        available_moves=list(active.moves.values()),
        available_switches=[bench],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        can_tera=None,
        won=False,
        lost=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_team_context_matches_training_header_shape_with_preview():
    battle = _battle(gen=5)

    words = team_context_words(battle, "gen5ou")

    assert words[:2] == ["<begin_team>", "<poke1>"]
    assert "<begin_opponent_team>" in words
    assert words[words.index("<begin_opponent_team>"):] == [
        "<begin_opponent_team>",
        "<poke1>",
        "starmie",
        "<end_poke1>",
        "<end_opponent_team>",
    ]

    move_idx = words.index("<move>")
    assert words[move_idx:move_idx + 3] == ["<move>", "psychic", "<end_move>"]


def test_forced_switch_state_uses_training_conditions_shape():
    battle = _battle(
        gen=9,
        weather={"RainDance": 3},
        fields={"Trick Room": 2},
        side_conditions={"Reflect": 1, "Spikes": 2},
        opponent_side_conditions={"Stealth Rock": 1},
        can_tera="Water",
    )

    words = state_words(battle, "gen9ou", is_force_switch=True)

    conditions = words[words.index("<conditions>") : words.index("<eos>")]
    assert conditions == [
        "<conditions>",
        "raindance",
        "trickroom",
        "<you>",
        "forceswitch",
        "cantera",
        "reflect",
        "spikes_2",
        "<end_you>",
        "<opponent>",
        "stealthrock",
        "<end_opponent>",
        "<end_conditions>",
    ]
    assert "<conditions_empty>" not in conditions


def test_force_switch_detection_accepts_request_flag_and_switch_only_request():
    battle = _battle(force_switch=[True])
    assert is_force_switch_state(battle)

    switch_only = _battle(force_switch=False, available_moves=[])
    assert is_force_switch_state(switch_only)


def test_action_words_use_training_action_tags():
    assert action_words("switch Alakazam", opponent=False) == [
        "<chosen_move>",
        "switch",
        "alakazam",
        "<end_chosen_move>",
    ]
    assert action_words("switch: Alakazam", opponent=False) == [
        "<chosen_move>",
        "switch",
        "alakazam",
        "<end_chosen_move>",
    ]
