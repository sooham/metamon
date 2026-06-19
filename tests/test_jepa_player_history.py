from types import SimpleNamespace

from metamon.jepa.player import BattleHistory, JEPAWorldModelPlayer


def _player():
    return JEPAWorldModelPlayer.__new__(JEPAWorldModelPlayer)


def test_previous_turn_results_put_cant_in_state_not_action():
    player = _player()
    battle = SimpleNamespace(_player_role="p1", opponent_active_pokemon=SimpleNamespace(previous_move=None))
    hist = BattleHistory(
        pending_player_action_text="move: Lovely Kiss",
        raw_messages=[
            "|move|p2a: Chansey|Thunderbolt|p1a: Jynx",
            "|cant|p1a: Jynx|par",
        ],
        pending_raw_message_start=0,
    )

    result = player._infer_previous_turn_results(battle, hist)

    assert result == (
        "move: Lovely Kiss",
        "cant par",
        "opponent move: Thunderbolt",
        "success",
    )


def test_previous_turn_results_keep_unknown_opponent_cant_out_of_action():
    player = _player()
    battle = SimpleNamespace(
        _player_role="p1",
        opponent_active_pokemon=SimpleNamespace(previous_move=SimpleNamespace(id="thunderbolt")),
    )
    hist = BattleHistory(
        pending_player_action_text="move: Blizzard",
        raw_messages=["|cant|p2a: Chansey|par"],
        pending_raw_message_start=0,
    )

    result = player._infer_previous_turn_results(battle, hist)

    assert result == (
        "move: Blizzard",
        "success",
        "unknown",
        None,
    )


def test_previous_turn_results_do_not_treat_forced_replacement_as_opponent_action():
    player = _player()
    battle = SimpleNamespace(_player_role="p1", opponent_active_pokemon=SimpleNamespace(previous_move=None))
    hist = BattleHistory(
        pending_player_action_text="move: Hyper Beam",
        raw_messages=[
            "|move|p2a: Chansey|Thunder Wave|p1a: Tauros",
            "|move|p1a: Tauros|Hyper Beam|p2a: Chansey",
            "|faint|p2a: Chansey",
            "|switch|p2a: Alakazam|Alakazam, M|100/100",
        ],
        pending_raw_message_start=0,
    )

    result = player._infer_previous_turn_results(battle, hist)

    assert result == (
        "move: Hyper Beam",
        "success",
        "opponent move: Thunder Wave",
        "success",
    )
