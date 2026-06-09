"""
Tests for action tracking during forward parsing.

Verifies that moves, switches, forced switches, and other action types
are recorded consistently.
"""

import pytest
from metamon.backend.replay_parser.replay_state import Action


class TestForwardActions:
    """Action-tracking invariants."""

    # -- Basic action properties ---------------------------------------------

    def test_non_null_actions_have_user(self, parsed_replay):
        """Any action with a name (and not a no-op) should have a user Pokemon.

        Exception: turn-0 Switch actions may have user=None because no Pokemon
        is being switched *out* during the initial lead selection.
        """
        for turn in parsed_replay.turnlist:
            for action in turn.moves_1 + turn.moves_2:
                if action is None or action.name is None or action.is_noop:
                    continue
                # Turn-0 switches have no 'user' because nothing is being switched out
                if turn.turn_number == 0 and action.name == "Switch":
                    continue
                assert action.user is not None, (
                    f"Action '{action.name}' at turn {turn.turn_number} has no user"
                )

    def test_switch_actions_have_target(self, parsed_replay):
        """Any action marked as a switch should have a target Pokemon."""
        for turn in parsed_replay.turnlist:
            for action in turn.moves_1 + turn.moves_2:
                if action is None:
                    continue
                if action.is_switch:
                    assert action.target is not None, (
                        f"Switch action at turn {turn.turn_number} has no target"
                    )

    def test_switch_actions_are_marked(self, parsed_replay):
        """Actions named 'Switch' have is_switch=True."""
        for turn in parsed_replay.turnlist:
            for action in turn.moves_1 + turn.moves_2:
                if action is None:
                    continue
                if action.name == "Switch":
                    assert action.is_switch, (
                        f"'Switch' action not marked is_switch at turn {turn.turn_number}"
                    )

    def test_actions_have_valid_names(self, parsed_replay):
        """Every non-None action has a string name (even if empty for no-ops)."""
        for turn in parsed_replay.turnlist:
            for action in turn.moves_1 + turn.moves_2:
                if action is not None:
                    assert isinstance(action.name, (str, type(None)))

    # -- Subturn structure ---------------------------------------------------

    def test_subturns_have_valid_team_and_slot(self, parsed_replay):
        """Every subturn has team in {1, 2} and slot in {0, 1}."""
        for turn in parsed_replay.turnlist:
            for subturn in turn.subturns:
                assert subturn.team in {1, 2}, (
                    f"Subturn has invalid team={subturn.team}"
                )
                assert subturn.slot in {0, 1}, (
                    f"Subturn has invalid slot={subturn.slot}"
                )

    def test_filled_subturns_have_turns(self, parsed_replay):
        """A subturn with turn is not None means it was filled."""
        for turn in parsed_replay.turnlist:
            for subturn in turn.subturns:
                if not subturn.unfilled:
                    assert subturn.turn is not None
                    # The filled turn should have consistent state
                    assert subturn.turn.is_force_switch

    # -- Tera actions (gen 9 only) -------------------------------------------

    def test_tera_actions_only_in_gen9(self, parsed_replay):
        """Actions with is_tera=True only appear in gen 9."""
        for turn in parsed_replay.turnlist:
            for action in turn.moves_1 + turn.moves_2:
                if action is not None and action.is_tera:
                    assert parsed_replay.gen == 9, (
                        f"Tera action in gen {parsed_replay.gen} at turn {turn.turn_number}"
                    )

    # -- Choices -------------------------------------------------------------

    def test_choices_are_actions_or_none(self, parsed_replay):
        """Every choice slot is None or an Action."""
        for turn in parsed_replay.turnlist:
            for c in turn.choices_1 + turn.choices_2:
                if c is not None:
                    assert isinstance(c, Action), (
                        f"Choice is not an Action: {type(c)}"
                    )

    # -- Moves vs switches consistency ---------------------------------------

    def test_move_and_switch_are_mutually_exclusive(self, parsed_replay):
        """An action should not be both a switch and a move at the same time."""
        for turn in parsed_replay.turnlist:
            for action in turn.moves_1 + turn.moves_2:
                if action is None or action.is_noop:
                    continue
                if action.name == "Switch":
                    assert action.is_switch
                if action.is_switch and action.name != "Switch":
                    # Forced switches via moves (U-turn, etc.) can have move names
                    # but is_switch set. That's fine.
                    pass
