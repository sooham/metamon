"""
Structural invariant tests for forward-parsed replays.

These tests check properties of the ``ParsedReplay`` object that should hold
for *every* successfully-parsed replay, regardless of generation.
"""

import pytest
from metamon.backend.replay_parser.replay_state import Winner


class TestForwardStructure:
    """Basic structural checks on ParsedReplay."""

    def test_gen_is_set(self, parsed_replay):
        """Generation is always one of {1, 2, 3, 4, 9}."""
        assert parsed_replay.gen in {1, 2, 3, 4, 9}, f"Unexpected gen: {parsed_replay.gen}"

    def test_players_are_set(self, parsed_replay):
        """Both player names are non-None strings."""
        assert isinstance(parsed_replay.players[0], str)
        assert isinstance(parsed_replay.players[1], str)
        assert len(parsed_replay.players[0]) > 0
        assert len(parsed_replay.players[1]) > 0

    def test_winner_is_set(self, parsed_replay):
        """Winner is one of the Winner enum values."""
        assert parsed_replay.winner is not None
        assert isinstance(parsed_replay.winner, Winner)

    def test_turnlist_nonempty(self, parsed_replay):
        """Every parsed replay has at least one turn (turn 0)."""
        assert len(parsed_replay.turnlist) >= 1

    def test_turn_numbers_are_sequential(self, parsed_replay):
        """Turn numbers start at 0 and increase by 1 each turn."""
        for i, turn in enumerate(parsed_replay.turnlist):
            assert turn.turn_number == i, (
                f"Expected turn_number={i}, got {turn.turn_number}"
            )

    def test_teamsize_at_most_6(self, parsed_replay):
        """No team exceeds 6 Pokemon slots (Species Clause)."""
        for turn in parsed_replay.turnlist:
            assert len(turn.pokemon_1) <= 6, (
                f"Team 1 has {len(turn.pokemon_1)} slots at turn {turn.turn_number}"
            )
            assert len(turn.pokemon_2) <= 6, (
                f"Team 2 has {len(turn.pokemon_2)} slots at turn {turn.turn_number}"
            )

    def test_active_slots_are_length_2(self, parsed_replay):
        """Active slots are always length 2 (singles format; slot 'b' is unused)."""
        for turn in parsed_replay.turnlist:
            assert len(turn.active_pokemon_1) == 2
            assert len(turn.active_pokemon_2) == 2

    def test_pokemon_stay_on_same_team(self, parsed_replay):
        """No Pokemon unique_id appears on both teams."""
        p1_ids = set()
        p2_ids = set()
        for turn in parsed_replay.turnlist:
            for p in turn.pokemon_1:
                if p is not None:
                    p1_ids.add(p.unique_id)
            for p in turn.pokemon_2:
                if p is not None:
                    p2_ids.add(p.unique_id)
        overlap = p1_ids & p2_ids
        assert not overlap, (
            f"Pokemon IDs found on both teams: {overlap}"
        )

    def test_pokemon_ids_never_vanish_from_team(self, parsed_replay):
        """Once a Pokemon appears on a team, it stays on that team's roster."""
        p1_seen = {}   # unique_id -> first_turn_seen
        p2_seen = {}
        for turn in parsed_replay.turnlist:
            for p in turn.pokemon_1:
                if p is not None and p.unique_id not in p1_seen:
                    p1_seen[p.unique_id] = turn.turn_number
            for p in turn.pokemon_2:
                if p is not None and p.unique_id not in p2_seen:
                    p2_seen[p.unique_id] = turn.turn_number

        # After first seen, it should appear in every subsequent turn
        for turn in parsed_replay.turnlist:
            for uid, first_turn in p1_seen.items():
                if turn.turn_number >= first_turn:
                    ids_in_team = {p.unique_id for p in turn.pokemon_1 if p is not None}
                    assert uid in ids_in_team, (
                        f"Pokemon {uid} vanished from team 1 at turn {turn.turn_number}"
                    )
            for uid, first_turn in p2_seen.items():
                if turn.turn_number >= first_turn:
                    ids_in_team = {p.unique_id for p in turn.pokemon_2 if p is not None}
                    assert uid in ids_in_team, (
                        f"Pokemon {uid} vanished from team 2 at turn {turn.turn_number}"
                    )

    def test_showteam_data_is_dict_or_none(self, parsed_replay):
        """showteam_data is either None or a dict (not malformed)."""
        if parsed_replay.showteam_data is not None:
            assert isinstance(parsed_replay.showteam_data, dict)

    def test_format_is_nonempty_string(self, parsed_replay):
        """The format string is present and non-empty."""
        assert isinstance(parsed_replay.format, str)
        assert len(parsed_replay.format) > 0
