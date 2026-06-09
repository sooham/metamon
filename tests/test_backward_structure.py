"""
Structural invariant tests for backward-filled POVReplay objects.

Checks that the output of ``backward_fill`` is well-formed.
"""

import pytest
from metamon.backend.replay_parser.replay_state import Winner


class TestBackwardStructure:
    """Basic structural checks on POVReplay."""

    def test_povturnlist_nonempty(self, pov_replays):
        """Every POV replay has at least one turn."""
        for fmt, (p1, p2) in pov_replays.items():
            assert len(p1.povturnlist) > 0, f"{fmt} p1 has empty povturnlist"
            assert len(p2.povturnlist) > 0, f"{fmt} p2 has empty povturnlist"

    def test_actionlist_length_matches_povturnlist(self, pov_replays):
        """actionlist and povturnlist are the same length."""
        for fmt, (p1, p2) in pov_replays.items():
            assert len(p1.actionlist) == len(p1.povturnlist), (
                f"{fmt} p1: actions={len(p1.actionlist)} != turns={len(p1.povturnlist)}"
            )
            assert len(p2.actionlist) == len(p2.povturnlist), (
                f"{fmt} p2: actions={len(p2.actionlist)} != turns={len(p2.povturnlist)}"
            )

    def test_pov_winner_is_consistent(self, pov_replays):
        """p1 wins ⇔ p2 loses (excluding ties)."""
        for fmt, (p1, p2) in pov_replays.items():
            if p1.replay.winner == Winner.TIE:
                assert not p1.winner, f"{fmt}: tie but p1.winner is True"
                assert not p2.winner, f"{fmt}: tie but p2.winner is True"
            else:
                assert p1.winner != p2.winner, (
                    f"{fmt}: p1.winner={p1.winner}, p2.winner={p2.winner} — "
                    f"should be opposites"
                )

    def test_format_is_preserved(self, pov_replays):
        """The battle format string is set correctly."""
        for fmt, (p1, p2) in pov_replays.items():
            assert p1.format == fmt, f"{fmt} p1 format is {p1.format}"
            assert p2.format == fmt, f"{fmt} p2 format is {p2.format}"

    def test_gen_is_set(self, pov_replays):
        """Generation is set on both POV replays."""
        for fmt, (p1, p2) in pov_replays.items():
            assert p1.gen is not None
            assert p2.gen is not None
            assert p1.gen == p2.gen

    def test_gameid_is_set(self, pov_replays):
        """Every POV replay has a non-empty gameid."""
        for fmt, (p1, p2) in pov_replays.items():
            assert isinstance(p1.gameid, str) and len(p1.gameid) > 0
            assert p1.gameid == p2.gameid

    def test_revealed_team_is_not_none(self, pov_replays):
        """revealed_team is set on both POV replays."""
        for fmt, (p1, p2) in pov_replays.items():
            assert p1.revealed_team is not None, f"{fmt} p1 has no revealed_team"
            assert p2.revealed_team is not None, f"{fmt} p2 has no revealed_team"

    def test_actionlist_items_are_lists_of_length_2(self, pov_replays):
        """Each entry in actionlist is a list of length 2 (singles format)."""
        for fmt, (p1, p2) in pov_replays.items():
            for alist in p1.actionlist:
                assert isinstance(alist, list)
                assert len(alist) == 2
            for alist in p2.actionlist:
                assert isinstance(alist, list)
                assert len(alist) == 2

    def test_rating_is_set(self, pov_replays):
        """Rating is set (may be 'Unrated' string or int)."""
        for fmt, (p1, p2) in pov_replays.items():
            assert p1.rating is not None
            assert p2.rating is not None
