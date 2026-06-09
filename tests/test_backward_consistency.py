"""
Consistency tests for backward-filled POVReplay objects.

Checks properties that span across turns and between the two POVs.
"""

import pytest
from metamon.backend.replay_parser.replay_state import PEStatus


class TestBackwardConsistency:
    """Cross-turn and cross-POV consistency checks."""

    def test_observed_pokemon_have_hp(self, pov_replays):
        """Any Pokemon that has current_hp set also has max_hp set."""
        for fmt, (p1, p2) in pov_replays.items():
            for pov in (p1, p2):
                for turn in pov.povturnlist:
                    for p in turn.all_pokemon:
                        if p is None:
                            continue
                        if p.current_hp is not None:
                            assert p.max_hp is not None, (
                                f"{p.name} has current_hp={p.current_hp} but max_hp is None "
                                f"in {fmt} turn {turn.turn_number}"
                            )

    def test_fainted_pokemon_not_in_available_switches(self, pov_replays):
        """Pokemon with FNT status should not appear as available switches."""
        for fmt, (p1, p2) in pov_replays.items():
            for turn in p1.povturnlist:
                for p in turn.available_switches_1:
                    if p is not None:
                        assert p.status != PEStatus.FNT, (
                            f"Fainted {p.name} in available_switches_1 at turn {turn.turn_number}"
                        )
            for turn in p2.povturnlist:
                for p in turn.available_switches_2:
                    if p is not None:
                        assert p.status != PEStatus.FNT, (
                            f"Fainted {p.name} in available_switches_2 at turn {turn.turn_number}"
                        )

    def test_active_pokemon_not_in_available_switches(self, pov_replays):
        """Active Pokemon should not appear as available switches."""
        for fmt, (p1, p2) in pov_replays.items():
            for turn in p1.povturnlist:
                active_ids = {
                    a.unique_id for a in turn.active_pokemon_1 if a is not None
                }
                switch_ids = {
                    s.unique_id for s in turn.available_switches_1 if s is not None
                }
                overlap = active_ids & switch_ids
                assert not overlap, (
                    f"Active Pokemon also in switches at turn {turn.turn_number}: {overlap}"
                )
            for turn in p2.povturnlist:
                active_ids = {
                    a.unique_id for a in turn.active_pokemon_2 if a is not None
                }
                switch_ids = {
                    s.unique_id for s in turn.available_switches_2 if s is not None
                }
                overlap = active_ids & switch_ids
                assert not overlap, (
                    f"Active Pokemon also in switches at turn {turn.turn_number}: {overlap}"
                )

    def test_both_povs_share_same_underlying_replay(self, pov_replays):
        """Both POVs are derived from the same gameid and gen."""
        for fmt, (p1, p2) in pov_replays.items():
            assert p1.gameid == p2.gameid
            assert p1.gen == p2.gen
            assert p1.format == p2.format

    def test_opponent_info_is_forward_only(self, pov_replays):
        """Opponent team info must not be backfilled from future turns.

        On turn 0, the opponent bench should be empty (no Pokemon revealed yet
        besides the active). Opponent moves/items/abilities should only appear
        as they are revealed during the forward pass.
        """
        for fmt, (p1, p2) in pov_replays.items():
            # POV1: player=pokemon_1, opponent=pokemon_2
            turn0 = p1.povturnlist[0]
            opponent_team = turn0.pokemon_2
            # Count how many opponent Pokemon are known (non-None)
            known_opponents = [p for p in opponent_team if p is not None]
            # On turn 0, only the active should be known. Bench should be None.
            # Allow up to 2 known (active + maybe one revealed in team preview
            # for gen9) but never the full team.
            max_known = 1 if "gen9" not in fmt else 6  # gen9 has team preview
            assert len(known_opponents) <= max_known, (
                f"{fmt}: turn 0 has {len(known_opponents)} known opponent Pokemon "
                f"(expected <= {max_known}). Backfill may be leaking."
            )
            # If the full team is known (gen9 team preview), they should have
            # no moves revealed on turn 0 (only the active might have moves).
            if len(known_opponents) > 1:
                for p in known_opponents:
                    # Active may have some moves; bench should have none
                    if p is not turn0.active_pokemon_2[0]:
                        assert len(p.had_moves) == 0, (
                            f"{fmt}: turn 0 bench Pokemon {p.name} has "
                            f"{len(p.had_moves)} moves (expected 0). "
                            f"Opponent backfill may be leaking."
                        )


class TestPredictionConsistency:
    """Tests that verify NaiveUsagePredictor fills player moves correctly
    while keeping opponent info forward-observed across both POVs."""

    def test_player_pokemon_have_predicted_moves(self, pov_replays_predicted):
        """Every player Pokemon on turn 0 should have > 0 moves (predicted).

        Even Pokemon that haven't used a move yet should have predicted
        moves from usage stats, since the player knows their own moveset.
        """
        for fmt, (p1, p2) in pov_replays_predicted.items():
            for pov, side_name in [(p1, "p1"), (p2, "p2")]:
                turn0 = pov.povturnlist[0]
                player_team = turn0.pokemon_1 if side_name == "p1" else turn0.pokemon_2
                for p in player_team:
                    if p is None:
                        continue
                    assert len(p.had_moves) > 0, (
                        f"{fmt} {side_name}: player Pokemon {p.name} has "
                        f"0 moves on turn 0. Prediction should have filled them."
                    )

    def test_prediction_fills_missing_moves(self, pov_replays_predicted):
        """Prediction should increase the number of known moves per Pokemon.

        The forward pass only sees moves that were actually used. After
        prediction, the player's Pokemon should have MORE moves than what
        was forward-observed (typically 4, unless moveset < 4).
        """
        for fmt, (p1, p2) in pov_replays_predicted.items():
            for pov, side_name in [(p1, "p1"), (p2, "p2")]:
                final_turn = pov.povturnlist[-1]
                player_team = (
                    final_turn.pokemon_1 if side_name == "p1" else final_turn.pokemon_2
                )
                for p in player_team:
                    if p is None:
                        continue
                    n_moves = len(p.had_moves)
                    assert n_moves >= 1, (
                        f"{fmt} {side_name} final turn: {p.name} has "
                        f"{n_moves} moves, expected >= 1"
                    )

    def test_opponent_moves_are_strictly_forward_observed(self, pov_replays_predicted):
        """Opponent Pokemon must NOT have predicted moves.

        Their move count should be exactly what was forward-observed
        (no usage-stat filling). On turn 0, this should be 0.
        """
        for fmt, (p1, p2) in pov_replays_predicted.items():
            for pov, side_name in [(p1, "p1"), (p2, "p2")]:
                turn0 = pov.povturnlist[0]
                opponent_team = (
                    turn0.pokemon_2 if side_name == "p1" else turn0.pokemon_1
                )
                active_opponent = (
                    turn0.active_pokemon_2 if side_name == "p1" else turn0.active_pokemon_1
                )
                for i, p in enumerate(opponent_team):
                    if p is None:
                        continue
                    if p not in active_opponent:
                        assert len(p.had_moves) == 0, (
                            f"{fmt} {side_name}: opponent bench[{i}] {p.name} has "
                            f"{len(p.had_moves)} moves on turn 0 (expected 0). "
                            f"Opponent prediction leak."
                        )

    def test_cross_pov_teams_are_swapped(self, pov_replays_predicted):
        """P1's opponent should be a subset of P2's player (prediction adds).

        Both POVs observe the same battle. The predictor fills unrevealed
        Pokemon for the PLAYER (filling None slots), so the player team
        may have more species than what the opponent sees. Therefore:
        - P1's opponent species ⊆ P2's player species
        - P2's opponent species ⊆ P1's player species
        """
        for fmt, (p1, p2) in pov_replays_predicted.items():
            final1 = p1.povturnlist[-1]
            final2 = p2.povturnlist[-1]

            p1_opponent_names = {
                p.name for p in final1.pokemon_2 if p is not None
            }
            p2_player_names = {
                p.name for p in final2.pokemon_2 if p is not None
            }
            assert p1_opponent_names <= p2_player_names, (
                f"{fmt}: P1 opponent {p1_opponent_names} ⊄ P2 player {p2_player_names}"
            )

            p2_opponent_names = {
                p.name for p in final2.pokemon_1 if p is not None
            }
            p1_player_names = {
                p.name for p in final1.pokemon_1 if p is not None
            }
            assert p2_opponent_names <= p1_player_names, (
                f"{fmt}: P2 opponent {p2_opponent_names} ⊄ P1 player {p1_player_names}"
            )

            assert len(p1_opponent_names & p2_player_names) > 0, (
                f"{fmt}: no shared species between P1 opponent and P2 player"
            )

    def test_prediction_only_on_player_side(self, pov_replays_predicted):
        """Player Pokemon should have moves filled; opponent should not.

        On the FINAL turn, P1's player team (pokemon_1) should have
        predicted moves, while P1's opponent team (pokemon_2) should
        only have forward-observed moves (typically fewer).
        """
        for fmt, (p1, p2) in pov_replays_predicted.items():
            final1 = p1.povturnlist[-1]
            final2 = p2.povturnlist[-1]

            for p in final1.pokemon_1:
                if p is None:
                    continue
                assert len(p.had_moves) >= 1, (
                    f"{fmt} P1 player {p.name}: {len(p.had_moves)} moves"
                )

            for p in final2.pokemon_2:
                if p is None:
                    continue
                assert len(p.had_moves) >= 1, (
                    f"{fmt} P2 player {p.name}: {len(p.had_moves)} moves"
                )
