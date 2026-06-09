"""
Tests for Pokemon state tracking during forward parsing.

Checks HP, status, moves, abilities, items, boosts, and other per-Pokemon
properties for consistency.
"""

import pytest
from metamon.backend.replay_parser.replay_state import Nothing, PEStatus
from metamon.backend.replay_parser.exceptions import WarningFlags


class TestPokemonState:
    """State-tracking invariants for individual Pokemon."""

    # -- HP -----------------------------------------------------------------

    def test_hp_in_valid_range(self, parsed_replay):
        """0 <= current_hp <= max_hp for every Pokemon at every turn."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None or p.current_hp is None or p.max_hp is None:
                    continue
                assert 0 <= p.current_hp <= p.max_hp, (
                    f"{p.name} HP {p.current_hp}/{p.max_hp} at turn {turn.turn_number}"
                )

    def test_fainted_pokemon_have_zero_hp(self, parsed_replay):
        """If status == FNT, current_hp must be 0."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is not None and p.status == PEStatus.FNT:
                    assert p.current_hp == 0, (
                        f"{p.name} is FNT but has {p.current_hp} HP at turn {turn.turn_number}"
                    )

    # -- Moves ---------------------------------------------------------------

    def test_moveset_size_at_most_4(self, parsed_replay):
        """No Pokemon exceeds 4 moves unless explained by Transform/Mimic/Zoroark."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None or p.transformed_into is not None:
                    continue
                # Known exceptions that can cause extra moves
                skip = (
                    "Mimic" in p.had_moves
                    or parsed_replay.has_warning(WarningFlags.ZOROARK)
                    or parsed_replay.has_warning(WarningFlags.TRANSFORM)
                )
                if skip:
                    continue
                assert len(p.moves) <= 4, (
                    f"{p.name} has {len(p.moves)} moves: {list(p.moves.keys())} "
                    f"at turn {turn.turn_number}"
                )
                assert len(p.had_moves) <= 4, (
                    f"{p.name} has {len(p.had_moves)} had_moves: {list(p.had_moves.keys())} "
                    f"at turn {turn.turn_number}"
                )

    def test_pp_is_non_negative(self, parsed_replay):
        """Move PP is never negative."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None:
                    continue
                for m in p.moves.values():
                    assert m.pp >= 0, (
                        f"{p.name} move {m.name} has PP={m.pp} at turn {turn.turn_number}"
                    )
                for m in p.had_moves.values():
                    assert m.pp >= 0, (
                        f"{p.name} had_move {m.name} has PP={m.pp} at turn {turn.turn_number}"
                    )

    def test_moves_have_valid_names(self, parsed_replay):
        """Every Move object has a non-empty name."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None:
                    continue
                for m in p.moves.values():
                    assert isinstance(m.name, str) and len(m.name) > 0
                for m in p.had_moves.values():
                    assert isinstance(m.name, str) and len(m.name) > 0

    # -- Boosts --------------------------------------------------------------

    def test_boosts_in_valid_range(self, parsed_replay):
        """Stat stages are always in [-6, 6]."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None:
                    continue
                for attr in p.boosts.stat_attrs:
                    val = getattr(p.boosts, attr)
                    assert -6 <= val <= 6, (
                        f"{p.name} {attr}={val} at turn {turn.turn_number}"
                    )

    # -- Names & identity ----------------------------------------------------

    def test_had_name_is_stable(self, parsed_replay):
        """had_name never changes for a given unique_id."""
        seen = {}
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None or p.had_name is None:
                    continue
                if p.unique_id in seen:
                    assert seen[p.unique_id] == p.had_name, (
                        f"had_name changed: {seen[p.unique_id]} -> {p.had_name}"
                    )
                else:
                    seen[p.unique_id] = p.had_name

    def test_every_pokemon_has_unique_id(self, parsed_replay):
        """Every non-None Pokemon has a unique_id string."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is not None:
                    assert isinstance(p.unique_id, str) and len(p.unique_id) > 0

    def test_every_pokemon_has_name(self, parsed_replay):
        """Every non-None Pokemon has a non-empty name."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is not None:
                    assert isinstance(p.name, str) and len(p.name) > 0

    # -- Gen-specific --------------------------------------------------------

    def test_tera_only_in_gen9(self, parsed_replay):
        """Tera types are only set (non-NO_TERA_TYPE) in gen 9."""
        for turn in parsed_replay.turnlist:
            for p in turn.all_pokemon:
                if p is None or p.tera_type is None:
                    continue
                if p.tera_type != Nothing.NO_TERA_TYPE:
                    assert parsed_replay.gen == 9, (
                        f"{p.name} has tera_type={p.tera_type} in gen {parsed_replay.gen}"
                    )
