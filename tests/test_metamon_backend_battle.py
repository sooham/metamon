"""Tests for MetamonBackendBattle integration with UniversalState.from_Battle.

Verifies that opponent_bench, opponent_fainted, and fainted_pokemon are
correctly populated when converting a MetamonBackendBattle to a UniversalState
during live play.
"""

import logging

import pytest

from metamon.env.metamon_battle import MetamonBackendBattle
from metamon.interface import UniversalState
from metamon.backend.replay_parser.pe_datatypes import PEStatus


# ── helpers ──────────────────────────────────────────────────────────────


def _make_battle(player_role="p1", gen=1):
    """Create a MetamonBackendBattle and feed it basic init + first-turn messages.

    Sets up a 2-player Gen 1 OU battle with 6 Pokémon per side.  The player
    leads Starmie; the opponent leads Alakazam.  A minimal request is parsed
    so that `_player_role` is set and the state is ready for `from_Battle`.
    """
    battle = MetamonBackendBattle(
        battle_tag="battle-gen1ou-1",
        username="TestBot" if player_role == "p1" else "Opponent",
        logger=logging.getLogger("test"),
        save_replays=False,
        gen=gen,
    )

    # ── init messages (equivalent to make_skeleton) ──
    init_messages = [
        ["", "gen", "1"],
        ["", "tier", "[Gen 1] OU"],
        ["", "rule", "Species Clause: Limit one of each Pokémon"],
        ["", "player", "p1", "TestBot", ""],
        ["", "player", "p2", "Opponent", ""],
        ["", "teamsize", "p1", "6"],
        ["", "teamsize", "p2", "6"],
        ["", "poke", "p1", "Starmie", ""],
        ["", "poke", "p1", "Alakazam", ""],
        ["", "poke", "p1", "Chansey", ""],
        ["", "poke", "p1", "Exeggutor", ""],
        ["", "poke", "p1", "Snorlax", ""],
        ["", "poke", "p1", "Tauros", ""],
        ["", "poke", "p2", "Alakazam", ""],
        ["", "poke", "p2", "Zapdos", ""],
        ["", "poke", "p2", "Gengar", ""],
        ["", "poke", "p2", "Rhydon", ""],
        ["", "poke", "p2", "Lapras", ""],
        ["", "poke", "p2", "Jolteon", ""],
        ["", "start", ""],
        ["", "switch", "p1a: Starmie", "Starmie", "100/100"],
        ["", "switch", "p2a: Alakazam", "Alakazam", "100/100"],
        ["", "turn", "1"],
    ]
    for msg in init_messages:
        battle.parse_message(msg)

    # ── minimal request to set _player_role ──
    request = {
        "side": {
            "pokemon": [
                {
                    "ident": "p1: Starmie" if player_role == "p1" else "p2: Alakazam",
                    "details": "Starmie" if player_role == "p1" else "Alakazam",
                    "condition": "100/100",
                    "active": True,
                    "stats": {"atk": 5, "def": 5, "spa": 5, "spd": 5, "spe": 5},
                    "moves": ["psychic", "blizzard", "recover", "thunderwave"],
                    "baseAbility": "noability",
                    "item": "",
                }
            ]
        },
    }
    battle.parse_request(request)
    return battle


def _opponent_switch(battle, turn_num, species, hp="100/100"):
    """Feed messages for the opponent switching to *species* at *turn_num*."""
    battle.parse_message(["", "turn", str(turn_num)])
    battle.parse_message(
        ["", "switch", f"p2a: {species}", species, hp]
    )


def _our_switch(battle, turn_num, species, hp="100/100"):
    """Feed messages for our own Pokémon switching to *species* at *turn_num*."""
    battle.parse_message(["", "turn", str(turn_num)])
    battle.parse_message(
        ["", "switch", f"p1a: {species}", species, hp]
    )


def _opponent_faint(battle, turn_num, active_species):
    """Feed messages for the opponent's *active_species* fainting."""
    battle.parse_message(["", "turn", str(turn_num)])
    battle.parse_message(["", "faint", f"p2a: {active_species}"])


def _our_faint(battle, turn_num, active_species):
    """Feed messages for our own *active_species* fainting."""
    battle.parse_message(["", "turn", str(turn_num)])
    battle.parse_message(["", "faint", f"p1a: {active_species}"])


# ── tests ─────────────────────────────────────────────────────────────────


class TestMetamonBackendOpponentBench:
    """Tests that opponent_bench tracks switched-out opponent Pokémon."""

    def test_no_bench_on_turn_1(self):
        """On the first turn (no switches yet), opponent_bench is empty."""
        battle = _make_battle()
        us = UniversalState.from_Battle(battle)
        assert us.opponent_bench == []
        assert us.opponent_fainted == []

    def test_bench_after_one_opponent_switch(self):
        """After the opponent switches once, the old active appears in bench."""
        battle = _make_battle()
        _opponent_switch(battle, 2, "Zapdos")

        us = UniversalState.from_Battle(battle)

        assert len(us.opponent_bench) == 1
        assert us.opponent_bench[0].name == "alakazam"
        # Alakazam switched out at full HP
        assert us.opponent_bench[0].hp_pct == 1.0
        assert us.opponent_fainted == []

    def test_bench_after_two_opponent_switches(self):
        """After two opponent switches, both old actives appear in bench."""
        battle = _make_battle()
        _opponent_switch(battle, 2, "Zapdos")
        _opponent_switch(battle, 3, "Gengar")

        us = UniversalState.from_Battle(battle)

        assert len(us.opponent_bench) == 2
        bench_names = {p.name for p in us.opponent_bench}
        assert bench_names == {"alakazam", "zapdos"}
        assert us.opponent_fainted == []

    def test_bench_excludes_active(self):
        """The currently active opponent is NOT in opponent_bench."""
        battle = _make_battle()
        _opponent_switch(battle, 2, "Zapdos")

        us = UniversalState.from_Battle(battle)

        # Zapdos is active, Alakazam is on bench
        assert us.opponent_active_pokemon.name == "zapdos"
        bench_names = {p.name for p in us.opponent_bench}
        assert "zapdos" not in bench_names

    def test_bench_after_switch_back(self):
        """After switching back to a previously-seen Pokémon, bench is correct."""
        battle = _make_battle()
        # Alakazam -> Zapdos -> Alakazam (switch back)
        _opponent_switch(battle, 2, "Zapdos")
        _opponent_switch(battle, 3, "Alakazam")
        # Parse a request to refresh state
        _parse_minimal_request(battle, active_species="Alakazam")

        us = UniversalState.from_Battle(battle)

        assert us.opponent_active_pokemon.name == "alakazam"
        bench_names = {p.name for p in us.opponent_bench}
        assert bench_names == {"zapdos"}
        assert "alakazam" not in bench_names


def _parse_minimal_request(battle, active_species="Starmie"):
    """Parse a minimal request to keep the battle state consistent."""
    request = {
        "side": {
            "pokemon": [
                {
                    "ident": f"p1: {active_species}",
                    "details": active_species,
                    "condition": "100/100",
                    "active": True,
                    "stats": {"atk": 5, "def": 5, "spa": 5, "spd": 5, "spe": 5},
                    "moves": ["psychic"],
                    "baseAbility": "noability",
                    "item": "",
                }
            ]
        },
    }
    battle.parse_request(request)


class TestMetamonBackendFainted:
    """Tests that fainted_pokemon and opponent_fainted are correct.

    In the Showdown protocol, a fainted Pokémon stays in the active slot
    until a replacement switches in.  It only moves to the fainted lists
    AFTER the replacement switch-in (when it becomes non-active).
    """

    def test_opponent_fainted_after_ko_and_replace(self):
        """After opponent faints AND replacement switches in, it appears in
        opponent_fainted."""
        battle = _make_battle()
        _opponent_faint(battle, 2, "Alakazam")
        # Immediately after faint, still active — not in fainted yet
        us = UniversalState.from_Battle(battle)
        assert us.opponent_fainted == []
        assert us.opponent_active_pokemon.status == "fnt"

        # Replacement switches in (e.g. Zapdos replaces fainted Alakazam)
        _opponent_switch(battle, 3, "Zapdos")
        us = UniversalState.from_Battle(battle)
        assert len(us.opponent_fainted) == 1
        assert us.opponent_fainted[0].name == "alakazam"
        assert us.opponent_fainted[0].hp_pct == 0.0
        assert us.opponent_active_pokemon.name == "zapdos"

    def test_fainted_not_in_bench_after_replace(self):
        """After faint + replace, the fainted mon is in fainted, not bench."""
        battle = _make_battle()
        _opponent_switch(battle, 2, "Zapdos")   # Alakazam on bench
        _opponent_faint(battle, 3, "Zapdos")     # Zapdos faints (still active)
        _opponent_switch(battle, 4, "Gengar")    # Replace with Gengar

        us = UniversalState.from_Battle(battle)

        bench_names = {p.name for p in us.opponent_bench}
        fainted_names = {p.name for p in us.opponent_fainted}
        # Alakazam switched out → bench
        assert "alakazam" in bench_names
        # Zapdos fainted + replaced → opponent_fainted (not bench)
        assert "zapdos" in fainted_names
        assert "zapdos" not in bench_names

    def test_our_fainted_after_replace(self):
        """After our Pokémon faints AND we switch in a replacement, it appears
        in fainted_pokemon."""
        battle = _make_battle()
        _our_faint(battle, 2, "Starmie")
        # Still active after faint
        us = UniversalState.from_Battle(battle)
        assert us.fainted_pokemon == []
        assert us.player_active_pokemon.status == "fnt"

        # Switch in replacement
        _our_switch(battle, 3, "Alakazam")
        us = UniversalState.from_Battle(battle)
        assert len(us.fainted_pokemon) == 1
        assert us.fainted_pokemon[0].name == "starmie"
        assert us.fainted_pokemon[0].hp_pct == 0.0
        assert us.player_active_pokemon.name == "alakazam"

    def test_our_fainted_not_in_switches_after_replace(self):
        """After faint + replace, our fainted Pokémon is not in switches."""
        battle = _make_battle()
        _our_faint(battle, 2, "Starmie")
        _our_switch(battle, 3, "Alakazam")

        us = UniversalState.from_Battle(battle)

        switch_names = {p.name for p in us.available_switches}
        fainted_names = {p.name for p in us.fainted_pokemon}
        assert "starmie" not in switch_names
        assert "starmie" in fainted_names

    def test_fainted_and_bench_coexist(self):
        """After switches AND a KO + replace, bench and fainted are correct."""
        battle = _make_battle()
        _opponent_switch(battle, 2, "Zapdos")   # Alakazam on bench
        _opponent_switch(battle, 3, "Gengar")   # Zapdos on bench
        _opponent_faint(battle, 4, "Gengar")    # Gengar faints (still active)
        _opponent_switch(battle, 5, "Rhydon")   # Replace with Rhydon

        us = UniversalState.from_Battle(battle)

        bench_names = {p.name for p in us.opponent_bench}
        fainted_names = {p.name for p in us.opponent_fainted}

        # Alakazam and Zapdos were switched out (not fainted)
        assert "alakazam" in bench_names
        assert "zapdos" in bench_names

        # Gengar was fainted + replaced → fainted
        assert "gengar" in fainted_names

        # No overlap
        assert bench_names.isdisjoint(fainted_names)


class TestMetamonBackendEdgeCases:
    """Edge cases for MetamonBackendBattle + UniversalState interoperability."""

    def test_empty_state_is_well_formed(self):
        """A freshly initialized battle produces a well-formed UniversalState."""
        battle = _make_battle()
        us = UniversalState.from_Battle(battle)

        # Required string fields are non-empty
        assert us.format == "gen1ou"
        assert us.player_active_pokemon.name == "starmie"
        assert us.opponent_active_pokemon.name == "alakazam"
        # Opponent bench is empty on turn 1
        assert us.opponent_bench == []
        assert us.opponent_fainted == []
        assert us.fainted_pokemon == []

    def test_switch_then_from_battle_is_idempotent(self):
        """Calling from_Battle twice on the same state gives identical results."""
        battle = _make_battle()
        _opponent_switch(battle, 2, "Zapdos")

        us1 = UniversalState.from_Battle(battle)
        us2 = UniversalState.from_Battle(battle)

        assert us1.opponent_bench == us2.opponent_bench
        assert us1.opponent_active_pokemon.name == us2.opponent_active_pokemon.name
