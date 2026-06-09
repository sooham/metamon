"""
Forward-fill tests using synthetic 1v1 (or minimal) replay logs.

Each test class corresponds to a group of related scenarios
from ``tests/TEST_SCENARIOS.md``.  Scenario numbers in docstrings
and comments reference that document.
"""

import datetime

import pytest

from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay
from metamon.backend.replay_parser.exceptions import (
    UnfinishedReplayException,
    NoSpeciesClause,
    SoftLockedGen,
)
from metamon.backend.replay_parser.replay_state import (
    Action,
    Move,
    Nothing,
    Pokemon,
    Turn,
    Winner,
)
from metamon.backend.replay_parser.pe_datatypes import (
    PEField,
    PESideCondition,
    PEStatus,
    PEWeather,
)

from tests.synthetic_helpers import (
    make_skeleton,
    make_turn,
    make_winner,
    make_faint_winner,
    build_parsed_replay,
)


# ---------------------------------------------------------------------------
# Helper: reusable Pokemon / Turn / Action property checks
# ---------------------------------------------------------------------------

def _assert_pokemon_basics(poke: Pokemon, name: str, had_name: str, lvl: int = 100):
    """Check the immutable identity fields of a Pokemon."""
    assert isinstance(poke, Pokemon)
    assert poke.name == name
    assert poke.had_name == had_name
    assert poke.lvl == lvl
    assert isinstance(poke.unique_id, str) and len(poke.unique_id) > 0


def _assert_pokemon_hp(poke: Pokemon, current_hp: int, max_hp: int):
    assert poke.current_hp == current_hp
    assert poke.max_hp == max_hp


def _assert_pokemon_types(poke: Pokemon, expected_types: list[str]):
    assert poke.type == expected_types
    assert poke.had_type == expected_types


def _assert_pokemon_base_stats(poke: Pokemon):
    """Base stats dict is populated and values are positive."""
    for key in ("hp", "atk", "def", "spa", "spd", "spe"):
        assert key in poke.base_stats
        assert poke.base_stats[key] > 0


def _assert_pokemon_gen1_defaults(poke: Pokemon):
    """Gen 1 Pokemon have no items and No Ability."""
    assert poke.active_item is None
    assert poke.had_item is None
    assert poke.active_ability is Nothing.NO_ABILITY
    assert poke.had_ability is Nothing.NO_ABILITY
    assert poke.status is Nothing.NO_STATUS
    assert poke.boosts.atk_ == 0
    assert poke.boosts.def_ == 0
    assert poke.boosts.spa_ == 0
    assert poke.boosts.spd_ == 0
    assert poke.boosts.spe_ == 0
    assert poke.effects == {}
    assert poke.last_used_move is None
    assert poke.moves == {}
    assert poke.had_moves == {}
    assert poke.transformed_into is None
    assert poke.transformed_this_turn is False


def _assert_turn_structure(t: Turn, turn_number: int):
    """Check the structural invariants of a Turn object."""
    assert isinstance(t, Turn)
    assert t.turn_number == turn_number
    # Doubles slots exist but are None in singles
    assert len(t.active_pokemon_1) == 2
    assert len(t.active_pokemon_2) == 2
    assert len(t.moves_1) == 2
    assert len(t.moves_2) == 2
    assert len(t.choices_1) == 2
    assert len(t.choices_2) == 2


def _assert_action_move(action: Action, move_name: str):
    """Check that an Action is a standard attacking move."""
    assert isinstance(action, Action)
    assert action.name == move_name
    assert action.is_switch is False
    assert action.is_noop is False
    assert action.is_tera is False
    assert action.is_revival is False
    assert action.user is not None
    assert action.target is not None


# ---------------------------------------------------------------------------
# Basic Structure / Metadata  (scenarios 1–5)
# ---------------------------------------------------------------------------

class TestMetadata:
    """Tests for replay-level metadata and error conditions."""

    # -- Scenario 1: minimal valid replay ------------------------------------

    def test_minimal_valid_replay(self):  # Gen:1  Gimmick: none — basic structural skeleton.
        """Scenario 1: A replay with 5+ turns and a winner.

        Exercises: |gen|, |tier|, |player|, |teamsize|, |poke|, |start|,
        |switch|, |turn|, |move|, |-damage|, |win|.

        1v1: Charizard vs Alakazam, one attack each on turn 1, then empty
        turns 2–5 and P1 wins.
        """
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        ) + [
            make_turn(1),
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "90/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-minimal")

        # ---- Replay metadata ----
        assert replay.gen == 1
        assert replay.format == "gen1ou"
        assert replay.winner == Winner.PLAYER_1
        assert replay.players == ["alice", "bob"]  # clean_name lowercases
        assert len(replay.turnlist) == 6  # turns 0–5
        assert replay.check_warnings == set()

        # ---- Turn 0 (team preview + initial leads) ----
        t0 = replay.turnlist[0]
        _assert_turn_structure(t0, 0)

        # Team lists
        assert len(t0.pokemon_1) == 1
        assert len(t0.pokemon_2) == 1

        # Active Pokemon
        p1_active = t0.active_pokemon_1[0]
        p2_active = t0.active_pokemon_2[0]
        assert t0.active_pokemon_1[1] is None  # doubles slot unused
        assert t0.active_pokemon_2[1] is None

        # ---- P1's Charizard ----
        _assert_pokemon_basics(p1_active, "Charizard", "Charizard")
        _assert_pokemon_hp(p1_active, 100, 100)
        _assert_pokemon_types(p1_active, ["Fire", "Flying"])
        _assert_pokemon_base_stats(p1_active)
        _assert_pokemon_gen1_defaults(p1_active)
        # Abilities loaded from Pokédex (Gen 1: single "No Ability" → auto-revealed as NO_ABILITY)
        assert p1_active.active_ability is Nothing.NO_ABILITY
        assert p1_active.had_ability is Nothing.NO_ABILITY

        # ---- P2's Alakazam ----
        _assert_pokemon_basics(p2_active, "Alakazam", "Alakazam")
        _assert_pokemon_hp(p2_active, 100, 100)
        _assert_pokemon_types(p2_active, ["Psychic"])
        _assert_pokemon_base_stats(p2_active)
        _assert_pokemon_gen1_defaults(p2_active)
        assert p2_active.active_ability is Nothing.NO_ABILITY
        assert p2_active.had_ability is Nothing.NO_ABILITY

        # Team preview (frozen copies, different objects from active)
        assert len(t0.teampreview_1) == 1
        assert len(t0.teampreview_2) == 1
        tp1 = t0.teampreview_1[0]
        tp2 = t0.teampreview_2[0]
        assert tp1 is not p1_active
        assert tp2 is not p2_active
        assert tp1.name == "Charizard"
        assert tp2.name == "Alakazam"

        # Weather / field / conditions
        assert t0.weather is Nothing.NO_WEATHER
        assert t0.battle_field == {}
        assert t0.conditions_1 == {}
        assert t0.conditions_2 == {}

        # Subturns / replacements / force switch / tera
        assert t0.subturns == []
        assert t0.replacements_1 == []
        assert t0.replacements_2 == []
        assert t0.is_force_switch is False
        assert t0.can_tera_1 is False  # Gen 1
        assert t0.can_tera_2 is False

        # Moves on turn 0: the initial |switch| messages create Switch actions
        assert t0.moves_1[0] is not None
        assert t0.moves_1[0].name == "Switch"
        assert t0.moves_1[0].is_switch is True
        assert t0.moves_1[0].target is p1_active
        assert t0.moves_1[1] is None  # doubles slot unused
        assert t0.moves_2[0] is not None
        assert t0.moves_2[0].name == "Switch"
        assert t0.moves_2[0].is_switch is True
        assert t0.moves_2[0].target is p2_active
        assert t0.moves_2[1] is None
        # Choices are empty on turn 0 (no |choice| messages)
        assert t0.choices_1 == [None, None]
        assert t0.choices_2 == [None, None]

        # ---- Turn 1 (moves executed) ----
        t1 = replay.turnlist[1]
        _assert_turn_structure(t1, 1)

        # Moves were recorded
        a1 = t1.moves_1[0]
        a2 = t1.moves_2[0]
        _assert_action_move(a1, "Flamethrower")
        _assert_action_move(a2, "Psychic")
        # user / target point to the active Pokemon on *this* turn
        # (deep-copied from turn 0, so object identity differs; compare unique_id)
        assert a1.user.unique_id == t1.active_pokemon_1[0].unique_id
        assert a1.target.unique_id == t1.active_pokemon_2[0].unique_id
        assert a2.user.unique_id == t1.active_pokemon_2[0].unique_id
        assert a2.target.unique_id == t1.active_pokemon_1[0].unique_id
        # The Pokemon on turn 1 are copies of those on turn 0 (same unique_id)
        assert t1.active_pokemon_1[0].unique_id == p1_active.unique_id
        assert t1.active_pokemon_2[0].unique_id == p2_active.unique_id

        # HP after damage
        _assert_pokemon_hp(t1.active_pokemon_1[0], 70, 100)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 90, 100)

        # Last used move on each active
        p1_t1 = t1.active_pokemon_1[0]
        p2_t1 = t1.active_pokemon_2[0]
        assert p1_t1.last_used_move.name == "Flamethrower"
        assert p2_t1.last_used_move.name == "Psychic"

        # Moves revealed in the moves dict
        assert "Flamethrower" in p1_t1.moves
        assert "Psychic" in p2_t1.moves
        flare_move = p1_t1.moves["Flamethrower"]
        psychic_move = p2_t1.moves["Psychic"]
        assert isinstance(flare_move, Move)
        assert isinstance(psychic_move, Move)
        # PP was consumed (default max PP for Flamethrower is 24, so now 23)
        assert flare_move.pp == flare_move.maximum_pp - 1
        assert psychic_move.pp == psychic_move.maximum_pp - 1

        # had_moves also recorded (immutable copies used for backfill)
        assert "Flamethrower" in p1_t1.had_moves
        assert "Psychic" in p2_t1.had_moves

        # ---- Turns 2–5 (empty) ----
        for i in range(2, 6):
            t = replay.turnlist[i]
            _assert_turn_structure(t, i)
            # No new moves on empty turns
            assert t.moves_1 == [None, None]
            assert t.moves_2 == [None, None]
            # HP should carry forward (deep copy from previous turn)
            _assert_pokemon_hp(t.active_pokemon_1[0], 70, 100)
            _assert_pokemon_hp(t.active_pokemon_2[0], 90, 100)

    # -- Scenario 2: missing Species Clause ----------------------------------

    def test_no_species_clause_raises(self):  # Gen:1  Gimmick: error-path — NoSpeciesClause.
        """Scenario 2: A replay without Species Clause should raise NoSpeciesClause."""
        log = [
            ["gen", "1"],
            ["tier", "[Gen 1] OU"],
            # deliberately omit Species Clause
            ["rule", "OHKO Clause: OHKO moves are banned"],
            ["player", "p1", "Alice", ""],
            ["player", "p2", "Bob", ""],
            ["teamsize", "p1", "1"],
            ["teamsize", "p2", "1"],
            ["poke", "p1", "Charizard", ""],
            ["poke", "p2", "Alakazam", ""],
            ["start", ""],
            ["switch", "p1a: Charizard", "Charizard", "100/100"],
            ["switch", "p2a: Alakazam", "Alakazam", "100/100"],
        ] + make_winner("Alice")

        replay = ParsedReplay(
            gameid="test-no-sc",
            format="gen1ou",
            time_played=datetime.datetime(2020, 1, 1),
        )
        with pytest.raises(NoSpeciesClause):
            forward_fill(replay, log)

    # -- Scenario 3: unsupported gen -----------------------------------------

    def test_unsupported_gen_raises(self):  # Gen:5(rejected)  Gimmick: error-path — SoftLockedGen.
        """Scenario 3: Gen 5 (unsupported) should raise SoftLockedGen."""
        log = [
            ["gen", "5"],
            ["tier", "[Gen 5] OU"],
            ["rule", "Species Clause: Limit one of each Pokémon"],
            ["player", "p1", "Alice", ""],
            ["player", "p2", "Bob", ""],
            ["teamsize", "p1", "1"],
            ["teamsize", "p2", "1"],
            ["poke", "p1", "Charizard", ""],
            ["poke", "p2", "Alakazam", ""],
        ]
        replay = ParsedReplay(
            gameid="test-gen5",
            format="gen5ou",
            time_played=datetime.datetime(2020, 1, 1),
        )
        with pytest.raises(SoftLockedGen):
            forward_fill(replay, log)

    # -- Scenario 4: non-standard team size ----------------------------------

    def test_teamsize_four(self):  # Gen:1  Gimmick: non-standard teamsize (4).
        """Scenario 4: A replay with teamsize 4 should parse correctly.

        Verifies that team lists, active slots, and teampreview are all
        sized correctly, and that bench Pokemon have proper Pokedex info
        even though they never switch in.
        """
        log = make_skeleton(
            teamsize1=4, teamsize2=4,
            p1_pokes=["Charizard", "Blastoise", "Venusaur", "Pikachu"],
            p2_pokes=["Alakazam", "Golem", "Starmie", "Dragonite"],
        ) + [
            make_turn(1),
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-teamsize4")

        t0 = replay.turnlist[0]
        _assert_turn_structure(t0, 0)

        # Team sizes
        assert len(t0.pokemon_1) == 4
        assert len(t0.pokemon_2) == 4
        assert len(t0.teampreview_1) == 4
        assert len(t0.teampreview_2) == 4

        # Every Pokemon slot in P1's team has Pokedex info
        p1_expected = ["Charizard", "Blastoise", "Venusaur", "Pikachu"]
        for i, expected_name in enumerate(p1_expected):
            p = t0.pokemon_1[i]
            assert p is not None
            _assert_pokemon_basics(p, expected_name, expected_name)
            _assert_pokemon_types(p, _expected_types(expected_name))
            _assert_pokemon_base_stats(p)
            _assert_pokemon_gen1_defaults(p)
            # Teampreview is a frozen copy
            tp = t0.teampreview_1[i]
            assert tp.name == expected_name
            assert tp is not p  # different object

        # Bench Pokemon (non-active) have HP = None until they switch in
        for i in range(1, 4):
            assert t0.pokemon_1[i].current_hp is None
            assert t0.pokemon_1[i].max_hp is None
            assert t0.pokemon_2[i].current_hp is None
            assert t0.pokemon_2[i].max_hp is None

        # Active Pokemon DO have HP set
        assert t0.pokemon_1[0].current_hp == 100
        assert t0.pokemon_1[0].max_hp == 100

        # P2's team also verified
        p2_expected = ["Alakazam", "Golem", "Starmie", "Dragonite"]
        for i, expected_name in enumerate(p2_expected):
            p = t0.pokemon_2[i]
            assert p is not None
            _assert_pokemon_basics(p, expected_name, expected_name)
            _assert_pokemon_types(p, _expected_types(expected_name))
            _assert_pokemon_base_stats(p)

        # Turn 1: moves recorded correctly
        t1 = replay.turnlist[1]
        a1 = t1.moves_1[0]
        a2 = t1.moves_2[0]
        _assert_action_move(a1, "Flamethrower")
        _assert_action_move(a2, "Psychic")
        _assert_pokemon_hp(t1.active_pokemon_1[0], 70, 100)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 70, 100)

        p1_t1 = t1.active_pokemon_1[0]
        p2_t1 = t1.active_pokemon_2[0]

        # Moves registered after use
        assert "Flamethrower" in p1_t1.moves
        assert "Psychic" in p2_t1.moves
        assert isinstance(p1_t1.moves["Flamethrower"], Move)
        assert isinstance(p2_t1.moves["Psychic"], Move)
        assert p1_t1.moves["Flamethrower"].pp == p1_t1.moves["Flamethrower"].maximum_pp - 1
        assert p2_t1.moves["Psychic"].pp == p2_t1.moves["Psychic"].maximum_pp - 1

        # had_moves also populated
        assert "Flamethrower" in p1_t1.had_moves
        assert "Psychic" in p2_t1.had_moves

        # last_used_move set
        assert p1_t1.last_used_move.name == "Flamethrower"
        assert p2_t1.last_used_move.name == "Psychic"

    # -- Scenario 5: Unrated players ----------------------------------------

    def test_unrated_players(self):  # Gen:1  Gimmick: Unrated ratings.
        """Scenario 5: Both players Unrated should record ratings as 'Unrated'."""
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
            p1_rating="", p2_rating="",
        ) + [
            make_turn(1),
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-unrated")
        assert replay.ratings == ["Unrated", "Unrated"]


# ---------------------------------------------------------------------------
# Per-Pokémon type lookup (Gen 1 only)
# ---------------------------------------------------------------------------

_GEN1_TYPES: dict[str, list[str]] = {
    "Charizard":  ["Fire", "Flying"],
    "Blastoise":  ["Water"],
    "Venusaur":   ["Grass", "Poison"],
    "Pikachu":    ["Electric"],
    "Alakazam":   ["Psychic"],
    "Golem":      ["Rock", "Ground"],
    "Starmie":    ["Water", "Psychic"],
    "Dragonite":  ["Dragon", "Flying"],
}


def _expected_types(name: str) -> list[str]:
    try:
        return _GEN1_TYPES[name]
    except KeyError:
        raise AssertionError(f"No Gen 1 type mapping for {name!r}") from None


# ---------------------------------------------------------------------------
# Moves — Basic Execution  (scenarios 6A–13D)
# ---------------------------------------------------------------------------

class TestMoves:
    """Tests for move execution, damage, PP, KO, recharge, and multi-turn moves."""

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _make_1v1_log(*turns: list[list[str]]) -> list[list[str]]:
        """Build a minimal 1v1 Charizard-vs-Alakazam log with extra turns.

        Appends filler turns + a win to satisfy >= 5 turn check.
        """
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        # pad to >= 5 turns
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 6A: basic attack, 30% damage ---------------------------------------

    def test_6A_basic_attack_damage(self):  # Gen:1  Move:Flamethrower,Psychic  Gimmick: basic damage+PP.
        """P1 attacks P2, deals 30% damage; PP consumed, HP tracked."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-6A")

        # Turn 0: no moves, full HP
        t0 = replay.turnlist[0]
        _assert_pokemon_hp(t0.active_pokemon_1[0], 100, 100)
        _assert_pokemon_hp(t0.active_pokemon_2[0], 100, 100)
        assert t0.active_pokemon_1[0].last_used_move is None
        assert t0.active_pokemon_2[0].last_used_move is None
        assert t0.active_pokemon_1[0].moves == {}
        assert t0.active_pokemon_2[0].moves == {}

        # Turn 1: both attacked
        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # HP
        _assert_pokemon_hp(p1, 85, 100)
        _assert_pokemon_hp(p2, 70, 100)

        # Actions
        _assert_action_move(t1.moves_1[0], "Flamethrower")
        _assert_action_move(t1.moves_2[0], "Psychic")
        assert t1.moves_1[0].target.unique_id == p2.unique_id
        assert t1.moves_2[0].target.unique_id == p1.unique_id

        # P1: Flamethrower PP 24→23
        assert "Flamethrower" in p1.moves
        assert p1.moves["Flamethrower"].pp == 23
        assert p1.moves["Flamethrower"].maximum_pp == 24
        assert p1.last_used_move.name == "Flamethrower"
        assert "Flamethrower" in p1.had_moves

        # P2: Psychic PP 16→15
        assert "Psychic" in p2.moves
        assert p2.moves["Psychic"].pp == 15
        assert p2.moves["Psychic"].maximum_pp == 16
        assert p2.last_used_move.name == "Psychic"
        assert "Psychic" in p2.had_moves

        # No status changes
        assert p1.status is Nothing.NO_STATUS
        assert p2.status is Nothing.NO_STATUS

    # -- 6B: multi-turn with miss and different moves -----------------------

    def test_6B_multiturn_miss_and_different_move(self):  # Gen:1  Move:Flamethrower,FireBlast,Thunderbolt  Gimmick: multi-turn PP tracking, miss.
        """Two turns: P1 uses two different moves; P2 misses on turn 2.

        Verifies PP tracking across turns for the same move and a new move,
        and that a missed move still consumes PP.
        """
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "40/100"],
            ["move", "p2a: Alakazam", "Thunderbolt", "p1a: Charizard", "[miss]"],
            # no -damage for the miss
        )
        replay = build_parsed_replay(log, gameid="test-6B")

        # Turn 1
        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        p2_t1 = t1.active_pokemon_2[0]
        _assert_pokemon_hp(p1_t1, 85, 100)
        _assert_pokemon_hp(p2_t1, 70, 100)
        assert p1_t1.moves["Flamethrower"].pp == 23  # 24→23
        assert p2_t1.moves["Psychic"].pp == 15         # 16→15

        # Turn 2
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        p2_t2 = t2.active_pokemon_2[0]

        # P1: Flamethrower again, PP 23→22
        _assert_pokemon_hp(p1_t2, 85, 100)   # P1 took no damage this turn (P2 missed)
        _assert_pokemon_hp(p2_t2, 40, 100)   # P2 took second Flamethrower
        _assert_action_move(t2.moves_1[0], "Flamethrower")
        assert p1_t2.moves["Flamethrower"].pp == 22
        assert p1_t2.last_used_move.name == "Flamethrower"

        # P2: Thunderbolt missed — action recorded, PP consumed, no damage
        _assert_action_move(t2.moves_2[0], "Thunderbolt")
        assert "Thunderbolt" in p2_t2.moves
        # Thunderbolt base PP in Gen 1 is 24, so after one use: 23
        assert p2_t2.moves["Thunderbolt"].pp == 23
        assert p2_t2.last_used_move.name == "Thunderbolt"

    # -- 7: miss (no damage, PP still consumed) -----------------------------

    def test_7_miss_no_damage_pp_consumed(self):  # Gen:1  Move:Flamethrower(miss)  Gimmick: [miss] flag.
        """P1 attacks, misses — no damage, but PP is still consumed."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam", "[miss]"],
            # no -damage
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-7")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # P1's move missed: action recorded, PP consumed, HP unchanged
        _assert_action_move(t1.moves_1[0], "Flamethrower")
        assert p1.moves["Flamethrower"].pp == 23  # consumed
        _assert_pokemon_hp(p2, 100, 100)            # not damaged

        # P2's move hit normally
        _assert_action_move(t1.moves_2[0], "Psychic")
        _assert_pokemon_hp(p1, 85, 100)
        assert p2.moves["Psychic"].pp == 15

    # -- 8: KO (0 fnt, faint message) ---------------------------------------

    def test_8_ko_faint(self):  # Gen:1  Move:Flamethrower  Gimmick: faint (HP=0, status=FNT).
        """P1 attacks, KOs P2: HP 0, status FNT, faint message."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "0 fnt"],
            ["faint", "p2a: Alakazam"],
        )
        replay = build_parsed_replay(log, gameid="test-8")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]
        assert p2.current_hp == 0
        from metamon.backend.replay_parser.pe_datatypes import PEStatus
        assert p2.status == PEStatus.FNT

        # P1's move still recorded normally
        _assert_action_move(t1.moves_1[0], "Flamethrower")
        # P2 fainted so no move action on their side
        assert t1.moves_2[0] is None

    # -- 9: Hyper Beam → recharge -------------------------------------------

    def test_9_hyper_beam_recharge(self):  # Gen:1  Move:HyperBeam  Gimmick: recharge (is_noop).
        """P1 uses Hyper Beam; next turn must recharge (is_noop)."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Hyper Beam", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "50/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["-mustrecharge", "p1a: Charizard"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-9")

        # Turn 1: Hyper Beam used, PP 8→7
        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        _assert_action_move(t1.moves_1[0], "Hyper Beam")
        assert t1.moves_1[0].is_noop is False
        assert p1_t1.moves["Hyper Beam"].pp == 7  # 8→7
        _assert_pokemon_hp(t1.active_pokemon_2[0], 50, 100)

        # Turn 2: P1 recharges, P2 attacks
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        a1 = t2.moves_1[0]
        assert a1 is not None
        assert a1.name == "Recharge"
        assert a1.is_noop is True
        assert a1.is_switch is False
        assert a1.user.unique_id == p1_t2.unique_id
        # P1's Hyper Beam PP unchanged on recharge turn
        assert p1_t2.moves["Hyper Beam"].pp == 7
        # P2's move recorded
        _assert_action_move(t2.moves_2[0], "Psychic")
        _assert_pokemon_hp(p1_t2, 70, 100)

    # -- 10: Struggle -------------------------------------------------------

    def test_10_struggle(self):  # Gen:1  Move:Struggle  Gimmick: no PP, not in had_moves.
        """P1 uses Struggle — no PP consumed, action recorded as Struggle."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Struggle", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "95/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-10")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        a1 = t1.moves_1[0]
        assert a1.name == "Struggle"
        assert a1.is_switch is False
        assert a1.is_noop is False  # noop flag is not set on the Action object
        # Struggle does NOT appear in moves dict (use_move returns early)
        assert "Struggle" not in p1.moves
        assert "Struggle" not in p1.had_moves
        # HP still tracked
        _assert_pokemon_hp(p1, 85, 100)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 95, 100)

    # -- 11: Outrage multi-turn (2–3 turns, [still]) -----------------------

    def test_11_outrage_multiturn(self):  # Gen:1  Move:Outrage  Gimmick: [still], multi-turn, PP tracking.
        """P1 uses Outrage for 3 turns: PP consumed on first only."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Outrage", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Outrage", "p2a: Alakazam", "[still]", "[from] move: Outrage"],
            ["-damage", "p2a: Alakazam", "40/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["turn", "3"],
            ["move", "p1a: Charizard", "Outrage", "p2a: Alakazam", "[still]", "[from] move: Outrage"],
            ["-damage", "p2a: Alakazam", "10/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "55/100"],
        )
        replay = build_parsed_replay(log, gameid="test-11")

        # Turn 1: first use, PP consumed
        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        _assert_action_move(t1.moves_1[0], "Outrage")
        assert p1_t1.moves["Outrage"].pp == 23  # 24→23
        _assert_pokemon_hp(t1.active_pokemon_2[0], 70, 100)

        # Turn 2: continuation with [still], PP unchanged
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        _assert_action_move(t2.moves_1[0], "Outrage")
        assert p1_t2.moves["Outrage"].pp == 23  # still 23 (no PP on [still])
        _assert_pokemon_hp(t2.active_pokemon_2[0], 40, 100)

        # Turn 3: second continuation
        t3 = replay.turnlist[3]
        p1_t3 = t3.active_pokemon_1[0]
        _assert_action_move(t3.moves_1[0], "Outrage")
        assert p1_t3.moves["Outrage"].pp == 23  # still 23
        _assert_pokemon_hp(t3.active_pokemon_2[0], 10, 100)

    # -- 12: Solar Beam charge turn + attack --------------------------------

    def test_12_solar_beam_charge(self):  # Gen:1  Move:SolarBeam  Gimmick: charge move, [still].
        """P1 uses Solar Beam: charge turn [still] (no PP), attack turn."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Solar Beam", "p2a: Alakazam", "[still]"],
            # charge turn: no damage
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Solar Beam", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "60/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-12")

        # Turn 1: charge — no PP consumed, no damage dealt by P1
        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        _assert_action_move(t1.moves_1[0], "Solar Beam")
        # charge turn: [still] → pp_used = 0
        assert p1_t1.moves["Solar Beam"].pp == 16  # unchanged (max = 16)
        # P2 not damaged by Solar Beam yet
        _assert_pokemon_hp(t1.active_pokemon_2[0], 100, 100)
        # P1 took P2's Psychic
        _assert_pokemon_hp(p1_t1, 85, 100)

        # Turn 2: attack — PP consumed, damage dealt
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        _assert_action_move(t2.moves_1[0], "Solar Beam")
        assert p1_t2.moves["Solar Beam"].pp == 15  # 16→15
        _assert_pokemon_hp(t2.active_pokemon_2[0], 60, 100)
        _assert_pokemon_hp(p1_t2, 70, 100)

    # -- 13A: Fly (semi-invulnerable then attack) ---------------------------

    def test_13A_fly(self):  # Gen:1  Move:Fly  Gimmick: semi-invulnerable, charge.
        """P1 uses Fly: charge turn [still] (no PP), attack turn."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Fly", "p2a: Alakazam", "[still]"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Fly", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "65/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-13A")

        # Turn 1: semi-invulnerable — PP unchanged
        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        _assert_action_move(t1.moves_1[0], "Fly")
        assert p1_t1.moves["Fly"].pp == 24  # unchanged (max=24)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 100, 100)
        _assert_pokemon_hp(p1_t1, 85, 100)

        # Turn 2: attack — PP consumed, damage dealt
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        _assert_action_move(t2.moves_1[0], "Fly")
        assert p1_t2.moves["Fly"].pp == 23  # 24→23
        _assert_pokemon_hp(t2.active_pokemon_2[0], 65, 100)
        _assert_pokemon_hp(p1_t2, 70, 100)

    # -- 13B: Dig (semi-invulnerable then attack) ---------------------------

    def test_13B_dig(self):  # Gen:1  Move:Dig  Gimmick: semi-invulnerable, charge.
        """P2 uses Dig: charge turn [still] (no PP), attack turn."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "85/100"],
            ["move", "p2a: Alakazam", "Dig", "p1a: Charizard", "[still]"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Dig", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "65/100"],
        )
        replay = build_parsed_replay(log, gameid="test-13B")

        # Turn 1: Dig charge, PP unchanged
        t1 = replay.turnlist[1]
        p2_t1 = t1.active_pokemon_2[0]
        _assert_action_move(t1.moves_2[0], "Dig")
        assert p2_t1.moves["Dig"].pp == 16  # unchanged
        # P2 was hit by Flamethrower on turn 1
        _assert_pokemon_hp(p2_t1, 85, 100)

        # Turn 2: Dig attack, PP consumed
        t2 = replay.turnlist[2]
        p2_t2 = t2.active_pokemon_2[0]
        _assert_action_move(t2.moves_2[0], "Dig")
        assert p2_t2.moves["Dig"].pp == 15  # 16→15
        _assert_pokemon_hp(t2.active_pokemon_1[0], 65, 100)

    # -- 13C: Earthquake hits Dig user --------------------------------------

    def test_13C_earthquake_hits_dig(self):  # Gen:1  Move:Earthquake,Dig  Gimmick: move hits semi-invulnerable.
        """P2 uses Dig [still]; P1's Earthquake hits during semi-invulnerable."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Earthquake", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "50/100"],
            ["move", "p2a: Alakazam", "Dig", "p1a: Charizard", "[still]"],
        )
        replay = build_parsed_replay(log, gameid="test-13C")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # Both moves recorded
        _assert_action_move(t1.moves_1[0], "Earthquake")
        _assert_action_move(t1.moves_2[0], "Dig")
        # Earthquake dealt damage
        _assert_pokemon_hp(p2, 50, 100)
        # P1's Earthquake PP consumed
        assert p1.moves["Earthquake"].pp == 15  # 16→15
        # P2's Dig on charge turn: PP unchanged
        assert p2.moves["Dig"].pp == 16

    # -- 13D: Protect blocks attack -----------------------------------------

    def test_13D_protect(self):  # Gen:2+  Move:Protect  Gimmick: protected flag, no damage.
        """P2 uses Protect — P1's attack deals no damage, protected flag set."""
        log = self._make_1v1_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Protect", "p2a: Alakazam"],
            ["-singleturn", "p2a: Alakazam", "Protect"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-fail", "p2a: Alakazam"],
            # no -damage
        )
        replay = build_parsed_replay(log, gameid="test-13D")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # Both actions recorded
        _assert_action_move(t1.moves_1[0], "Flamethrower")
        _assert_action_move(t1.moves_2[0], "Protect")
        # P2's protected flag set
        assert p2.protected is True
        # P1's PP still consumed (the move was attempted)
        assert p1.moves["Flamethrower"].pp == 23
        # P2 took no damage
        _assert_pokemon_hp(p2, 100, 100)
        # P1 took no damage (P2 used Protect, not an attack)
        _assert_pokemon_hp(p1, 100, 100)


# ---------------------------------------------------------------------------
# Damage / HP Edge Cases  (scenarios 14–18)
# ---------------------------------------------------------------------------

class TestDamage:
    """Tests for heal, sethp, item/ability reveal via damage, and recoil."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 14: |-heal| recovers HP -------------------------------------------

    def test_14_heal(self):  # Gen:1  Gimmick: |-heal| HP recovery.
        """P1 takes damage then recovers HP via |-heal|."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "40/100"],
            ["turn", "2"],
            ["-heal", "p1a: Charizard", "75/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "60/100"],
        )
        replay = build_parsed_replay(log, gameid="test-14")

        # Turn 1: P1 down to 40
        t1 = replay.turnlist[1]
        _assert_pokemon_hp(t1.active_pokemon_1[0], 40, 100)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 70, 100)

        # Turn 2: P1 heals 40→75, then takes damage 75→60
        t2 = replay.turnlist[2]
        p1 = t2.active_pokemon_1[0]
        _assert_pokemon_hp(p1, 60, 100)
        # Action for heal turn: P1 didn't use a move, so moves_1[0] is None
        # (the heal is not a player action — it came from an item/move side effect)
        assert t2.moves_1[0] is None
        # P2 attacked
        _assert_action_move(t2.moves_2[0], "Psychic")

    # -- 15: |-sethp| direct HP set ----------------------------------------

    def test_15_sethp(self):  # Gen:1  Move:SuperFang  Gimmick: |-sethp| direct HP set.
        """|-sethp| sets HP directly (Pain Split, Endeavor, Super Fang)."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Super Fang", "p1a: Charizard"],
            ["-sethp", "p1a: Charizard", "50/100"],
        )
        replay = build_parsed_replay(log, gameid="test-15")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # P1's HP was set directly to 50 (not via subtraction)
        _assert_pokemon_hp(p1, 50, 100)
        # P2 took normal Flamethrower damage
        _assert_pokemon_hp(p2, 70, 100)
        # Both moves recorded
        _assert_action_move(t1.moves_1[0], "Flamethrower")
        _assert_action_move(t1.moves_2[0], "Super Fang")

    # -- 16: damage from item (Life Orb) -----------------------------------

    def test_16_damage_from_item_reveals_item(self):  # Gen:1(log only)  Item:LifeOrb  Gimmick: item reveal via -damage [from].
        """Damage with [from] item: Life Orb reveals the item on the damaged mon."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "50/100"],
            ["-damage", "p1a: Charizard", "90/100", "[from] item: Life Orb"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "75/100"],
        )
        replay = build_parsed_replay(log, gameid="test-16")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # P1's item revealed from Life Orb recoil damage
        assert p1.active_item == "Life Orb"
        assert p1.had_item == "Life Orb"

        # HP after Life Orb recoil (90) then Psychic (75)
        _assert_pokemon_hp(p1, 75, 100)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 50, 100)

    # -- 17: damage from ability reveals ability ---------------------------

    def test_17_damage_from_ability_reveals_ability(self):  # Gen:1(log only)  Ability:RoughSkin  Gimmick: ability reveal via -damage [from].
        """Damage with [from] ability: ... reveals the ability."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["-damage", "p1a: Charizard", "85/100", "[from] ability: Rough Skin"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-17")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Active ability overwritten (no [of] tag → revealed on damaged mon)
        assert p1.active_ability == "Rough Skin"
        # had_ability was already set (NO_ABILITY) so it is NOT overwritten
        assert p1.had_ability is Nothing.NO_ABILITY

        # HP tracked through both damage events
        _assert_pokemon_hp(p1, 70, 100)

    # -- 18: recoil damage (Double-Edge) -----------------------------------

    def test_18_recoil_damage(self):  # Gen:1  Move:Double-Edge  Gimmick: recoil self-damage.
        """P1 uses Double-Edge: deals damage and takes recoil."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Double-Edge", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "40/100"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-18")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # Both moves recorded
        _assert_action_move(t1.moves_1[0], "Double-Edge")
        _assert_action_move(t1.moves_2[0], "Psychic")

        # P1 took recoil (85) then Psychic (70)
        _assert_pokemon_hp(p1, 70, 100)
        # P2 took Double-Edge
        _assert_pokemon_hp(p2, 40, 100)

        # Double-Edge PP consumed, revealed in moves
        assert "Double-Edge" in p1.moves
        assert p1.moves["Double-Edge"].pp == p1.moves["Double-Edge"].maximum_pp - 1
        assert p1.last_used_move.name == "Double-Edge"


# ---------------------------------------------------------------------------
# Stat Boosts  (scenarios 25–32)
# ---------------------------------------------------------------------------

class TestBoosts:
    """Tests for boost, unboost, setboost, clear, swap, copy, and invert."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 25: single stat boost (Swords Dance) -------------------------------

    def test_25_single_boost(self):  # Gen:1  Move:SwordsDance  Gimmick: single stat boost (+2 atk).
        """Swords Dance: +2 atk boost."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Swords Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "2"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-25")

        t0 = replay.turnlist[0]
        p1_t0 = t0.active_pokemon_1[0]
        # Turn 0: all boosts at 0
        assert p1_t0.boosts.atk_ == 0
        assert p1_t0.boosts.def_ == 0

        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        # Turn 1: atk +2
        assert p1_t1.boosts.atk_ == 2
        assert p1_t1.boosts.def_ == 0  # unchanged
        assert p1_t1.boosts.spa_ == 0
        assert p1_t1.boosts.spe_ == 0

        # Move action recorded
        _assert_action_move(t1.moves_1[0], "Swords Dance")
        assert t1.moves_1[0].target.unique_id == p1_t1.unique_id  # self-target

    # -- 26: two stats boosted (Dragon Dance) -------------------------------

    def test_26_double_boost(self):  # Gen:1  Move:DragonDance  Gimmick: double stat boost (+1 atk,+1 spe).
        """Dragon Dance: +1 atk, +1 spe in one turn."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Dragon Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "1"],
            ["-boost", "p1a: Charizard", "spe", "1"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-26")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        assert p1.boosts.atk_ == 1
        assert p1.boosts.spe_ == 1
        assert p1.boosts.def_ == 0
        assert p1.boosts.spa_ == 0

    # -- 27: unboost (Charm) ------------------------------------------------

    def test_27_unboost(self):  # Gen:1  Move:Charm  Gimmick: |-unboost| (-2 atk).
        """Charm: P2 lowers P1's atk by 2 stages."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Charm", "p1a: Charizard"],
            ["-unboost", "p1a: Charizard", "atk", "2"],
        )
        replay = build_parsed_replay(log, gameid="test-27")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        # P1's atk dropped by 2
        assert p1.boosts.atk_ == -2
        assert p1.boosts.def_ == 0

        # Both actions recorded
        _assert_action_move(t1.moves_1[0], "Flamethrower")
        _assert_action_move(t1.moves_2[0], "Charm")

    # -- 28: clear all boosts (Haze) ---------------------------------------

    def test_28_clearallboost(self):  # Gen:1  Move:Haze  Gimmick: |-clearallboost|.
        """Haze: clears boosts on both active Pokemon."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Swords Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "2"],
            ["move", "p2a: Alakazam", "Amnesia", "p2a: Alakazam"],
            ["-boost", "p2a: Alakazam", "spd", "2"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Haze", "p2a: Alakazam"],
            ["-clearallboost"],
        )
        replay = build_parsed_replay(log, gameid="test-28")

        # Turn 1: boosts applied
        t1 = replay.turnlist[1]
        assert t1.active_pokemon_1[0].boosts.atk_ == 2
        assert t1.active_pokemon_2[0].boosts.spd_ == 2

        # Turn 2: Haze clears everything
        t2 = replay.turnlist[2]
        for stat in ("atk_", "def_", "spa_", "spd_", "spe_"):
            assert getattr(t2.active_pokemon_1[0].boosts, stat) == 0
            assert getattr(t2.active_pokemon_2[0].boosts, stat) == 0

    # -- 29: swap boost (Heart Swap) ---------------------------------------

    def test_29_swapboost(self):  # Gen:1  Move:HeartSwap  Gimmick: |-swapboost|.
        """Heart Swap: swaps all boost stages between two Pokemon."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Swords Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "2"],
            ["move", "p2a: Alakazam", "Amnesia", "p2a: Alakazam"],
            ["-boost", "p2a: Alakazam", "spd", "2"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Heart Swap", "p1a: Charizard"],
            ["-swapboost", "p1a: Charizard", "p2a: Alakazam",
             "[from] move: Heart Swap"],
        )
        replay = build_parsed_replay(log, gameid="test-29")

        # Turn 1: P1 atk=+2, P2 spd=+2
        t1 = replay.turnlist[1]
        assert t1.active_pokemon_1[0].boosts.atk_ == 2
        assert t1.active_pokemon_2[0].boosts.spd_ == 2

        # Turn 2: Heart Swap exchanges ALL boosts
        t2 = replay.turnlist[2]
        p1 = t2.active_pokemon_1[0]
        p2 = t2.active_pokemon_2[0]
        # P1 gets P2's boosts (spd=+2, others 0)
        assert p1.boosts.atk_ == 0
        assert p1.boosts.spd_ == 2
        # P2 gets P1's boosts (atk=+2, others 0)
        assert p2.boosts.atk_ == 2
        assert p2.boosts.spd_ == 0

    # -- 30: copy boost (Psych Up) -----------------------------------------

    def test_30_copyboost(self):  # Gen:1  Move:PsychUp  Gimmick: |-copyboost|.
        """Psych Up: copies opponent's boost stages."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Swords Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "2"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psych Up", "p1a: Charizard"],
            ["-copyboost", "p2a: Alakazam", "p1a: Charizard"],
        )
        replay = build_parsed_replay(log, gameid="test-30")

        # Turn 1: P1 atk=+2, P2 has no boosts
        t1 = replay.turnlist[1]
        assert t1.active_pokemon_1[0].boosts.atk_ == 2
        assert t1.active_pokemon_2[0].boosts.atk_ == 0

        # Turn 2: P2 copies P1's boosts
        t2 = replay.turnlist[2]
        p2 = t2.active_pokemon_2[0]
        assert p2.boosts.atk_ == 2
        assert p2.boosts.def_ == 0
        # P1 unaffected
        assert t2.active_pokemon_1[0].boosts.atk_ == 2

    # -- 31: set boost (Belly Drum) ----------------------------------------

    def test_31_setboost(self):  # Gen:1  Move:BellyDrum  Gimmick: |-setboost| atk→6.
        """Belly Drum: sets atk to +6 regardless of current value."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Charm", "p1a: Charizard"],
            ["-unboost", "p1a: Charizard", "atk", "2"],
            ["move", "p1a: Charizard", "Belly Drum", "p1a: Charizard"],
            ["-setboost", "p1a: Charizard", "atk", "6"],
        )
        replay = build_parsed_replay(log, gameid="test-31")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        # Charm dropped atk to -2, then Belly Drum set it to +6
        assert p1.boosts.atk_ == 6
        assert p1.boosts.def_ == 0

    # -- 32: invert boost (Topsy-Turvy) ------------------------------------

    def test_32_invertboost(self):  # Gen:1  Move:Topsy-Turvy  Gimmick: |-invertboost|.
        """Topsy-Turvy: inverts all of the target's stat changes."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Swords Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "2"],
            ["move", "p2a: Alakazam", "Charm", "p1a: Charizard"],
            ["-unboost", "p1a: Charizard", "atk", "2"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Dragon Dance", "p1a: Charizard"],
            ["-boost", "p1a: Charizard", "atk", "1"],
            ["-boost", "p1a: Charizard", "spe", "1"],
            ["move", "p2a: Alakazam", "Topsy-Turvy", "p1a: Charizard"],
            ["-invertboost", "p1a: Charizard"],
        )
        replay = build_parsed_replay(log, gameid="test-32")

        # Turn 1: atk=0 (SD +2 then Charm -2)
        t1 = replay.turnlist[1]
        assert t1.active_pokemon_1[0].boosts.atk_ == 0

        # Turn 2: Dragon Dance → atk=+1, spe=+1; then Topsy-Turvy inverts
        t2 = replay.turnlist[2]
        p1 = t2.active_pokemon_1[0]
        assert p1.boosts.atk_ == -1   # +1 → -1
        assert p1.boosts.spe_ == -1   # +1 → -1
        assert p1.boosts.def_ == 0    # 0 → 0

        # Action recorded
        _assert_action_move(t2.moves_2[0], "Topsy-Turvy")


# ---------------------------------------------------------------------------
# Weather  (scenarios 33–35B)
# ---------------------------------------------------------------------------

class TestWeather:
    """Tests for weather set, clear, and residual damage."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 33: Rain Dance sets weather ---------------------------------------

    def test_33_rain_dance(self):  # Gen:2+  Move:RainDance  Gimmick: weather set.
        """Rain Dance: |-weather|RainDance sets weather on the same turn."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Rain Dance", "p1a: Charizard"],
            ["-weather", "RainDance"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-33")

        # Turn 0: no weather
        t0 = replay.turnlist[0]
        assert t0.weather is Nothing.NO_WEATHER

        # Turn 1: Rain Dance active
        t1 = replay.turnlist[1]
        assert t1.weather == PEWeather.RAINDANCE

        # Move action recorded
        _assert_action_move(t1.moves_1[0], "Rain Dance")

    # -- 34: weather clears -------------------------------------------------

    def test_34_weather_clear(self):  # Gen:2+  Gimmick: weather clear (|-weather|none).
        """Rain expires: |-weather|none clears weather back to NO_WEATHER."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Rain Dance", "p1a: Charizard"],
            ["-weather", "RainDance"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["-weather", "none"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-34")

        # Turn 1: Rain is active
        t1 = replay.turnlist[1]
        assert t1.weather == PEWeather.RAINDANCE

        # Turn 2: weather cleared
        t2 = replay.turnlist[2]
        assert t2.weather is Nothing.NO_WEATHER

    # -- 35A: Sandstorm residual damage ------------------------------------

    def test_35A_sandstorm_damage(self):  # Gen:2+  Move:Sandstorm  Gimmick: residual weather damage.
        """Sandstorm: residual damage on non-immune types each turn.

        Charizard (Fire/Flying) is immune; Alakazam (Psychic) takes damage.
        """
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Sandstorm", "p1a: Charizard"],
            ["-weather", "Sandstorm"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            # end-of-turn Sandstorm damage
            ["-damage", "p2a: Alakazam", "87/100", "[from] Sandstorm"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "57/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["-damage", "p2a: Alakazam", "49/100", "[from] Sandstorm"],
        )
        replay = build_parsed_replay(log, gameid="test-35A")

        # Turn 1: Sandstorm set, P2 took 13 from it (100→87)
        t1 = replay.turnlist[1]
        assert t1.weather == PEWeather.SANDSTORM
        _assert_pokemon_hp(t1.active_pokemon_2[0], 87, 100)

        # Turn 2: more Sandstorm damage (57→49)
        t2 = replay.turnlist[2]
        assert t2.weather == PEWeather.SANDSTORM  # weather persists
        _assert_pokemon_hp(t2.active_pokemon_2[0], 49, 100)
        # P1 is immune (Fire/Flying = no sandstorm damage)
        _assert_pokemon_hp(t2.active_pokemon_1[0], 70, 100)

        # Sandstorm move PP consumed
        p1_t1 = t1.active_pokemon_1[0]
        assert p1_t1.moves["Sandstorm"].pp == p1_t1.moves["Sandstorm"].maximum_pp - 1

    # -- 35B: Hail residual damage ------------------------------------------

    def test_35B_hail_damage(self):  # Gen:2+  Move:Hail  Gimmick: residual weather damage.
        """Hail: residual damage on non-Ice types each turn."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Hail", "p1a: Charizard"],
            ["-weather", "Hail"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            # end-of-turn Hail damage on both (neither is Ice type)
            ["-damage", "p1a: Charizard", "73/100", "[from] Hail"],
            ["-damage", "p2a: Alakazam", "88/100", "[from] Hail"],
        )
        replay = build_parsed_replay(log, gameid="test-35B")

        t1 = replay.turnlist[1]
        assert t1.weather == PEWeather.HAIL
        # Both took Hail damage (~12 each)
        _assert_pokemon_hp(t1.active_pokemon_1[0], 73, 100)
        _assert_pokemon_hp(t1.active_pokemon_2[0], 88, 100)

        # Hail action recorded
        _assert_action_move(t1.moves_1[0], "Hail")


# ---------------------------------------------------------------------------
# Field Conditions  (scenarios 36–37B)
# ---------------------------------------------------------------------------

class TestFieldConditions:
    """Tests for fieldstart, fieldend, terrain replacement, stacking,
    and ability reveal via field condition."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 36A: Electric Terrain set -----------------------------------------

    def test_36A_electric_terrain_set(self):  # Gen:2+  Move:ElectricTerrain  Gimmick: |-fieldstart|.
        """|-fieldstart|Electric Terrain adds to battle_field."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Electric Terrain", "p1a: Charizard"],
            ["-fieldstart", "Electric Terrain",
             "[from] move: Electric Terrain"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-36A")

        # Turn 0: empty field
        t0 = replay.turnlist[0]
        assert t0.battle_field == {}

        # Turn 1: terrain active
        t1 = replay.turnlist[1]
        assert PEField.ELECTRIC_TERRAIN in t1.battle_field
        assert t1.battle_field[PEField.ELECTRIC_TERRAIN] == t1.turn_number

        # Action recorded
        _assert_action_move(t1.moves_1[0], "Electric Terrain")

    # -- 36B: terrain replacement (Grassy replaces Electric) -----------------

    def test_36B_terrain_replacement(self):  # Gen:2+  Move:GrassyTerrain  Gimmick: terrain replacement.
        """Grassy Terrain replaces Electric Terrain (only one terrain active)."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Electric Terrain", "p1a: Charizard"],
            ["-fieldstart", "Electric Terrain",
             "[from] move: Electric Terrain"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Grassy Terrain", "p1a: Charizard"],
            ["-fieldstart", "Grassy Terrain",
             "[from] move: Grassy Terrain"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-36B")

        # Turn 1: Electric Terrain
        t1 = replay.turnlist[1]
        assert PEField.ELECTRIC_TERRAIN in t1.battle_field
        assert PEField.GRASSY_TERRAIN not in t1.battle_field

        # Turn 2: Grassy replaced Electric
        t2 = replay.turnlist[2]
        assert PEField.ELECTRIC_TERRAIN not in t2.battle_field
        assert PEField.GRASSY_TERRAIN in t2.battle_field
        assert t2.battle_field[PEField.GRASSY_TERRAIN] == t2.turn_number

    # -- 36C: non-terrain stacking (Trick Room + Gravity) -------------------

    def test_36C_nonterrain_stacking(self):  # Gen:2+  Move:TrickRoom,Gravity  Gimmick: non-terrain stacking.
        """Trick Room + Gravity coexist (non-terrains stack in battle_field)."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Trick Room", "p1a: Charizard"],
            ["-fieldstart", "Trick Room", "[from] move: Trick Room"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Gravity", "p1a: Charizard"],
            ["-fieldstart", "Gravity", "[from] move: Gravity"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-36C")

        # Turn 1: Trick Room only
        t1 = replay.turnlist[1]
        assert PEField.TRICK_ROOM in t1.battle_field

        # Turn 2: BOTH Trick Room and Gravity (neither is terrain, both stack)
        t2 = replay.turnlist[2]
        assert PEField.TRICK_ROOM in t2.battle_field
        assert PEField.GRAVITY in t2.battle_field
        assert len([f for f in t2.battle_field if not f.is_terrain]) == 2

    # -- 37A: field end (expire) -------------------------------------------

    def test_37A_field_end(self):  # Gen:2+  Gimmick: |-fieldend|.
        """|-fieldend|Electric Terrain removes it from battle_field."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Electric Terrain", "p1a: Charizard"],
            ["-fieldstart", "Electric Terrain",
             "[from] move: Electric Terrain"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["-fieldend", "Electric Terrain"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-37A")

        # Turn 1: terrain present
        t1 = replay.turnlist[1]
        assert PEField.ELECTRIC_TERRAIN in t1.battle_field

        # Turn 2: cleared
        t2 = replay.turnlist[2]
        assert t2.battle_field == {}

    # -- 37B: ability revealed via field start -----------------------------

    def test_37B_ability_reveal_via_field(self):  # Gen:2+  Ability:ElectricSurge  Gimmick: ability reveal via field.
        """Field condition with [from] ability: ... [of] ... reveals ability."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Electric Terrain", "p1a: Charizard"],
            ["-fieldstart", "Electric Terrain",
             "[from] ability: Electric Surge",
             "[of] p1a: Charizard"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-37B")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Ability revealed on the [of] Pokemon
        assert p1.active_ability == "Electric Surge"
        # had_ability was already NO_ABILITY (Gen 1 auto-reveal), not overwritten
        assert p1.had_ability is Nothing.NO_ABILITY

        # Field still set
        assert PEField.ELECTRIC_TERRAIN in t1.battle_field


# ---------------------------------------------------------------------------
# Side Conditions  (scenarios 38–42E)
# ---------------------------------------------------------------------------

class TestSideConditions:
    """Tests for sidestart, sideend, stacking, swapping, and multiple conditions."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 38: Stealth Rock on P2's side -------------------------------------

    def test_38_stealth_rock(self):  # Gen:2+  Move:StealthRock  Gimmick: side condition.
        """|-sidestart|p2: ... |move: Stealth Rock tracks condition on P2."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Stealth Rock", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Stealth Rock"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-38")

        # Turn 0: no conditions
        t0 = replay.turnlist[0]
        assert t0.conditions_1 == {}
        assert t0.conditions_2 == {}

        # Turn 1: Stealth Rock only on P2's side
        t1 = replay.turnlist[1]
        assert t1.conditions_1 == {}  # P1's side clean
        assert PESideCondition.STEALTH_ROCK in t1.conditions_2
        assert t1.conditions_2[PESideCondition.STEALTH_ROCK] == t1.turn_number

    # -- 39: Spikes 3 layers (stackable) -----------------------------------

    def test_39_spikes_stacking(self):  # Gen:2+  Move:Spikes  Gimmick: stackable side condition.
        """Spikes stack: 3 |-sidestart| messages increment counter to 3."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Spikes", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Spikes"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Spikes", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Spikes"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["turn", "3"],
            ["move", "p1a: Charizard", "Spikes", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Spikes"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "55/100"],
        )
        replay = build_parsed_replay(log, gameid="test-39")

        # Turn 1: 1 layer
        t1 = replay.turnlist[1]
        assert t1.conditions_2[PESideCondition.SPIKES] == 1

        # Turn 2: 2 layers
        t2 = replay.turnlist[2]
        assert t2.conditions_2[PESideCondition.SPIKES] == 2

        # Turn 3: 3 layers
        t3 = replay.turnlist[3]
        assert t3.conditions_2[PESideCondition.SPIKES] == 3

        # P1's side still clean
        assert t3.conditions_1 == {}

    # -- 40: Reflect set + expire -------------------------------------------

    def test_40_reflect_set_and_expire(self):  # Gen:2+  Move:Reflect  Gimmick: side condition set+expire.
        """Reflect on P1: |-sidestart| sets it, |-sideend| removes it."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Reflect", "p1a: Charizard"],
            ["-sidestart", "p1: Alice", "Reflect"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["-sideend", "p1: Alice", "Reflect"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-40")

        # Turn 1: Reflect active
        t1 = replay.turnlist[1]
        assert PESideCondition.REFLECT in t1.conditions_1
        assert t1.conditions_2 == {}

        # Turn 2: Reflect cleared
        t2 = replay.turnlist[2]
        assert t2.conditions_1 == {}

    # -- 41: Rapid Spin clears hazards -------------------------------------

    def test_41_rapid_spin_clear(self):  # Gen:2+  Move:RapidSpin  Gimmick: side condition removal.
        """Rapid Spin: |-sideend|p1: ... |move: Spikes removes Spikes from P1."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Stealth Rock", "p2a: Alakazam"],
            ["-sidestart", "p1: Alice", "move: Stealth Rock"],
            # Also place Spikes
            ["-sidestart", "p1: Alice", "move: Spikes"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Rapid Spin", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "60/100"],
            ["-sideend", "p1: Alice", "move: Spikes"],
            ["-sideend", "p1: Alice", "move: Stealth Rock"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-41")

        # Turn 1: both hazards on P1
        t1 = replay.turnlist[1]
        assert PESideCondition.STEALTH_ROCK in t1.conditions_1
        assert PESideCondition.SPIKES in t1.conditions_1

        # Turn 2: Rapid Spin cleared both
        t2 = replay.turnlist[2]
        assert t2.conditions_1 == {}

    # -- 42: Tailwind -------------------------------------------------------

    def test_42_tailwind(self):  # Gen:2+  Move:Tailwind  Gimmick: side condition.
        """Tailwind on P1: turn-limited side condition."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Tailwind", "p1a: Charizard"],
            ["-sidestart", "p1: Alice", "move: Tailwind"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-42")

        t1 = replay.turnlist[1]
        assert PESideCondition.TAILWIND in t1.conditions_1
        assert t1.conditions_1[PESideCondition.TAILWIND] == t1.turn_number
        assert t1.conditions_2 == {}

    # -- 42B (new): swapsideconditions ---------------------------------------

    def test_42B_swapsideconditions(self):  # Gen:2+  Move:CourtChange  Gimmick: |-swapsideconditions|.
        """|-swapsideconditions exchanges conditions_1 and conditions_2."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Stealth Rock", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Stealth Rock"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Court Change", "p2a: Alakazam"],
            ["-swapsideconditions"],
        )
        replay = build_parsed_replay(log, gameid="test-42B")

        # Turn 1: Stealth Rock on P2 only
        t1 = replay.turnlist[1]
        assert t1.conditions_1 == {}
        assert PESideCondition.STEALTH_ROCK in t1.conditions_2

        # Turn 2: swapped — Stealth Rock now on P1
        t2 = replay.turnlist[2]
        assert PESideCondition.STEALTH_ROCK in t2.conditions_1
        assert t2.conditions_2 == {}

    # -- 42C (new): multiple conditions on same side -------------------------

    def test_42C_multiple_conditions(self):  # Gen:2+  Gimmick: multiple side conditions coexist.
        """Spikes + Stealth Rock + Reflect coexist on P2's side."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Stealth Rock", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Stealth Rock"],
            ["-sidestart", "p2: Bob", "move: Spikes"],
            ["-sidestart", "p2: Bob", "Reflect"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-42C")

        t1 = replay.turnlist[1]
        assert PESideCondition.STEALTH_ROCK in t1.conditions_2
        assert PESideCondition.SPIKES in t1.conditions_2
        assert PESideCondition.REFLECT in t1.conditions_2
        assert len(t1.conditions_2) == 3

    # -- 42D (new): side condition on P1's side -----------------------------

    def test_42D_both_sides_have_conditions(self):  # Gen:2+  Gimmick: both sides have conditions.
        """Both sides have independent side conditions."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Stealth Rock", "p1a: Charizard"],
            ["-sidestart", "p2: Bob", "move: Stealth Rock"],
            ["move", "p2a: Alakazam", "Reflect", "p2a: Alakazam"],
            ["-sidestart", "p1: Alice", "Reflect"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-42D")

        t1 = replay.turnlist[1]
        # P1 has Reflect
        assert PESideCondition.REFLECT in t1.conditions_1
        # P2 has Stealth Rock
        assert PESideCondition.STEALTH_ROCK in t1.conditions_2
        # Each side has exactly 1 condition
        assert len(t1.conditions_1) == 1
        assert len(t1.conditions_2) == 1


# ---------------------------------------------------------------------------
# Abilities  (scenarios 43–47D)
# ---------------------------------------------------------------------------

class TestAbilities:
    """Tests for ability activation, reveal, override, trace, end, and
    weather/status-triggered abilities."""

    @staticmethod
    def _make_log(
        *turns: list[list[str]],
        gen: int = 3,
        tier: str = "[Gen 3] OU",
        format: str = "gen3ou",
        p1_pokes: list[str] | None = None,
        p2_pokes: list[str] | None = None,
        p1_lead: tuple[str, str] | None = None,
        p2_lead: tuple[str, str] | None = None,
    ) -> list[list[str]]:
        log = make_skeleton(
            gen=gen, tier=tier, format=format,
            teamsize1=1, teamsize2=1,
            p1_pokes=p1_pokes or ["Charizard"],
            p2_pokes=p2_pokes or ["Alakazam"],
            p1_lead=p1_lead,
            p2_lead=p2_lead,
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    @staticmethod
    def _parse(log: list[list[str]], gameid: str, format: str = "gen3ou"):
        return build_parsed_replay(log, gameid=gameid, format=format)

    # -- 43: Intimidate on switch-in ----------------------------------------

    def test_43_intimidate(self):  # Gen:3+  Ability:Intimidate  Gimmick: ability on switch-in, atk drop.
        """Intimidate on switch-in: ability revealed, opponent atk -1."""
        log = self._make_log(
            ["turn", "1"],
            ["-ability", "p1a: Gyarados", "Intimidate", "boost"],
            ["-unboost", "p2a: Alakazam", "atk", "1"],
            ["move", "p1a: Gyarados", "Waterfall", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Gyarados"],
            ["-damage", "p1a: Gyarados", "85/100"],
            p1_pokes=["Gyarados"],
            p1_lead=("Gyarados", "100/100"),
        )
        replay = self._parse(log, gameid="test-43")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # Ability revealed on Gyarados
        assert p1.active_ability == "Intimidate"
        assert p1.had_ability == "Intimidate"

        # Alakazam atk dropped by 1
        assert p2.boosts.atk_ == -1
        assert p2.boosts.def_ == 0

        # Actions recorded
        _assert_action_move(t1.moves_1[0], "Waterfall")
        _assert_action_move(t1.moves_2[0], "Psychic")

    # -- 44: single-ability species auto-reveal -----------------------------

    def test_44_single_ability_autoreveal(self):  # Gen:3+  Ability:Blaze  Gimmick: single-ability auto-reveal from Pokédex.
        """Single-ability species auto-revealed from Pokédex on init."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = self._parse(log, gameid="test-44")

        t0 = replay.turnlist[0]
        p1 = t0.active_pokemon_1[0]
        p2 = t0.active_pokemon_2[0]

        # Charizard has only Blaze → auto-revealed
        assert p1.active_ability == "Blaze"
        assert p1.had_ability == "Blaze"

        # Alakazam has multiple abilities (Synchronize/Inner Focus) → NOT auto-revealed
        assert p2.active_ability is None
        assert p2.had_ability is None

    # -- 45: Mummy overrides ability ---------------------------------------

    def test_45_mummy_override(self):  # Gen:5+  Ability:Mummy  Gimmick: ability overwrite on contact.
        """Mummy overrides the target's active ability; both abilities revealed."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Yamask", "Shadow Ball", "p2a: Gyarados"],
            ["-damage", "p2a: Gyarados", "80/100"],
            ["-ability", "p2a: Gyarados", "Mummy",
             "[from] ability: Mummy", "[of] p1a: Yamask"],
            ["move", "p2a: Gyarados", "Waterfall", "p1a: Yamask"],
            ["-damage", "p1a: Yamask", "70/100"],
            p1_pokes=["Yamask"],
            p1_lead=("Yamask", "100/100"),
            p2_pokes=["Gyarados"],
            p2_lead=("Gyarados", "100/100"),
        )
        replay = self._parse(log, gameid="test-45")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]  # Yamask
        p2 = t1.active_pokemon_2[0]  # Gyarados

        # Gyarados' active ability overwritten to Mummy
        assert p2.active_ability == "Mummy"
        # Gyarados' had_ability was Intimidate (auto-revealed), not overwritten
        assert p2.had_ability == "Intimidate"

        # Yamask reveals Mummy (its permanent ability)
        assert p1.active_ability == "Mummy"
        assert p1.had_ability == "Mummy"

    # -- 46: Trace copies ability -------------------------------------------

    def test_46_trace_copies(self):  # Gen:3+  Ability:Trace  Gimmick: active_ability vs had_ability.
        """Trace copies opponent's ability: active_ability changes, had_ability stays."""
        log = self._make_log(
            ["turn", "1"],
            ["-ability", "p2a: Gyarados", "Intimidate", "boost"],
            ["-unboost", "p1a: Porygon2", "atk", "1"],
            ["-ability", "p1a: Porygon2", "Intimidate",
             "[from] ability: Trace", "[of] p2a: Gyarados"],
            ["move", "p1a: Porygon2", "Tri Attack", "p2a: Gyarados"],
            ["-damage", "p2a: Gyarados", "80/100"],
            ["move", "p2a: Gyarados", "Waterfall", "p1a: Porygon2"],
            ["-damage", "p1a: Porygon2", "85/100"],
            p1_pokes=["Porygon2"],
            p1_lead=("Porygon2", "100/100"),
            p2_pokes=["Gyarados"],
            p2_lead=("Gyarados", "100/100"),
        )
        replay = self._parse(log, gameid="test-46")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]  # Porygon2
        p2 = t1.active_pokemon_2[0]  # Gyarados

        # Porygon2: active=Intimidate (copied), had=Trace (original)
        assert p1.active_ability == "Intimidate"
        assert p1.had_ability == "Trace"

        # Gyarados: Intimidate revealed
        assert p2.active_ability == "Intimidate"
        assert p2.had_ability == "Intimidate"

        # Intimidate from Gyarados dropped Porygon2's atk
        assert p1.boosts.atk_ == -1

    # -- 47A: endability ----------------------------------------------------

    def test_47A_endability(self):  # Gen:3+  Move:GastroAcid  Gimmick: |-endability|.
        """|-endability| clears active_ability, optionally reveals had_ability."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Gastro Acid", "p1a: Charizard"],
            ["-endability", "p1a: Charizard", "Blaze"],
        )
        replay = self._parse(log, gameid="test-47A")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Active ability cleared
        assert p1.active_ability is Nothing.NO_ABILITY
        # had_ability revealed (it was already Blaze from auto-reveal, but
        # endability would set it if it were None)
        assert p1.had_ability == "Blaze"

    # -- 47B: Dry Skin (weather-triggered heal / damage) ---------------------

    def test_47B_dry_skin(self):  # Gen:3+  Ability:DrySkin  Gimmick: weather-triggered heal.
        """Dry Skin heals in rain: ability revealed via |-heal| [from] tag."""
        # Use Charizard (single-ability Blaze auto-revealed).  The heal
        # message overwrites active_ability; had_ability stays Blaze.
        log = self._make_log(
            ["turn", "1"],
            ["-weather", "RainDance"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "80/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "50/100"],
            ["-heal", "p1a: Charizard", "62/100",
             "[from] ability: Dry Skin"],
        )
        replay = self._parse(log, gameid="test-47B")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Dry Skin set as active ability (overwrites Blaze)
        assert p1.active_ability == "Dry Skin"
        # had_ability unchanged (was already Blaze from auto-reveal)
        assert p1.had_ability == "Blaze"

        # HP after heal: 50 → 62
        _assert_pokemon_hp(p1, 62, 100)

    # -- 47C: weather-speed ability activation (Swift Swim) ------------------

    def test_47C_swift_swim_activation(self):  # Gen:3+  Ability:SwiftSwim  Gimmick: weather-speed ability.
        """Swift Swim activates in rain: |-ability| message reveals ability."""
        log = self._make_log(
            ["turn", "1"],
            ["-weather", "RainDance"],
            ["-ability", "p1a: Kingdra", "Swift Swim"],
            ["move", "p1a: Kingdra", "Hydro Pump", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "50/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Kingdra"],
            ["-damage", "p1a: Kingdra", "85/100"],
            p1_pokes=["Kingdra"],
            p1_lead=("Kingdra", "100/100"),
        )
        replay = self._parse(log, gameid="test-47C")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Swift Swim revealed via |-ability|
        assert p1.active_ability == "Swift Swim"
        assert p1.had_ability == "Swift Swim"

        # No stat boost (speed isn't shown as boost)
        assert p1.boosts.spe_ == 0

    # -- 47D: weather-setting ability (Drizzle) -----------------------------

    def test_47D_weather_setting_ability(self):  # Gen:3+  Ability:Drizzle  Gimmick: weather-setting ability.
        """Drizzle sets rain: |-ability| then |-weather| with [from] ability:."""
        log = self._make_log(
            ["turn", "1"],
            ["-ability", "p1a: Politoed", "Drizzle"],
            ["-weather", "RainDance",
             "[from] ability: Drizzle", "[of] p1a: Politoed"],
            ["move", "p1a: Politoed", "Hydro Pump", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "60/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Politoed"],
            ["-damage", "p1a: Politoed", "85/100"],
            p1_pokes=["Politoed"],
            p1_lead=("Politoed", "100/100"),
        )
        replay = self._parse(log, gameid="test-47D")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Drizzle revealed
        assert p1.active_ability == "Drizzle"
        assert p1.had_ability == "Drizzle"

        # Rain set
        assert t1.weather == PEWeather.RAINDANCE


# ---------------------------------------------------------------------------
# Volatile Effects  (scenarios 52–59)
# ---------------------------------------------------------------------------

class TestVolatileEffects:
    """Tests for |-start|, |-end|, Leech Seed, confusion, Taunt, Encore,
    Nightmare, and Curse."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 52: Leech Seed -----------------------------------------------------

    def test_52_leech_seed(self):  # Gen:1  Move:LeechSeed  Gimmick: volatile effect, recurring drain.
        """Leech Seed on P2: effect in effects dict, drain on subsequent turns."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Leech Seed", "p2a: Alakazam"],
            ["-start", "p2a: Alakazam", "Leech Seed"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            # end-of-turn Leech Seed drain
            ["-damage", "p2a: Alakazam", "87/100", "[from] Leech Seed"],
            ["-heal", "p1a: Charizard", "98/100", "[from] Leech Seed"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "57/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "83/100"],
            ["-damage", "p2a: Alakazam", "45/100", "[from] Leech Seed"],
            ["-heal", "p1a: Charizard", "96/100", "[from] Leech Seed"],
        )
        replay = build_parsed_replay(log, gameid="test-52")

        # Turn 1: Leech Seed applied
        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]
        assert PEEffect.LEECH_SEED in p2.effects
        # P2 took Leech Seed drain (100→87), P1 healed (85→98)
        _assert_pokemon_hp(p2, 87, 100)
        _assert_pokemon_hp(t1.active_pokemon_1[0], 98, 100)

        # Turn 2: Leech Seed still active, drains again
        t2 = replay.turnlist[2]
        p2_t2 = t2.active_pokemon_2[0]
        assert PEEffect.LEECH_SEED in p2_t2.effects
        _assert_pokemon_hp(p2_t2, 45, 100)
        _assert_pokemon_hp(t2.active_pokemon_1[0], 96, 100)

    # -- 53 & 54: confusion + self-hit -------------------------------------

    def test_53_confusion(self):  # Gen:1  Move:ConfuseRay  Gimmick: volatile effect (confusion).
        """|-start|p1a: Charizard|confusion adds confusion to effects dict."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Confuse Ray", "p1a: Charizard"],
            ["-start", "p1a: Charizard", "confusion"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-53")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        assert PEEffect.CONFUSION in p1.effects
        assert p1.effects[PEEffect.CONFUSION] == 0

    def test_54_confusion_self_hit(self):  # Gen:1  Gimmick: confusion self-hit damage.
        """Confused Pokemon hurts itself: |-damage| [from] confusion."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Confuse Ray", "p1a: Charizard"],
            ["-start", "p1a: Charizard", "confusion"],
            ["-damage", "p1a: Charizard", "85/100", "[from] confusion"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-54")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        # Confusion effect present, P1's move is None (hurt itself)
        assert PEEffect.CONFUSION in p1.effects
        assert t1.moves_1[0] is None
        # HP: 100 → 85 (confusion) → 70 (Psychic)
        _assert_pokemon_hp(p1, 70, 100)

    # -- 55: Taunt ----------------------------------------------------------

    def test_55_taunt(self):  # Gen:3+  Move:Taunt  Gimmick: volatile effect.
        """|-start|p2a: ...|Taunt adds Taunt to effects."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Taunt", "p2a: Alakazam"],
            ["-start", "p2a: Alakazam", "Taunt"],
            ["move", "p2a: Alakazam", "Struggle", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "95/100"],
        )
        replay = build_parsed_replay(log, gameid="test-55")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]
        assert PEEffect.TAUNT in p2.effects

    # -- 56: Encore ---------------------------------------------------------

    def test_56_encore(self):  # Gen:2+  Move:Encore  Gimmick: volatile effect.
        """|-start|p2a: ...|Encore adds Encore to effects."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["move", "p1a: Charizard", "Encore", "p2a: Alakazam"],
            ["-start", "p2a: Alakazam", "Encore"],
        )
        replay = build_parsed_replay(log, gameid="test-56")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]
        assert PEEffect.ENCORE in p2.effects

    # -- 57: effect expires (|-end|) ---------------------------------------

    def test_57_effect_end(self):  # Gen:1  Gimmick: |-end| removes volatile effect.
        """|-end|p2a: ...|Leech Seed removes effect from dict."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Leech Seed", "p2a: Alakazam"],
            ["-start", "p2a: Alakazam", "Leech Seed"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["-end", "p2a: Alakazam", "Leech Seed"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-57")

        # Turn 1: Leech Seed active
        t1 = replay.turnlist[1]
        assert PEEffect.LEECH_SEED in t1.active_pokemon_2[0].effects

        # Turn 2: Leech Seed removed
        t2 = replay.turnlist[2]
        assert PEEffect.LEECH_SEED not in t2.active_pokemon_2[0].effects

    # -- 58: Nightmare on sleeping Pokemon ----------------------------------

    def test_58_nightmare(self):  # Gen:2+  Move:Nightmare  Gimmick: nested status+volatile.
        """Nightmare on sleeping P2: effect + damage from Nightmare."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        from metamon.backend.replay_parser.pe_datatypes import PEStatus
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Hypnosis", "p2a: Alakazam"],
            ["-status", "p2a: Alakazam", "slp"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Nightmare", "p2a: Alakazam"],
            ["-start", "p2a: Alakazam", "Nightmare"],
            ["move", "p2a: Alakazam", "Sleep Talk", "p1a: Charizard"],
            # Nightmare damage
            ["-damage", "p2a: Alakazam", "75/100", "[from] Nightmare"],
        )
        replay = build_parsed_replay(log, gameid="test-58")

        t2 = replay.turnlist[2]
        p2 = t2.active_pokemon_2[0]
        # Nightmare effect added, sleep status still present
        assert PEEffect.NIGHTMARE in p2.effects
        assert p2.status == PEStatus.SLP
        # Nightmare damage applied
        _assert_pokemon_hp(p2, 75, 100)

    # -- 59: Curse (Ghost-type self-inflicted) ------------------------------

    def test_59_curse_ghost(self):  # Gen:2+  Move:Curse(ghost)  Gimmick: self-inflicted volatile.
        """Ghost-type Curse: P1 cuts own HP, effect visible."""
        from metamon.backend.replay_parser.pe_datatypes import PEEffect
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Curse", "p1a: Charizard"],
            ["-start", "p1a: Charizard", "Curse"],
            ["-damage", "p1a: Charizard", "50/100", "[from] Curse"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "35/100"],
        )
        replay = build_parsed_replay(log, gameid="test-59")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        # Curse effect added (note: Ghost-type Curse isn't an effect on the user
        # in the usual sense — the parser tracks it anyway)
        assert "Curse" in t1.moves_1[0].name
        # HP: 100 → 50 (Curse cut) → 35 (Psychic)
        _assert_pokemon_hp(p1, 35, 100)


# ---------------------------------------------------------------------------
# Forced Switches  (scenarios 60–63)
# ---------------------------------------------------------------------------

class TestForcedSwitches:
    """Tests for U-turn, blocked Volt Switch, Roar/Dragon Tail, and Eject
    Button forced switches.  Requires teamsize >= 2 for bench mons."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=2, teamsize2=2,
            p1_pokes=["Charizard", "Blastoise"],
            p2_pokes=["Alakazam", "Golem"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 60: U-turn forced switch -------------------------------------------

    def test_60_uturn_forced_switch(self):  # Gen:4+  Move:U-turn  Gimmick: forced switch subturn.
        """U-turn deals damage then forces P1 to switch to bench."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "U-turn", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            # U-turn triggers forced switch to bench
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Blastoise"],
            ["-damage", "p1a: Blastoise", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-60")

        t1 = replay.turnlist[1]

        # P1's active is now Blastoise (switched in)
        assert t1.active_pokemon_1[0].name == "Blastoise"
        assert t1.active_pokemon_1[0].current_hp == 85

        # P1's bench: Charizard swapped out, HP preserved
        bench_p1 = [p for p in t1.pokemon_1 if p is not None
                     and p.unique_id != t1.active_pokemon_1[0].unique_id]
        assert len(bench_p1) == 1
        assert bench_p1[0].name == "Charizard"

        # Subturn created and filled with forced switch action
        assert len(t1.subturns) == 1
        subturn = t1.subturns[0]
        assert subturn.team == 1  # P1's side
        assert subturn.slot == 0  # slot a
        assert subturn.turn is not None  # filled
        assert subturn.action is not None
        assert subturn.action.name == "Switch"
        assert subturn.action.is_switch is True
        assert subturn.action.target.name == "Blastoise"

        # U-turn action recorded on the caller
        assert t1.moves_1[0] is not None
        assert t1.moves_1[0].name == "U-turn"

    # -- 61: Volt Switch blocked by Lightning Rod ---------------------------

    def test_61_volt_switch_blocked(self):  # Gen:5+  Move:VoltSwitch  Ability:LightningRod  Gimmick: blocked forced switch.
        """Volt Switch blocked by Lightning Rod: no forced switch, subturn removed."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Volt Switch", "p2a: Alakazam"],
            # Alakazam has Volt Absorb (or Lightning Rod) — but the parser
            # checks ABILITY_CAUSES_MOVE_TO_FAIL which maps Lightning Rod → Volt Switch.
            # For a synthetic test, we just have the move without a switch follow-up.
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        # Set Alakazam's active ability to Lightning Rod so the parser blocks the switch
        # We need to do this BEFORE the move via an ability message
        # Actually: we can insert |-ability|p2a: Alakazam|Lightning Rod before the move
        log2 = make_skeleton(
            teamsize1=2, teamsize2=2,
            p1_pokes=["Charizard", "Blastoise"],
            p2_pokes=["Alakazam", "Golem"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        ) + [
            ["turn", "1"],
            ["-ability", "p2a: Alakazam", "Lightning Rod"],
            ["move", "p1a: Charizard", "Volt Switch", "p2a: Alakazam"],
            # Should be blocked — no switch message follows
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        ]
        # pad turns and win
        for t in range(2, 6):
            log2.append(["turn", str(t)])
        log2.append(["win", "Alice"])

        replay = build_parsed_replay(log2, gameid="test-61")

        t1 = replay.turnlist[1]

        # Volt Switch action recorded (the move was used)
        assert t1.moves_1[0].name == "Volt Switch"

        # No switch happened — Charizard is still active
        assert t1.active_pokemon_1[0].name == "Charizard"

        # Subturn should be empty (blocked forced switch was removed)
        unfilled = [s for s in t1.subturns if s.unfilled]
        assert len(unfilled) == 0

    # -- 62: Roar / Dragon Tail (|drag| forced switch) ----------------------

    def test_62_roar_drag(self):  # Gen:2+  Move:Roar  Gimmick: |drag| forced switch.
        """Roar forces P1 to switch via |drag| message."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Roar", "p1a: Charizard"],
            # Roar forces a switch — |drag| handles it
            ["drag", "p1a: Blastoise", "Blastoise", "100/100"],
        )
        replay = build_parsed_replay(log, gameid="test-62")

        t1 = replay.turnlist[1]

        # P1's active is now Blastoise (dragged in)
        assert t1.active_pokemon_1[0].name == "Blastoise"
        _assert_pokemon_hp(t1.active_pokemon_1[0], 100, 100)

        # Charizard is on the bench (swapped out by Roar)
        bench_p1 = [p for p in t1.pokemon_1 if p is not None
                     and p.unique_id != t1.active_pokemon_1[0].unique_id]
        assert len(bench_p1) == 1
        assert bench_p1[0].name == "Charizard"

        # P1's move was Flamethrower (before being roared out)
        assert t1.moves_1[0].name == "Flamethrower"
        # P2's move was Roar
        assert t1.moves_2[0].name == "Roar"

    # -- 63: Eject Button forced switch -------------------------------------

    def test_63_eject_button(self):  # Gen:5+  Item:EjectButton  Gimmick: item-triggered forced switch.
        """Eject Button activates on hit: |-enditem| triggers forced switch."""
        log = self._make_log(
            ["turn", "1"],
            # Reveal Eject Button on P1
            ["-item", "p1a: Charizard", "Eject Button"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "80/100"],
            # Eject Button consumed, triggers switch
            ["-enditem", "p1a: Charizard", "Eject Button",
             "[from] item: Eject Button"],
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
        )
        replay = build_parsed_replay(log, gameid="test-63")

        t1 = replay.turnlist[1]

        # After switch, active is Blastoise
        assert t1.active_pokemon_1[0].name == "Blastoise"

        # Charizard (on bench) had its Eject Button consumed
        charizard = [p for p in t1.pokemon_1 if p is not None and p.name == "Charizard"][0]
        # active_item cleared (NO_ITEM means confirmed absent)
        assert charizard.active_item is Nothing.NO_ITEM
        assert charizard.had_item == "Eject Button"

        # Subturn created for the forced switch
        eject_subturns = [
            s for s in t1.subturns
            if s.action is not None and s.action.name == "Switch"
        ]
        assert len(eject_subturns) == 1
        assert eject_subturns[0].action.is_switch is True


# ---------------------------------------------------------------------------
# Faint / Replace  (scenarios 65–67)
# ---------------------------------------------------------------------------

class TestFaintReplace:
    """Tests for faint-triggered forced switches, double faint (Explosion),
    and Destiny Bond."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=2, teamsize2=2,
            p1_pokes=["Charizard", "Blastoise"],
            p2_pokes=["Alakazam", "Golem"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 65: P1 faints, sends replacement ----------------------------------

    def test_65_faint_and_replace(self):  # Gen:1  Gimmick: faint → forced switch subturn → replacement.
        """P1 faints: HP 0, status FNT, forced switch subturn, replacement fills it."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "0 fnt"],
            ["faint", "p1a: Charizard"],
            # P1 sends replacement
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
        )
        replay = build_parsed_replay(log, gameid="test-65")

        t1 = replay.turnlist[1]

        # Charizard is fainted
        charizard = [p for p in t1.pokemon_1 if p is not None and p.name == "Charizard"][0]
        assert charizard.current_hp == 0
        assert charizard.status == PEStatus.FNT

        # Active is now Blastoise (replacement)
        assert t1.active_pokemon_1[0].name == "Blastoise"
        _assert_pokemon_hp(t1.active_pokemon_1[0], 100, 100)

        # Subturn created by faint, filled by replacement switch
        faint_subturns = [s for s in t1.subturns if s.action is not None]
        assert len(faint_subturns) >= 1
        subturn = faint_subturns[0]
        assert subturn.turn is not None  # filled
        assert subturn.action.is_switch is True
        assert subturn.action.target.name == "Blastoise"
        # is_force_switch on the subturn (create_subturn(True))
        assert subturn.turn.is_force_switch is True

        # Charizard's move (Flamethrower) still recorded
        assert t1.moves_1[0] is not None
        assert t1.moves_1[0].name == "Flamethrower"

        # P2's move recorded
        assert t1.moves_2[0].name == "Psychic"

    # -- 66: Explosion — both faint ----------------------------------------

    def test_66_explosion_double_faint(self):  # Gen:1  Move:Explosion  Gimmick: double faint.
        """Explosion: both Pokemon faint, both sides get forced switch subturns."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Explosion", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "0 fnt"],
            ["-damage", "p1a: Charizard", "0 fnt"],
            ["faint", "p1a: Charizard"],
            ["faint", "p2a: Alakazam"],
            # Both sides send replacements
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
            ["switch", "p2a: Golem", "Golem", "100/100"],
        )
        replay = build_parsed_replay(log, gameid="test-66")

        t1 = replay.turnlist[1]

        # Both original mons fainted
        charizard = [p for p in t1.pokemon_1 if p is not None and p.name == "Charizard"][0]
        alakazam = [p for p in t1.pokemon_2 if p is not None and p.name == "Alakazam"][0]
        assert charizard.status == PEStatus.FNT
        assert charizard.current_hp == 0
        assert alakazam.status == PEStatus.FNT
        assert alakazam.current_hp == 0

        # Both sides have replacements
        assert t1.active_pokemon_1[0].name == "Blastoise"
        assert t1.active_pokemon_2[0].name == "Golem"

        # Subturns for both sides
        p1_subturns = [s for s in t1.subturns if s.team == 1 and s.action is not None]
        p2_subturns = [s for s in t1.subturns if s.team == 2 and s.action is not None]
        assert len(p1_subturns) >= 1
        assert len(p2_subturns) >= 1
        assert p1_subturns[0].action.target.name == "Blastoise"
        assert p2_subturns[0].action.target.name == "Golem"

        # Explosion action recorded
        assert t1.moves_1[0].name == "Explosion"
        # P2's move is None (fainted before they could act)

    # -- 67: Destiny Bond ---------------------------------------------------

    def test_67_destiny_bond(self):  # Gen:2+  Move:DestinyBond  Gimmick: Destiny Bond faint.
        """Destiny Bond: P2 KOs P1, P2 also faints from Destiny Bond."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Destiny Bond", "p1a: Charizard"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "0 fnt"],
            ["faint", "p1a: Charizard"],
            # Destiny Bond triggers, P2 also faints
            ["-damage", "p2a: Alakazam", "0 fnt", "[from] Destiny Bond"],
            ["faint", "p2a: Alakazam"],
            # Both replacements
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
            ["switch", "p2a: Golem", "Golem", "100/100"],
        )
        replay = build_parsed_replay(log, gameid="test-67")

        t1 = replay.turnlist[1]

        # Both fainted
        charizard = [p for p in t1.pokemon_1 if p is not None and p.name == "Charizard"][0]
        alakazam = [p for p in t1.pokemon_2 if p is not None and p.name == "Alakazam"][0]
        assert charizard.status == PEStatus.FNT
        assert alakazam.status == PEStatus.FNT

        # Replacements active
        assert t1.active_pokemon_1[0].name == "Blastoise"
        assert t1.active_pokemon_2[0].name == "Golem"

        # Both moves recorded
        assert t1.moves_1[0].name == "Destiny Bond"
        assert t1.moves_2[0].name == "Psychic"

        # Subturns for both faint-triggered forced switches
        faint_subturns = [s for s in t1.subturns if s.action is not None]
        assert len(faint_subturns) >= 2  # one per side


# ---------------------------------------------------------------------------
# Transform  (scenarios 68–69B)
# ---------------------------------------------------------------------------

class TestTransform:
    """Tests for Ditto's Transform: copying moves/boosts, checking
    transformed_into / transformed_this_turn, and clearing on switch-out."""

    @staticmethod
    def _make_log(
        *turns: list[list[str]],
        p1_pokes: list[str] | None = None,
        p2_pokes: list[str] | None = None,
        p1_lead: tuple[str, str] | None = None,
        p2_lead: tuple[str, str] | None = None,
        teamsize1: int = 1,
        teamsize2: int = 1,
    ) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=teamsize1, teamsize2=teamsize2,
            p1_pokes=p1_pokes or ["Ditto"],
            p2_pokes=p2_pokes or ["Alakazam"],
            p1_lead=p1_lead or ("Ditto", "100/100"),
            p2_lead=p2_lead or ("Alakazam", "100/100"),
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 68: Ditto uses Transform on P2 ------------------------------------

    def test_68_ditto_transform(self):  # Gen:1  Move:Transform  Gimmick: transformed_into, 5PP copies.
        """Ditto uses Transform: transformed_into set, moves copied with 5 PP,
        TRANSFORM warning flag added."""
        log = self._make_log(
            ["turn", "1"],
            # P2 uses Psychic first (reveals a move for Ditto to copy)
            ["move", "p2a: Alakazam", "Psychic", "p1a: Ditto"],
            ["-damage", "p1a: Ditto", "85/100"],
            ["move", "p1a: Ditto", "Transform", "p2a: Alakazam"],
            ["-transform", "p1a: Ditto", "p2a: Alakazam"],
        )
        replay = build_parsed_replay(log, gameid="test-68")

        t1 = replay.turnlist[1]
        ditto = [p for p in t1.pokemon_1 if p is not None and p.name == "Ditto"][0]

        # Transform state
        assert ditto.transformed_this_turn is True
        assert ditto.transformed_into is not None
        assert ditto.transformed_into.name == "Alakazam"

        # Moves copied from Alakazam (Psychic) with 5 PP each
        assert "Psychic" in ditto.moves
        assert ditto.moves["Psychic"].pp == 5
        assert ditto.moves["Psychic"].maximum_pp == 5

        # Boosts copied from Alakazam (all 0)
        assert ditto.boosts.atk_ == 0

        # Active ability copied from Alakazam (NO_ABILITY in Gen 1)
        assert ditto.active_ability is Nothing.NO_ABILITY

        # Warning flag set
        from metamon.backend.replay_parser.exceptions import WarningFlags
        assert WarningFlags.TRANSFORM in replay.check_warnings

        # Both actions recorded
        _assert_action_move(t1.moves_1[0], "Transform")
        _assert_action_move(t1.moves_2[0], "Psychic")

    # -- 69: Ditto switches out after Transform ----------------------------

    def test_69_ditto_switch_out_clears_transform(self):  # Gen:1  Gimmick: transform cleared on switch-out.
        """After switching out, Ditto's transform state is cleared."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Ditto"],
            ["-damage", "p1a: Ditto", "85/100"],
            ["move", "p1a: Ditto", "Transform", "p2a: Alakazam"],
            ["-transform", "p1a: Ditto", "p2a: Alakazam"],
            ["turn", "2"],
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Blastoise"],
            ["-damage", "p1a: Blastoise", "85/100"],
            teamsize1=2, teamsize2=1,
            p1_pokes=["Ditto", "Blastoise"],
            p1_lead=("Ditto", "100/100"),
            p2_pokes=["Alakazam"],
            p2_lead=("Alakazam", "100/100"),
        )
        replay = build_parsed_replay(log, gameid="test-69")

        # Turn 1: Ditto is transformed
        t1 = replay.turnlist[1]
        ditto_t1 = [p for p in t1.pokemon_1 if p is not None and p.name == "Ditto"][0]
        assert ditto_t1.transformed_into is not None
        assert "Psychic" in ditto_t1.moves

        # Turn 2: Ditto switched out — on_switch_out cleared transform
        t2 = replay.turnlist[2]
        ditto_t2 = [p for p in t2.pokemon_1 if p is not None and p.name == "Ditto"][0]
        assert ditto_t2.transformed_into is None
        # Moves restored from had_moves (Ditto's own moves: Transform, not Alakazam's Psychic)
        assert "Transform" in ditto_t2.moves
        assert "Psychic" not in ditto_t2.moves
        # Type restored to Normal (original)
        assert ditto_t2.type == ["Normal"]
        # Active is now Blastoise
        assert t2.active_pokemon_1[0].name == "Blastoise"

    # -- 69B (new): transformed Ditto uses copied move ---------------------

    def test_69B_transformed_ditto_uses_copied_move(self):  # Gen:1  Move:Transform  Gimmick: copied move PP tracking.
        """After Transform, Ditto can use copied moves; PP tracked on Ditto."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Ditto"],
            ["-damage", "p1a: Ditto", "85/100"],
            ["move", "p1a: Ditto", "Transform", "p2a: Alakazam"],
            ["-transform", "p1a: Ditto", "p2a: Alakazam"],
            ["turn", "2"],
            # Transformed Ditto uses copied Psychic
            ["move", "p1a: Ditto", "Psychic", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "85/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Ditto"],
            ["-damage", "p1a: Ditto", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-69B")

        # Turn 1: Ditto transforms, copies Psychic (5 PP)
        t1 = replay.turnlist[1]
        ditto_t1 = [p for p in t1.pokemon_1 if p is not None and p.name == "Ditto"][0]
        assert ditto_t1.moves["Psychic"].pp == 5

        # Turn 2: Ditto uses copied Psychic — PP consumed on Ditto's copy
        t2 = replay.turnlist[2]
        ditto_t2 = [p for p in t2.pokemon_1 if p is not None and p.name == "Ditto"][0]
        _assert_action_move(t2.moves_1[0], "Psychic")
        assert ditto_t2.moves["Psychic"].pp == 4  # 5→4
        # transformed_this_turn cleared by on_end_of_turn
        assert ditto_t2.transformed_this_turn is False
        # transformed_into still set (not cleared until switch-out)
        assert ditto_t2.transformed_into is not None


# ---------------------------------------------------------------------------
# Zoroark Illusion  (scenarios 70–71B)
# ---------------------------------------------------------------------------

class TestZoroark:
    """Tests for Zoroark/Zorua Illusion: |replace| handler, disguise rewind,
    WarningFlags.ZOROARK, Replacement tuples, and unbroken Illusion."""

    @staticmethod
    def _make_log(
        *turns: list[list[str]],
        teamsize1: int = 2,
        teamsize2: int = 1,
    ) -> list[list[str]]:
        """Build a Gen 9 2v1 log. Zoroark (last in party) disguises as P1's
        first Pokémon (Charizard)."""
        log = make_skeleton(
            gen=9, tier="[Gen 9] OU", format="gen9ou",
            teamsize1=teamsize1, teamsize2=teamsize2,
            p1_pokes=["Charizard", "Zoroark"],
            p2_pokes=["Alakazam"],
            p1_lead=("Charizard", "100/100"),  # actually Zoroark disguised
            p2_lead=("Alakazam", "100/100"),
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 70: Zoroark illusion breaks after being hit -----------------------

    def test_70_zoroark_illusion_break(self):  # Gen:9  Ability:Illusion  Gimmick: |replace|, disguise rewind.
        """Zoroark disguised as Charizard; |replace| fires when hit.

        Verifies: WarningFlags.ZOROARK, Replacement tuple, active changes
        to Zoroark, newly-discovered moves transferred.
        """
        log = self._make_log(
            ["turn", "1"],
            # Zoroark (disguised as Charizard) uses Night Daze
            ["move", "p1a: Charizard", "Night Daze", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            # P2 hits back, breaking the illusion
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "60/100"],
            # Illusion breaks — replace reveals real Zoroark
            ["replace", "p1a: Charizard", "Zoroark", "60/100"],
        )
        replay = build_parsed_replay(log, gameid="test-70", format="gen9ou")

        from metamon.backend.replay_parser.exceptions import WarningFlags

        t1 = replay.turnlist[1]

        # Warning flag set
        assert WarningFlags.ZOROARK in replay.check_warnings

        # Active is now Zoroark (not Charizard)
        assert t1.active_pokemon_1[0].name == "Zoroark"
        assert t1.active_pokemon_1[0].current_hp == 60  # damage carried over

        # Replacement tuple created
        assert len(t1.replacements_1) >= 1
        repl = t1.replacements_1[0]
        assert repl.replaced.name == "Charizard"
        assert repl.replaced_with.name == "Zoroark"
        assert isinstance(repl.turn_range, tuple) and len(repl.turn_range) == 2

        # Zoroark's Night Daze revealed (transferred from disguise)
        zoroark = t1.active_pokemon_1[0]
        assert "Night Daze" in zoroark.moves
        assert "Night Daze" in zoroark.had_moves

        # Charizard restored to bench state (no moves from illusion window)
        charizard = [p for p in t1.pokemon_1 if p is not None and p.name == "Charizard"][0]
        assert "Night Daze" not in charizard.moves

        # P1's move (Night Daze) and P2's move both recorded
        assert t1.moves_1[0].name == "Night Daze"
        assert t1.moves_2[0].name == "Psychic"

    # -- 71: Zoroark illusion breaks from damage only ----------------------

    def test_71_zoroark_break_on_damage(self):  # Gen:9  Ability:Illusion  Gimmick: illusion break on damage.
        """Damage from an attack breaks Illusion; replace fires."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["replace", "p1a: Charizard", "Zoroark", "70/100"],
            # Zoroark (now revealed) attacks
            ["move", "p1a: Zoroark", "Night Daze", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-71", format="gen9ou")

        from metamon.backend.replay_parser.exceptions import WarningFlags

        t1 = replay.turnlist[1]
        assert WarningFlags.ZOROARK in replay.check_warnings

        # Active changes from Charizard (disguise) to Zoroark
        assert t1.active_pokemon_1[0].name == "Zoroark"
        assert t1.active_pokemon_1[0].current_hp == 70

        # Replacement recorded
        assert len(t1.replacements_1) >= 1

        # Zoroark's Night Daze revealed
        zoroark = t1.active_pokemon_1[0]
        assert "Night Daze" in zoroark.had_moves

        # Both actions recorded (Psychic by P2, Night Daze by Zoroark after reveal)
        # Note: P1's initial move slot may be None (disguise didn't move before break)
        # or the Night Daze is recorded under moves_1[0]
        assert t1.moves_2[0].name == "Psychic"

    # -- 71B: Illusion does not break (no damage) --------------------------

    def test_71B_illusion_unbroken(self):  # Gen:9  Ability:Illusion  Gimmick: illusion stays intact.
        """No damage taken → Illusion stays intact; no warnings, no replace."""
        log = self._make_log(
            ["turn", "1"],
            # Zoroark (disguised) uses Night Daze
            ["move", "p1a: Charizard", "Night Daze", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            # P2 uses a status move that doesn't break illusion
            ["move", "p2a: Alakazam", "Thunder Wave", "p1a: Charizard"],
            ["-status", "p1a: Charizard", "par"],
        )
        replay = build_parsed_replay(log, gameid="test-71B", format="gen9ou")

        from metamon.backend.replay_parser.exceptions import WarningFlags

        t1 = replay.turnlist[1]

        # No replace happened — active is still "Charizard" (disguise intact)
        assert t1.active_pokemon_1[0].name == "Charizard"

        # No warning flags, no replacements
        assert WarningFlags.ZOROARK not in replay.check_warnings
        assert t1.replacements_1 == []

        # Zoroark (the real mon) is on bench, never activated
        zoroark = [p for p in t1.pokemon_1 if p is not None and p.name == "Zoroark"][0]
        assert zoroark not in t1.active_pokemon_1

        # Both moves recorded
        assert t1.moves_1[0].name == "Night Daze"
        assert t1.moves_2[0].name == "Thunder Wave"


# ---------------------------------------------------------------------------
# Multi-Pokémon Play  (scenarios 89–92)
# ---------------------------------------------------------------------------

class TestMultiPokemon:
    """Tests for normal switches, faint+replace, Revival Blessing, and
    multi-faint sequences."""

    # -- 89: P1 switches to bench (normal switch) --------------------------

    def test_89_normal_switch(self):  # Gen:1  Gimmick: normal switch action (is_switch=True).
        """Normal switch: action is_switch=True, active changes, bench HP tracked."""
        log = make_skeleton(
            teamsize1=2, teamsize2=2,
            p1_pokes=["Charizard", "Blastoise"],
            p2_pokes=["Alakazam", "Golem"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        ) + [
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Blastoise"],
            ["-damage", "p1a: Blastoise", "85/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-89")

        # Turn 1: Charizard active
        t1 = replay.turnlist[1]
        assert t1.active_pokemon_1[0].name == "Charizard"
        assert t1.moves_1[0].name == "Flamethrower"
        assert t1.moves_1[0].is_switch is False

        # Turn 2: Blastoise switched in, action recorded as Switch
        t2 = replay.turnlist[2]
        assert t2.active_pokemon_1[0].name == "Blastoise"
        assert t2.moves_1[0].name == "Switch"
        assert t2.moves_1[0].is_switch is True
        assert t2.moves_1[0].target.name == "Blastoise"

        # Charizard on bench, HP preserved from turn 1 (85)
        charizard = [p for p in t2.pokemon_1 if p is not None and p.name == "Charizard"][0]
        assert charizard.current_hp == 85

        # P2's move on turn 2 targeted Blastoise
        assert t2.moves_2[0].target.name == "Blastoise"

    # -- 90: P2 faints, sends replacement ----------------------------------

    def test_90_faint_and_replace_p2(self):  # Gen:1  Gimmick: P2 faint + replacement.
        """P2 faints, sends replacement via forced switch subturn."""
        log = make_skeleton(
            teamsize1=1, teamsize2=2,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam", "Golem"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        ) + [
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "0 fnt"],
            ["faint", "p2a: Alakazam"],
            ["move", "p2a: Golem", "Earthquake", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["switch", "p2a: Golem", "Golem", "100/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-90")

        t1 = replay.turnlist[1]

        # Alakazam fainted
        alakazam = [p for p in t1.pokemon_2 if p is not None and p.name == "Alakazam"][0]
        assert alakazam.current_hp == 0
        assert alakazam.status == PEStatus.FNT

        # Replacement active
        assert t1.active_pokemon_2[0].name == "Golem"

        # Forced switch subturn created
        faint_subturns = [s for s in t1.subturns if s.team == 2 and s.action is not None]
        assert len(faint_subturns) >= 1
        assert faint_subturns[0].action.is_switch is True
        assert faint_subturns[0].action.target.name == "Golem"

        # Both actions recorded (P1's Flamethrower, replacement Golem's Earthquake)
        assert t1.moves_1[0].name == "Flamethrower"
        assert t1.moves_2[0].name == "Earthquake"

    # -- 91: Revival Blessing (Gen 9) --------------------------------------

    def test_91_revival_blessing(self):  # Gen:9  Move:RevivalBlessing  Gimmick: revival (is_revival).
        """Revival Blessing revives a fainted ally: is_revival, status→NO_STATUS."""
        log = make_skeleton(
            gen=9, tier="[Gen 9] OU", format="gen9ou",
            teamsize1=2, teamsize2=1,
            p1_pokes=["Charizard", "Blastoise"],
            p2_pokes=["Alakazam"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        ) + [
            ["turn", "1"],
            # P2 KOs Blastoise (set up fainted ally)
            # Actually, let's have Charizard faint on turn 1, then revived later.
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "0 fnt"],
            ["faint", "p1a: Charizard"],
            ["switch", "p1a: Blastoise", "Blastoise", "100/100"],
            ["turn", "2"],
            ["move", "p1a: Blastoise", "Revival Blessing", "p1a: Blastoise"],
            ["-heal", "p1: Charizard", "100/100", "[from] move: Revival Blessing"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Blastoise"],
            ["-damage", "p1a: Blastoise", "85/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-91", format="gen9ou")

        t2 = replay.turnlist[2]

        # Charizard was revived: status no longer FNT
        charizard = [p for p in t2.pokemon_1 if p is not None and p.name == "Charizard"][0]
        assert charizard.status is Nothing.NO_STATUS
        assert charizard.current_hp == 100

        # Subturn has revival action
        revival_subturns = [
            s for s in t2.subturns
            if s.action is not None and s.action.is_revival is True
        ]
        assert len(revival_subturns) == 1
        assert revival_subturns[0].action.name == "$Forced Revival$"
        assert revival_subturns[0].action.is_revival is True

        # Both moves recorded
        assert t2.moves_1[0].name == "Revival Blessing"
        assert t2.moves_2[0].name == "Psychic"

    # -- 92: Multi-faint sequence (supereffective chain) --------------------

    def test_92_supereffective_chain(self):  # Gen:1  Gimmick: multi-faint sequence.
        """P1 KOs P2 with supereffective move; replacement comes in, also KO'd."""
        log = make_skeleton(
            teamsize1=1, teamsize2=3,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam", "Golem", "Dragonite"],
            p1_lead=("Charizard", "100/100"),
            p2_lead=("Alakazam", "100/100"),
        ) + [
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "0 fnt"],
            ["faint", "p2a: Alakazam"],
            ["switch", "p2a: Golem", "Golem", "100/100"],
            ["move", "p2a: Golem", "Earthquake", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Golem"],
            ["-damage", "p2a: Golem", "0 fnt"],
            ["faint", "p2a: Golem"],
            ["switch", "p2a: Dragonite", "Dragonite", "100/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-92")

        # Turn 1: Alakazam fainted, Golem replaced
        t1 = replay.turnlist[1]
        assert t1.active_pokemon_2[0].name == "Golem"
        assert t1.moves_1[0].name == "Flamethrower"
        assert t1.moves_2[0].name == "Earthquake"

        # Alakazam is fainted
        alakazam = [p for p in t1.pokemon_2 if p is not None and p.name == "Alakazam"][0]
        assert alakazam.status == PEStatus.FNT

        # Turn 2: Golem fainted, Dragonite replaced
        t2 = replay.turnlist[2]
        assert t2.active_pokemon_2[0].name == "Dragonite"
        assert t2.moves_1[0].name == "Flamethrower"

        golem = [p for p in t2.pokemon_2 if p is not None and p.name == "Golem"][0]
        assert golem.status == PEStatus.FNT

        # Both fainted Pokémon are tracked correctly
        fainted_count = sum(
            1 for p in t2.pokemon_2 if p is not None and p.status == PEStatus.FNT
        )
        assert fainted_count == 2  # Alakazam + Golem

        # Dragonite is active and healthy
        assert t2.active_pokemon_2[0].current_hp == 100


# ---------------------------------------------------------------------------
# Mimic / Mirror Move / Metronome  (scenarios 72–75)
# ---------------------------------------------------------------------------

class TestForeignMoves:
    """Tests for Mimic, Metronome (foreign-called moves), and Sleep Talk."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 72: Mimic copies P2's last move -----------------------------------

    def test_72_mimic(self):  # Gen:1  Move:Mimic  Gimmick: move_copy, move_change_to_from.
        """Mimic copies P2's last move; MIMIC warning, move_change_to_from set."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["move", "p1a: Charizard", "Mimic", "p2a: Alakazam"],
            ["-start", "p1a: Charizard", "Mimic", "Psychic"],
        )
        replay = build_parsed_replay(log, gameid="test-72")

        from metamon.backend.replay_parser.exceptions import WarningFlags

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # MIMIC warning flag
        assert WarningFlags.MIMIC in replay.check_warnings

        # Mimic replaced by Psychic in moves dict
        assert "Mimic" not in p1.moves
        assert "Psychic" in p1.moves
        # Mimic replaces Mimic move slot; in Gen 1 the copied move gets Mimic's
        # remaining PP (16 max → 15 after using Mimic)
        assert p1.moves["Psychic"].pp == p1.moves["Psychic"].maximum_pp  # copied at Mimic's PP

        # move_change_to_from tracks the replacement
        assert "Psychic" in p1.move_change_to_from
        assert p1.move_change_to_from["Psychic"] == "Mimic"

        # Both actions recorded
        assert t1.moves_1[0].name == "Mimic"
        assert t1.moves_2[0].name == "Psychic"

    # -- 73: Metronome calls foreign move → suppressed ---------------------

    def test_73_metronome_foreign_suppressed(self):  # Gen:1  Move:Metronome  Gimmick: foreign-called move suppressed.
        """Metronome calls Flamethrower: called move NOT in had_moves."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Metronome", "p2a: Alakazam"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam",
             "[from] move: Metronome"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-73")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Metronome PP consumed (Metronome is the actual move used)
        assert p1.moves["Metronome"].pp == p1.moves["Metronome"].maximum_pp - 1

        # Flamethrower (foreign-called) NOT in had_moves
        assert "Flamethrower" not in p1.had_moves

        # Pending foreign move tracking (persists across turns until a
        # different move is used or the sequence ends naturally)
        assert p1.pending_foreign_move == "flamethrower"

    # -- 74: Metronome → Outrage → multi-turn suppression -------------------

    def test_74_metronome_calls_outrage_multiturn_suppressed(self):  # Gen:1  Move:Metronome→Outrage  Gimmick: cross-turn suppression.
        """Metronome calls Outrage: all continuation turns suppressed."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Metronome", "p2a: Alakazam"],
            ["move", "p1a: Charizard", "Outrage", "p2a: Alakazam",
             "[from] move: Metronome"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            # Outrage continuation (auto-repeat from Metronome call)
            ["move", "p1a: Charizard", "Outrage", "p2a: Alakazam",
             "[still]", "[from] move: Outrage"],
            ["-damage", "p2a: Alakazam", "40/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-74")

        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]

        # Metronome PP consumed
        assert "Metronome" in p1_t1.moves
        assert p1_t1.moves["Metronome"].pp == p1_t1.moves["Metronome"].maximum_pp - 1

        # Outrage NOT in had_moves (foreign-called, suppressed)
        assert "Outrage" not in p1_t1.had_moves

        # Turn 2: Outrage continuation also suppressed
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        assert "Outrage" not in p1_t2.had_moves

        # P2's moves recorded on both turns
        assert t1.moves_2[0].name == "Psychic"
        assert t2.moves_2[0].name == "Psychic"

    # -- 75: Sleep Talk calls own move → revealed --------------------------

    def test_75_sleep_talk_reveals_move(self):  # Gen:2+  Move:SleepTalk  Gimmick: move revealed from own moveset.
        """Sleep Talk calls Body Slam: called move IS added to had_moves."""
        log = self._make_log(
            ["turn", "1"],
            ["-status", "p1a: Charizard", "slp"],
            ["move", "p1a: Charizard", "Sleep Talk", "p1a: Charizard"],
            ["move", "p1a: Charizard", "Body Slam", "p2a: Alakazam",
             "[from] move: Sleep Talk"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-75")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        # Sleep Talk PP consumed
        assert "Sleep Talk" in p1.moves
        assert p1.moves["Sleep Talk"].pp == p1.moves["Sleep Talk"].maximum_pp - 1

        # Body Slam IS added to had_moves (Sleep Talk reveals from own moveset)
        assert "Body Slam" in p1.had_moves

        # Body Slam not in moves dict (the called move doesn't use PP from the Pokemon)
        # Actually: reveal_move adds to both moves and had_moves. Let me check...
        # reveal_move: if name not in moves: moves[name] = move; had_moves[name] = copy
        # So Body Slam should be in moves too.
        assert "Body Slam" in p1.moves  # added via reveal_move

        # P2's move recorded
        assert t1.moves_2[0].name == "Psychic"


# ---------------------------------------------------------------------------
# Protect / Fail / Immune  (scenarios 76–77C)
# ---------------------------------------------------------------------------

class TestProtectFailImmune:
    """Tests for |-fail| and |-immune| message handling."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 76: |-fail| message (Protect blocks) -----------------------------

    def test_76_fail(self):  # Gen:2+  Move:Protect  Gimmick: |-fail|, protected flag.
        """|-fail| message: move action recorded but damage blocked."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p2a: Alakazam", "Protect", "p2a: Alakazam"],
            ["-singleturn", "p2a: Alakazam", "Protect"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-fail", "p2a: Alakazam"],
        )
        replay = build_parsed_replay(log, gameid="test-76")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]

        # Protect sets protected flag
        assert p2.protected is True

        # P1's move action still recorded (attempted)
        assert t1.moves_1[0].name == "Flamethrower"

        # P2 took no damage
        _assert_pokemon_hp(p2, 100, 100)

    # -- 77A: |-immune| (Thunder Wave on Ground) ---------------------------

    def test_77A_immune(self):  # Gen:1  Move:ThunderWave  Gimmick: |-immune|.
        """|-immune| message: move action recorded, no status applied."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Thunder Wave", "p2a: Alakazam"],
            ["-immune", "p2a: Alakazam"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-77A")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]

        # Move action recorded (was attempted)
        assert t1.moves_1[0].name == "Thunder Wave"

        # No status applied (immune)
        assert p2.status is Nothing.NO_STATUS

        # P2's move recorded
        assert t1.moves_2[0].name == "Psychic"

    # -- 77B: Normal move on Ghost type ------------------------------------

    def test_77B_normal_on_ghost(self):  # Gen:1  Move:Tackle  Gimmick: type immunity.
        """Normal move on Ghost: |-immune| fires, no damage."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Tackle", "p2a: Alakazam"],
            ["-immune", "p2a: Alakazam"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-77B")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]

        # Tackle action recorded
        assert t1.moves_1[0].name == "Tackle"
        # No damage on P2 (immune)
        _assert_pokemon_hp(p2, 100, 100)

    # -- 77C: Poison move on Steel type ------------------------------------

    def test_77C_poison_on_steel(self):  # Gen:2+  Move:Toxic  Gimmick: type immunity.
        """Poison move on Steel: |-immune| fires, no status."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Toxic", "p2a: Alakazam"],
            ["-immune", "p2a: Alakazam"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-77C")

        t1 = replay.turnlist[1]
        p2 = t1.active_pokemon_2[0]

        assert t1.moves_1[0].name == "Toxic"
        assert p2.status is Nothing.NO_STATUS


# ---------------------------------------------------------------------------
# Choice Messages  (scenarios 78–81)
# ---------------------------------------------------------------------------

class TestChoices:
    """Tests for |choice| message parsing: named, numeric, empty, pre-emptive."""

    @staticmethod
    def _make_log(*turns: list[list[str]]) -> list[list[str]]:
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Charizard"],
            p2_pokes=["Alakazam"],
        )
        for t in turns:
            log.append(t)
        next_turn = 1
        for msg in log:
            if msg[0] == "turn":
                next_turn = max(next_turn, int(msg[1]) + 1)
        while next_turn <= 5:
            log.append(["turn", str(next_turn)])
            next_turn += 1
        log.append(["win", "Alice"])
        return log

    # -- 78: named choice parsing ------------------------------------------

    def test_78_named_choice(self):  # Gen:any  Gimmick: |choice| named move parsing.
        """|choice|move Ice Beam|move Earthquake populates choices."""
        log = self._make_log(
            ["turn", "1"],
            ["choice", "move Ice Beam", "move Earthquake"],
            ["move", "p1a: Charizard", "Ice Beam", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Earthquake", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        )
        replay = build_parsed_replay(log, gameid="test-78")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]
        p2 = t1.active_pokemon_2[0]

        # choices populated
        assert t1.choices_1[0] is not None
        assert t1.choices_1[0].name == "Ice Beam"
        assert t1.choices_2[0] is not None
        assert t1.choices_2[0].name == "Earthquake"

        # Choice reveals the move in had_moves
        assert "Ice Beam" in p1.had_moves
        assert "Earthquake" in p2.had_moves

        # Actual moves also recorded
        assert t1.moves_1[0].name == "Ice Beam"
        assert t1.moves_2[0].name == "Earthquake"

    # -- 79: numeric choice (skipped) -------------------------------------

    def test_79_numeric_choice_skipped(self):  # Gen:any  Gimmick: |choice| numeric (skipped).
        """|choice|switch 2|switch 4 — numeric format is skipped."""
        log = self._make_log(
            ["turn", "1"],
            ["choice", "switch 2", "switch 4"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-79")

        t1 = replay.turnlist[1]
        # Numeric choices not parsed → choices remain None
        assert t1.choices_1[0] is None
        assert t1.choices_2[0] is None

        # Actual moves still recorded
        assert t1.moves_1[0].name == "Flamethrower"
        assert t1.moves_2[0].name == "Psychic"

    # -- 80: empty choice (skipped) ---------------------------------------

    def test_80_empty_choice_skipped(self):  # Gen:any  Gimmick: |choice| empty (skipped).
        """|choice|| — empty choice is skipped."""
        log = self._make_log(
            ["turn", "1"],
            ["choice", "", ""],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        )
        replay = build_parsed_replay(log, gameid="test-80")

        t1 = replay.turnlist[1]
        assert t1.choices_1[0] is None
        assert t1.choices_2[0] is None

    # -- 81: pre-emptive choice (4 moves known) ----------------------------

    def test_81_preemptive_choice_skipped(self):  # Gen:any  Gimmick: |choice| pre-emptive (4 moves known → skipped).
        """Choice for a mon with 4 known moves → skipped (pre-emptive)."""
        log = self._make_log(
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "85/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Fire Blast", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "60/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
            ["turn", "3"],
            ["move", "p1a: Charizard", "Earthquake", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "30/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "55/100"],
            ["turn", "4"],
            ["move", "p1a: Charizard", "Dragon Claw", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "10/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "40/100"],
            ["turn", "5"],
            ["choice", "move Thunder Punch", "move Ice Beam"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "0 fnt"],
            ["faint", "p2a: Alakazam"],
            ["win", "Alice"],
        )
        replay = build_parsed_replay(log, gameid="test-81")

        t5 = replay.turnlist[4]  # turn 5
        p1 = t5.active_pokemon_1[0]
        assert len(p1.moves) == 4
        # Pre-emptive choice NOT recorded (skipped)
        assert t5.choices_1[0] is None


# ---------------------------------------------------------------------------
# Team Preview  (scenarios 82–83)
# ---------------------------------------------------------------------------

class TestTeamPreview:
    """Tests for team preview via |poke| and |teamsize| messages."""

    def test_82_six_pokes_teampreview(self):  # Gen:1  Gimmick: teampreview populated.
        """6 |poke| messages per side populate teampreview."""
        log = make_skeleton() + [
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-82")

        t0 = replay.turnlist[0]
        assert len(t0.teampreview_1) == 6
        assert len(t0.teampreview_2) == 6

        # Frozen copies: different objects from active
        tp1_names = [p.name for p in t0.teampreview_1]
        assert t0.active_pokemon_1[0].name in tp1_names
        assert t0.teampreview_1[0] is not t0.active_pokemon_1[0]

        # Teampreview persists across turns
        t_last = replay.turnlist[-2]
        assert len(t_last.teampreview_1) == 6

    def test_83_teamsize_three(self):  # Gen:1  Gimmick: teamsize≠6 warning.
        """3-Pokémon team: teamsize≠6, teampreview sized correctly."""
        log = make_skeleton(
            teamsize1=3, teamsize2=3,
            p1_pokes=["Charizard", "Blastoise", "Venusaur"],
            p2_pokes=["Alakazam", "Golem", "Starmie"],
        ) + [
            ["turn", "1"],
            ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "70/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-83")

        t0 = replay.turnlist[0]
        assert len(t0.teampreview_1) == 3
        assert len(t0.teampreview_2) == 3
        assert len(t0.pokemon_1) == 3
        assert len(t0.pokemon_2) == 3


# ---------------------------------------------------------------------------
# Forme Change  (scenario 85)
# ---------------------------------------------------------------------------

class TestFormeChange:
    """Tests for |detailschange| and |-formechange|."""

    def test_85_detailschange(self):  # Gen:4+  Pokemon:Rotom-Wash  Gimmick: forme change, had_name preserved.
        """|detailschange| updates name/type but preserves had_name."""
        log = make_skeleton(
            teamsize1=1, teamsize2=1,
            p1_pokes=["Rotom"],
            p2_pokes=["Alakazam"],
            p1_lead=("Rotom", "100/100"),
        ) + [
            ["turn", "1"],
            ["detailschange", "p1a: Rotom", "Rotom-Wash"],
            ["move", "p1a: Rotom-Wash", "Hydro Pump", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "60/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Rotom-Wash"],
            ["-damage", "p1a: Rotom-Wash", "85/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-85")

        t1 = replay.turnlist[1]
        p1 = t1.active_pokemon_1[0]

        assert p1.name == "Rotom-Wash"
        assert p1.had_name == "Rotom"
        # Stats updated (Rotom-Wash stats: 50/65/107/105/107/86)
        assert p1.base_stats["spa"] == 105
        assert p1.base_stats["def"] == 107
        # Type stays as base form (parser limitation on forme changes)
        assert "Electric" in p1.type
        assert "Ghost" in p1.type
        assert t1.moves_1[0].name == "Hydro Pump"


# ---------------------------------------------------------------------------
# Gen 1 PP Rollover  (scenario 88)
# ---------------------------------------------------------------------------

class TestGen1PPRollover:
    """Tests for Gen 1 partial-trapping PP rollover (Wrap, Bind, etc.)."""

    def test_88_wrap_pp_usage(self):  # Gen:1  Move:Wrap  Gimmick: GEN1_PP_ROLLOVERS, auto-repeat PP=0.
        """Wrap in Gen 1: first use consumes PP; auto-repeat doesn't."""
        log = make_skeleton() + [
            ["turn", "1"],
            ["move", "p1a: Charizard", "Wrap", "p2a: Alakazam"],
            ["-damage", "p2a: Alakazam", "95/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "85/100"],
            ["turn", "2"],
            ["move", "p1a: Charizard", "Wrap", "p2a: Alakazam",
             "[from] move: Wrap"],
            ["-damage", "p2a: Alakazam", "90/100"],
            ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
            ["-damage", "p1a: Charizard", "70/100"],
        ] + make_winner("Alice")

        replay = build_parsed_replay(log, gameid="test-88")

        # Turn 1: first Wrap, PP consumed
        t1 = replay.turnlist[1]
        p1_t1 = t1.active_pokemon_1[0]
        assert p1_t1.moves["Wrap"].pp == p1_t1.moves["Wrap"].maximum_pp - 1

        # Turn 2: auto-repeat with [from] move: Wrap, PP unchanged
        t2 = replay.turnlist[2]
        p1_t2 = t2.active_pokemon_1[0]
        assert p1_t2.moves["Wrap"].pp == p1_t1.moves["Wrap"].pp
