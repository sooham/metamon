"""
Tests for forward-fill error handling, including cross-generation Pokémon
detection and other expected exception paths.
"""

from metamon.backend.replay_parser.exceptions import (
    CrossGenPokemonException,
    ForwardException,
)
from metamon.backend.replay_parser.forward import (
    forward_fill,
    ParsedReplay,
)
from metamon.backend.replay_parser.parse_replays import ReplayParser


def _build_gen1_replay_with_poke(pokemon_name: str, turns: int = 6) -> dict:
    """Build a minimal synthetic raw replay JSON for Gen 1 OU containing the given
    Pokémon in team preview.

    The battle runs *turns* of Tackle spam and ends with p1 winning.
    Default is 6 turns to satisfy check_finished().
    """
    log_lines = [
        "|gametype|singles",
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        "|teamsize|p1|1",
        "|teamsize|p2|1",
        "|gen|1",
        "|tier|[Gen 1] OU",
        "|rule|Species Clause: Limit one of each Pokémon",
        f"|poke|p1|{pokemon_name}|",
        "|poke|p2|Snorlax|",
        "|teampreview",
        "|",
        "|start",
        f"|switch|p1a: {pokemon_name}|{pokemon_name}|100/100",
        "|switch|p2a: Snorlax|Snorlax|100/100",
    ]
    for t in range(1, turns + 1):
        log_lines += [
            f"|turn|{t}",
            "|",
            f"|move|p1a: {pokemon_name}|Tackle|p2a: Snorlax",
            f"|-damage|p2a: Snorlax|{100 - t * 5}/100",
            f"|move|p2a: Snorlax|Tackle|p1a: {pokemon_name}",
            f"|-damage|p1a: {pokemon_name}|{100 - t * 3}/100",
            "|",
        ]
    log_lines.append("|win|Alice")
    return {
        "id": f"test-crossgen-{pokemon_name.lower()}",
        "formatid": "gen1ou",
        "players": ["Alice", "Bob"],
        "uploadtime": "1700000000",
        "log": "\n".join(log_lines),
    }


def _run_forward_on_synthetic(data: dict):
    """Run forward_fill on a synthetic replay dict."""
    log = ReplayParser.clean_log(data)
    replay = ParsedReplay(
        gameid=data["id"],
        format=data["formatid"],
        time_played=None,
    )
    return forward_fill(replay, log)


class TestCrossGenPokemonException:
    """Verify that cross-generation Pokémon in team preview raise
    CrossGenPokemonException during forward fill."""

    def test_gen5_pokemon_in_gen1_replay_raises(self):
        """Cinccino (Gen 5) in a Gen 1 OU replay should raise."""
        data = _build_gen1_replay_with_poke("Cinccino")
        try:
            _run_forward_on_synthetic(data)
            assert False, "Expected CrossGenPokemonException"
        except CrossGenPokemonException as e:
            assert "Cinccino" in str(e)
            assert "Generation 5" in str(e)
            assert "Generation 1" in str(e)

    def test_gen3_pokemon_in_gen1_replay_raises(self):
        """Slaking (Gen 3) in a Gen 1 OU replay should raise."""
        data = _build_gen1_replay_with_poke("Slaking")
        try:
            _run_forward_on_synthetic(data)
            assert False, "Expected CrossGenPokemonException"
        except CrossGenPokemonException as e:
            assert "Slaking" in str(e)

    def test_gen1_pokemon_in_gen1_replay_does_not_raise(self):
        """Snorlax (Gen 1) in a Gen 1 OU replay should parse successfully."""
        data = _build_gen1_replay_with_poke("Snorlax")
        replay = _run_forward_on_synthetic(data)
        assert replay.gen == 1
        assert len(replay.turnlist) >= 1

    def test_gen2_pokemon_in_gen2_replay_does_not_raise(self):
        """Typhlosion (Gen 2) in a Gen 2 OU replay should parse successfully."""
        # Build as Gen 2
        data = _build_gen1_replay_with_poke("Typhlosion")
        data["formatid"] = "gen2ou"
        data["id"] = "test-crossgen-typhlosion-gen2"
        # Patch log gen and tier
        log = data["log"]
        log = log.replace("|gen|1", "|gen|2")
        log = log.replace("[Gen 1] OU", "[Gen 2] OU")
        data["log"] = log
        replay = _run_forward_on_synthetic(data)
        assert replay.gen == 2
        assert len(replay.turnlist) >= 1

    def test_gen1_pokemon_does_not_raise_in_later_gen(self):
        """Snorlax (Gen 1) in a Gen 4 OU replay should parse successfully
        (lower-gen Pokémon are always legal in higher-gen formats)."""
        data = _build_gen1_replay_with_poke("Snorlax")
        data["formatid"] = "gen4ou"
        data["id"] = "test-crossgen-snorlax-gen4"
        log = data["log"]
        log = log.replace("|gen|1", "|gen|4")
        log = log.replace("[Gen 1] OU", "[Gen 4] OU")
        data["log"] = log
        replay = _run_forward_on_synthetic(data)
        assert replay.gen == 4
        assert len(replay.turnlist) >= 1
