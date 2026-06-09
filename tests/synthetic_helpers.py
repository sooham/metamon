"""
Shared helpers for constructing synthetic replay logs in tests.

Import from test files to build minimal raw replay logs without
creating JSON files on disk.
"""

from __future__ import annotations

import datetime
from typing import Optional


def make_skeleton(
    gen: int = 1,
    tier: str = "[Gen 1] OU",
    format: str = "gen1ou",
    p1: str = "Alice",
    p2: str = "Bob",
    teamsize1: int = 6,
    teamsize2: int = 6,
    p1_pokes: Optional[list[str]] = None,
    p2_pokes: Optional[list[str]] = None,
    p1_lead: Optional[tuple[str, str]] = None,
    p2_lead: Optional[tuple[str, str]] = None,
    p1_rating: Optional[str] = "",
    p2_rating: Optional[str] = "",
) -> list[list[str]]:
    """Return the common log prefix shared by all synthetic replays.

    Parameters
    ----------
    gen : int
        Generation number (1, 2, 3, 4, or 9).
    tier : str
        Raw tier string as it appears in the log (e.g. ``"[Gen 1] OU"``).
    format : str
        Short format id (e.g. ``"gen1ou"``).
    p1 : str
        Player 1's username.
    p2 : str
        Player 2's username.
    teamsize1, teamsize2 : int
        Number of Pokémon per team (default 6).
    p1_pokes, p2_pokes : list[str] | None
        Species names for team preview ``|poke|`` messages.
        Defaults to two distinct 6-mon teams.
    p1_lead, p2_lead : tuple[str, str] | None
        ``(species, hp_string)`` for the initial ``|switch|``.
        Defaults to the first Pokémon in each team at 100/100.
    p1_rating, p2_rating : str
        Player ratings (empty string = Unrated).

    Returns
    -------
    list[list[str]]
        Log prefix ending with the two initial ``|switch|`` messages.
    """
    if p1_pokes is None:
        p1_pokes = [
            "Charizard", "Blastoise", "Venusaur",
            "Pikachu", "Gengar", "Snorlax",
        ]
    if p2_pokes is None:
        p2_pokes = [
            "Alakazam", "Golem", "Starmie",
            "Dragonite", "Machamp", "Jolteon",
        ]
    if p1_lead is None:
        p1_lead = (p1_pokes[0], "100/100")
    if p2_lead is None:
        p2_lead = (p2_pokes[0], "100/100")

    log: list[list[str]] = [
        ["gen", str(gen)],
        ["tier", tier],
        ["rule", "Species Clause: Limit one of each Pokémon"],
        ["player", "p1", p1, p1_rating],
        ["player", "p2", p2, p2_rating],
        ["teamsize", "p1", str(teamsize1)],
        ["teamsize", "p2", str(teamsize2)],
    ]
    for poke in p1_pokes:
        log.append(["poke", "p1", poke, ""])
    for poke in p2_pokes:
        log.append(["poke", "p2", poke, ""])
    log.append(["start", ""])
    log.append(
        ["switch", f"p1a: {p1_lead[0]}", p1_lead[0], p1_lead[1]]
    )
    log.append(
        ["switch", f"p2a: {p2_lead[0]}", p2_lead[0], p2_lead[1]]
    )
    return log


def make_turn(turn_number: int) -> list[str]:
    """Return a ``|turn|N`` message."""
    return ["turn", str(turn_number)]


def make_winner(player_name: str) -> list[list[str]]:
    """Return empty turns + win for a battle that ends by decision.

    Adds turns 2–5 and a ``|win|`` so the replay satisfies the
    ``>= 5 turns`` check.  Use when the actual battle action ends
    on turn 1 (e.g. KO or forfeit-like scenarios).
    """
    return [
        ["turn", "2"],
        ["turn", "3"],
        ["turn", "4"],
        ["turn", "5"],
        ["win", player_name],
    ]


def make_faint_winner(winner: str, loser_active: str) -> list[list[str]]:
    """Return turns for a KO + win, satisfying ``>= 5 turns``.

    Adds extra turn boundaries (2–5), a ``|faint|`` for the loser's
    active Pokémon, and a ``|win|``.
    """
    return [
        ["turn", "2"],
        ["turn", "3"],
        ["turn", "4"],
        ["turn", "5"],
        ["faint", loser_active],
        ["win", winner],
    ]


def build_parsed_replay(
    log: list[list[str]],
    gameid: str = "test",
    format: str = "gen1ou",
):
    """Run ``forward_fill`` on a synthetic log and return the ``ParsedReplay``.

    This is the standard entry point for forward-only synthetic tests.
    """
    from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay

    replay = ParsedReplay(
        gameid=gameid,
        format=format,
        time_played=datetime.datetime(2020, 1, 1),
    )
    return forward_fill(replay, log)
