"""Save online Showdown protocol logs as parser-format replay text files."""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import orjson

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.team_prediction.predictor import NoPredictor


@dataclass(frozen=True)
class OnlineReplaySaveResult:
    """Summary of a parser-backed online replay save."""

    format_name: str
    output_dir: Path
    game_id: str
    saved_files: tuple[Path, ...]


def format_from_battle_tag(battle_tag: str, fallback: str) -> str:
    """Infer the Showdown format id from a battle room tag."""
    tag = battle_tag[1:] if battle_tag.startswith(">") else battle_tag
    if tag.startswith("battle-"):
        tag = tag.removeprefix("battle-")
    match = re.match(r"^([A-Za-z0-9]+)-\d+$", tag)
    if match:
        return match.group(1).lower()
    return (fallback or "unknown").lower()


def game_id_from_battle_tag(battle_tag: str, fallback_format: str) -> str:
    """Return the parser-style game id for an online battle tag."""
    tag = battle_tag[1:] if battle_tag.startswith(">") else battle_tag
    if tag.startswith("battle-"):
        tag = tag.removeprefix("battle-")
    if re.match(r"^[A-Za-z0-9]+-\d+$", tag):
        return tag
    fmt = format_from_battle_tag(battle_tag, fallback_format)
    suffix = re.sub(r"[^A-Za-z0-9]+", "", tag.rsplit("-", 1)[-1]) or "online"
    return f"{fmt}-{suffix}"


def players_from_raw_messages(raw_messages: Iterable[str]) -> tuple[str, str]:
    """Extract p1/p2 names from captured Showdown protocol messages."""
    players = {"p1": "p1", "p2": "p2"}
    for msg in raw_messages:
        parts = msg.split("|")
        if parts and parts[0] == "":
            parts = parts[1:]
        if len(parts) >= 3 and parts[0] == "player" and parts[1] in players:
            players[parts[1]] = parts[2] or parts[1]
    return players["p1"], players["p2"]


def online_replay_payload(
    raw_messages: list[str],
    battle_tag: str,
    fallback_format: str,
    *,
    upload_time: int | None = None,
) -> dict:
    """Build the raw-replay JSON shape consumed by ``ReplayParser``."""
    format_name = format_from_battle_tag(battle_tag, fallback_format)
    return {
        "players": list(players_from_raw_messages(raw_messages)),
        "uploadtime": int(time.time() if upload_time is None else upload_time),
        "formatid": format_name,
        "format": format_name,
        "log": "\n".join(raw_messages),
    }


def save_online_replay_as_parsed(
    *,
    raw_messages: list[str],
    battle_tag: str,
    fallback_format: str,
    output_root: str | os.PathLike[str],
    upload_time: int | None = None,
    replay_parser_cls=ReplayParser,
) -> OnlineReplaySaveResult:
    """Run the replay parser on a captured online battle and save text outputs.

    ``output_root`` is the root directory, not the format directory.  Files are
    written to ``output_root/<format>/`` to match parsed replay cache layout.
    """
    if not raw_messages:
        raise ValueError("Cannot save an online replay without raw protocol messages.")

    format_name = format_from_battle_tag(battle_tag, fallback_format)
    game_id = game_id_from_battle_tag(battle_tag, format_name)
    output_dir = Path(output_root) / format_name
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = online_replay_payload(
        raw_messages,
        battle_tag,
        format_name,
        upload_time=upload_time,
    )
    with tempfile.TemporaryDirectory(prefix="metamon-online-replay-") as tmp:
        raw_path = Path(tmp) / f"{game_id}.json"
        raw_path.write_bytes(orjson.dumps(payload))
        parser = replay_parser_cls(
            replay_output_dir=str(output_dir),
            team_output_dir=None,
            verbose=False,
            compress=False,
            pretty=True,
            team_predictor=NoPredictor(),
        )
        parser.parse_replay(str(raw_path))

    saved_files = tuple(sorted(output_dir.glob(f"{game_id}_*.txt")))
    return OnlineReplaySaveResult(
        format_name=format_name,
        output_dir=output_dir,
        game_id=game_id,
        saved_files=saved_files,
    )
