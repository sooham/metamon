"""
Shared helper functions for the metamon test suite.

These are plain functions (not fixtures) that can be imported by any test module.
"""

import os
import glob
import random
import datetime
from typing import Optional

import orjson

from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay
from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.backward import backward_fill
from metamon.backend.team_prediction.predictor import (
    NoPredictor,
    NaiveUsagePredictor,
    TeamPredictor,
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

RAW_REPLAY_DIR = os.path.join(
    os.environ.get("METAMON_CACHE_DIR", os.path.expanduser("~/Repositories/poke-datasets")),
    "raw-replays",
)

# mapping of format string to generation and format
SUPPORTED_GENS = {
    "gen1ou": ("gen1", "ou"),
    "gen2ou": ("gen2", "ou"),
    "gen3ou": ("gen3", "ou"),
    "gen4ou": ("gen4", "ou"),
    "gen9ou": ("gen9", "ou"),
}

# fixed test files per gen for specific input output parsing tests
FIXED_TEST_BATTLES = {
    "gen1ou": ["gen1ou-316031019.json"]
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_raw_replay(path: str) -> dict:
    """Load a raw replay JSON file from disk."""
    with open(path, "rb") as f:
        return orjson.loads(f.read())

def load_battle(format_name : str, battle_id: str): 
    """
    For the given battle_id, get the battle id's associated raw replay files
    """
    entry = SUPPORTED_GENS.get(format_name)
    if entry is None:
        raise ValueError(f"Unsupported format: {format_name}")
    gen, tier = entry
    filename = os.path.join(RAW_REPLAY_DIR, gen, tier, battle_id + ".json"),
    return load_raw_replay(filename)

def load_fixed_battle(format_name: str):
    fixed_battle_id = FIXED_TEST_BATTLES[format_name]
    return load_battle(format_name, fixed_battle_id)


def find_random_replay_files(format_name: str, n: int = 10) -> list[str]:
    """Find up to *n* raw replay file paths for *format_name* (e.g. 'gen1ou').

    Searches both ``{raw_replays}/{gen}/{tier}/`` and ``{raw_replays}/{format}/``.
    Results are shuffled with a fixed seed for reproducibility.
    """
    entry = SUPPORTED_GENS.get(format_name)
    if entry is None:
        raise ValueError(f"Unsupported format: {format_name}")
    gen, tier = entry
    search_roots = [
        os.path.join(RAW_REPLAY_DIR, gen, tier),
        os.path.join(RAW_REPLAY_DIR, format_name),
    ]
    files = []
    for root in search_roots:
        if os.path.isdir(root):
            files.extend(glob.glob(os.path.join(root, "**/*.json"), recursive=True))
    # Deterministic shuffle
    rng = random.Random(42)
    rng.shuffle(files)
    return files[:n]


def run_forward_fill(path: str) -> ParsedReplay:
    """Load a raw replay and run *only* forward fill on it.

    Returns the ``ParsedReplay``.  Raises ``ForwardException`` on parse errors.
    """
    data = load_raw_replay(path)
    log = ReplayParser.clean_log(data)
    replay = ParsedReplay(
        gameid=data["id"],
        format=data.get("formatid", data.get("format", "unknown")),
        time_played=datetime.datetime.fromtimestamp(int(data["uploadtime"])),
    )
    return forward_fill(replay, log)


def run_full_parse(
    path: str,
    team_predictor: Optional[TeamPredictor] = None,
) -> tuple:
    """Run forward + backward fill and return the two POVReplays.

    Args:
        path: Raw replay JSON file path.
        team_predictor: Predictor to use. Defaults to NoPredictor for isolation.
    """
    if team_predictor is None:
        team_predictor = NoPredictor()
    data = load_raw_replay(path)
    log = ReplayParser.clean_log(data)
    replay = ParsedReplay(
        gameid=data["id"],
        format=data.get("formatid", data.get("format", "unknown")),
        time_played=datetime.datetime.fromtimestamp(int(data["uploadtime"])),
    )
    replay = forward_fill(replay, log)
    pov_p1, pov_p2 = backward_fill(replay, team_predictor=team_predictor)
    return pov_p1, pov_p2


def run_e2e_parse(
    path: str,
    output_dir: str,
    team_predictor: Optional[TeamPredictor] = None,
) -> str:
    """Run the full ReplayParser pipeline on a single replay file, saving output.

    Returns the path to the first output JSON file found.
    """
    if team_predictor is None:
        team_predictor = NoPredictor()
    parser = ReplayParser(
        replay_output_dir=output_dir,
        team_output_dir=None,
        verbose=False,
        team_predictor=team_predictor,
        compress=False,
    )
    parser.parse_replay(path)
    out_files = glob.glob(os.path.join(output_dir, "*.json"))
    assert out_files, f"No output produced for {path}"
    return out_files[0]
