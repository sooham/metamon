"""
Shared pytest fixtures for the metamon replay parser test suite.

Fixtures are automatically discovered by pytest.  Plain helper functions
live in ``tests/helpers.py`` so that test modules can import them directly.
"""

import os

# Ensure METAMON_CACHE_DIR is set before any metamon imports
if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = os.path.expanduser(
        os.environ.get("METAMON_CACHE_DIR", "~/Repositories/poke-datasets")
    )

import pytest

from tests.helpers import (
    find_random_replay_files,
    load_fixed_battle,
    run_forward_fill,
    run_full_parse,

    SUPPORTED_GENS,
)
from metamon.backend.team_prediction.predictor import (
    NoPredictor,
    NaiveUsagePredictor,
    # TODO: add a random TeamPredictor which predicts random pokemon without 
    # any heed to stats
)

from metamon.backend.replay_parser.exceptions import ForwardException, BackwardException


# ---------------------------------------------------------------------------
# Module-scoped fixtures: one known-good ParsedReplay per gen (forward only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gen1_replay():
    """A successfully forward-parsed gen1 OU replay."""
    files = find_random_replay_files("gen1ou", 1)
    assert files, "No gen1ou replays found"
    return run_forward_fill(files[0])


@pytest.fixture(scope="module")
def gen2_replay():
    files = find_random_replay_files("gen2ou", 1)
    assert files, "No gen2ou replays found"
    return run_forward_fill(files[0])


@pytest.fixture(scope="module")
def gen3_replay():
    files = find_random_replay_files("gen3ou", 1)
    assert files, "No gen3ou replays found"
    return run_forward_fill(files[0])


@pytest.fixture(scope="module")
def gen4_replay():
    files = find_random_replay_files("gen4ou", 1)
    assert files, "No gen4ou replays found"
    return run_forward_fill(files[0])


@pytest.fixture(scope="module")
def gen9_replay():
    files = find_random_replay_files("gen9ou", 1)
    assert files, "No gen9ou replays found"
    return run_forward_fill(files[0])


# Parametrized fixture: one replay per gen
@pytest.fixture(
    scope="module",
    params=["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou"],
)
def parsed_replay(request):
    """Parametrized fixture yielding one forward-parsed replay per gen."""
    files = find_random_replay_files(request.param, 1)
    assert files, f"No replays for {request.param}"
    return run_forward_fill(files[0])

# ---------------------------------------------------------------------------
# Module-scoped fixtures: raw replay files (one per gen) 
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gen1_raw_replay():
    raw_replay_file = load_fixed_battle("gen1ou")
    return raw_replay_file

@pytest.fixture(scope="module")
def gen2_raw_replay():
    raw_replay_file = load_fixed_battle("gen2ou")
    return raw_replay_file

@pytest.fixture(scope="module")
def gen3_raw_replay():
    raw_replay_file = load_fixed_battle("gen3ou")
    return raw_replay_file

@pytest.fixture(scope="module")
def gen4_raw_replay():
    raw_replay_file = load_fixed_battle("gen4ou")
    return raw_replay_file

@pytest.fixture(scope="module")
def gen9_raw_replay():
    raw_replay_file = load_fixed_battle("gen9ou")
    return raw_replay_file


@pytest.fixture(
    scope="module",
    params=["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou"],
)
def raw_replay(request):
    """Parametrized fixture yielding one fixed battle raw replay per gen"""
    raw_replay_file = load_fixed_battle(request.param)
    return raw_replay_file


# ---------------------------------------------------------------------------
# Module-scoped fixtures: full forward+backward POV replays (one per gen)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pov_replays():
    """Dict mapping format_name -> (POVReplay_p1, POVReplay_p2) for one replay per gen.

    Uses NoPredictor for isolation — no usage stats, no team prediction.
    Suitable for testing forward/backward pass structure independently.
    """
    results = {}
    for fmt in SUPPORTED_GENS:
        files = find_random_replay_files(fmt, 1)
        if not files:
            continue
        try:
            pov_p1, pov_p2 = run_full_parse(files[0], team_predictor=NoPredictor())
            results[fmt] = (pov_p1, pov_p2)
        except (ForwardException, BackwardException):
            pass
    return results


@pytest.fixture(scope="module")
def pov_replays_predicted():
    """Dict mapping format_name -> (POVReplay_p1, POVReplay_p2) with prediction.

    Uses NaiveUsagePredictor to fill player moves from usage stats.
    Opponent info remains forward-observed only.
    """
    results = {}
    for fmt in SUPPORTED_GENS:
        files = find_random_replay_files(fmt, 1)
        if not files:
            continue
        try:
            pov_p1, pov_p2 = run_full_parse(
                files[0], team_predictor=NaiveUsagePredictor()
            )
            results[fmt] = (pov_p1, pov_p2)
        except (ForwardException, BackwardException):
            pass
    return results
