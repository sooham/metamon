"""
Smoke tests for backward fill.

Verifies that ``backward_fill`` (with NoPredictor) completes without
crashing on a sample of raw replays across all supported generations.
"""

import datetime
import pytest

from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay
from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.backward import backward_fill
from metamon.backend.replay_parser.exceptions import ForwardException, BackwardException
from metamon.backend.team_prediction.predictor import NoPredictor

from tests.helpers import load_raw_replay, find_random_replay_files, SUPPORTED_GENS


class TestBackwardSmoke:
    """Run forward + backward fill on a handful of replays per gen."""

    @pytest.mark.parametrize("fmt", list(SUPPORTED_GENS.keys()))
    def test_backward_fill_N_replays(self, fmt):
        """Parse up to 5 replays from *fmt* with NoPredictor (isolation).

        Some may fail with expected exceptions; the test only fails if
        *every* replay crashes.
        """
        paths = find_random_replay_files(fmt, 5)
        assert paths, f"No replays for {fmt}"

        ok, skipped = 0, 0
        for p in paths:
            try:
                data = load_raw_replay(p)
                log = ReplayParser.clean_log(data)
                replay = ParsedReplay(
                    gameid=data["id"],
                    format=data.get("formatid", data.get("format", "unknown")),
                    time_played=datetime.datetime.fromtimestamp(int(data["uploadtime"])),
                )
                replay = forward_fill(replay, log)
                pov_p1, pov_p2 = backward_fill(replay, team_predictor=NoPredictor())
                assert len(pov_p1.povturnlist) > 0
                assert len(pov_p2.povturnlist) > 0
                ok += 1
            except (ForwardException, BackwardException):
                skipped += 1

        assert ok > 0, (
            f"All {len(paths)} replays for {fmt} failed backward fill."
        )
        print(f"  {fmt}: {ok} OK, {skipped} skipped")
