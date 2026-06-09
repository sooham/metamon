"""
End-to-end smoke tests using the full ReplayParser pipeline.

These tests run ``ReplayParser.parse_replay()`` with ``NaiveUsagePredictor`` on
real raw replay files and verify that output files are produced correctly.
"""

import os
import glob
import pytest
import orjson

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.exceptions import ForwardException, BackwardException
from metamon.backend.team_prediction.predictor import NaiveUsagePredictor

from tests.helpers import find_random_replay_files, SUPPORTED_GENS


class TestE2ESmoke:
    """Full pipeline tests on real replays."""

    @pytest.mark.parametrize("fmt", list(SUPPORTED_GENS.keys()))
    def test_full_pipeline_on_3_replays(self, fmt, tmp_path):
        """Run ReplayParser.parse_replay() on up to 3 replays per format.

        Verifies that output JSON files are created and are valid.
        """
        paths = find_random_replay_files(fmt, 3)
        assert paths, f"No raw replays for {fmt}"

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        parser = ReplayParser(
            replay_output_dir=output_dir,
            team_output_dir=None,
            verbose=False,
            team_predictor=NaiveUsagePredictor(),
            compress=False,
        )

        ok = 0
        for p in paths:
            try:
                parser.parse_replay(p)
                ok += 1
            except (ForwardException, BackwardException):
                pass

        assert ok > 0, f"All replays for {fmt} failed the full pipeline"

        # Check output files exist
        out_files = glob.glob(os.path.join(output_dir, "*.json"))
        assert len(out_files) >= ok, (
            f"Expected >= {ok} output files, found {len(out_files)}"
        )

        # Verify each output file has the expected structure
        for f in out_files:
            with open(f, "r") as fh:
                data = orjson.loads(fh.read())
            assert "states" in data, f"Missing 'states' in {f}"
            assert "actions" in data, f"Missing 'actions' in {f}"
            assert len(data["states"]) == len(data["actions"]), (
                f"states ({len(data['states'])}) != actions ({len(data['actions'])}) in {f}"
            )
            assert len(data["states"]) > 0, f"Empty states in {f}"

    def test_full_pipeline_single_replay(self, tmp_path):
        """Parse one replay end-to-end and do detailed validation."""
        paths = find_random_replay_files("gen1ou", 1)
        assert paths, "No gen1ou replays"

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        parser = ReplayParser(
            replay_output_dir=output_dir,
            team_output_dir=None,
            verbose=False,
            team_predictor=NaiveUsagePredictor(),
            compress=False,
        )
        parser.parse_replay(paths[0])

        out_files = glob.glob(os.path.join(output_dir, "*.json"))
        assert len(out_files) == 2, (
            f"Expected 2 POV files, got {len(out_files)}"
        )

        for f in out_files:
            with open(f, "r") as fh:
                data = orjson.loads(fh.read())

            # States must be a list of dicts
            assert isinstance(data["states"], list)
            assert all(isinstance(s, dict) for s in data["states"])

            # Actions must be a list of ints
            assert isinstance(data["actions"], list)
            assert all(isinstance(a, int) for a in data["actions"])

            # Action indices in valid range
            for a in data["actions"]:
                assert -1 <= a <= 13, f"Action index {a} out of range"

            # Validate required state keys on first and last state
            required_keys = {
                "format",
                "player_active_pokemon",
                "opponent_active_pokemon",
                "available_switches",
                "opponent_bench",
                "fainted_pokemon",
                "opponent_fainted",
                "player_prev_move",
                "opponent_prev_move",
                "opponents_remaining",
                "player_conditions",
                "opponent_conditions",
                "weather",
                "battle_field",
                "forced_switch",
                "battle_won",
                "battle_lost",
                "can_tera",
                "opponent_teampreview",
            }
            for state in data["states"]:
                for key in required_keys:
                    assert key in state, f"Missing key '{key}' in state"
