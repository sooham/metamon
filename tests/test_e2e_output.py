"""
End-to-end output validation tests.

Checks the shape, types, and content of the final parsed output files
produced by ``ReplayParser.parse_replay()`` with ``NaiveUsagePredictor``.
"""

import os
import glob
import pytest
import orjson

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.team_prediction.predictor import NaiveUsagePredictor

from tests.helpers import find_random_replay_files


class TestE2EOutput:
    """Validate the structure of output JSON files."""

    @pytest.fixture(scope="module")
    def output_files(self, tmp_path_factory):
        """Parse one gen1ou replay and return paths to both POV output files."""
        paths = find_random_replay_files("gen1ou", 1)
        assert paths, "No gen1ou replays"

        output_dir = str(tmp_path_factory.mktemp("e2e_output"))
        parser = ReplayParser(
            replay_output_dir=output_dir,
            team_output_dir=None,
            verbose=False,
            team_predictor=NaiveUsagePredictor(),
            compress=False,
        )
        parser.parse_replay(paths[0])

        files = sorted(glob.glob(os.path.join(output_dir, "*.json")))
        assert len(files) == 2
        return files

    @pytest.fixture(scope="module")
    def parsed_output(self, output_files):
        """Load both POV output files into memory."""
        results = []
        for f in output_files:
            with open(f, "r") as fh:
                results.append(orjson.loads(fh.read()))
        return results

    # -- State shape ---------------------------------------------------------

    def test_states_and_actions_same_length(self, parsed_output):
        """states and actions arrays are the same length."""
        for data in parsed_output:
            assert len(data["states"]) == len(data["actions"])

    def test_states_is_list_of_dicts(self, parsed_output):
        """Every state is a dict."""
        for data in parsed_output:
            for s in data["states"]:
                assert isinstance(s, dict)

    def test_actions_is_list_of_ints(self, parsed_output):
        """Every action is an int."""
        for data in parsed_output:
            for a in data["actions"]:
                assert isinstance(a, int)

    # -- Action indices ------------------------------------------------------

    def test_action_indices_in_range(self, parsed_output):
        """Action indices are in [-1, 13]."""
        for data in parsed_output:
            for a in data["actions"]:
                assert -1 <= a <= 13, f"Action index {a} out of valid range"

    def test_last_action_is_minus_one(self, parsed_output):
        """The final action (terminal state) is always -1 (no action taken)."""
        for data in parsed_output:
            assert data["actions"][-1] == -1, (
                f"Last action should be -1, got {data['actions'][-1]}"
            )

    # -- State keys ----------------------------------------------------------

    def test_required_state_keys_present(self, parsed_output):
        """Every state dict contains all required keys."""
        required = {
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
        for data in parsed_output:
            for i, state in enumerate(data["states"]):
                missing = required - set(state.keys())
                assert not missing, f"State {i} missing keys: {missing}"

    # -- Terminal state ------------------------------------------------------

    def test_terminal_state_has_battle_end_flag(self, parsed_output):
        """The last state has battle_won=True or battle_lost=True."""
        for data in parsed_output:
            last = data["states"][-1]
            assert last["battle_won"] or last["battle_lost"], (
                "Final state does not have battle_won or battle_lost"
            )

    def test_non_terminal_states_are_not_ended(self, parsed_output):
        """Non-final states should have battle_won=False and battle_lost=False."""
        for data in parsed_output:
            for state in data["states"][:-1]:
                # Earlier states should not be terminal
                assert not state["battle_won"], "Non-terminal state has battle_won=True"
                assert not state["battle_lost"], "Non-terminal state has battle_lost=True"

    # -- Format string -------------------------------------------------------

    def test_format_string_is_gen1ou(self, parsed_output):
        """The format string matches the input replay format."""
        for data in parsed_output:
            for state in data["states"]:
                assert state["format"] == "gen1ou", (
                    f"Expected gen1ou, got {state['format']}"
                )

    # -- Weather & battle field ----------------------------------------------

    def test_weather_is_valid(self, parsed_output):
        """Weather is either a string, None, or an int (Nothing enum values)."""
        for data in parsed_output:
            for state in data["states"]:
                w = state["weather"]
                assert w is None or isinstance(w, (str, int)), (
                    f"Unexpected weather type: {type(w)}"
                )

    def test_battle_field_is_dict_or_nofield(self, parsed_output):
        """battle_field is a dict or the sentinel 'nofield' string."""
        for data in parsed_output:
            for state in data["states"]:
                bf = state["battle_field"]
                assert isinstance(bf, (dict, str)), (
                    f"battle_field has unexpected type: {type(bf)}"
                )

    # -- Opponent team preview -----------------------------------------------

    def test_opponent_teampreview_is_list(self, parsed_output):
        """opponent_teampreview is a list (may be empty)."""
        for data in parsed_output:
            for state in data["states"]:
                assert isinstance(state["opponent_teampreview"], list)
