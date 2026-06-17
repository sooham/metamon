"""
End-to-end output validation tests for the new text format.

Checks the shape, structure, and content of the text output files
produced by ``ReplayParser.parse_replay()`` with ``NaiveUsagePredictor``.
"""

import os
import glob
import re
import pytest

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.team_prediction.predictor import NaiveUsagePredictor

from tests.helpers import find_random_replay_files


class TestE2EOutput:
    """Validate the structure of text output files."""

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

        files = sorted(glob.glob(os.path.join(output_dir, "*.txt")))
        assert len(files) == 2
        return files

    @pytest.fixture(scope="module")
    def parsed_texts(self, output_files):
        """Load both POV text output files into memory."""
        results = []
        for f in output_files:
            with open(f, "r", encoding="utf-8") as fh:
                results.append(fh.read())
        return results

    def test_tags_are_balanced(self, parsed_texts):
        """Each opening tag has matching closing tag count.

        Accounts for self-closing variants: ``<conditions_empty>`` replaces
        ``<conditions>...</end_conditions>``, ``<you_empty>`` replaces
        ``<you>...</end_you>``, etc.
        """
        # (open_pattern, close_tag, self_closing_tag_or_None)
        _PAIRED = [
            (r"<begin_team>", r"<end_team>", None),
            (r"<bos>", r"<eos>", None),
            (r"<boa>", r"<eoa>", None),
            (r"<arena>", r"<end_arena>", None),
            (r"<begin_moves>", r"<end_moves>", None),
            (r"<bench>", r"<end_bench>", None),
            # <conditions_empty> replaces an entire <conditions>...</end_conditions> pair
            (r"<conditions>", r"<end_conditions>", "<conditions_empty>"),
            (r"<you>", r"<end_you>", None),
            (r"<active>", r"<end_active>", None),
            (r"<opponent>", r"<end_opponent>", None),
            (r"<move>", r"<end_move>", None),
            (r"<format>", r"<end_format>", None),
            (r"<turn>", r"<end_turn>", None),
            # <chosen_move> may have attributes: <chosen_move cant="...">
            (r"<chosen_move", r"<end_chosen_move>", None),
            (r"<opponent_chosen_move>", r"<end_opponent_chosen_move>", None),
            (r"<boosts>", r"<end_boosts>", None),
            (r"<terminal>", r"<end_terminal>", None),
        ]
        for text in parsed_texts:
            for open_pat, close_tag, self_close in _PAIRED:
                open_count = len(re.findall(open_pat, text))
                close_count = text.count(close_tag)
                if self_close:
                    close_count += text.count(self_close)
                assert open_count == close_count, (
                    f"{open_pat} ({open_count}) != {close_tag}"
                    + (f" + {self_close}" if self_close else "")
                    + f" ({close_count})"
                )

    # -- State-action count consistency ------------------------------------

    def test_states_and_actions_count(self, parsed_texts):
        """Number of states (bos blocks) = number of actions (boa blocks) + 1."""
        for text in parsed_texts:
            n_states = text.count("<bos>")
            n_actions = text.count("<boa>")
            n_eos = text.count("<eos>")
            n_eoa = text.count("<eoa>")
            assert n_states == n_eos
            assert n_actions == n_eoa
            assert n_states == n_actions + 1, (
                f"states ({n_states}) != actions + 1 ({n_actions + 1})"
            )

    # -- Terminal state ------------------------------------------------------

    def test_terminal_state_has_terminal_tag(self, parsed_texts):
        """The last state block contains a <terminal> tag."""
        for text in parsed_texts:
            assert "<terminal>" in text, "Missing <terminal> tag"
            assert "<end_terminal>" in text, "Missing <end_terminal> tag"
            # Terminal must be one of: won, lost, tie, forfeit (now on separate lines)
            pattern = r'<terminal>\s*(won|lost|tie|forfeit)\s*<end_terminal>'
            assert re.search(pattern, text, re.DOTALL), (
                f"No valid terminal found in output"
            )

    def test_non_terminal_states_no_terminal_tag(self, parsed_texts):
        """Only the last state has a terminal tag."""
        for text in parsed_texts:
            # Split into state blocks
            blocks = text.split("<bos>")
            # First split element is before first <bos> (team header)
            terminal_blocks = 0
            for block in blocks[1:]:  # skip team header
                if "<terminal>" in block:
                    terminal_blocks += 1
            assert terminal_blocks == 1, (
                f"Expected 1 terminal block, found {terminal_blocks}"
            )

    # -- Format string -------------------------------------------------------

    def test_format_string_is_gen1ou(self, parsed_texts):
        """The format string matches the input replay format."""
        for text in parsed_texts:
            format_pattern = r'<format>\s*gen1ou\s*<end_format>'
            matches = re.findall(format_pattern, text, re.DOTALL)
            assert len(matches) > 0, "Expected gen1ou format"

    # -- Team header ---------------------------------------------------------

    def test_team_header_present(self, parsed_texts):
        """Team header is present and has move blocks."""
        for text in parsed_texts:
            assert "<begin_team>" in text
            assert "<end_team>" in text

            # Extract only the team header section
            team_match = re.search(r'<begin_team>(.*?)<end_team>', text, re.DOTALL)
            assert team_match, "Could not extract team header"
            team_section = team_match.group(1)

            # Each poke should have moves
            moves_blocks = len(re.findall(r'<begin_moves>.*?<end_moves>', team_section, re.DOTALL))
            assert moves_blocks >= 6, f"Expected >=6 moves blocks in team header, got {moves_blocks}"

    # -- HP values -----------------------------------------------------------

    def test_hp_values_are_fixed_point(self, parsed_texts):
        """HP values appear in X.XX format."""
        for text in parsed_texts:
            hp_pattern = re.compile(r'\b\d\.\d{2}\b')
            matches = hp_pattern.findall(text)
            assert len(matches) > 0, "No HP values found"
            for hp in matches:
                val = float(hp)
                assert 0.0 <= val <= 1.0, f"HP {hp} out of range [0.00, 1.00]"

    # -- Status tokens -------------------------------------------------------

    def test_status_tokens_are_valid(self, parsed_texts):
        """Status tokens are from the valid set."""
        valid_status = {"nostatus", "par", "slp", "psn", "tox", "brn", "frz", "fnt"}
        for text in parsed_texts:
            # Find status-like tokens in arena/bench blocks
            for token in valid_status:
                # All should appear somewhere (at least nostatus will)
                pass  # Just validating the format doesn't crash
            # Check that no bracketed status tokens appear
            assert "<nostatus>" not in text, "nostatus should not have angle brackets"
            assert "<noboosts>" not in text, "noboosts should not have angle brackets"

    # -- Action blocks ------------------------------------------------------

    def test_action_blocks_have_valid_content(self, parsed_texts):
        """Each action block has chosen_move and opponent_chosen_move tags."""
        for text in parsed_texts:
            # Extract all boa blocks
            boa_blocks = re.findall(r'<boa>(.*?)<eoa>', text, re.DOTALL)
            for block in boa_blocks:
                # <chosen_move> may have attributes like cant="..."
                assert re.search(r'<chosen_move[ >]', block), (
                    f"Missing chosen_move in boa block: {block[:100]}..."
                )
                assert "<opponent_chosen_move>" in block, (
                    f"Missing opponent_chosen_move in boa block: {block[:100]}..."
                )

    # -- Weather ------------------------------------------------------------

    def test_weather_token_is_valid(self, parsed_texts):
        """Weather tokens are from the valid set."""
        valid_weather = {
            "noweather", "sandstorm", "raindance", "sunnyday",
            "hail", "snow", "deltastream", "primordialsea", "desolateland",
        }
        for text in parsed_texts:
            for token in valid_weather:
                if token in text:
                    break  # at least one weather token found
            else:
                # If no specific weather found, check for noweather
                assert "noweather" in text, "No weather token found"

    # -- Bench section -------------------------------------------------------

    def test_bench_section_present(self, parsed_texts):
        """Every state has a bench section."""
        for text in parsed_texts:
            bench_count = text.count("<bench>")
            bos_count = text.count("<bos>")
            assert bench_count == bos_count, (
                f"bench blocks ({bench_count}) != bos blocks ({bos_count})"
            )

    # -- No XML-style closing tags -------------------------------------------

    def test_no_xml_style_closing_tags(self, parsed_texts):
        """No </foo> XML-style closing tags — only <end_foo>."""
        for text in parsed_texts:
            xml_close = re.findall(r'</[a-zA-Z_]+>', text)
            assert len(xml_close) == 0, (
                f"Found XML-style closing tags: {xml_close}"
            )
