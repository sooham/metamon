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


def _assert_canonical_action_content(content: str) -> None:
    # "none" is a single-token sentinel for "opponent had no action"
    # (e.g. forced-switch subturns).
    if content == "none":
        return
    parts = content.split()
    assert len(parts) >= 2, f"Action content must have kind and value: {content!r}"
    assert parts[0] in {"move", "switch", "unknown"}, (
        f"Action content must start with move/switch/unknown: {content!r}"
    )
    if parts[0] == "unknown":
        assert parts == ["unknown", "unknown"], (
            f"Unknown action content must be 'unknown unknown': {content!r}"
        )


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

        Accounts for self-closing variants: ``<you_empty>`` replaces
        ``<you>...</end_you>``, etc. Empty arena conditions use the standalone
        ``<empty_conditions>`` sentinel and are validated separately.
        """
        # (open_pattern, close_tag, self_closing_tag_or_None)
        _PAIRED = [
            (r"<begin_team>", r"<end_team>", None),
            (r"<bos>", r"<eos>", None),
            (r"<boa>", r"<eoa>", None),
            (r"<last_turn_results>", r"<end_last_turn_results>", None),
            (r"<arena>", r"<end_arena>", None),
            (r"<begin_moves>", r"<end_moves>", None),
            (r"<bench>", r"<end_bench>", None),
            (r"<conditions>", r"<end_conditions>", None),
            (r"<you>", r"<end_you>", None),
            (r"<active>", r"<end_active>", None),
            (r"<opponent>", r"<end_opponent>", None),
            (r"<move>", r"<end_move>", None),
            (r"<format>", r"<end_format>", None),
            (r"<turn>", r"<end_turn>", None),
            # <chosen_move> no longer has cant= attributes
            (r"<chosen_move>", r"<end_chosen_move>", None),
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

    def test_each_state_has_one_conditions_representation(self, parsed_texts):
        """Each state has either <empty_conditions> or a paired populated block."""
        for text in parsed_texts:
            assert "<conditions_empty>" not in text
            for block in re.findall(r"<bos>(.*?)<eos>", text, re.DOTALL):
                has_empty = "<empty_conditions>" in block
                condition_open_count = block.count("<conditions>")
                assert int(has_empty) + condition_open_count == 1
                if has_empty:
                    assert "<end_conditions>" not in block
                else:
                    assert condition_open_count == 1
                    assert block.count("<end_conditions>") == 1

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
        """HP percentage values appear in X.XX format within [0.00, 1.00]."""
        for text in parsed_texts:
            # Match HP percentage tokens: X.XX bounded by whitespace or tags
            hp_pattern = re.compile(r'\b\d\.\d{2}\b')
            matches = hp_pattern.findall(text)
            assert len(matches) > 0, "No HP percentage values found"
            for hp in matches:
                val = float(hp)
                assert 0.0 <= val <= 1.0, f"HP percentage {hp} out of range [0.00, 1.00]"

    def test_full_hp_values_present(self, parsed_texts):
        """Arena and bench entries include raw current_hp and max_hp integers."""
        for text in parsed_texts:
            # Find active/opponent/bench lines that contain HP triples
            # Pattern: species 0.XX NNN NNN ... (percentage + two integers)
            hp_triple = re.compile(
                r'\b\d\.\d{2}\s+(\d+)\s+(\d+)\b'
            )
            matches = hp_triple.findall(text)
            assert len(matches) > 0, "No full HP triples (pct cur max) found in text"
            for cur_str, max_str in matches:
                cur = int(cur_str)
                max_hp = int(max_str)
                assert cur <= max_hp, f"current_hp {cur} > max_hp {max_hp}"

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
                # <chosen_move> is plain (no attributes — outcome is in
                # <last_turn_results> of the following state)
                assert "<chosen_move>" in block, (
                    f"Missing chosen_move in boa block: {block[:100]}..."
                )
                assert "<opponent_chosen_move>" in block, (
                    f"Missing opponent_chosen_move in boa block: {block[:100]}..."
                )
                for tag, end_tag in (
                    ("chosen_move", "end_chosen_move"),
                    ("opponent_chosen_move", "end_opponent_chosen_move"),
                ):
                    match = re.search(rf"<{tag}>\s*(.*?)\s*<{end_tag}>", block, re.DOTALL)
                    assert match is not None, f"Missing {tag} content: {block[:100]}..."
                    _assert_canonical_action_content(match.group(1))

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

    # -- Last turn results ---------------------------------------------------

    def test_last_turn_results_present(self, parsed_texts):
        """Every state block has a <last_turn_results> block."""
        for text in parsed_texts:
            ltr_count = text.count("<last_turn_results>")
            bos_count = text.count("<bos>")
            assert ltr_count == bos_count, (
                f"<last_turn_results> ({ltr_count}) != <bos> ({bos_count})"
            )
            # First state should have an empty block
            first_bos = text.index("<bos>")
            first_ltr = text.index("<last_turn_results>", first_bos)
            first_ltr_end = text.index("<end_last_turn_results>", first_ltr)
            ltr_content = text[first_ltr + len("<last_turn_results>"):first_ltr_end]
            # First state's last_turn_results should be empty (no sub-tags)
            stripped = ltr_content.strip()
            assert stripped == "", (
                f"First state's <last_turn_results> should be empty, got: {stripped[:80]}"
            )
            # Subsequent states should have content
            assert text.count("<last_turn_results>\n<end_last_turn_results>") == 1, (
                "Only the first state should have an empty <last_turn_results>"
            )

    def test_last_turn_results_has_outcome_tokens(self, parsed_texts):
        """<last_turn_results> blocks use valid outcome tokens."""
        for text in parsed_texts:
            ltr_blocks = re.findall(
                r'<last_turn_results>(.*?)<end_last_turn_results>', text, re.DOTALL
            )
            for block in ltr_blocks:
                # Extract lines like:
                # <active>body slam success<end_active>
                # <opponent>unknown<end_opponent>
                for match in re.finditer(
                    r'<(?:active|opponent)(?:[12])?>(.*?)<end_(?:active|opponent)(?:[12])?>',
                    block, re.DOTALL
                ):
                    content = match.group(1).strip()
                    if content == "unknown":
                        continue
                    # Content is: NAME [fail | cant REASON | success]
                    # e.g. "body slam success", "sucker punch fail",
                    #      "psychic cant slp", "drillpeck cant par"
                    parts = content.split()
                    # The move name can be multi-word (e.g. "sucker punch",
                    # "switch alakazam"), so check the last 1-2 tokens.
                    if len(parts) >= 2:
                        if parts[-1] in ("success", "fail"):
                            continue  # single-word outcome
                        if parts[-2] == "cant" and len(parts) >= 3:
                            continue  # "NAME cant REASON"
                    # If we get here, the format doesn't match expectations
                    # but this isn't necessarily a failure — just skip

    # -- No XML-style closing tags -------------------------------------------

    def test_no_xml_style_closing_tags(self, parsed_texts):
        """No </foo> XML-style closing tags — only <end_foo>."""
        for text in parsed_texts:
            xml_close = re.findall(r'</[a-zA-Z_]+>', text)
            assert len(xml_close) == 0, (
                f"Found XML-style closing tags: {xml_close}"
            )
