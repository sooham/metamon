"""
End-to-end smoke tests using the full ReplayParser pipeline.

These tests run ``ReplayParser.parse_replay()`` with ``NaiveUsagePredictor`` on
real raw replay files and verify that text output files are produced correctly
in the new stateful text format.
"""

import os
import glob
import pytest

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.exceptions import ForwardException, BackwardException
from metamon.backend.team_prediction.predictor import NaiveUsagePredictor

from tests.helpers import find_random_replay_files, SUPPORTED_GENS


class TestE2ESmoke:
    """Full pipeline tests on real replays."""

    @pytest.mark.parametrize("fmt", list(SUPPORTED_GENS.keys()))
    def test_full_pipeline_on_3_replays(self, fmt, tmp_path):
        """Run ReplayParser.parse_replay() on up to 3 replays per format.

        Verifies that text output files are created and contain valid structure.
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

        # Check output files exist (.txt extension for new format)
        out_files = glob.glob(os.path.join(output_dir, "*.txt"))
        assert len(out_files) >= ok * 2, (
            f"Expected >= {ok * 2} output files (2 POV per replay), found {len(out_files)}"
        )

        # Verify each output file has the expected structure
        for f in out_files:
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()

            assert "<begin_team>" in text, f"Missing <begin_team> in {f}"
            assert "<end_team>" in text, f"Missing <end_team> in {f}"
            assert "<bos>" in text, f"Missing <bos> in {f}"
            assert "<eos>" in text, f"Missing <eos> in {f}"
            assert "<boa>" in text, f"Missing <boa> in {f}"
            assert "<eoa>" in text, f"Missing <eoa> in {f}"
            assert "<last_turn_results>" in text, f"Missing <last_turn_results> in {f}"
            assert "<terminal>" in text, f"Missing <terminal> in {f}"

            # Count state-action pairs: bos count should equal eos count,
            # and boa count should equal eoa count
            bos_count = text.count("<bos>")
            eos_count = text.count("<eos>")
            boa_count = text.count("<boa>")
            eoa_count = text.count("<eoa>")
            assert bos_count == eos_count, f"<bos> ({bos_count}) != <eos> ({eos_count})"
            assert boa_count == eoa_count, f"<boa> ({boa_count}) != <eoa> ({eoa_count})"
            # Actions = states - 1 (no action after terminal state)
            assert bos_count == boa_count + 1, (
                f"states ({bos_count}) != actions + 1 ({boa_count + 1})"
            )
            assert bos_count > 0, f"Empty states in {f}"

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

        out_files = glob.glob(os.path.join(output_dir, "*.txt"))
        assert len(out_files) == 2, (
            f"Expected 2 POV files, got {len(out_files)}"
        )

        for f in out_files:
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()

            # Must start with team header
            assert text.startswith("<begin_team>"), f"File {f} doesn't start with team header"

            # Must contain format tag
            assert "<format>" in text, f"Missing <format> in {f}"

            # Must contain arena with active and opponent
            assert "<active>" in text, f"Missing <active> in {f}"
            assert "<opponent>" in text, f"Missing <opponent> in {f}"

            # Must have bench section
            assert "<bench>" in text, f"Missing <bench> in {f}"

            # Must have a conditions representation
            assert "<conditions>" in text or "<empty_conditions>" in text, (
                f"Missing conditions representation in {f}"
            )
            assert "<conditions_empty>" not in text, f"Legacy <conditions_empty> in {f}"

            # Must have last_turn_results in every state
            ltr_count = text.count("<last_turn_results>")
            eos_count = text.count("<eos>")
            assert ltr_count == eos_count, f"<last_turn_results>=({ltr_count}) != <eos>=({eos_count})"

            # Must have a terminal marker
            assert "<terminal>" in text, f"Missing <terminal> in {f}"
            assert "<end_terminal>" in text, f"Missing <end_terminal> in {f}"

            # Validate terminal content
            import re
            terminal_options = ["won", "lost", "tie", "forfeit"]
            found_terminal = False
            for opt in terminal_options:
                # Terminal is now on separate lines: <terminal>\nwon\n<end_terminal>
                if re.search(rf'<terminal>\s*{opt}\s*<end_terminal>', text, re.DOTALL):
                    found_terminal = True
                    break
            assert found_terminal, f"No valid terminal found in {f}"

            # HP values should be in fixed-point format (X.XX)
            import re
            hp_pattern = re.compile(r'\b\d\.\d{2}\b')
            hp_matches = hp_pattern.findall(text)
            assert len(hp_matches) > 0, f"No HP values found in {f}"

            # Check that noboosts token is bracketless
            assert "noboosts" in text, f"Missing noboosts in {f}"
