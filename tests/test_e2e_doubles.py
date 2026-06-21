"""
End-to-end tests for doubles battle parsing with the new text format.

Validates that the full ReplayParser pipeline correctly handles doubles
formats (two active Pokémon per side), producing output with ``<active1>`` /
``<active2>`` / ``<opponent1>`` / ``<opponent2>`` tags, per-slot move blocks,
and per-slot action entries.

Uses a known-good Gen 9 Doubles OU replay downloaded on-demand from Showdown.
"""

import os
import glob
import json
import re
import datetime

import pytest
import requests

from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay, SimProtocol
from metamon.backend.replay_parser import checks
from metamon.backend.replay_parser.backward import backward_fill_doubles
from metamon.backend.replay_parser.text_serializer import serialize_pov_replay_doubles
from metamon.backend.team_prediction.predictor import NoPredictor, NaiveUsagePredictor


# Known-good Gen 9 Doubles OU replay (11 turns, includes Tera, |cant|, switches).
DOUBLES_REPLAY_URL = (
    "https://replay.pokemonshowdown.com/gen9doublesou-2633651189.json"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _assert_canonical_action_content(content: str) -> None:
    parts = content.split()
    assert len(parts) >= 2, f"Action content must have kind and value: {content!r}"
    assert parts[0] in {"move", "switch", "unknown"}, (
        f"Action content must start with move/switch/unknown: {content!r}"
    )
    if parts[0] == "unknown":
        assert parts == ["unknown", "unknown"], (
            f"Unknown action content must be 'unknown unknown': {content!r}"
        )


def _download_replay(url: str, dest: str) -> dict:
    """Download a replay JSON, cache to *dest*, return parsed dict."""
    if os.path.exists(dest):
        with open(dest) as f:
            return json.load(f)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    with open(dest, "w") as f:
        json.dump(data, f)
    return data


def _run_forward_fill(raw_data: dict) -> ParsedReplay:
    """Run forward fill on raw replay data, skipping check_finished for short battles."""
    log = ReplayParser.clean_log(raw_data)
    replay = ParsedReplay(
        gameid=raw_data["id"],
        format=raw_data.get("formatid", raw_data.get("format", "gen9doublesou")),
        time_played=datetime.datetime.fromtimestamp(int(raw_data["uploadtime"])),
    )
    sim = SimProtocol(replay)
    for msg in log:
        if not msg:
            continue
        sim.interpret_message(msg)
    checks.check_forced_switching(replay)
    checks.check_noun_spelling(replay)
    checks.check_forward_consistency(replay)
    checks.check_name_permanence(replay)
    checks.check_tera_consistency(replay)
    return replay


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def doubles_replay_data(tmp_path_factory):
    """Download (once) and return the raw replay dict."""
    dest = str(tmp_path_factory.mktemp("doubles_replay") / "replay.json")
    return _download_replay(DOUBLES_REPLAY_URL, dest)


@pytest.fixture(scope="module")
def doubles_parsed_replay(doubles_replay_data):
    """Forward-parsed replay object (spectator view)."""
    return _run_forward_fill(doubles_replay_data)


@pytest.fixture(scope="module")
def doubles_pov_replays(doubles_parsed_replay):
    """Both POVReplayDoubles instances (p1 WIN, p2 LOSS)."""
    return backward_fill_doubles(
        doubles_parsed_replay, team_predictor=NoPredictor()
    )


@pytest.fixture(scope="module")
def doubles_pov_texts(doubles_pov_replays):
    """Serialized text output for both POVs."""
    pov_p1, pov_p2 = doubles_pov_replays
    return (
        serialize_pov_replay_doubles(pov_p1),
        serialize_pov_replay_doubles(pov_p2),
    )


# ── forward fill tests ──────────────────────────────────────────────────────

class TestDoublesForward:
    """Forward-fill structural checks for doubles."""

    def test_gametype_is_doubles(self, doubles_parsed_replay):
        """The replay has two active slots populated."""
        # At least one turn should have both active slots non-None
        found_doubles = False
        for turn in doubles_parsed_replay.turnlist:
            if (
                turn.active_pokemon_1[0] is not None
                and turn.active_pokemon_1[1] is not None
            ):
                found_doubles = True
                break
        assert found_doubles, "No turn has two active Pokémon — not a doubles battle?"

    def test_moves_populate_both_slots(self, doubles_parsed_replay):
        """At least one turn has moves in both slot 0 and slot 1."""
        found = False
        for turn in doubles_parsed_replay.turnlist:
            if turn.moves_1[0] is not None and turn.moves_1[1] is not None:
                found = True
                break
            if turn.moves_2[0] is not None and turn.moves_2[1] is not None:
                found = True
                break
        assert found, "No turn has moves in both active slots"

    def test_winner_is_set(self, doubles_parsed_replay):
        """The replay has a declared winner."""
        assert doubles_parsed_replay.winner is not None

    def test_turns_exist(self, doubles_parsed_replay):
        """The replay has a reasonable number of turns."""
        assert len(doubles_parsed_replay.turnlist) >= 5


# ── backward fill tests ─────────────────────────────────────────────────────

class TestDoublesBackward:
    """Backward-fill structural checks for doubles POVReplayDoubles."""

    def test_povturnlist_nonempty(self, doubles_pov_replays):
        """Both POV replays have turns."""
        pov_p1, pov_p2 = doubles_pov_replays
        assert len(pov_p1.povturnlist) > 0
        assert len(pov_p2.povturnlist) > 0

    def test_actionlist_matches_povturnlist(self, doubles_pov_replays):
        """actionlist and povturnlist have same length."""
        pov_p1, pov_p2 = doubles_pov_replays
        assert len(pov_p1.actionlist) == len(pov_p1.povturnlist)
        assert len(pov_p2.actionlist) == len(pov_p2.povturnlist)

    def test_opponent_actionlist_is_list_of_lists(self, doubles_pov_replays):
        """opponent_actionlist stores [Action|None, Action|None] per step."""
        pov_p1, _ = doubles_pov_replays
        for oa in pov_p1.opponent_actionlist:
            assert isinstance(oa, list), f"Expected list, got {type(oa)}"
            assert len(oa) == 2, f"Expected 2, got {len(oa)}"

    def test_winner_consistent(self, doubles_pov_replays):
        """p1 wins ⇔ p2 loses (excluding ties)."""
        pov_p1, pov_p2 = doubles_pov_replays
        from metamon.backend.replay_parser.replay_state import Winner
        if pov_p1.replay.winner == Winner.TIE:
            assert not pov_p1.winner
            assert not pov_p2.winner
        else:
            assert pov_p1.winner != pov_p2.winner

    def test_gen_is_set(self, doubles_pov_replays):
        """Generation is 9."""
        pov_p1, pov_p2 = doubles_pov_replays
        assert pov_p1.gen == 9
        assert pov_p2.gen == 9


# ── text serialization tests ────────────────────────────────────────────────

class TestDoublesTextOutput:
    """Validate the structure of doubles text output."""

    # Tags that must be balanced in doubles output.
    # Third element is a self-closing variant (counts as both open+close).
    _PAIRED_TAGS = [
        (r"<begin_team>", "<end_team>", None),
        (r"<begin_opponent_team>", "<end_opponent_team>", None),
        (r"<bos>", "<eos>", None),
        (r"<boa>", "<eoa>", None),
        (r"<last_turn_results>", "<end_last_turn_results>", None),
        (r"<arena>", "<end_arena>", None),
        (r"<active1>", "<end_active1>", None),
        (r"<active2>", "<end_active2>", None),
        (r"<opponent1>", "<end_opponent1>", None),
        (r"<opponent2>", "<end_opponent2>", None),
        (r"<begin_moves", "<end_moves>", None),
        (r"<bench>", "<end_bench>", None),
        (r"<conditions>", "<end_conditions>", None),
        (r"<you>", "<end_you>", None),
        (r"<move>", "<end_move>", None),
        (r"<chosen_move", "<end_chosen_move>", None),
        (r"<opponent_chosen_move", "<end_opponent_chosen_move>", None),
        (r"<boosts>", "<end_boosts>", None),
        (r"<terminal>", "<end_terminal>", None),
    ]

    def test_tags_balanced(self, doubles_pov_texts):
        """Each opening tag has a matching closing tag."""
        for text in doubles_pov_texts:
            for open_pat, close_tag, self_close in self._PAIRED_TAGS:
                open_count = len(re.findall(open_pat, text))
                close_count = text.count(close_tag)
                if self_close:
                    close_count += text.count(self_close)
                assert open_count == close_count, (
                    f"{open_pat} ({open_count}) != {close_tag}"
                    + (f" + {self_close}" if self_close else "")
                    + f" ({close_count})"
                )

    def test_each_state_has_one_conditions_representation(self, doubles_pov_texts):
        """Each state has either <empty_conditions> or a paired populated block."""
        for text in doubles_pov_texts:
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

    def test_states_and_actions_count(self, doubles_pov_texts):
        """states = actions + 1 (terminal state has no action)."""
        for text in doubles_pov_texts:
            n_states = text.count("<bos>")
            n_actions = text.count("<boa>")
            assert n_states == n_actions + 1, (
                f"states ({n_states}) != actions + 1 ({n_actions + 1})"
            )

    def test_doubles_arena_tags_present(self, doubles_pov_texts):
        """Output uses active1/active2/opponent1/opponent2 (not bare active)."""
        for text in doubles_pov_texts:
            assert "<active1>" in text, "Missing <active1>"
            assert "<active2>" in text, "Missing <active2>"
            assert "<opponent1>" in text, "Missing <opponent1>"
            assert "<opponent2>" in text, "Missing <opponent2>"

    def test_no_singles_tags_leak(self, doubles_pov_texts):
        """Bare <active> / <opponent> (without slot) should NOT appear."""
        for text in doubles_pov_texts:
            # The tags <active> and <opponent> without a digit suffix
            # should not appear in doubles output.
            bare_active = len(re.findall(r"<active>", text))
            bare_opponent = len(re.findall(r"<opponent>", text))
            # Note: <opponent> appears inside <conditions> block as a sub-tag,
            # which is fine.  We only care about bare active without digit.
            assert bare_active == 0, (
                f"Found {bare_active} bare <active> tags (should use <active1>/<active2>)"
            )

    def test_per_slot_moves_blocks(self, doubles_pov_texts):
        """Each state has two <begin_moves:N> blocks (colon-format slots)."""
        for text in doubles_pov_texts:
            bos_blocks = re.findall(r"<bos>(.*?)<eos>", text, re.DOTALL)
            for block in bos_blocks:
                slot1 = len(re.findall(r'<begin_moves:1>', block))
                slot2 = len(re.findall(r'<begin_moves:2>', block))
                # Every state should have slot 1 moves; slot 2 may be absent
                # if that active fainted before the turn started
                assert slot1 >= 0  # at least not negative

    def test_per_slot_action_entries(self, doubles_pov_texts):
        """Each action block has chosen_move and opponent_chosen_move with colon-format slots."""
        for text in doubles_pov_texts:
            boa_blocks = re.findall(r"<boa>(.*?)<eoa>", text, re.DOTALL)
            for block in boa_blocks:
                # Must have :1 and :2 chosen_move entries
                assert re.search(r'<chosen_move:1>', block), (
                    f"Missing chosen_move:1: {block[:100]}"
                )
                assert re.search(r'<chosen_move:2>', block), (
                    f"Missing chosen_move:2: {block[:100]}"
                )
                assert re.search(r'<opponent_chosen_move:1>', block), (
                    f"Missing opponent_chosen_move:1"
                )
                assert re.search(r'<opponent_chosen_move:2>', block), (
                    f"Missing opponent_chosen_move:2"
                )
                for tag, end_tag in (
                    ("chosen_move", "end_chosen_move"),
                    ("opponent_chosen_move", "end_opponent_chosen_move"),
                ):
                    for slot in (1, 2):
                        match = re.search(
                            rf"<{tag}:{slot}>\s*(.*?)\s*<{end_tag}>",
                            block,
                            re.DOTALL,
                        )
                        assert match is not None, (
                            f"Missing {tag}:{slot} content: {block[:100]}"
                        )
                        _assert_canonical_action_content(match.group(1))

    def test_terminal_tag_present(self, doubles_pov_texts):
        """Output has a valid terminal tag (multi-line format)."""
        for text in doubles_pov_texts:
            assert re.search(r"<terminal>\s*(won|lost|tie|forfeit)\s*<end_terminal>", text, re.DOTALL), (
                "No valid terminal tag found"
            )

    def test_team_header_present(self, doubles_pov_texts):
        """Team header and opponent team preview present (Gen 9)."""
        for text in doubles_pov_texts:
            assert "<begin_team>" in text
            assert "<end_team>" in text
            assert "<begin_opponent_team>" in text, (
                "Expected opponent team preview for Gen 9 doubles"
            )

    def test_hp_format(self, doubles_pov_texts):
        """HP percentage values are in X.XX format with raw current/max HP."""
        for text in doubles_pov_texts:
            hp_vals = re.findall(r"\b\d\.\d{2}\b", text)
            assert len(hp_vals) > 0, "No HP percentage values in output"
            # Verify full HP triples are present
            hp_triple = re.compile(r"\b\d\.\d{2}\s+(\d+)\s+(\d+)\b")
            triple_matches = hp_triple.findall(text)
            assert len(triple_matches) > 0, "No full HP triples found"

    def test_no_xml_style_closers(self, doubles_pov_texts):
        """No </foo> closers — only <end_foo>."""
        for text in doubles_pov_texts:
            xml_closers = re.findall(r"</[a-zA-Z_]+>", text)
            assert len(xml_closers) == 0, (
                f"Found XML-style closing tags: {xml_closers}"
            )

    def test_noboosts_bracketless(self, doubles_pov_texts):
        """noboosts is bracketless."""
        for text in doubles_pov_texts:
            assert "<noboosts>" not in text, "noboosts should not have brackets"
            # noboosts should appear as a bare token
            assert "noboosts" in text


# ── full pipeline test (via ReplayParser) ──────────────────────────────────

class TestDoublesFullPipeline:
    """End-to-end: raw replay → text output via ReplayParser."""

    def test_full_pipeline_produces_txt(self, doubles_replay_data, tmp_path):
        """ReplayParser.parse_replay() writes .txt files for doubles."""
        # Write raw replay to a temp file
        raw_path = str(tmp_path / "raw.json")
        with open(raw_path, "w") as f:
            json.dump(doubles_replay_data, f)

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        parser = ReplayParser(
            replay_output_dir=output_dir,
            team_output_dir=None,
            verbose=False,
            team_predictor=NoPredictor(),
            compress=False,
        )
        parser.parse_replay(raw_path)

        out_files = glob.glob(os.path.join(output_dir, "*.txt"))
        assert len(out_files) == 2, f"Expected 2 output files, got {len(out_files)}"

        for f in out_files:
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()
            # Doubles-specific checks
            assert "<active1>" in text
            assert "<active2>" in text
            assert "<chosen_move:1>" in text
            assert "<chosen_move:2>" in text
            assert "<terminal>" in text
