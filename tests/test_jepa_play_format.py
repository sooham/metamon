from pathlib import Path
from types import SimpleNamespace
import os

import orjson

from metamon.jepa.online_replay_saver import (
    format_from_battle_tag,
    game_id_from_battle_tag,
    save_online_replay_as_parsed,
)
from metamon.jepa.play import (
    _active_battle_count,
    random_battle_format_for,
    uses_random_battle_team,
)
from metamon.tui import BattleHistory, TuiMixin


def test_random_battle_format_matches_requested_generation():
    assert random_battle_format_for("gen1ou") == "gen1randombattle"
    assert random_battle_format_for("gen9ou") == "gen9randombattle"
    assert random_battle_format_for("gen4ubers") == "gen4randombattle"


def test_random_battle_formats_do_not_use_fixed_teams():
    assert uses_random_battle_team("gen1randombattle")
    assert uses_random_battle_team("gen9randombattle")
    assert not uses_random_battle_team("gen1ou")


def test_active_battle_count_ignores_finished_battles():
    player = SimpleNamespace(_battles={
        "a": SimpleNamespace(finished=False),
        "b": SimpleNamespace(finished=True),
        "c": SimpleNamespace(finished=False),
    })

    assert _active_battle_count(player) == 2


def test_online_replay_ids_use_actual_battle_format():
    tag = "battle-gen9randombattle-12345"

    assert format_from_battle_tag(tag, "gen9ou") == "gen9randombattle"
    assert game_id_from_battle_tag(tag, "gen9ou") == "gen9randombattle-12345"


def test_online_replay_save_routes_to_format_directory(tmp_path):
    raw_messages = [
        "|player|p1|JEPABot|",
        "|player|p2|Opponent|",
        "|win|JEPABot",
    ]

    class FakeReplayParser:
        payload = None
        raw_path_name = None

        def __init__(self, replay_output_dir, **_kwargs):
            self.replay_output_dir = Path(replay_output_dir)

        def parse_replay(self, path):
            raw_path = Path(path)
            FakeReplayParser.raw_path_name = raw_path.name
            FakeReplayParser.payload = orjson.loads(raw_path.read_bytes())
            out = self.replay_output_dir / (
                "gen9randombattle-12345_Unrated_JEPABot_vs_Opponent_"
                "07-07-2026_WIN.txt"
            )
            out.write_text("<begin_team>\n<end_team>\n", encoding="utf-8")

    result = save_online_replay_as_parsed(
        raw_messages=raw_messages,
        battle_tag="battle-gen9randombattle-12345",
        fallback_format="gen9ou",
        output_root=tmp_path,
        upload_time=1783382400,
        replay_parser_cls=FakeReplayParser,
    )

    assert result.format_name == "gen9randombattle"
    assert result.output_dir == tmp_path / "gen9randombattle"
    assert FakeReplayParser.raw_path_name == "gen9randombattle-12345.json"
    assert FakeReplayParser.payload["players"] == ["JEPABot", "Opponent"]
    assert FakeReplayParser.payload["formatid"] == "gen9randombattle"
    assert FakeReplayParser.payload["uploadtime"] == 1783382400
    assert FakeReplayParser.payload["log"] == "\n".join(raw_messages)
    assert [path.name for path in result.saved_files] == [
        "gen9randombattle-12345_Unrated_JEPABot_vs_Opponent_07-07-2026_WIN.txt"
    ]


def test_tui_raw_replay_save_routes_by_battle_format(tmp_path):
    TuiMixin._repl_save_raw_dir = str(tmp_path)
    TuiMixin._repl_save_raw_by_format = True
    try:
        tui = TuiMixin()
        tui.username = "JEPABot"
        hist = BattleHistory(
            raw_messages=[
                "|player|p1|JEPABot|",
                "|player|p2|Opponent|",
                "|win|JEPABot",
            ],
            finished=True,
            outcome="win",
        )

        tui._tui_save_raw_replay("battle-gen9randombattle-12345", hist)

        saved = list((tmp_path / "gen9" / "randombattle").glob("*.txt"))
        assert len(saved) == 1
        assert saved[0].name.startswith("gen9randombattle-12345_JEPABot_vs_Opponent_")
    finally:
        TuiMixin._repl_save_raw_dir = None
        TuiMixin._repl_save_raw_by_format = False


def test_tui_overview_reports_ongoing_battle_count(capsys):
    histories = {
        "battle-gen1randombattle-1": BattleHistory(finished=False),
        "battle-gen1randombattle-2": BattleHistory(finished=False),
        "battle-gen1randombattle-3": BattleHistory(finished=True, outcome="win"),
    }
    inst = TuiMixin()
    inst._histories = histories
    inst._battles = {}

    old_instances = TuiMixin._repl_all_instances
    TuiMixin._repl_all_instances = [inst]
    try:
        inst._tui_render_overview()
    finally:
        TuiMixin._repl_all_instances = old_instances

    output = capsys.readouterr().out
    assert "ongoing battles: 2  finished: 1  total: 3" in output


def test_tui_q_key_stops_repl_from_listener(monkeypatch):
    import metamon.tui.player as tui_player

    class FakeTerm:
        def cbreak(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def exit_fullscreen(self):
            return ""

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"q")
    os.close(write_fd)

    old_running = TuiMixin._repl_running
    old_lock = TuiMixin._repl_lock
    old_term = TuiMixin._repl_term
    TuiMixin._repl_running = True
    TuiMixin._repl_lock = None
    TuiMixin._repl_term = FakeTerm()
    monkeypatch.setattr(tui_player._sys, "stdin", SimpleNamespace(fileno=lambda: read_fd))
    try:
        TuiMixin._key_listener()
    finally:
        os.close(read_fd)
        TuiMixin._repl_running = old_running
        TuiMixin._repl_lock = old_lock
        TuiMixin._repl_term = old_term

    assert TuiMixin._repl_running is False
