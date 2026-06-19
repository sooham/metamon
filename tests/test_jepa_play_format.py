from pathlib import Path

from metamon.jepa.play import random_battle_format_for
from metamon.tui import BattleHistory, TuiMixin


def test_random_battle_format_matches_requested_generation():
    assert random_battle_format_for("gen1ou") == "gen1randombattle"
    assert random_battle_format_for("gen9ou") == "gen9randombattle"
    assert random_battle_format_for("gen4ubers") == "gen4randombattle"


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
