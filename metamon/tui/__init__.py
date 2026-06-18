"""Model‑agnostic TUI for Pokémon Showdown players.

Provides :class:`TuiMixin` — a mixin class that adds an interactive
full‑screen REPL on top of any MetamonPlayer subclass.  Import
:class:`BattleHistory` from here so model‑specific player code can
store turn data that the TUI will render.
"""

from metamon.tui.theme import (
    TURN_BG_COLORS,
    colorize_raw_line,
    colorize_state_line,
    turn_background,
)
from metamon.tui.player import TuiMixin, BattleHistory, TurnContext, TuiDiagnostic

__all__ = [
    "TuiMixin",
    "BattleHistory",
    "TurnContext",
    "TuiDiagnostic",
    "TURN_BG_COLORS",
    "colorize_raw_line",
    "colorize_state_line",
    "turn_background",
]
