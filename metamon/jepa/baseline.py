"""Registered JEPA baseline for the compete/head2head framework.

Usage:
    from metamon.jepa.baseline import register_jepa_baseline
    from metamon.jepa.model import PairedJEPAModel

    model = PairedJEPAModel(...).eval()
    register_jepa_baseline(model, tokenizer, fmt="gen1ou", heuristic="max-rank")

    # Now JEPA is in ALL_BASELINES and can be used like any baseline:
    from metamon.baselines import get_baseline
    jepa_cls = get_baseline("JEPABaseline")
    player = jepa_cls(battle_format="gen1ou", team=..., ...)
"""

from __future__ import annotations

from typing import Optional
import torch

from poke_env.player import BattleOrder
from poke_env.environment import AbstractBattle

from metamon.baselines import register_baseline
from metamon.baselines.base import Baseline
from metamon.jepa.model import PairedJEPAModel
from metamon.jepa.player import JEPAWorldModelPlayer
from metamon.tokenizer import PokemonTokenizer


# Module-level shared model (set once, used by all instances).
_shared_model: Optional[PairedJEPAModel] = None
_shared_tokenizer: Optional[PokemonTokenizer] = None
_shared_fmt: str = "gen1ou"
_shared_heuristic: str = "max-rank"
_registered: bool = False


def register_jepa_baseline(
    model: PairedJEPAModel,
    tokenizer: PokemonTokenizer,
    fmt: str = "gen1ou",
    heuristic: str = "max-rank",
) -> None:
    """Register JEPA as a baseline so it can be used in head2head / compete."""
    global _shared_model, _shared_tokenizer, _shared_fmt, _shared_heuristic, _registered

    _shared_model = model
    _shared_tokenizer = tokenizer
    _shared_fmt = fmt
    _shared_heuristic = heuristic

    if not _registered:
        # Register for all gens/formats — the JEPA model is format-specific
        # but we register broadly so it shows up in BASELINES_BY_GEN.
        register_baseline()(JEPABaseline)
        _registered = True


class JEPABaseline(Baseline):
    """Baseline wrapper around JEPAWorldModelPlayer.

    Shares the module-level model and tokenizer.  The TUI is disabled
    so no fullscreen takeover happens during competition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _shared_model is None or _shared_tokenizer is None:
            raise RuntimeError(
                "JEPA model not loaded. Call register_jepa_baseline(model, tokenizer) first."
            )
        # The inner JEPAWorldModelPlayer is logic-only; it must NOT
        # receive server/auth kwargs (account_configuration,
        # server_configuration) or it will double-connect.
        inner_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("account_configuration", "server_configuration",
                         "start_timer_on_battle_start")
        }
        self._jepa_player = JEPAWorldModelPlayer(
            *args,
            model=_shared_model,
            tokenizer=_shared_tokenizer,
            fmt=_shared_fmt,
            heuristic=_shared_heuristic,
            verbose=False,
            verbose_blocks=False,
            **inner_kwargs,
        )
        # Share the same histories dict so the TUI (if active) can see battles.
        self._histories = self._jepa_player._histories
        self._last_active_battle_tag = self._jepa_player._last_active_battle_tag

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self._jepa_player.choose_move(battle)

    def randomize(self) -> None:
        pass

    def reset_battles(self) -> None:
        try:
            super().reset_battles()
        except OSError:
            # Stale battles from a previous run may still be on the server.
            # Force-clear the battle dict so we can start fresh.
            self._battles = {}
        self._jepa_player.reset_battles()

    # Forward battle-count properties to the wrapped JEPA player.
    # NOTE: The outer JEPABaseline (a Player connected to the server) tracks
    # wins/losses via the base Player mechanism.  The inner player is logic-only
    # and has no battle tracking, so we must NOT delegate these properties.
    # Removing the overrides lets the base Player counters work correctly.

