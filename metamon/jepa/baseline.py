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


@register_baseline()
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
        self._jepa_player = JEPAWorldModelPlayer(
            *args,
            model=_shared_model,
            tokenizer=_shared_tokenizer,
            fmt=_shared_fmt,
            heuristic=_shared_heuristic,
            verbose=False,
            verbose_blocks=False,
            **kwargs,
        )
        # Share the same histories dict so the TUI (if active) can see battles.
        self._histories = self._jepa_player._histories
        self._last_active_battle_tag = self._jepa_player._last_active_battle_tag

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self._jepa_player.choose_move(battle)

    def reset_battles(self) -> None:
        super().reset_battles()
        self._jepa_player.reset_battles()

    # Forward battle-count properties to the wrapped JEPA player.
    @property
    def n_won_battles(self) -> int:
        return self._jepa_player.n_won_battles

    @property
    def n_lost_battles(self) -> int:
        return self._jepa_player.n_lost_battles

    @property
    def n_tied_battles(self) -> int:
        return self._jepa_player.n_tied_battles

    @property
    def n_finished_battles(self) -> int:
        return self._jepa_player.n_finished_battles
