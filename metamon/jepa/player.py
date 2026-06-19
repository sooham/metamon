"""JEPAWorldModelPlayer — Showdown player using JEPA latent rollouts for action selection.

Includes interactive REPL (R=raw logs, P=blocks, V=verbose toggle, O=overview),
battle history tracking, JEPA diagnostics, and opponent ``|cant|`` detection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder

from metamon.backend.replay_parser.str_parsing import move_name
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalState, consistent_move_order, consistent_pokemon_order
from metamon.jepa.model import PairedJEPAModel
from metamon.jepa.online_serializer import (
    action_block,
    is_force_switch_state,
    state_block,
    team_context_block,
    tokenize_words,
)
from metamon.tui import TuiMixin, BattleHistory, TurnContext, TuiDiagnostic
from metamon.tokenizer import PokemonTokenizer


# ── tiny display helpers ──────────────────────────────────────────────

def _display_hp(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{max(0.0, min(1.0, float(value))):.2f}"


def _display_species(pokemon) -> str:
    if pokemon is None:
        return "unknown"
    return str(getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None) or "unknown")


# ── BattleHistory ─────────────────────────────────────────────────────

@dataclass
class BattleHistory:
    """Per-battle state tracked across turns for model input and REPL display."""

    state_blocks: list[np.ndarray] = field(default_factory=list)
    player_actions: list[np.ndarray] = field(default_factory=list)
    opponent_actions: list[np.ndarray] = field(default_factory=list)
    pending_player_action: Optional[np.ndarray] = None
    pending_player_action_text: Optional[str] = None
    pending_forced_switch: bool = False
    last_state_key: Optional[tuple[int, ...]] = None
    current_forced_switch: bool = False
    # Raw protocol messages (pipe-delimited strings) for the "R" REPL key.
    raw_messages: list[str] = field(default_factory=list)
    # Per-turn JEPA diagnostics history for REPL views.
    diag_history: list[dict] = field(default_factory=list)
    action_names_history: list[dict[int, str]] = field(default_factory=list)
    # Convenience aliases for the most recent turn (kept for backward compat).
    last_diag: Optional[dict] = None
    last_action_names: Optional[dict[int, str]] = None
    # Per-turn structured diagnostics for TUI REPL views.
    turn_contexts: list[TurnContext] = field(default_factory=list)
    turn_diagnostics: list[TuiDiagnostic] = field(default_factory=list)
    # Battle outcome tracking.
    finished: bool = False
    outcome: Optional[str] = None  # "win", "loss", "tie", or None
    # Indices into raw_messages where each turn started (for per‑turn raw log view).
    turn_msg_boundaries: list[int] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════
# JEPAWorldModelPlayer
# ═════════════════════════════════════════════════════════════════════════

class JEPAWorldModelPlayer(TuiMixin, MetamonPlayer):
    """Showdown player that uses JEPA for latent diagnostics and action logging."""

    def __init__(
        self,
        *args,
        model: PairedJEPAModel,
        tokenizer: PokemonTokenizer,
        fmt: str,
        heuristic: str = "max-self-state-delta",
        verbose: bool = True,
        verbose_blocks: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._jepa = model
        self._tokenizer = tokenizer
        self._fmt = fmt
        self._heuristic = heuristic
        self._verbose = verbose
        TuiMixin._repl_verbose_blocks = verbose_blocks
        self._histories: dict[str, BattleHistory] = {}
        self._last_active_battle_tag: Optional[str] = None
        self._tui_register()

    # ── TuiMixin hooks ────────────────────────────────────────────────

    def _tui_detok(self, tokens: np.ndarray) -> str:
        return " ".join(self._tokenizer.detokenize(tokens.tolist()))

    def _tui_store_turn(self, ctx: TurnContext, diag: TuiDiagnostic) -> None:
        hist = self._current_hist()
        if hist is None:
            return
        hist.turn_contexts.append(ctx)
        hist.turn_diagnostics.append(diag)
        hist.turn_msg_boundaries.append(len(hist.raw_messages))

    def _tui_current_hist(self) -> Optional[BattleHistory]:
        tag = self._last_active_battle_tag
        if tag is None:
            return None
        hist = self._histories.get(tag)
        if hist is None and tag.startswith(">"):
            hist = self._histories.get(tag[1:])
        elif hist is None:
            hist = self._histories.get(">" + tag)
        return hist

    # ── utility ───────────────────────────────────────────────────────

    def _detok(self, tokens: np.ndarray) -> str:
        return self._tui_detok(tokens)

    def _current_hist(self) -> Optional[BattleHistory]:
        return self._tui_current_hist()



    def _history(self, battle: AbstractBattle) -> BattleHistory:
        hist = self._histories.get(battle.battle_tag)
        if hist is None:
            hist = BattleHistory()
            self._histories[battle.battle_tag] = hist
        # Add team header if history was created by _handle_battle_message (no blocks yet).
        if not hist.state_blocks:
            hist.state_blocks.append(team_context_block(self._tokenizer, battle, self._fmt))

        force_switch = self._is_force_switch(battle)
        current_state = state_block(
            self._tokenizer, battle, self._fmt, is_force_switch=force_switch,
        )
        current_key = tuple(int(x) for x in current_state.tolist())
        hist.current_forced_switch = force_switch

        if hist.last_state_key is None:
            hist.state_blocks.append(current_state)
            hist.last_state_key = current_key
            return hist

        if current_key != hist.last_state_key:
            if hist.pending_player_action is not None:
                hist.player_actions.append(hist.pending_player_action)
                opp_block = self._infer_opponent_previous_action(battle, hist)
                hist.opponent_actions.append(opp_block)
                hist.pending_player_action = None
                hist.pending_player_action_text = None
                hist.pending_forced_switch = False
            elif self._verbose and TuiMixin._repl_verbose_blocks:
                print("  [state advanced without a pending player action]")

            hist.state_blocks.append(current_state)
            hist.last_state_key = current_key
        return hist

    # ═══════════════════════════════════════════════════════════════════
    # Action resolution
    # ═══════════════════════════════════════════════════════════════════

    def _is_force_switch(self, battle: AbstractBattle) -> bool:
        return is_force_switch_state(battle)

    def _legal_action_indices(self, battle: AbstractBattle) -> list[int]:
        state = UniversalState.from_Battle(battle)
        legal = UniversalAction.definitely_valid_actions(state, battle)
        actions = sorted(action.action_idx for action in legal)
        if self._is_force_switch(battle):
            actions = [idx for idx in actions if 4 <= idx <= 8]
        return actions

    def _resolve_action_names(self, battle: AbstractBattle, action_indices: list[int]) -> dict[int, str]:
        available_moves = getattr(battle, "available_moves", None) or []
        valid_moves = {getattr(m, "id", str(m)) for m in available_moves}
        if valid_moves == {"recharge"}:
            move_names = ["recharge"] * 4
        elif valid_moves == {"struggle"}:
            move_names = ["struggle"] * 4
        elif "fight" in valid_moves:
            move_names = ["struggle"] * 4
        else:
            active = getattr(battle, "active_pokemon", None)
            active_moves = list(getattr(active, "moves", {}).values()) if active is not None else []
            move_names = [getattr(m, "id", str(m)) for m in consistent_move_order(active_moves)]

        if not getattr(battle, "reviving", False):
            switches = consistent_pokemon_order(
                [p for p in battle.team.values() if not p.fainted and not p.active]
            )
        else:
            switches = consistent_pokemon_order(
                [p for p in battle.team.values() if p.fainted and not p.active]
            )
        switch_names = [getattr(p, "species", getattr(p, "name", "unknown")) for p in switches]

        names: dict[int, str] = {}
        for idx in action_indices:
            if 0 <= idx <= 3:
                names[idx] = f"move: {move_names[idx]}" if idx < len(move_names) else f"move_{idx}"
            elif 4 <= idx <= 8:
                i = idx - 4
                names[idx] = f"switch: {switch_names[i]}" if i < len(switch_names) else f"switch_{idx}"
            elif 9 <= idx <= 12:
                i = idx - 9
                names[idx] = f"tera+move: {move_names[i]}" if i < len(move_names) else f"tera_move_{idx}"
            else:
                names[idx] = f"action_{idx}"
        return names

    def _infer_opponent_previous_action(self, battle: AbstractBattle, hist: BattleHistory) -> np.ndarray:
        """Return the tokenised opponent action block.

        Detects ``|cant|`` from raw protocol messages so that moves that were
        clicked but not executed (paralysis, sleep, freeze) carry the same
        ``cant=X`` token the training data uses.
        """
        prev = getattr(battle.opponent_active_pokemon, "previous_move", None)
        if prev is None:
            return self._opponent_action_block("unknown")

        raw = getattr(prev, "id", None) or getattr(prev, "name", None) or str(prev)
        move = move_name(str(raw))

        # Check raw protocol messages for a recent |cant| affecting the opponent.
        cant_reason: Optional[str] = None
        opponent_prefix = self._opponent_role_prefix(battle)
        if opponent_prefix is not None:
            for msg in reversed(hist.raw_messages[-4:]):
                if msg.startswith(f"|cant|{opponent_prefix}a:"):
                    parts = msg.split("|")
                    # |cant|p2a: Jynx|par → parts = ['', 'cant', 'p2a: Jynx', 'par']
                    if len(parts) >= 4:
                        reason = parts[3].strip().lower()
                        if reason in {"par", "slp", "frz"}:
                            cant_reason = reason
                    break

        return self._opponent_action_block(move, cant=cant_reason)

    @staticmethod
    def _opponent_role_prefix(battle: AbstractBattle) -> Optional[str]:
        """Return ``'p1'`` or ``'p2'`` for the opponent's role, or None if unknown."""
        our_role = getattr(battle, "_player_role", None)
        if our_role == "p1":
            return "p2"
        elif our_role == "p2":
            return "p1"
        return None

    def _opponent_action_block(self, move: str, *, cant: Optional[str] = None) -> np.ndarray:
        """Build an opponent action token block, optionally with a cant=X token.

        The training-data format for a cant'd opponent move is::

            <opponent_chosen_move> blizzard cant=par <end_opponent_chosen_move>

        where ``cant=par`` is a **single** token in the vocabulary.
        """
        words: list[str] = ["<opponent_chosen_move>"]
        if cant is not None:
            words.append(move)
            words.append(f"cant={cant}")
        else:
            words.append(move)
        words.append("<end_opponent_chosen_move>")
        return tokenize_words(self._tokenizer, words, warn_unknown=False)

    # ═══════════════════════════════════════════════════════════════════
    # Tensor encoding for JEPA forward pass
    # ═══════════════════════════════════════════════════════════════════

    def _encode_history_tensors(self, hist: BattleHistory):
        pad_id = self._tokenizer.pad_token_id

        def pad_blocks(blocks: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
            max_blocks = max(len(blocks), 1)
            max_tokens = max((len(block) for block in blocks), default=1)
            padded = torch.full((1, max_blocks, max_tokens), pad_id, dtype=torch.long)
            valid = torch.zeros((1, max_blocks), dtype=torch.bool)
            for block_idx, block in enumerate(blocks):
                tokens = torch.from_numpy(block.astype(np.int64, copy=False))
                padded[0, block_idx, :len(tokens)] = tokens
                valid[0, block_idx] = True
            return padded, valid

        state_tokens, state_valid = pad_blocks(hist.state_blocks)
        player_hist_tokens, player_hist_valid = pad_blocks(hist.player_actions)
        opponent_hist_tokens, opponent_hist_valid = pad_blocks(hist.opponent_actions)
        return (
            state_tokens,
            state_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        )

    # ═══════════════════════════════════════════════════════════════════
    # JEPA diagnostics
    # ═══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _jepa_diagnostics(self, battle: AbstractBattle, hist: BattleHistory, action_names: dict[int, str]):
        device = next(self._jepa.parameters()).device
        (
            state_tokens,
            state_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        ) = (tensor.to(device) for tensor in self._encode_history_tensors(hist))

        z_self = self._jepa.encode_current_state(state_tokens, state_valid)
        c_self = self._jepa.encode_history_context(
            state_tokens, state_valid,
            player_hist_tokens, player_hist_valid,
            opponent_hist_tokens, opponent_hist_valid,
        )
        (pred_opponent_state_mu, pred_opponent_state_logvar,
         pred_opponent_action_mu, pred_opponent_action_logvar) = self._jepa.opponent_belief_predictor(
            c_self, z_self,
        )
        pred_opponent_state = pred_opponent_state_mu
        pred_opponent_action = pred_opponent_action_mu

        z_self_norm = torch.norm(z_self, dim=-1).item()
        pred_state_norm = torch.norm(pred_opponent_state, dim=-1).item()
        pred_state_delta = torch.norm(pred_opponent_state - z_self, dim=-1).item()
        pred_action_norm = torch.norm(pred_opponent_action, dim=-1).item()
        pred_state_logvar_mean = pred_opponent_state_logvar.mean().item()
        pred_action_logvar_mean = pred_opponent_action_logvar.mean().item()
        current_rank = self._jepa.rank_head(z_self, pred_opponent_state).item()

        rows = []
        for idx, name in action_names.items():
            pa = action_block(self._tokenizer, name, opponent=False)
            pa_tensor = torch.from_numpy(pa.astype(np.int64)).unsqueeze(0).to(device)
            own_action = self._jepa.action_encoder(pa_tensor)
            pred_next, _ = self._jepa.next_state_predictor(
                z_self, own_action, pred_opponent_state, pred_opponent_action,
            )
            next_norm = torch.norm(pred_next, dim=-1).item()
            state_delta = torch.norm(pred_next - z_self, dim=-1).item()
            action_norm = torch.norm(own_action, dim=-1).item()
            if self._heuristic == "max-rank":
                rank_score = self._jepa.rank_head(pred_next, pred_opponent_state).item()
                delta = rank_score
            elif self._heuristic == "max-opponent-state-delta":
                next_pred_opponent, _ = self._jepa.opponent_belief_predictor(c_self, pred_next)[:2]
                delta = torch.norm(next_pred_opponent - pred_opponent_state, dim=-1).item()
                rank_score = None
            else:
                delta = torch.norm(pred_next - z_self, dim=-1).item()
                rank_score = None
            rows.append((idx, name, delta, rank_score, next_norm, state_delta, action_norm))

        return {
            "z_self_norm": z_self_norm,
            "pred_state_norm": pred_state_norm,
            "pred_state_delta": pred_state_delta,
            "pred_action_norm": pred_action_norm,
            "pred_state_logvar_mean": pred_state_logvar_mean,
            "pred_action_logvar_mean": pred_action_logvar_mean,
            "current_rank": current_rank,
            "rows": rows,
        }

    # ═══════════════════════════════════════════════════════════════════
    # choose_move — the main decision entry point
    # ═══════════════════════════════════════════════════════════════════

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        import sys as _sys
        _sys.stderr.write(f"[TUI] choose_move called for {battle.battle_tag}\n")
        _sys.stderr.flush()
        TuiMixin._repl_active_instance = self
        self._process_repl_keys()
        self._last_active_battle_tag = battle.battle_tag

        hist = self._history(battle)
        legal_actions = self._legal_action_indices(battle)
        if not legal_actions:
            return self.choose_random_move(battle)

        action_names = self._resolve_action_names(battle, legal_actions)
        diag = self._jepa_diagnostics(battle, hist, action_names)

        # Build  TurnContext + TuiDiagnostic for the TUI contract.
        opponent_name = getattr(battle, "opponent_username", "?")
        ctx = TurnContext(
            battle_tag=battle.battle_tag,
            turn=battle.turn,
            opponent_name=opponent_name,
            active_species=_display_species(battle.active_pokemon),
            active_hp=_display_hp(getattr(battle.active_pokemon, "current_hp_fraction", None)),
            opponent_species=_display_species(battle.opponent_active_pokemon),
            opponent_hp=_display_hp(getattr(battle.opponent_active_pokemon, "current_hp_fraction", None)),
            heuristic=self._heuristic,
            forced_switch=hist.current_forced_switch,
            num_state_blocks=len(hist.state_blocks),
            num_player_actions=len(hist.player_actions),
            num_opponent_actions=len(hist.opponent_actions),
        )

        rows = diag["rows"]
        best_idx = max(rows, key=lambda row: row[2])[0]

        # Build TuiDiagnostic from JEPA‑specific output.
        if self._heuristic == "max-rank":
            columns = ["action", "name", "rank", "Δrank", "|z_next|", "|Δz|", "|a|"]
            str_rows: list[list[str]] = [
                [str(idx), name, f"{rank:+8.4f}", f"{rank - diag['current_rank']:+8.4f}",
                 f"{next_n:8.4f}", f"{state_d:8.4f}", f"{act_n:8.4f}"]
                for idx, name, rank, _, next_n, state_d, act_n in rows
            ]
        else:
            columns = ["action", "name", "delta"]
            str_rows = [
                [str(idx), name, f"{delta:+.4f}"]
                for idx, name, delta, *_ in rows
            ]

        diagnostic = TuiDiagnostic(
            columns=columns,
            rows=str_rows,
            chosen_idx=best_idx,
            chosen_label=action_names.get(best_idx, "?"),
            context_lines=[
                f"z_self: {diag['z_self_norm']:.4f}  pred_opp: {diag['pred_state_norm']:.4f}  "
                f"delta: {diag['pred_state_delta']:.4f}  rank: {diag['current_rank']:+.4f}",
                f"opp-state logvar: {diag['pred_state_logvar_mean']:.4f}  "
                f"opp-action logvar: {diag['pred_action_logvar_mean']:.4f}",
            ],
        )

        self._tui_store_turn(ctx, diagnostic)

        # Let the TUI own all screen output so the footer stays at the bottom.
        if TuiMixin._repl_view == "live":
            TuiMixin._repl_selected_turn = -1  # auto‑select latest
            TuiMixin._repl_last_auto_redraw = 0.0  # bypass throttle
            self._tui_redraw()

        order = UniversalAction.action_idx_to_BattleOrder(battle, action_idx=best_idx)
        if order is None:
            return self.choose_random_move(battle)

        hist.pending_player_action = action_block(self._tokenizer, action_names[best_idx], opponent=False)
        hist.pending_player_action_text = action_names[best_idx]
        hist.pending_forced_switch = hist.current_forced_switch
        return order
