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

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalState, consistent_move_order, consistent_pokemon_order
from metamon.jepa.model import PairedJEPAModel
from metamon.jepa.online_serializer import (
    format_action_choice_text,
    format_action_text,
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
    pending_raw_message_start: int = 0
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
        verbose: bool = True,
        verbose_blocks: bool = False,
        max_history_blocks: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._jepa = model
        self._tokenizer = tokenizer
        self._fmt = fmt
        self._verbose = verbose
        self._max_history_blocks = max_history_blocks  # 0 = unlimited, default 300
        TuiMixin._repl_verbose_blocks = verbose_blocks
        TuiMixin._repl_max_history_blocks = max_history_blocks
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
        force_revival = self._is_force_revival(battle)
        state_key_block = state_block(
            self._tokenizer, battle, self._fmt, is_force_switch=force_switch,
            is_force_revival=force_revival,
        )
        current_key = tuple(int(x) for x in state_key_block.tolist())
        hist.current_forced_switch = force_switch

        if hist.last_state_key is None:
            hist.state_blocks.append(state_key_block)
            hist.last_state_key = current_key
            return hist

        if current_key != hist.last_state_key:
            if hist.pending_player_action is not None:
                (
                    player_action_text,
                    player_outcome,
                    opponent_action_text,
                    opponent_outcome,
                ) = self._infer_previous_turn_results(battle, hist)
                current_state = state_block(
                    self._tokenizer,
                    battle,
                    self._fmt,
                    is_force_switch=force_switch,
                    is_force_revival=force_revival,
                    last_turn_player_action=player_action_text,
                    last_turn_player_outcome=player_outcome,
                    last_turn_opponent_action=opponent_action_text,
                    last_turn_opponent_outcome=opponent_outcome,
                )
                hist.player_actions.append(hist.pending_player_action)
                opp_block = self._opponent_action_block(opponent_action_text or "unknown")
                hist.opponent_actions.append(opp_block)
                hist.pending_player_action = None
                hist.pending_player_action_text = None
                hist.pending_forced_switch = False
                hist.pending_raw_message_start = len(hist.raw_messages)
            elif self._verbose and TuiMixin._repl_verbose_blocks:
                print("  [state advanced without a pending player action]")
                current_state = state_key_block
            else:
                current_state = state_key_block

            hist.state_blocks.append(current_state)
            hist.last_state_key = current_key
        return hist

    # ═══════════════════════════════════════════════════════════════════
    # Action resolution
    # ═══════════════════════════════════════════════════════════════════

    def _is_force_switch(self, battle: AbstractBattle) -> bool:
        return is_force_switch_state(battle)

    def _is_force_revival(self, battle: AbstractBattle) -> bool:
        return bool(getattr(battle, "reviving", False))

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

    def _infer_previous_turn_results(
        self,
        battle: AbstractBattle,
        hist: BattleHistory,
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Infer previous action choices and outcomes for the next state block.

        Action embeddings stay choice-only.  This helper returns the outcome
        tokens used in ``<last_turn_results>`` for the state reached by the
        pending player action.
        """
        player_action = hist.pending_player_action_text
        player_outcome = self._default_outcome(player_action)
        opponent_action: Optional[str] = None
        opponent_outcome: Optional[str] = None

        our_prefix = self._player_role_prefix(battle)
        opponent_prefix = self._opponent_role_prefix(battle)
        raw_messages = hist.raw_messages[hist.pending_raw_message_start:]
        fainted_sides: set[str] = set()
        for msg in raw_messages:
            kind, args = self._split_protocol_message(msg)
            if kind is None:
                continue

            if kind == "move" and len(args) >= 2:
                side = self._side_from_identifier(args[0], our_prefix, opponent_prefix)
                if side == "opponent":
                    opponent_action = f"opponent move: {args[1]}"
                    opponent_outcome = "success"
                elif side == "player" and player_outcome is None:
                    player_outcome = self._default_outcome(player_action)
            elif kind == "switch" and len(args) >= 2:
                side = self._side_from_identifier(args[0], our_prefix, opponent_prefix)
                if side in fainted_sides:
                    continue
                species = self._species_from_protocol_details(args[1])
                if side == "opponent":
                    opponent_action = f"opponent switch: {species}"
                    opponent_outcome = "success"
                elif side == "player" and player_outcome is None:
                    player_outcome = self._default_outcome(player_action)
            elif kind == "cant" and len(args) >= 2:
                side = self._side_from_identifier(args[0], our_prefix, opponent_prefix)
                reason = args[1].strip()
                if side == "player" and self._known_action(player_action):
                    player_outcome = f"cant {reason}"
                elif side == "opponent":
                    if len(args) >= 3 and args[2].strip():
                        opponent_action = f"opponent move: {args[2]}"
                        opponent_outcome = f"cant {reason}"
                    elif self._known_action(opponent_action):
                        opponent_outcome = f"cant {reason}"
                    else:
                        opponent_action = "unknown"
                        opponent_outcome = None
            elif kind == "-fail" and args:
                side = self._side_from_identifier(args[0], our_prefix, opponent_prefix)
                if side == "player" and self._known_action(player_action):
                    player_outcome = "fail"
                elif side == "opponent" and self._known_action(opponent_action):
                    opponent_outcome = "fail"
            elif kind == "faint" and args:
                side = self._side_from_identifier(args[0], our_prefix, opponent_prefix)
                if side is not None:
                    fainted_sides.add(side)

        if opponent_action is None:
            opponent_action, opponent_outcome = self._fallback_opponent_previous_action(
                battle, raw_messages,
            )

        return player_action, player_outcome, opponent_action, opponent_outcome

    @staticmethod
    def _split_protocol_message(msg: str) -> tuple[Optional[str], list[str]]:
        parts = msg.split("|")
        if parts and parts[0] == "":
            parts = parts[1:]
        if not parts or not parts[0]:
            return None, []
        return parts[0], parts[1:]

    @staticmethod
    def _side_from_identifier(
        identifier: str,
        our_prefix: Optional[str],
        opponent_prefix: Optional[str],
    ) -> Optional[str]:
        if our_prefix is not None and identifier.startswith(our_prefix):
            return "player"
        if opponent_prefix is not None and identifier.startswith(opponent_prefix):
            return "opponent"
        return None

    @staticmethod
    def _species_from_protocol_details(details: str) -> str:
        return pokemon_name(details.split(",", 1)[0].strip()) or "unknown"

    @staticmethod
    def _known_action(action_text: Optional[str]) -> bool:
        if action_text is None:
            return False
        return format_action_text(action_text) != "unknown"

    @classmethod
    def _default_outcome(cls, action_text: Optional[str]) -> Optional[str]:
        return "success" if cls._known_action(action_text) else None

    def _fallback_opponent_previous_action(
        self,
        battle: AbstractBattle,
        raw_messages: list[str],
    ) -> tuple[str, Optional[str]]:
        if raw_messages:
            return "unknown", None
        opponent_active = getattr(battle, "opponent_active_pokemon", None)
        prev = getattr(opponent_active, "previous_move", None)
        if prev is None:
            return "unknown", None
        raw = getattr(prev, "id", None) or getattr(prev, "name", None) or str(prev)
        return f"opponent move: {move_name(str(raw))}", "success"

    @staticmethod
    def _player_role_prefix(battle: AbstractBattle) -> Optional[str]:
        """Return ``'p1'`` or ``'p2'`` for our role, or None if unknown."""
        our_role = getattr(battle, "_player_role", None)
        if our_role in {"p1", "p2"}:
            return our_role
        return None

    @staticmethod
    def _opponent_role_prefix(battle: AbstractBattle) -> Optional[str]:
        """Return ``'p1'`` or ``'p2'`` for the opponent's role, or None if unknown."""
        our_role = getattr(battle, "_player_role", None)
        if our_role == "p1":
            return "p2"
        elif our_role == "p2":
            return "p1"
        return None

    def _opponent_action_block(self, action_text: str) -> np.ndarray:
        """Build an opponent action token block.

        Action blocks contain only the chosen move name — no outcome
        tokens.  Outcomes (success / fail / cant <reason>) now live in
        ``<last_turn_results>`` inside the *following* state block.
        """
        return self._action_content_block(action_text)

    def _action_content_block(self, action_text: str) -> np.ndarray:
        words = format_action_choice_text(action_text).split()
        return tokenize_words(self._tokenizer, words, warn_unknown=False)

    # ═══════════════════════════════════════════════════════════════════
    # Tensor encoding for JEPA forward pass
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _window_history(
        state_blocks: list[np.ndarray],
        player_actions: list[np.ndarray],
        opponent_actions: list[np.ndarray],
        max_hist: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Apply history windowing matching the training dataset's _resolve_window logic.

        State index 0 is the team header (always retained).  Subsequent states
        are indexed 1..S-1.  Actions connect state_{i+1}→state_{i+2}.

        Returns (windowed_states, windowed_player_actions, windowed_opponent_actions).
        """
        if max_hist <= 0:
            return state_blocks, player_actions, opponent_actions

        S = len(state_blocks)         # total state blocks (incl. team header)
        battle_start = 0
        state_end = S

        # Match _resolve_window logic from train_paired.py:
        #   state_start = max(battle_start + 1, state_end - max_hist)
        #   action_start = max(0, state_start - battle_start - 1)
        #   action_end   = max(0, state_end - battle_start - 2)
        state_start = max(battle_start + 1, state_end - max_hist)
        action_start = max(0, state_start - battle_start - 1)
        action_end = max(0, state_end - battle_start - 2)
        action_end = max(action_start, action_end)

        # _slice_state_window: always keep team header, then states [state_start, state_end)
        windowed_states = [state_blocks[0]]
        if state_start > battle_start + 1:
            windowed_states.extend(state_blocks[state_start:state_end])
        else:
            windowed_states.extend(state_blocks[battle_start + 1:state_end])

        windowed_player = player_actions[action_start:action_end]
        windowed_opponent = opponent_actions[action_start:action_end]

        return windowed_states, windowed_player, windowed_opponent

    def _encode_history_tensors(self, hist: BattleHistory):
        pad_id = self._tokenizer.pad_token_id

        # Apply history windowing so the model sees the same context length it was trained on.
        states, p_actions, o_actions = self._window_history(
            hist.state_blocks, hist.player_actions, hist.opponent_actions,
            self._max_history_blocks,
        )

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

        state_tokens, state_valid = pad_blocks(states)
        player_hist_tokens, player_hist_valid = pad_blocks(p_actions)
        opponent_hist_tokens, opponent_hist_valid = pad_blocks(o_actions)
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
        decision_state = self._jepa.decision_state_encoder(
            z_self,
            pred_opponent_state_mu,
            pred_opponent_state_logvar,
        )
        value_logit_t = self._jepa.value_head(decision_state)
        value_logit = value_logit_t.item()
        value_prob = torch.sigmoid(value_logit_t).item()

        rows = []
        for idx, name in action_names.items():
            pa = self._action_content_block(name)
            pa_tensor = torch.from_numpy(pa.astype(np.int64)).unsqueeze(0).to(device)
            own_action = self._jepa.action_encoder(pa_tensor)
            action_h = self._jepa.action_projector(own_action)
            q_logit_t = self._jepa.action_value_head(decision_state, action_h)
            q_logit = q_logit_t.item()
            q_prob = torch.sigmoid(q_logit_t).item()
            pred_next, _ = self._jepa.next_state_predictor(
                z_self, own_action, pred_opponent_state, pred_opponent_action,
            )
            next_norm = torch.norm(pred_next, dim=-1).item()
            state_delta = torch.norm(pred_next - z_self, dim=-1).item()
            action_norm = torch.norm(own_action, dim=-1).item()
            rows.append({
                "idx": idx,
                "name": name,
                "score": q_logit,
                "q_logit": q_logit,
                "q_prob": q_prob,
                "next_norm": next_norm,
                "state_delta": state_delta,
                "action_norm": action_norm,
            })

        return {
            "z_self_norm": z_self_norm,
            "pred_state_norm": pred_state_norm,
            "pred_state_delta": pred_state_delta,
            "pred_action_norm": pred_action_norm,
            "pred_state_logvar_mean": pred_state_logvar_mean,
            "pred_action_logvar_mean": pred_action_logvar_mean,
            "value_logit": value_logit,
            "value_prob": value_prob,
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
            scorer="actor-critic",
            forced_switch=hist.current_forced_switch,
            num_state_blocks=len(hist.state_blocks),
            num_player_actions=len(hist.player_actions),
            num_opponent_actions=len(hist.opponent_actions),
        )

        rows = diag["rows"]
        best_row = max(rows, key=lambda row: row["score"])
        best_idx = best_row["idx"]

        # Build TuiDiagnostic from JEPA‑specific output.
        columns = ["action", "name", "Q", "Pwin", "V", "|z_next|", "|Δz|", "|a|"]
        str_rows: list[list[str]] = [
            [
                str(row["idx"]),
                row["name"],
                f"{row['q_logit']:+8.4f}",
                f"{row['q_prob']:.4f}",
                f"{diag['value_prob']:.4f}",
                f"{row['next_norm']:8.4f}",
                f"{row['state_delta']:8.4f}",
                f"{row['action_norm']:8.4f}",
            ]
            for row in rows
        ]

        diagnostic = TuiDiagnostic(
            columns=columns,
            rows=str_rows,
            chosen_idx=best_idx,
            chosen_label=action_names.get(best_idx, "?"),
            context_lines=[
                f"z_self: {diag['z_self_norm']:.4f}  pred_opp: {diag['pred_state_norm']:.4f}  "
                f"delta: {diag['pred_state_delta']:.4f}  V: {diag['value_logit']:+.4f} "
                f"Pwin: {diag['value_prob']:.4f}",
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

        hist.pending_player_action = self._action_content_block(action_names[best_idx])
        hist.pending_player_action_text = action_names[best_idx]
        hist.pending_forced_switch = hist.current_forced_switch
        hist.pending_raw_message_start = len(hist.raw_messages)
        return order
