"""Online player for the simple p1-only world model.

This reuses the JEPA online player's battle-history tracking, raw protocol
capture, action-name resolution, and TUI hooks.  Only the model diagnostics and
action scoring are specific to the simple world model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder

from metamon.interface import UniversalAction
from metamon.jepa.player import JEPAWorldModelPlayer, _display_hp, _display_species
from metamon.simple_world_model.model import SimpleWorldModel, TERMINAL_CLASSES
from metamon.tokenizer import PokemonTokenizer
from metamon.tui import TuiMixin, TurnContext, TuiDiagnostic


class SimpleWorldModelPlayer(JEPAWorldModelPlayer):
    """Showdown player driven by ``SimpleWorldModel`` controller logits."""

    def __init__(
        self,
        *args,
        model: SimpleWorldModel,
        tokenizer: PokemonTokenizer,
        fmt: str,
        verbose: bool = True,
        verbose_blocks: bool = False,
        max_history_blocks: int = 1,
        save_online_play_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            model=model,  # stored by the parent; overridden diagnostics use _swm
            tokenizer=tokenizer,
            fmt=fmt,
            verbose=verbose,
            verbose_blocks=verbose_blocks,
            max_history_blocks=max_history_blocks,
            save_online_play_root=save_online_play_root,
            **kwargs,
        )
        self._swm = model
        TuiMixin._repl_title = "Simple World Model REPL"

    @staticmethod
    def _concat_np_blocks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate([left.astype(np.int16, copy=False), right.astype(np.int16, copy=False)])

    @staticmethod
    def _pad_action_candidates(
        candidates: list[np.ndarray],
        *,
        pad_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        max_tokens = max((len(candidate) for candidate in candidates), default=1)
        out = torch.full(
            (1, len(candidates), max_tokens),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for idx, candidate in enumerate(candidates):
            tokens = torch.as_tensor(candidate.astype(np.int64, copy=False), dtype=torch.long, device=device)
            out[0, idx, :tokens.numel()] = tokens
        return out

    @torch.no_grad()
    def _simple_world_model_diagnostics(
        self,
        hist,
        action_names: dict[int, str],
    ) -> dict:
        device = next(self._swm.parameters()).device
        pad_id = self._tokenizer.pad_token_id
        if not hist.state_blocks:
            raise ValueError("Battle history has no state blocks")

        team_header = hist.state_blocks[0]
        current_state = hist.state_blocks[-1]
        state_tokens_np = self._concat_np_blocks(team_header, current_state)
        state_tokens = torch.as_tensor(
            state_tokens_np.astype(np.int64, copy=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        action_blocks = [
            self._action_content_block(action_names[idx])
            for idx in action_names
        ]
        legal_tokens = self._pad_action_candidates(
            action_blocks,
            pad_id=pad_id,
            device=device,
        )

        controller_out = self._swm.forward_controller(state_tokens, legal_tokens)
        logits = controller_out["controller_logits"][0].float()
        probs = F.softmax(logits, dim=-1)
        z_mu = controller_out["controller_z_mu"]
        h = controller_out["controller_h"]
        legal_embs = controller_out["legal_action_embs"][0]

        z_rep = z_mu.expand(legal_embs.shape[0], -1)
        m_out = self._swm.m(z_rep, legal_embs)
        mixture_probs = F.softmax(m_out["mixture_logits"].float(), dim=-1)
        expected_next_z = (mixture_probs.unsqueeze(-1) * m_out["mixture_means"].float()).sum(dim=-2)
        terminal_probs = F.softmax(m_out["terminal_logits"].float(), dim=-1)

        win_idx = TERMINAL_CLASSES.index("won")
        forfeit_win_idx = TERMINAL_CLASSES.index("forfeit_won")
        loss_idx = TERMINAL_CLASSES.index("lost")
        forfeit_loss_idx = TERMINAL_CLASSES.index("forfeit_lost")
        ongoing_idx = TERMINAL_CLASSES.index("ongoing")

        rows = []
        for row_idx, action_idx in enumerate(action_names):
            term = terminal_probs[row_idx]
            rows.append({
                "row_idx": row_idx,
                "idx": action_idx,
                "name": action_names[action_idx],
                "score": float(logits[row_idx].item()),
                "prob": float(probs[row_idx].item()),
                "p_win": float((term[win_idx] + term[forfeit_win_idx]).item()),
                "p_loss": float((term[loss_idx] + term[forfeit_loss_idx]).item()),
                "p_ongoing": float(term[ongoing_idx].item()),
                "next_norm": float(expected_next_z[row_idx].norm().item()),
                "mixture_entropy": float(
                    -(mixture_probs[row_idx] * mixture_probs[row_idx].clamp_min(1e-12).log()).sum().item()
                ),
            })

        return {
            "scorer": "controller_bc",
            "z_norm": float(z_mu.float().norm(dim=-1).mean().item()),
            "h_norm": float(h.float().norm(dim=-1).mean().item()),
            "state_token_count": int(state_tokens.shape[-1]),
            "team_token_count": int(len(team_header)),
            "current_state_token_count": int(len(current_state)),
            "rows": rows,
        }

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
        diag = self._simple_world_model_diagnostics(hist, action_names)
        rows = diag["rows"]
        if not rows:
            return self.choose_random_move(battle)
        best_row_pos, best_row = max(enumerate(rows), key=lambda item: item[1]["score"])
        best_idx = int(best_row["idx"])

        ctx = TurnContext(
            battle_tag=battle.battle_tag,
            turn=battle.turn,
            opponent_name=getattr(battle, "opponent_username", "?"),
            active_species=_display_species(battle.active_pokemon),
            active_hp=_display_hp(getattr(battle.active_pokemon, "current_hp_fraction", None)),
            opponent_species=_display_species(battle.opponent_active_pokemon),
            opponent_hp=_display_hp(getattr(battle.opponent_active_pokemon, "current_hp_fraction", None)),
            scorer="controller_bc",
            forced_switch=hist.current_forced_switch,
            num_state_blocks=len(hist.state_blocks),
            num_player_actions=len(hist.player_actions),
            num_opponent_actions=len(hist.opponent_actions),
        )

        diagnostic = TuiDiagnostic(
            columns=["action", "name", "C", "P", "Pwin", "Ploss", "Pongo", "|z_next|"],
            rows=[
                [
                    str(row["idx"]),
                    row["name"],
                    f"{row['score']:+8.4f}",
                    f"{row['prob']:.3f}",
                    f"{row['p_win']:.3f}",
                    f"{row['p_loss']:.3f}",
                    f"{row['p_ongoing']:.3f}",
                    f"{row['next_norm']:.3f}",
                ]
                for row in rows
            ],
            chosen_idx=best_row_pos,
            chosen_label=action_names.get(best_idx, "?"),
            context_lines=[
                f"input: team||state  tokens={diag['state_token_count']} "
                f"(team={diag['team_token_count']}, state={diag['current_state_token_count']})",
                f"|z|={diag['z_norm']:.4f}  |h|={diag['h_norm']:.4f}",
                "score: C behavior-cloning logit; terminal columns are M predictions for each legal action",
            ],
        )
        self._tui_store_turn(ctx, diagnostic)

        if TuiMixin._repl_view == "live":
            TuiMixin._repl_selected_turn = -1
            TuiMixin._repl_last_auto_redraw = 0.0
            self._tui_redraw()

        order = UniversalAction.action_idx_to_BattleOrder(battle, action_idx=best_idx)
        if order is None:
            return self.choose_random_move(battle)

        hist.pending_player_action = self._action_content_block(action_names[best_idx])
        hist.pending_player_action_text = action_names[best_idx]
        hist.pending_forced_switch = hist.current_forced_switch
        hist.pending_raw_message_start = len(hist.raw_messages)
        return order
