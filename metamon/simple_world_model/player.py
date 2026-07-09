"""Online V/M/C rollout player.

The player encodes each currently observed state with V, uses M to retain the
latent/action history, and blends rollout value with C's legal-action prior.
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
from metamon.simple_world_model.action_vocab import ActionVocabulary
from metamon.simple_world_model.model import OUTCOME_CLASSES, SimpleWorldModel
from metamon.tokenizer import PokemonTokenizer
from metamon.tui import TuiDiagnostic, TuiMixin, TurnContext


class SimpleWorldModelPlayer(JEPAWorldModelPlayer):
    """Showdown player that makes M predictions decision inputs, not diagnostics."""

    def __init__(
        self,
        *args,
        model: SimpleWorldModel,
        tokenizer: PokemonTokenizer,
        action_vocabulary: ActionVocabulary,
        fmt: str,
        verbose: bool = True,
        verbose_blocks: bool = False,
        max_context_transitions: int = 32,
        rollout_horizon: int = 4,
        rollouts_per_action: int = 8,
        save_online_play_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            model=model,
            tokenizer=tokenizer,
            fmt=fmt,
            verbose=verbose,
            verbose_blocks=verbose_blocks,
            max_history_blocks=max_context_transitions,
            save_online_play_root=save_online_play_root,
            **kwargs,
        )
        self._swm = model
        self._action_vocabulary = action_vocabulary
        self._max_context_transitions = int(max_context_transitions)
        self._rollout_horizon = int(rollout_horizon)
        self._rollouts_per_action = int(rollouts_per_action)
        TuiMixin._repl_title = "V/M/C World Model REPL"

    @staticmethod
    def _pad_blocks(blocks: list[np.ndarray], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        width = max((len(block) for block in blocks), default=1)
        tokens = torch.full((len(blocks), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(blocks), width), dtype=torch.bool, device=device)
        for row, block in enumerate(blocks):
            count = len(block)
            tokens[row, :count] = torch.as_tensor(block, dtype=torch.long, device=device)
            mask[row, :count] = True
        return tokens, mask

    def _action_id_from_block(self, block: np.ndarray | None) -> int:
        if block is None or len(block) == 0:
            return self._action_vocabulary.none_id
        words = self._tokenizer.detokenize([int(value) for value in block if int(value) != self._tokenizer.pad_token_id])
        return self._action_vocabulary.encode(words, fmt=self._fmt)

    @torch.no_grad()
    def _observed_latent_history(self, hist) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return ``team_z, states, own-actions, opponent-actions, token_count``."""
        if len(hist.state_blocks) < 2:
            raise ValueError("battle history requires a team header and one visible state")
        device = next(self._swm.parameters()).device
        team = hist.state_blocks[0]
        visible = hist.state_blocks[1:]
        # Keep at most context transitions + current state.  Actions align
        # with the state *before* the following state.
        first = max(0, len(visible) - (self._max_context_transitions + 1))
        states_np = visible[first:]
        count = len(states_np)
        own_blocks = hist.player_actions[first : first + max(count - 1, 0)]
        opp_blocks = hist.opponent_actions[first : first + max(count - 1, 0)]
        own_ids = [self._action_id_from_block(block) for block in own_blocks]
        opp_ids = [self._action_id_from_block(block) for block in opp_blocks]
        while len(own_ids) < count - 1:
            own_ids.append(self._action_vocabulary.none_id)
        while len(opp_ids) < count - 1:
            opp_ids.append(self._action_vocabulary.unknown_id)
        team_tokens, team_mask = self._pad_blocks([team], self._tokenizer.pad_token_id, device)
        state_tokens, state_mask = self._pad_blocks(list(states_np), self._tokenizer.pad_token_id, device)
        headers = team_tokens.expand(count, -1)
        header_masks = team_mask.expand(count, -1)
        state_mu, _ = self._swm.v.encode(
            headers, state_tokens, header_valid_mask=header_masks, state_valid_mask=state_mask
        )
        team_mu, _ = self._swm.v.encode(team_tokens, header_valid_mask=team_mask)
        own = torch.as_tensor(own_ids, dtype=torch.long, device=device).unsqueeze(0)
        opp = torch.as_tensor(opp_ids, dtype=torch.long, device=device).unsqueeze(0)
        return (
            team_mu,
            state_mu.unsqueeze(0),
            own,
            opp,
            int(len(team) + sum(len(state) for state in states_np)),
        )

    def _rollout_value(
        self,
        *,
        team_z: torch.Tensor,
        states: torch.Tensor,
        own_history: torch.Tensor,
        opponent_history: torch.Tensor,
        proposed_own_action: int,
    ) -> tuple[float, float, float]:
        """One imagined trajectory; returns value, P(win), P(loss)."""
        device = team_z.device
        format_mask = torch.as_tensor(self._action_vocabulary.format_mask(self._fmt), dtype=torch.bool, device=device).unsqueeze(0)
        imagined_states = states.clone()
        imagined_own = own_history.clone()
        imagined_opp = opponent_history.clone()
        h = None
        for _ in range(self._rollout_horizon):
            history = self._swm.encode_history(team_z, imagined_states, imagined_own, imagined_opp)
            h = history["h"]
            own_id = torch.tensor([proposed_own_action], dtype=torch.long, device=device)
            own_embedding = self._swm.action_embedding(own_id)
            opponent_logits = self._swm.opponent_head(torch.cat([h, own_embedding], dim=-1)).masked_fill(
                ~format_mask, torch.finfo(h.dtype).min
            )
            opponent_id = torch.distributions.Categorical(logits=opponent_logits.float()).sample()
            opponent_embedding = self._swm.action_embedding(opponent_id)
            params = self._swm.transition_head(h, own_embedding, opponent_embedding)
            next_z = self._swm.transition_head.sample(params)
            done_prob = torch.sigmoid(self._swm.done_head(h).squeeze(-1))
            imagined_states = torch.cat([imagined_states, next_z.unsqueeze(1)], dim=1)
            imagined_own = torch.cat([imagined_own, own_id.unsqueeze(1)], dim=1)
            imagined_opp = torch.cat([imagined_opp, opponent_id.unsqueeze(1)], dim=1)
            if imagined_states.shape[1] > self._max_context_transitions + 1:
                imagined_states = imagined_states[:, -self._max_context_transitions - 1 :]
                imagined_own = imagined_own[:, -self._max_context_transitions :]
                imagined_opp = imagined_opp[:, -self._max_context_transitions :]
            if float(done_prob.item()) > 0.5:
                break
        if h is None:  # defensive only; horizon is validated by play.py
            h = self._swm.encode_history(team_z, states, own_history, opponent_history)["h"]
        # Value always sees the final imagined state, including the terminal
        # prediction that caused an early stop.
        h = self._swm.encode_history(team_z, imagined_states, imagined_own, imagined_opp)["h"]
        probs = F.softmax(self._swm.value_head(h).float(), dim=-1)[0]
        p_win = float(probs[OUTCOME_CLASSES.index("win")])
        p_loss = float(probs[OUTCOME_CLASSES.index("loss")])
        return p_win - p_loss, p_win, p_loss

    @torch.no_grad()
    def _simple_world_model_diagnostics(self, hist, action_names: dict[int, str]) -> dict:
        team_z, states, own_history, opp_history, token_count = self._observed_latent_history(hist)
        history = self._swm.encode_history(team_z, states, own_history, opp_history)
        h = history["h"]
        z_t = states[:, -1]
        action_indices = list(action_names)
        legal_ids = torch.as_tensor(
            [self._action_vocabulary.encode(action_names[index], fmt=self._fmt) for index in action_indices],
            dtype=torch.long, device=h.device,
        ).unsqueeze(0)
        legal_mask = torch.ones_like(legal_ids, dtype=torch.bool)
        controller_logits = self._swm.forward_c(
            z_t=z_t, h_t=h, legal_action_ids=legal_ids, legal_action_mask=legal_mask
        )["controller_logits"][0].float()
        bc_probs = F.softmax(controller_logits, dim=-1)
        bc_min, bc_max = bc_probs.min(), bc_probs.max()
        bc_normalized = (bc_probs - bc_min) / (bc_max - bc_min).clamp_min(1e-6)
        rows = []
        for row, (action_index, action_id) in enumerate(zip(action_indices, legal_ids[0].tolist(), strict=True)):
            values: list[float] = []
            wins: list[float] = []
            losses: list[float] = []
            for _ in range(self._rollouts_per_action):
                value, p_win, p_loss = self._rollout_value(
                    team_z=team_z, states=states, own_history=own_history, opponent_history=opp_history,
                    proposed_own_action=int(action_id),
                )
                values.append(value); wins.append(p_win); losses.append(p_loss)
            values_t = torch.tensor(values)
            risk_adjusted = float(values_t.mean() - 0.25 * values_t.std(unbiased=False))
            rollout_score = (risk_adjusted + 1.0) / 2.0
            blend = 0.75 * rollout_score + 0.25 * float(bc_normalized[row])
            rows.append({
                "row_idx": row,
                "idx": action_index,
                "name": action_names[action_index],
                "score": blend,
                "prob": float(bc_probs[row]),
                "bc_score": float(bc_normalized[row]),
                "rollout_value": risk_adjusted,
                "p_win": float(np.mean(wins)),
                "p_loss": float(np.mean(losses)),
                "p_ongoing": 0.0,
                "next_norm": 0.0,
            })
        return {
            "scorer": "rollout_plus_bc",
            "z_norm": float(z_t.float().norm(dim=-1).mean()),
            "h_norm": float(h.float().norm(dim=-1).mean()),
            "history_token_count": token_count,
            "team_token_count": int(len(hist.state_blocks[0])),
            "current_state_token_count": int(len(hist.state_blocks[-1])),
            "rows": rows,
        }

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
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
        best_pos, best = max(enumerate(rows), key=lambda item: item[1]["score"])
        best_idx = int(best["idx"])
        ctx = TurnContext(
            battle_tag=battle.battle_tag, turn=battle.turn,
            opponent_name=getattr(battle, "opponent_username", "?"),
            active_species=_display_species(battle.active_pokemon),
            active_hp=_display_hp(getattr(battle.active_pokemon, "current_hp_fraction", None)),
            opponent_species=_display_species(battle.opponent_active_pokemon),
            opponent_hp=_display_hp(getattr(battle.opponent_active_pokemon, "current_hp_fraction", None)),
            scorer="rollout_plus_bc", forced_switch=hist.current_forced_switch,
            num_state_blocks=len(hist.state_blocks), num_player_actions=len(hist.player_actions),
            num_opponent_actions=len(hist.opponent_actions),
        )
        diagnostic = TuiDiagnostic(
            columns=["action", "name", "blend", "BC", "rollout", "Pwin", "Ploss"],
            rows=[[str(row["idx"]), row["name"], f"{row['score']:.3f}", f"{row['prob']:.3f}",
                   f"{row['rollout_value']:+.3f}", f"{row['p_win']:.3f}", f"{row['p_loss']:.3f}"] for row in rows],
            chosen_idx=best_pos, chosen_label=action_names.get(best_idx, "?"),
            context_lines=[
                f"latent history tokens={diag['history_token_count']}; |z|={diag['z_norm']:.3f}; |h|={diag['h_norm']:.3f}",
                "score = 0.75 risk-adjusted M rollout value + 0.25 normalized C prior",
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
