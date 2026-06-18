"""Play Pokémon Showdown battles with JEPA world-model diagnostics.

Paired JEPA is a latent predictive model: from this player's visible POV it
predicts the hidden opponent POV, the opponent's next action embedding, and the
next visible POV embedding. It does not decode a next state into text. This
player therefore uses JEPA for introspection and latent rollouts while choosing
the action whose predicted next-state embedding is farthest from the current
state embedding. The verbose logs show:

  - the block history being encoded,
  - JEPA's predicted opponent-state/action embedding norms,
  - each legal action's encoded text and latent next-state delta,
  - the max-delta action selected for play.

Usage:
    uv run python -m metamon.jepa.play \\
        --checkpoint /workspace/poke-datasets/jepa-checkpoints/paired_best.pt \\
        --tokenizer_path /workspace/poke-datasets/tokenizers/WorldModelObservationSpace-v1.json \\
        --format gen1ou \\
        --username JEPABot
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import yaml

from poke_env.environment import AbstractBattle
from poke_env.ps_client import AccountConfiguration
from poke_env.player import BattleOrder

from metamon.backend.replay_parser.str_parsing import move_name
from metamon.data.download import METAMON_CACHE_DIR
from metamon.env import get_metamon_teams
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalState, consistent_move_order, consistent_pokemon_order
from metamon.jepa.model import PairedJEPAModel
from metamon.jepa.online_serializer import (
    action_block,
    is_force_switch_state,
    state_block,
    team_context_block,
)
from metamon.tokenizer import PokemonTokenizer


def _display_hp(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{max(0.0, min(1.0, float(value))):.2f}"


def _display_species(pokemon) -> str:
    if pokemon is None:
        return "unknown"
    return str(getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None) or "unknown")


@dataclass
class BattleHistory:
    state_blocks: list[np.ndarray] = field(default_factory=list)
    player_actions: list[np.ndarray] = field(default_factory=list)
    opponent_actions: list[np.ndarray] = field(default_factory=list)
    pending_player_action: Optional[np.ndarray] = None
    pending_player_action_text: Optional[str] = None
    pending_forced_switch: bool = False
    last_state_key: Optional[tuple[int, ...]] = None
    current_forced_switch: bool = False


class JEPAWorldModelPlayer(MetamonPlayer):
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
        self._verbose_blocks = verbose_blocks
        self._histories: dict[str, BattleHistory] = {}

    def _detok(self, tokens: np.ndarray) -> str:
        words = self._tokenizer.detokenize(tokens.tolist())
        return " ".join(words)

    def _print_history_blocks(self, hist: BattleHistory) -> None:
        print("  JEPA input blocks:")
        for i, block in enumerate(hist.state_blocks):
            label = "team_header" if i == 0 else f"state_{i - 1}"
            print(f"    [{label}] {self._detok(block)}")
            if i >= 1:
                action_idx = i - 1
                if action_idx < len(hist.player_actions):
                    print(f"    [player_action_{action_idx}] {self._detok(hist.player_actions[action_idx])}")
                if action_idx < len(hist.opponent_actions):
                    print(f"    [opponent_action_{action_idx}] {self._detok(hist.opponent_actions[action_idx])}")

    def _history(self, battle: AbstractBattle) -> BattleHistory:
        hist = self._histories.get(battle.battle_tag)
        if hist is None:
            hist = BattleHistory()
            hist.state_blocks.append(team_context_block(self._tokenizer, battle, self._fmt))
            self._histories[battle.battle_tag] = hist

        force_switch = self._is_force_switch(battle)
        current_state = state_block(
            self._tokenizer,
            battle,
            self._fmt,
            is_force_switch=force_switch,
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
                opp_text = (
                    "unknown"
                    if hist.pending_forced_switch
                    else self._infer_opponent_previous_action(battle)
                )
                hist.opponent_actions.append(
                    action_block(self._tokenizer, opp_text, opponent=True)
                )
                hist.pending_player_action = None
                hist.pending_player_action_text = None
                hist.pending_forced_switch = False
            elif self._verbose and self._verbose_blocks:
                print("  [state advanced without a pending player action]")

            hist.state_blocks.append(current_state)
            hist.last_state_key = current_key
        return hist

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

    def _infer_opponent_previous_action(self, battle: AbstractBattle) -> str:
        prev = getattr(battle.opponent_active_pokemon, "previous_move", None)
        if prev is not None:
            raw = getattr(prev, "id", None) or getattr(prev, "name", None) or str(prev)
            return f"opponent move: {move_name(str(raw))}"
        return "unknown"

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

        enc_self = self._jepa.encode_history(
            state_tokens,
            state_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        )
        pred_opponent_state = self._jepa.opponent_state_predictor(enc_self)
        pred_opponent_action = self._jepa.action_predictor(enc_self, pred_opponent_state)
        pred_state_norm = torch.norm(pred_opponent_state, dim=-1).item()
        pred_state_delta = torch.norm(pred_opponent_state - enc_self, dim=-1).item()
        pred_action_norm = torch.norm(pred_opponent_action, dim=-1).item()

        rows = []
        for idx, name in action_names.items():
            pa = action_block(self._tokenizer, name, opponent=False)
            pa_tensor = torch.from_numpy(pa.astype(np.int64)).unsqueeze(0).to(device)
            own_action = self._jepa.action_encoder(pa_tensor)
            pred_next = self._jepa.next_state_predictor(
                enc_self,
                pred_opponent_state,
                own_action,
                pred_opponent_action,
            )
            if self._heuristic == "max-opponent-state-delta":
                next_pred_opponent = self._jepa.opponent_state_predictor(pred_next)
                delta = torch.norm(next_pred_opponent - pred_opponent_state, dim=-1).item()
            else:
                delta = torch.norm(pred_next - enc_self, dim=-1).item()
            rows.append((idx, name, delta))
        return pred_state_norm, pred_state_delta, pred_action_norm, rows

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        hist = self._history(battle)
        legal_actions = self._legal_action_indices(battle)
        if not legal_actions:
            return self.choose_random_move(battle)

        action_names = self._resolve_action_names(battle, legal_actions)
        pred_state_norm, pred_state_delta, pred_action_norm, rows = self._jepa_diagnostics(
            battle, hist, action_names
        )

        best_idx = max(rows, key=lambda row: row[2])[0]

        if self._verbose:
            opponent_name = getattr(battle, "opponent_username", "?")
            replay_url = f"https://replay.pokemonshowdown.com/{battle.battle_tag}"
            print(f"\n── JEPA turn {battle.turn} ({battle.battle_tag}) [{self._heuristic}] ──")
            print(f"  vs: {opponent_name}  |  replay: {replay_url}")
            print(
                f"  active: {_display_species(battle.active_pokemon)} "
                f"hp={_display_hp(getattr(battle.active_pokemon, 'current_hp_fraction', None))}"
            )
            print(
                f"  opponent: {_display_species(battle.opponent_active_pokemon)} "
                f"hp={_display_hp(getattr(battle.opponent_active_pokemon, 'current_hp_fraction', None))}"
            )
            if hist.current_forced_switch:
                print("  state: forceswitch")
            if self._verbose_blocks:
                print(f"  blocks: states={len(hist.state_blocks)} player_actions={len(hist.player_actions)} opponent_actions={len(hist.opponent_actions)}")
                self._print_history_blocks(hist)
                print(f"  predicted opponent state embedding norm: {pred_state_norm:.4f}  delta_from_self={pred_state_delta:.4f}")
                print(f"  predicted opponent action embedding norm: {pred_action_norm:.4f}")
                print("  legal actions:")
                for idx, name, latent_delta in rows:
                    marker = " <- chosen" if idx == best_idx else ""
                    print(f"    {idx:3d} {name:28s} delta={latent_delta:7.3f}{marker}")
            else:
                # Minimal: just the chosen action
                chosen_name = action_names.get(best_idx, "?")
                print(f"  chosen: {chosen_name}")

        order = UniversalAction.action_idx_to_BattleOrder(battle, action_idx=best_idx)
        if order is None:
            return self.choose_random_move(battle)

        hist.pending_player_action = action_block(self._tokenizer, action_names[best_idx], opponent=False)
        hist.pending_player_action_text = action_names[best_idx]
        hist.pending_forced_switch = hist.current_forced_switch
        return order


async def main() -> None:
    parser = argparse.ArgumentParser(description="Play Showdown with JEPA diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=os.path.join(METAMON_CACHE_DIR, "tokenizers", "WorldModelObservationSpace-v1.json"))
    parser.add_argument("--format", type=str, default="gen1ou",
                        help="Battle format (default: gen1ou).")
    parser.add_argument("--username", default="JEPABot")
    parser.add_argument("--num_battles", type=int, default=30,
                        help="Number of battles to play (default: 30).")
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose_blocks", action="store_true",
                        help="Print full JEPA input state/action blocks for each decision.")
    parser.add_argument("--ladder", action="store_true",
                        help="Search for random ladder battles instead of waiting for challenges.")
    parser.add_argument("--server", default="localhost",
                        choices=["localhost", "showdown"],
                        help="Server to connect to (default: localhost).")
    parser.add_argument("--password", default=None,
                        help="Showdown password (required for real server).")
    parser.add_argument("--heuristic", default="max-self-state-delta",
                        choices=["max-self-state-delta", "max-opponent-state-delta"],
                        help="Action selection heuristic (default: max-self-state-delta).")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config")
    if not model_cfg:
        with open(args.config, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]

    model = PairedJEPAModel(
        vocab_size=ckpt.get("vocab_size", len(tokenizer)),
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer["<bos>"],
        eos_id=tokenizer["<eos>"],
        latent_dim=model_cfg.get("latent_dim", 192),
        action_latent_dim=model_cfg.get("action_latent_dim", 32),
        encoder_cfg=model_cfg.get("encoder", {}),
        temporal_encoder_cfg=model_cfg.get("temporal_encoder", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        opponent_state_predictor_cfg=model_cfg.get("opponent_state_predictor", {}),
        action_predictor_cfg=model_cfg.get("paired_action_predictor", model_cfg.get("action_predictor", {})),
        next_state_predictor_cfg=model_cfg.get("next_state_predictor", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded paired JEPA checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create lightweight bot instances sharing one model.
    # gen1ou needs a competitive team; gen1randombattle gets team from server.
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration, ShowdownServerConfiguration
    server_config = ShowdownServerConfiguration if args.server == "showdown" else LocalhostServerConfiguration
    team_set = get_metamon_teams("gen1ou", args.team_set)

    # Generate unique usernames for the real server.
    # With password: use the registered name as-is.
    # Without password: lowercase + random suffix (guest access).
    import random, string
    if args.server == "showdown":
        if args.password:
            username = args.username
            username_rb = args.username + "-rb"
        else:
            suffix = "-" + "".join(random.choices(string.ascii_lowercase, k=6))
            username = args.username.lower() + suffix
            username_rb = args.username.lower() + "-rb" + suffix
    else:
        username = args.username
        username_rb = args.username + "-rb"

    player_ou = JEPAWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt="gen1ou",
        heuristic=args.heuristic,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        account_configuration=AccountConfiguration(username, args.password),
        server_configuration=server_config,
        battle_format="gen1ou",
        team=team_set,
        start_timer_on_battle_start=False,
        max_concurrent_battles=30,
    )
    player_rb = JEPAWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt="gen1randombattle",
        heuristic=args.heuristic,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        account_configuration=AccountConfiguration(username_rb, args.password),
        server_configuration=server_config,
        battle_format="gen1randombattle",
        team=None,
        start_timer_on_battle_start=False,
        max_concurrent_battles=30,
    )

    await asyncio.sleep(2)
    tasks = []
    if args.ladder:
        print(f"Searching for {args.num_battles} gen1ou ladder battles...")
        tasks.append(player_ou.ladder(args.num_battles))
    else:
        print(f"Bot online: {username} (gen1ou)")
        print(f"Challenge with: /challenge {username}, gen1ou")
        tasks.append(player_ou.accept_challenges(None, args.num_battles))

    # Random battle bot always ladders.
    print(f"Searching for {args.num_battles} gen1randombattle ladder battles as {username_rb}...")
    tasks.append(player_rb.ladder(args.num_battles))

    await asyncio.gather(*tasks)
    print(f"\nResults for {username} (gen1ou):")
    print(f"  Wins: {player_ou.n_won_battles}  Losses: {player_ou.n_lost_battles}  Ties: {player_ou.n_tied_battles}")
    print(f"Results for {username_rb} (gen1randombattle):")
    print(f"  Wins: {player_rb.n_won_battles}  Losses: {player_rb.n_lost_battles}  Ties: {player_rb.n_tied_battles}")


if __name__ == "__main__":
    asyncio.run(main())
