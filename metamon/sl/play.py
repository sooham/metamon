"""Play Pokémon Showdown battles using a trained WorldModelTransformer.

Loads a world-model checkpoint and battles on the local Showdown server.
At each turn the model predicts the next state for every legal action
(batched for efficiency), then picks the action whose predicted outcome
is best according to terminal tokens and HP advantage.

Usage:
    uv run python -m metamon.sl.play \\
        --checkpoint /path/to/best.pt \\
        --format gen1ou \\
        --username MyWorldModel \\
        --num_battles 5 \\
        --team_set competitive

Requirements:
    - Local Showdown server running (node pokemon-showdown start --no-security)
    - Trained world-model checkpoint
"""

import argparse
import asyncio
import os
import re
import time
from typing import Optional

import numpy as np
import torch
import yaml

from metamon.sl.model import WorldModelTransformer
from metamon.tokenizer import PokemonTokenizer
from metamon.interface import (
    UniversalState,
    UniversalAction,
    WorldModelObservationSpace,
    consistent_pokemon_order,
)
from metamon.env import QueueOnLocalLadder, TeamSet, get_metamon_teams
from metamon.env.metamon_player import MetamonPlayer
from metamon.backend.showdown_dex import Dex
from metamon.data.download import METAMON_CACHE_DIR

from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder


# ── State scoring ───────────────────────────────────────────────────────

def _parse_terminal_token(tokens: list[int], tokenizer: PokemonTokenizer) -> str:
    """Extract the terminal token from a generated sequence."""
    eos = tokenizer["<eos>"]
    pad = tokenizer.pad_token_id
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == eos and i > 0:
            terminal = tokens[i - 1]
            if terminal != pad:
                return tokenizer.detokenize([terminal])[0]
    return "<ongoing>"


def _extract_hp(tokens: list[int], tokenizer: PokemonTokenizer) -> tuple[float, float]:
    """Extract our HP and opponent HP from generated tokens.

    HP in WorldModelObservationSpace is 4 separate tokens per value:
    e.g. '1', '.', '0', '0' for 1.00, or '0', '.', '7', '3' for 0.73.
    We join adjacent digit/dot tokens, match the HP pattern, and return
    the first two HP values found (our active, opponent active).
    """
    our_hp = 1.0
    opp_hp = 1.0
    try:
        text = tokenizer.detokenize(tokens)
        # Walk through tokens, collecting runs of digits and dots
        hp_pattern = re.compile(r"^\d\.\d{2}$")
        hp_values = []
        i = 0
        while i <= len(text) - 4:
            candidate = "".join(text[i : i + 4])
            if hp_pattern.match(candidate):
                hp_values.append(float(candidate))
                i += 4
            else:
                i += 1
        if len(hp_values) >= 2:
            our_hp = hp_values[0]
            opp_hp = hp_values[1]
    except Exception:
        pass
    return our_hp, opp_hp


def score_state(
    generated_tokens: torch.Tensor,  # (len,) — one generated sequence
    tokenizer: PokemonTokenizer,
) -> float:
    """Score a predicted next state. Higher is better.

    - Terminal <won> → +100
    - Terminal <lost> → -100
    - <ongoing> → HP advantage (our_hp - opp_hp)
    """
    tok_list = generated_tokens.tolist()
    terminal = _parse_terminal_token(tok_list, tokenizer)
    if terminal == "<won>":
        return 100.0
    elif terminal == "<lost>":
        return -100.0
    else:
        our_hp, opp_hp = _extract_hp(tok_list, tokenizer)
        return our_hp - opp_hp


# ── World Model Player ──────────────────────────────────────────────────

class WorldModelPlayer(MetamonPlayer):
    """A Showdown player that uses the world model for action selection.

    At each turn, predicts the next state for every legal action (batched)
    and picks the action with the best predicted outcome.
    """

    def __init__(
        self,
        *args,
        model: WorldModelTransformer,
        tokenizer: PokemonTokenizer,
        obs_space: WorldModelObservationSpace,
        dex: Dex,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._wm = model
        self._tokenizer = tokenizer
        self._obs_space = obs_space
        self._dex = dex
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._bos_id = tokenizer["<bos>"]
        self._eos_id = tokenizer["<eos>"]
        self._boa_id = tokenizer["<boa>"]
        self._eoa_id = tokenizer["<eoa>"]

        # Special token IDs used by the model during generation
        self._model_bos = self._bos_id
        self._model_eos = self._eos_id
        self._model_boa = self._boa_id
        self._model_eoa = self._eoa_id

    def _state_to_universal(self, battle: AbstractBattle) -> UniversalState:
        """Convert the current poke-env battle to a UniversalState."""
        return UniversalState.from_Battle(battle)

    def _legal_action_indices(self, battle: AbstractBattle) -> list[int]:
        """Return the list of legal action indices for the current battle state."""
        us = UniversalState.from_Battle(battle)
        legal = UniversalAction.definitely_valid_actions(us, battle)
        return sorted([a.action_idx for a in legal])

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Select an action using the world model."""
        legal_actions = self._legal_action_indices(battle)
        if not legal_actions:
            # Fallback: random legal move via poke-env
            return self.choose_random_move(battle)
        if len(legal_actions) == 1:
            return UniversalAction.action_idx_to_BattleOrder(
                battle, action_idx=legal_actions[0]
            )

        # Get current state tokens
        us = UniversalState.from_Battle(battle)
        obs = self._obs_space.state_to_obs(us)
        state_tokens = self._tokenizer.tokenize(obs["text"].tolist())
        state_t = torch.from_numpy(state_tokens.astype(np.int64)).unsqueeze(0)  # (1, L)

        # Batch all legal actions
        B = len(legal_actions)
        state_t_batch = state_t.repeat(B, 1)  # (B, L)
        actions_batch = torch.tensor(legal_actions, dtype=torch.long)  # (B,)
        state_lens = torch.tensor([state_t.shape[1]] * B, dtype=torch.long)

        # Generate all next states in parallel
        device = next(self._wm.parameters()).device
        state_t_batch = state_t_batch.to(device)
        actions_batch = actions_batch.to(device)
        state_lens = state_lens.to(device)

        with torch.no_grad():
            generated, gen_lens = self._wm.generate(
                state_t_batch,
                actions_batch,
                state_lens,
                bos_id=self._model_bos,
                eos_id=self._model_eos,
                boa_id=self._model_boa,
                eoa_id=self._model_eoa,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )

        # Score each generated next state and pick the best action
        best_score = float("-inf")
        best_action = legal_actions[0]
        for i in range(B):
            length = int(gen_lens[i].item())
            if length == 0:
                continue
            gen_tokens = generated[i, :length]
            score = score_state(gen_tokens.cpu(), self._tokenizer)
            if score > best_score:
                best_score = score
                best_action = legal_actions[i]

        # Convert to BattleOrder
        order = UniversalAction.action_idx_to_BattleOrder(
            battle, action_idx=best_action
        )
        if order is None:
            return self.choose_random_move(battle)
        return order


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Play Pokémon Showdown with a trained WorldModel."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to world model checkpoint (.pt).")
    parser.add_argument("--format", type=str, default="gen1ou",
                        choices=["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou"],
                        help="Battle format.")
    parser.add_argument("--username", type=str, default="WorldModelBot",
                        help="Username on the local Showdown server.")
    parser.add_argument("--num_battles", type=int, default=5,
                        help="Number of battles to play.")
    parser.add_argument("--team_set", type=str, default="competitive",
                        help="Team set to use (competitive, elite, etc.).")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Max tokens to generate per rollout.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
                        help="Model config YAML.")
    args = parser.parse_args()

    # ---- Load checkpoint ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model_cfg = ckpt.get("config", {})
    if not model_cfg:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})

    vocab_size = ckpt.get("vocab_size", model_cfg.get("vocab_size", 491))
    bos_id = ckpt.get("bos_id", 30)
    eos_id = ckpt.get("eos_id", 33)
    boa_id = ckpt.get("boa_id", 28)
    eoa_id = ckpt.get("eoa_id", 32)

    # ---- Tokenizer ----
    tokenizer = PokemonTokenizer()
    tok_path = os.path.join(
        METAMON_CACHE_DIR, "tokenizers", "WorldModelObservationSpace-v1.json"
    )
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tokenizer.load_tokens_from_disk(tok_path)
    pad_id = tokenizer.pad_token_id

    action_to_token_id = {
        i: tokenizer.get_action_token_id(i) for i in range(-1, 13)
    }

    # ---- Model ----
    d_model = model_cfg.get("d_model", 256)
    n_heads = model_cfg.get("n_heads", 8)
    n_layers = model_cfg.get("n_layers", 6)
    d_ff = model_cfg.get("d_ff", 1024)
    dropout = model_cfg.get("dropout", 0.1)
    max_seq_len = model_cfg.get("max_seq_len", 2048)

    model = WorldModelTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        action_to_token_id=action_to_token_id,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ---- Dex ----
    dex = Dex.from_format(args.format)

    # ---- Observation space ----
    obs_space = WorldModelObservationSpace()

    # ---- Team set ----
    team_set = get_metamon_teams(args.format, args.team_set)

    # ---- Create player ----
    player = WorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        obs_space=obs_space,
        dex=dex,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        battle_format=args.format,
        team=team_set.sample_team(),
        start_timer_on_battle_start=True,
        max_concurrent_battles=1,
        username=args.username,
    )

    # ---- Battle ----
    print(f"Starting {args.num_battles} battles as '{args.username}' on the ladder...")
    print("(Make sure the local Showdown server is running)")
    await player.ladder(args.num_battles)

    # Print results
    print(f"\nResults for {args.username}:")
    print(f"  Wins: {player.n_won_battles}")
    print(f"  Losses: {player.n_lost_battles}")
    print(f"  Ties: {player.n_tied_battles}")
    if player.n_finished_battles > 0:
        win_rate = player.n_won_battles / player.n_finished_battles * 100
        print(f"  Win rate: {win_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
