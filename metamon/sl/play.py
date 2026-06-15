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
from poke_env.ps_client import AccountConfiguration
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
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._wm = model
        self._tokenizer = tokenizer
        self._obs_space = obs_space
        self._dex = dex
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._verbose = verbose
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

        # Resolve action names for logging
        action_names = self._resolve_action_names(battle, legal_actions)

        # Batch all legal actions
        B = len(legal_actions)
        state_t_batch = state_t.repeat(B, 1)  # (B, L)
        actions_batch = torch.tensor(legal_actions, dtype=torch.long)  # (B,)
        state_lens = torch.tensor([state_t.shape[1]] * B, dtype=torch.long)

        device = next(self._wm.parameters()).device
        state_t_batch = state_t_batch.to(device)
        actions_batch = actions_batch.to(device)
        state_lens = state_lens.to(device)

        t0 = time.time()
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
        gen_time = time.time() - t0

        # Score each generated next state and pick the best action
        scores = []
        best_score = float("-inf")
        best_action = legal_actions[0]
        best_tokens = None
        for i in range(B):
            length = int(gen_lens[i].item())
            if length == 0:
                scores.append((legal_actions[i], float("-inf"), "<empty>"))
                continue
            gen_tokens = generated[i, :length]
            score = score_state(gen_tokens.cpu(), self._tokenizer)
            terminal = _parse_terminal_token(gen_tokens.cpu().tolist(), self._tokenizer)
            scores.append((legal_actions[i], score, terminal))
            if score > best_score:
                best_score = score
                best_action = legal_actions[i]
                best_tokens = gen_tokens.cpu()

        # Log decision
        if self._verbose:
            turn = battle.turn
            print(f"\n── Turn {turn} ──")
            print(f"  Active: {battle.active_pokemon.species} (HP: {battle.active_pokemon.current_hp_fraction:.0%})")
            print(f"  Opponent: {battle.opponent_active_pokemon.species} (HP: {battle.opponent_active_pokemon.current_hp_fraction:.0%})")
            # Print state_t (the input to the model)
            st_detok = self._tokenizer.detokenize(state_tokens.tolist())
            print(f"  Input state_t ({len(state_tokens)} tokens):")
            print(f"    {' '.join(st_detok)}")
            # Print action scores
            print(f"  Legal actions ({B}):")
            for action_idx, score, terminal in scores:
                name = action_names.get(action_idx, f"action_{action_idx}")
                marker = " ← CHOSEN" if action_idx == best_action else ""
                print(f"    {action_idx:3d} {name:30s} score={score:+.3f}  terminal={terminal}{marker}")
            print(f"  Generation time: {gen_time:.1f}s")
            # Print detokenized predicted state for the chosen action
            if best_tokens is not None:
                best_len = int(gen_lens[legal_actions.index(best_action)].item())
                detok = self._tokenizer.detokenize(best_tokens[:best_len].tolist())
                print(f"  Predicted state_t+1 ({best_len} tokens):")
                print(f"    {' '.join(detok)}")

        # Convert to BattleOrder
        order = UniversalAction.action_idx_to_BattleOrder(
            battle, action_idx=best_action
        )
        if order is None:
            return self.choose_random_move(battle)
        return order

    def _resolve_action_names(
        self, battle: AbstractBattle, action_indices: list[int]
    ) -> dict[int, str]:
        """Map action indices to human-readable names.

        Must match the ordering used by :meth:`UniversalAction.action_idx_to_BattleOrder`
        exactly — otherwise the logging will claim the wrong action was chosen.
        """
        from metamon.interface import consistent_move_order, consistent_pokemon_order

        valid_moves = {m.id for m in battle.available_moves}
        if valid_moves == {"recharge"}:
            move_names = ["recharge"] * 4
        elif valid_moves == {"struggle"}:
            move_names = ["struggle"] * 4
        elif "fight" in valid_moves:
            move_names = ["fight"] * 4
        else:
            moves = consistent_move_order(list(battle.active_pokemon.moves.values()))
            move_names = [m.id for m in moves]

        if not battle.reviving:
            switches = consistent_pokemon_order(
                [p for p in battle.team.values() if not p.fainted and not p.active]
            )
        else:
            switches = consistent_pokemon_order(
                [p for p in battle.team.values() if p.fainted and not p.active]
            )
        switch_names = [p.species for p in switches]

        names = {}
        for idx in action_indices:
            if idx == -1:
                names[idx] = "<missing>"
            elif idx == 0:
                if valid_moves == {"recharge"}:
                    names[idx] = "move: recharge"
                elif valid_moves == {"struggle"}:
                    names[idx] = "move: struggle"
                elif "fight" in valid_moves:
                    names[idx] = "move: fight"
                elif len(move_names) > 0:
                    names[idx] = f"move: {move_names[0]}"
                else:
                    names[idx] = "<noop>"
            elif 1 <= idx <= 3:
                i = idx - 1
                if i < len(move_names):
                    names[idx] = f"move: {move_names[i]}"
                else:
                    names[idx] = f"move_{idx}"
            elif 4 <= idx <= 8:
                i = idx - 4
                if i < len(switch_names):
                    names[idx] = f"switch: {switch_names[i]}"
                else:
                    names[idx] = f"switch_{idx}"
            elif 9 <= idx <= 12:
                i = idx - 9
                if i < len(move_names):
                    names[idx] = f"tera+move: {move_names[i]}"
                else:
                    names[idx] = f"tera_move_{idx}"
            else:
                names[idx] = f"action_{idx}"
        return names


# ── Main ─────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Play Pokémon Showdown with a trained WorldModel."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to world model checkpoint (.pt).")
    parser.add_argument("--format", type=str, nargs="+", default=["gen1ou"],
                        help="Battle format(s). Use any Showdown format (gen1ou, gen9randombattle, etc).")
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
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Log per-turn action scores and predicted states (default: on).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-turn logging.")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
                        help="Model config YAML.")
    args = parser.parse_args()

    # ---- Load checkpoint ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

    # ---- Observation space ----
    obs_space = WorldModelObservationSpace()

    # ---- Launch one player per format ----
    players = []
    for fmt in args.format:
        dex = Dex.from_format(fmt)
        team_set = get_metamon_teams(fmt, args.team_set)
        # Each format gets a short suffix (Showdown caps usernames at 18 chars)
        fmt_short = fmt.replace("gen", "g").replace("ou", "")
        fmt_user = f"{args.username}-{fmt_short}" if len(args.format) > 1 else args.username
        player = WorldModelPlayer(
            model=model,
            tokenizer=tokenizer,
            obs_space=obs_space,
            dex=dex,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            verbose=not args.quiet,
            account_configuration=AccountConfiguration(fmt_user, None),
            battle_format=fmt,
            team=team_set,
            start_timer_on_battle_start=False,
            max_concurrent_battles=1,
        )
        players.append((fmt, fmt_user, player))

    # ---- Listen for challenges ----
    await asyncio.sleep(2)
    fmt_list = ", ".join(f"{u} ({f})" for f, u, _ in players)
    print(f"Bots online: {fmt_list}")
    print(f"  Open http://localhost:8000 (via SSH forward)")
    print(f"  Challenge with: /challenge <username>")
    print(f"  Accepting up to {args.num_battles} challenges each...")

    # Accept challenges on all players concurrently
    async def accept_for(fmt: str, username: str, p: WorldModelPlayer):
        await p.accept_challenges(None, args.num_battles)
        print(f"\nResults for {username} ({fmt}):")
        print(f"  Wins: {p.n_won_battles}")
        print(f"  Losses: {p.n_lost_battles}")
        print(f"  Ties: {p.n_tied_battles}")
        if p.n_finished_battles > 0:
            print(f"  Win rate: {p.n_won_battles / p.n_finished_battles * 100:.1f}%")

    await asyncio.gather(*[
        accept_for(fmt, username, player) for fmt, username, player in players
    ])


if __name__ == "__main__":
    asyncio.run(main())
