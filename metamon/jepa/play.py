"""Play Pokémon Showdown battles with JEPA world-model diagnostics.

JEPA is a latent predictive model: it predicts opponent-action embeddings and
next-history embeddings, but it does not decode a next state into text.  This
player therefore uses JEPA for introspection and latent rollouts while choosing
the action whose predicted next-state embedding is farthest from the current
state embedding.  The verbose logs show:

  - the block history being encoded,
  - JEPA's predicted opponent-action embedding norm,
  - each legal action's encoded text and latent next-state delta,
  - the max-delta action selected for play.

Usage:
    uv run python -m metamon.jepa.play \\
        --checkpoint /workspace/poke-datasets/jepa-checkpoints/best.pt \\
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

from metamon.backend.replay_parser.str_parsing import clean_name, move_name, pokemon_name
from metamon.backend.showdown_dex import Dex
from metamon.data.download import METAMON_CACHE_DIR
from metamon.env import get_metamon_teams
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalState, consistent_move_order, consistent_pokemon_order
from metamon.jepa.model import JEPAModel
from metamon.jepa.train import collate_fn
from metamon.tokenizer import PokemonTokenizer


def _safe_token(tokenizer: PokemonTokenizer, word: str) -> str:
    """Return *word* if it is known, otherwise ``<unk>``."""
    return word if word in tokenizer else "<unk>"


def _hp_token(value: Optional[float]) -> str:
    if value is None:
        return "1.00"
    return f"{max(0.0, min(1.0, float(value))):.2f}"


def _status_token(pokemon) -> str:
    status = getattr(pokemon, "status", None)
    if status is None:
        return "nostatus"
    raw = getattr(status, "value", None)
    if raw is None:
        raw = getattr(status, "name", None)
    if raw is None:
        raw = str(status)
    name = str(raw).lower()
    if name in {"0", "none", "null"}:
        return "nostatus"
    return clean_name(name) or "nostatus"


def _species_token(pokemon) -> str:
    if pokemon is None:
        return "unknown"
    return pokemon_name(getattr(pokemon, "species", None) or getattr(pokemon, "base_species", "unknown"))


def _move_token(move) -> str:
    if move is None:
        return "unknown"
    return move_name(getattr(move, "id", None) or getattr(move, "entry", None) or str(move))


def _move_type(move) -> str:
    typ = getattr(move, "type", None)
    if typ is None:
        return "normal"
    raw = getattr(typ, "name", None)
    if raw is None:
        raw = getattr(typ, "value", None)
    if raw is None:
        raw = str(typ)
    return clean_name(str(raw)) or "normal"


def _move_category(move) -> str:
    category = getattr(move, "category", None)
    if category is None:
        return "physical"
    raw = getattr(category, "name", None)
    if raw is None:
        raw = getattr(category, "value", None)
    if raw is None:
        raw = str(category)
    return clean_name(str(raw)) or "physical"


def _pokemon_types(pokemon) -> list[str]:
    types = []
    for typ in getattr(pokemon, "types", []) or []:
        if typ is None:
            continue
        raw = getattr(typ, "name", None)
        if raw is None:
            raw = getattr(typ, "value", None)
        if raw is None:
            raw = str(typ)
        types.append(clean_name(str(raw)))
    return types or ["normal"]


def _tokenize_words(tokenizer: PokemonTokenizer, words: list[str]) -> np.ndarray:
    return np.array([tokenizer[_safe_token(tokenizer, word)] for word in words], dtype=np.int16)


def _format_action_text(action_name: str) -> str:
    if action_name.startswith("move: "):
        return move_name(action_name.removeprefix("move: "))
    if action_name.startswith("tera+move: "):
        return move_name(action_name.removeprefix("tera+move: "))
    if action_name.startswith("switch: "):
        return f"switch {pokemon_name(action_name.removeprefix('switch: '))}"
    if action_name.startswith("opponent move: "):
        return move_name(action_name.removeprefix("opponent move: "))
    if action_name.startswith("opponent switch: "):
        return f"switch {pokemon_name(action_name.removeprefix('opponent switch: '))}"
    return clean_name(action_name) or "unknown"


def _action_block(
    tokenizer: PokemonTokenizer,
    action_text: str,
    *,
    opponent: bool,
) -> np.ndarray:
    if opponent:
        start = "<opponent_chosen_move>"
        end = "<end_opponent_chosen_move>"
    else:
        start = "<chosen_move>"
        end = "<end_chosen_move>"
    words = [start, *_format_action_text(action_text).split(), end]
    return _tokenize_words(tokenizer, words)


def _state_block(tokenizer: PokemonTokenizer, battle: AbstractBattle, fmt: str) -> np.ndarray:
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon

    words: list[str] = [
        "<bos>",
        "<format>", fmt, "<end_format>",
        "<turn>", str(max(1, battle.turn)), "<end_turn>",
        "<arena>",
        "<active>",
        _species_token(active),
        _hp_token(getattr(active, "current_hp_fraction", None)),
        *_pokemon_types(active),
        "noeffect",
        _status_token(active),
        "noboosts",
        "<end_active>",
        "<opponent>",
        _species_token(opponent),
        _hp_token(getattr(opponent, "current_hp_fraction", None)),
        *_pokemon_types(opponent),
        "noeffect",
        _status_token(opponent),
        "noboosts",
        "<end_opponent>",
        "<end_arena>",
        "<begin_moves>",
    ]

    moves = consistent_move_order(list(getattr(active, "moves", {}).values()))
    for move in moves[:4]:
        words.extend([
            "<move>",
            _move_token(move),
            _move_type(move),
            _move_category(move),
            "<end_move>",
        ])
    words.append("<end_moves>")

    words.append("<bench>")
    bench = [p for p in battle.team.values() if p is not None and not p.active]
    for pokemon in consistent_pokemon_order(bench):
        words.extend([
            "<you>",
            _species_token(pokemon),
            _hp_token(getattr(pokemon, "current_hp_fraction", None)),
            *_pokemon_types(pokemon),
            _status_token(pokemon),
            "<end_you>",
        ])
    words.extend([
        "<end_bench>",
        "<conditions>",
        "noweather",
        "nocondition",
        "nocondition",
        "<end_conditions>",
        "<terminal>",
        "ongoing",
        "<end_terminal>",
        "<eos>",
    ])
    return _tokenize_words(tokenizer, words)


def _team_header_block(tokenizer: PokemonTokenizer, battle: AbstractBattle) -> np.ndarray:
    words = ["<begin_team>"]
    for i, pokemon in enumerate(consistent_pokemon_order(list(battle.team.values()))[:6], start=1):
        words.append(f"<poke{i}>")
        words.append(_species_token(pokemon))
        words.extend(_pokemon_types(pokemon))
        words.append("<begin_moves>")
        moves = consistent_move_order(list(getattr(pokemon, "moves", {}).values()))
        for move in moves[:4]:
            words.extend(["<move>", _move_token(move), "<end_move>"])
        words.extend(["<end_moves>", f"<end_poke{i}>"])
    words.append("<end_team>")
    return _tokenize_words(tokenizer, words)


@dataclass
class BattleHistory:
    state_blocks: list[np.ndarray] = field(default_factory=list)
    player_actions: list[np.ndarray] = field(default_factory=list)
    opponent_actions: list[np.ndarray] = field(default_factory=list)
    pending_player_action: Optional[np.ndarray] = None
    pending_player_action_text: Optional[str] = None
    last_turn: int = -1


class JEPAWorldModelPlayer(MetamonPlayer):
    """Showdown player that uses JEPA for latent diagnostics and action logging."""

    def __init__(
        self,
        *args,
        model: JEPAModel,
        tokenizer: PokemonTokenizer,
        fmt: str,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._jepa = model
        self._tokenizer = tokenizer
        self._fmt = fmt
        self._verbose = verbose
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
            hist.state_blocks.append(_team_header_block(self._tokenizer, battle))
            hist.last_turn = -1
            self._histories[battle.battle_tag] = hist

        if hist.last_turn < 0:
            hist.state_blocks.append(_state_block(self._tokenizer, battle, self._fmt))
            hist.last_turn = battle.turn
            return hist

        if battle.turn != hist.last_turn:
            if hist.pending_player_action is not None:
                hist.player_actions.append(hist.pending_player_action)
                opp_text = self._infer_opponent_previous_action(battle)
                hist.opponent_actions.append(
                    _action_block(self._tokenizer, opp_text, opponent=True)
                )
                hist.pending_player_action = None
                hist.pending_player_action_text = None
            hist.state_blocks.append(_state_block(self._tokenizer, battle, self._fmt))
            hist.last_turn = battle.turn
        return hist

    def _legal_action_indices(self, battle: AbstractBattle) -> list[int]:
        state = UniversalState.from_Battle(battle)
        legal = UniversalAction.definitely_valid_actions(state, battle)
        return sorted(action.action_idx for action in legal)

    def _resolve_action_names(self, battle: AbstractBattle, action_indices: list[int]) -> dict[int, str]:
        valid_moves = {m.id for m in battle.available_moves}
        if valid_moves == {"recharge"}:
            move_names = ["recharge"] * 4
        elif valid_moves == {"struggle"}:
            move_names = ["struggle"] * 4
        elif "fight" in valid_moves:
            move_names = ["fight"] * 4
        else:
            move_names = [m.id for m in consistent_move_order(list(battle.active_pokemon.moves.values()))]

        if not battle.reviving:
            switches = consistent_pokemon_order([p for p in battle.team.values() if not p.fainted and not p.active])
        else:
            switches = consistent_pokemon_order([p for p in battle.team.values() if p.fainted and not p.active])
        switch_names = [p.species for p in switches]

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
            return f"opponent move: {_move_token(prev)}"
        return "unknown"

    def _encode_sample(self, hist: BattleHistory):
        return collate_fn([
            (
                hist.state_blocks,
                hist.state_blocks,
                hist.player_actions,
                hist.opponent_actions,
                hist.player_actions,
                hist.opponent_actions,
                _action_block(self._tokenizer, "unknown", opponent=False),
                _action_block(self._tokenizer, "unknown", opponent=True),
            )
        ], pad_id=self._tokenizer.pad_token_id)

    @torch.no_grad()
    def _jepa_diagnostics(self, battle: AbstractBattle, hist: BattleHistory, action_names: dict[int, str]):
        device = next(self._jepa.parameters()).device
        batch = [tensor.to(device) for tensor in self._encode_sample(hist)]
        outputs = self._jepa(*batch)
        enc_N = outputs["enc_N"]
        pred_o = outputs["pred_o_emb"]
        pred_o_norm = torch.norm(pred_o, dim=-1).item()

        rows = []
        for idx, name in action_names.items():
            pa = _action_block(self._tokenizer, name, opponent=False)
            pa_tensor = torch.from_numpy(pa.astype(np.int64)).unsqueeze(0).to(device)
            pa_emb = self._jepa.action_encoder(pa_tensor)
            pred_next = self._jepa.predictor(enc_N, pa_emb, pred_o)
            latent_delta = torch.norm(pred_next - enc_N, dim=-1).item()
            rows.append((idx, name, latent_delta))
        return pred_o_norm, rows

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        legal_actions = self._legal_action_indices(battle)
        if not legal_actions:
            return self.choose_random_move(battle)

        hist = self._history(battle)
        action_names = self._resolve_action_names(battle, legal_actions)
        pred_o_norm, rows = self._jepa_diagnostics(battle, hist, action_names)

        best_idx = max(rows, key=lambda row: row[2])[0]

        if self._verbose:
            print(f"\n── JEPA turn {battle.turn} ({battle.battle_tag}) ──")
            print(f"  blocks: states={len(hist.state_blocks)} player_actions={len(hist.player_actions)} opponent_actions={len(hist.opponent_actions)}")
            print(f"  active: {_species_token(battle.active_pokemon)} hp={_hp_token(getattr(battle.active_pokemon, 'current_hp_fraction', None))}")
            print(f"  opponent: {_species_token(battle.opponent_active_pokemon)} hp={_hp_token(getattr(battle.opponent_active_pokemon, 'current_hp_fraction', None))}")
            self._print_history_blocks(hist)
            print(f"  predicted opponent action embedding norm: {pred_o_norm:.4f}")
            print("  legal actions:")
            for idx, name, latent_delta in rows:
                marker = " <- chosen" if idx == best_idx else ""
                print(f"    {idx:3d} {name:28s} latent_delta={latent_delta:7.3f}{marker}")

        order = UniversalAction.action_idx_to_BattleOrder(battle, action_idx=best_idx)
        if order is None:
            return self.choose_random_move(battle)

        hist.pending_player_action = _action_block(self._tokenizer, action_names[best_idx], opponent=False)
        hist.pending_player_action_text = action_names[best_idx]
        return order


async def main() -> None:
    parser = argparse.ArgumentParser(description="Play Showdown with JEPA diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=os.path.join(METAMON_CACHE_DIR, "tokenizers", "WorldModelObservationSpace-v1.json"))
    parser.add_argument("--format", type=str, nargs="+", default=["gen1ou"])
    parser.add_argument("--username", default="JEPABot")
    parser.add_argument("--num_battles", type=int, default=5)
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--quiet", action="store_true")
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

    model = JEPAModel(
        vocab_size=ckpt.get("vocab_size", len(tokenizer)),
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer["<bos>"],
        eos_id=tokenizer["<eos>"],
        latent_dim=model_cfg.get("latent_dim", 192),
        action_latent_dim=model_cfg.get("action_latent_dim", 32),
        encoder_cfg=model_cfg.get("encoder", {}),
        temporal_encoder_cfg=model_cfg.get("temporal_encoder", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        action_predictor_cfg=model_cfg.get("action_predictor", {}),
        predictor_cfg=model_cfg.get("predictor", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded JEPA checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    players = []
    for fmt in args.format:
        Dex.from_format(fmt)
        team_set = get_metamon_teams(fmt, args.team_set)
        fmt_short = fmt.replace("gen", "g").replace("ou", "")
        username = f"{args.username}-{fmt_short}" if len(args.format) > 1 else args.username
        player = JEPAWorldModelPlayer(
            model=model,
            tokenizer=tokenizer,
            fmt=fmt,
            verbose=not args.quiet,
            account_configuration=AccountConfiguration(username, None),
            battle_format=fmt,
            team=team_set,
            start_timer_on_battle_start=False,
            max_concurrent_battles=1,
        )
        players.append((fmt, username, player))

    await asyncio.sleep(2)
    print("Bots online:", ", ".join(f"{username} ({fmt})" for fmt, username, _ in players))
    print("Challenge one locally with: /challenge <username>")

    async def accept_for(fmt: str, username: str, player: JEPAWorldModelPlayer):
        await player.accept_challenges(None, args.num_battles)
        print(f"\nResults for {username} ({fmt}):")
        print(f"  Wins: {player.n_won_battles}")
        print(f"  Losses: {player.n_lost_battles}")
        print(f"  Ties: {player.n_tied_battles}")

    await asyncio.gather(*(accept_for(fmt, username, player) for fmt, username, player in players))


if __name__ == "__main__":
    asyncio.run(main())
