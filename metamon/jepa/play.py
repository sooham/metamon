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

from metamon.backend.replay_parser.str_parsing import clean_name, move_name, pokemon_name
from metamon.backend.showdown_dex import Dex
from metamon.data.download import METAMON_CACHE_DIR
from metamon.env import get_metamon_teams
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import UniversalAction, UniversalState, consistent_move_order, consistent_pokemon_order
from metamon.jepa.model import PairedJEPAModel
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
    # Handle poke-env Status enum (int values 1-7)
    name = getattr(status, "name", None)
    if name is not None:
        return clean_name(str(name)) or "nostatus"
    # Handle string status values
    raw = str(status).lower()
    if raw in {"0", "none", "null", ""}:
        return "nostatus"
    return clean_name(raw) or "nostatus"


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


def _boosts_words(pokemon) -> list[str]:
    """Return boost tokens for a Pokémon in training-data format.

    No boosts → ["noboosts"]
    With boosts → ["<boosts>", "spa-1", "spd-1", "<end_boosts>"]

    Gen 1 note: Special is a single stat; spa and spd always change together.
    We mirror whichever of the two poke-env reports.
    """
    boosts = getattr(pokemon, "boosts", None) or {}
    if not boosts:
        return ["noboosts"]
    # Mirror spa↔spd for Gen 1 linked Special stat
    if "spa" in boosts and "spd" not in boosts:
        boosts = dict(boosts, spd=boosts["spa"])
    elif "spd" in boosts and "spa" not in boosts:
        boosts = dict(boosts, spa=boosts["spd"])
    parts = ["<boosts>"]
    has_any = False
    for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]:
        amount = boosts.get(stat, 0)
        if amount != 0:
            parts.append(f"{stat}{amount:+d}")
            has_any = True
    if not has_any:
        return ["noboosts"]
    parts.append("<end_boosts>")
    return parts


def _tokenize_words(tokenizer: PokemonTokenizer, words: list[str]) -> np.ndarray:
    """Tokenize words, replacing unknowns with <unk> and warning."""
    ids = []
    unk_id = tokenizer["<unk>"]
    unknown_words: list[str] = []
    for word in words:
        if word in tokenizer:
            ids.append(tokenizer[word])
        else:
            ids.append(unk_id)
            unknown_words.append(word)
    if unknown_words:
        print(f"WARNING: <unk> tokens for words not in tokenizer: {unknown_words}")
    return np.array(ids, dtype=np.int16)


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


def _state_block(tokenizer: PokemonTokenizer, battle: AbstractBattle, fmt: str, *, is_force_switch: bool = False, override_active=None) -> np.ndarray:
    active = override_active if override_active is not None else battle.active_pokemon
    # Belt-and-suspenders: if active would render as HP 0.00, force forceswitch.
    if not is_force_switch and active is not None:
        if _hp_token(getattr(active, "current_hp_fraction", None)) == "0.00":
            is_force_switch = True
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
    ]

    status_str = _status_token(active)
    boost_words = _boosts_words(active)
    if status_str == "nostatus" and boost_words == ["noboosts"]:
        words.append("clean")
    else:
        words.append("noeffect")
        words.append(status_str)
        words.extend(boost_words)

    words.append("<end_active>")

    words.extend([
        "<opponent>",
        _species_token(opponent),
        _hp_token(getattr(opponent, "current_hp_fraction", None)),
        *_pokemon_types(opponent),
    ])

    opp_status_str = _status_token(opponent)
    opp_boost_words = _boosts_words(opponent)
    if opp_status_str == "nostatus" and opp_boost_words == ["noboosts"]:
        words.append("clean")
    else:
        words.append("noeffect")
        words.append(opp_status_str)
        words.extend(opp_boost_words)

    words.append("<end_opponent>")

    words.extend([
        "<end_arena>",
        "<begin_moves>",
    ])

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
    # Exclude the active Pokémon from bench (poke-env may use different
    # object identities, so compare by species+HP+status as fallback).
    active_obj = override_active if override_active is not None else battle.active_pokemon
    active_species = _species_token(active_obj) if active_obj else ""
    active_hp = _hp_token(getattr(active_obj, "current_hp_fraction", None)) if active_obj else ""
    bench = []
    for p in battle.team.values():
        if p is None:
            continue
        if p is active_obj:
            continue
        if active_obj is not None:
            if _species_token(p) == active_species and _hp_token(getattr(p, "current_hp_fraction", None)) == active_hp:
                continue
        bench.append(p)
    for i, pokemon in enumerate(consistent_pokemon_order(bench), start=1):
        words.extend([
            f"<poke{i}>",
            _species_token(pokemon),
            _hp_token(getattr(pokemon, "current_hp_fraction", None)),
            *_pokemon_types(pokemon),
        ])
        poke_status = _status_token(pokemon)
        if poke_status != "nostatus":
            words.append(poke_status)
        words.append(f"<end_poke{i}>")
    words.append("<end_bench>")
    if is_force_switch:
        words.extend([
            "<conditions>",
            "noweather",
            "<you>",
            "forceswitch",
            "<end_you>",
            "<opponent_empty>",
            "<end_conditions>",
        ])
    else:
        words.extend([
            "<conditions>",
            "<conditions_empty>",
        ])
    words.append("<eos>")
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
    _prev_active = None


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._jepa = model
        self._tokenizer = tokenizer
        self._fmt = fmt
        self._heuristic = heuristic
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
            hist._prev_active = battle.active_pokemon
            hist.state_blocks.append(_state_block(
                self._tokenizer, battle, self._fmt,
                is_force_switch=self._is_force_switch(battle),
            ))
            hist.last_turn = battle.turn
            return hist

        if battle.turn != hist.last_turn:
            if hist.pending_player_action is not None:
                prev_active = hist._prev_active
                hist._prev_active = battle.active_pokemon

                # Record the previous turn's actions first
                hist.player_actions.append(hist.pending_player_action)
                opp_text = self._infer_opponent_previous_action(battle)
                hist.opponent_actions.append(
                    _action_block(self._tokenizer, opp_text, opponent=True)
                )
                hist.pending_player_action = None
                hist.pending_player_action_text = None

                # If the previous active fainted, insert a forceswitch subturn
                # AFTER recording actions (matching training-data ordering).
                if prev_active is not None:
                    # Check if prev_active fainted — use _hp_token for robust HP check,
                    # and also check the bench as a fallback (poke-env may update
                    # the bench before the Pokemon object's HP property).
                    prev_fainted = _hp_token(getattr(prev_active, "current_hp_fraction", None)) == "0.00"
                    if not prev_fainted:
                        prev_species = _species_token(prev_active)
                        for p in battle.team.values():
                            if p is not None and _species_token(p) == prev_species and _status_token(p) == "fnt":
                                prev_fainted = True
                                prev_active = p  # use the bench object (has correct HP)
                                break
                    if prev_fainted:
                        if self._verbose:
                            print(f"  [inserting forceswitch subturn: prev_active {_species_token(prev_active)} fainted]")
                        hist.state_blocks.append(_state_block(
                            self._tokenizer, battle, self._fmt,
                            is_force_switch=True,
                            override_active=prev_active,
                        ))
            hist.state_blocks.append(_state_block(
                self._tokenizer, battle, self._fmt,
                is_force_switch=self._is_force_switch(battle),
            ))
            hist.last_turn = battle.turn
        return hist

    def _is_force_switch(self, battle: AbstractBattle) -> bool:
        """Return True if the bot must switch (fainted active, forced out, etc.)."""
        active = battle.active_pokemon
        if active is None:
            return True
        hp = getattr(active, "current_hp_fraction", None)
        if hp is not None and hp <= 0:
            return True
        # Fallback: check if active is fainted (status == fnt) — more reliable
        # than current_hp_fraction which may lag behind poke-env updates.
        if _status_token(active) == "fnt":
            return True
        # Check if only switches are available and this is NOT a recharge/struggle/fight turn.
        available_moves = getattr(battle, "available_moves", None) or []
        move_ids = {m.id for m in available_moves} if available_moves else set()
        if not move_ids or move_ids == {"recharge"}:
            return False  # recharge turn, not a faint
        if move_ids == {"struggle"} or move_ids == {"fight"}:
            return False  # out of PP or can't act, not a faint
        # Only switches available → forced switch
        legal = self._legal_action_indices(battle)
        result = all(idx >= 4 for idx in legal) if legal else False
        if result and self._verbose:
            print(f"  [forceswitch detected via legal-actions-only: moves={move_ids} legal={legal}]")
        return result

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
            move_names = ["struggle"] * 4
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
            pa = _action_block(self._tokenizer, name, opponent=False)
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
        legal_actions = self._legal_action_indices(battle)
        if not legal_actions:
            return self.choose_random_move(battle)

        hist = self._history(battle)
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
            print(f"  active: {_species_token(battle.active_pokemon)} hp={_hp_token(getattr(battle.active_pokemon, 'current_hp_fraction', None))}")
            print(f"  opponent: {_species_token(battle.opponent_active_pokemon)} hp={_hp_token(getattr(battle.opponent_active_pokemon, 'current_hp_fraction', None))}")
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

        hist.pending_player_action = _action_block(self._tokenizer, action_names[best_idx], opponent=False)
        hist.pending_player_action_text = action_names[best_idx]
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
        fmt="gen1ou",
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
