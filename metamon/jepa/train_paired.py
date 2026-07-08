"""Train paired-POV JEPA on synchronized battle perspectives.

This trainer consumes ``paired_shard_*.npz`` files produced by:

    uv run python scripts/generate_world_model_data.py --rollout_len K ...

For each K-step rollout sample, the dataset provides both player perspectives
with target-excluded history for state T, the target state T, both current
actions, and next state T+1. The active objective is prediction NLL plus
state SIGReg:

1. history context -> current self-state belief latent
2. history context -> hidden opponent-state belief latent
3. sampled self/opponent belief latents -> opponent action belief latent
4. sampled beliefs + own action -> next POV state belief latent
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path
from typing import Iterator

if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{existing + ',' if existing else ''}expandable_segments:True"
    )

import numpy as np
import torch
import yaml

# Optional wandb import
_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass

from metamon.jepa.checkpointing import save_paired_jepa_checkpoint
from metamon.jepa.model import (
    LATENT_DIM,
    SIGREG_DOMAIN,
    SIGREG_NUM_POINTS,
    SIGREG_NUM_SLICES,
    PairedJEPAModel,
    compute_paired_losses,
    format_tensor_debug,
)

from metamon.tokenizer import PokemonTokenizer


class PairedJEPADataset(torch.utils.data.IterableDataset):
    """Iterable paired-POV rollout dataset with pre-computed action blocks.

    On first access each shard is loaded into RAM and action blocks are
    canonicalized without player/opponent role delimiters.  The hot iterator
    path then yields K-step rollout samples as raw numpy views — collation
    handles padding and dtype conversion.
    """

    def __init__(
        self,
        shard_paths: list[str],
        structural_token_ids: dict[str, int],
        shuffle_shards: bool = True,
        max_history_blocks: int = 100,
        include_simple_world_model_fields: bool = False,
    ):
        super().__init__()
        if not shard_paths:
            raise ValueError("No paired shard paths provided")
        self.shard_paths = list(shard_paths)
        self.structural = structural_token_ids
        self.shuffle_shards = shuffle_shards
        self.shuffle_transitions = shuffle_shards
        self.max_history_blocks = max_history_blocks  # 0 = unlimited, default 100
        self.include_simple_world_model_fields = include_simple_world_model_fields

        # Pre-processed shards — populated lazily by _get_shard()
        self._shards: dict[str, dict] = {}

    @staticmethod
    def discover(
        data_root: str,
        formats: list[str],
        split: str,
        *,
        required: bool = True,
    ) -> list[str]:
        """Find paired shard paths under *data_root*.

        Supports two layouts, preferring the current flat/interleaved output:

        1. **Flat / interleaved** (multi-format generation):
           ``data_root/{split}/paired_shard_*.npz``

        2. **Per-format** (single-format / legacy):
           ``data_root/{fmt}/{split}/paired_shard_*.npz``

        When the flat layout exists for a split, it is authoritative. Mixing it
        with per-format directories can accidentally train on stale shards from
        a previous data generation run.
        """
        # Layout 1: flat (interleaved, multi-format)
        flat_paths: list[str] = []
        flat_dir = os.path.join(data_root, split)
        if os.path.isdir(flat_dir):
            for name in sorted(os.listdir(flat_dir)):
                if name.startswith("paired_shard_") and name.endswith(".npz"):
                    flat_paths.append(os.path.join(flat_dir, name))
        if flat_paths:
            return flat_paths

        # Layout 2: per-format (single-format / legacy)
        shard_paths: list[str] = []
        for fmt in formats:
            split_dir = os.path.join(data_root, fmt, split)
            if not os.path.isdir(split_dir):
                continue
            for name in sorted(os.listdir(split_dir)):
                if name.startswith("paired_shard_") and name.endswith(".npz"):
                    shard_paths.append(os.path.join(split_dir, name))

        if required and not shard_paths:
            raise FileNotFoundError(
                f"No paired {split!r} shards found under {data_root} for {formats}"
            )
        return shard_paths

    @staticmethod
    def count_transitions(shard_paths: list[str]) -> int:
        total = 0
        for path in shard_paths:
            data = np.load(path)
            idx = data["p1_target_state_idx"]
            if idx.ndim == 1:
                total += int(len(idx))
            else:
                total += int(idx.shape[0] * idx.shape[1])
        return total

    @staticmethod
    def _resolve_window(
        battle_start: int,
        target_state_idx: int,
        action_base: int,
        max_hist: int,
    ) -> tuple[int, int, int]:
        state_start = battle_start
        if max_hist > 0:
            state_start = max(battle_start + 1, target_state_idx - max_hist)
        # State index 0 within each battle is the team header, so action i
        # connects state i+1 -> state i+2. For target state T, keep actions
        # before T; the current transition action is encoded separately.
        action_start = action_base + max(0, state_start - battle_start - 1)
        action_end = action_base + max(0, target_state_idx - battle_start - 1)
        action_end = max(action_start, action_end)
        return state_start, action_start, action_end

    @staticmethod
    def _canonicalize_actions(
        flat: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        unknown_token: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return action content blocks with no role-specific delimiters.

        World-model shards already store only the content inside the action
        tags.  Empty/missing actions are canonicalized to ``unknown`` when the
        tokenizer id is available so the action encoder never sees an all-pad
        sequence for those blocks.
        """
        n = len(offsets)
        new_lengths = lengths.astype(np.int32, copy=True)
        if unknown_token is not None:
            new_lengths[new_lengths == 0] = 1
        new_offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(new_lengths, out=new_offsets[1:])
        total = int(new_offsets[-1])
        combined = np.empty(total, dtype=np.int16)
        combined[:] = 0
        for i in range(n):
            off = int(offsets[i])
            length = int(lengths[i])
            dest = int(new_offsets[i])
            if length > 0:
                combined[dest : dest + length] = flat[off : off + length]
            elif unknown_token is not None:
                combined[dest] = unknown_token
        return combined, new_offsets, new_lengths

    # Module-level cache: survives across epochs within a worker process.
    # Keyed by (path, mtime) so the cache self-invalidates when data changes.
    _shard_cache: dict[tuple, dict] = {}

    @staticmethod
    def _get_shard(path: str) -> dict:
        """Load shard into RAM, using process-level cache to avoid re-loading
        and re-combining action blocks on every epoch."""
        cache_key = (path, os.path.getmtime(path))
        cached = PairedJEPADataset._shard_cache.get(cache_key)
        if cached is not None:
            return cached
        data = dict(np.load(path))
        PairedJEPADataset._shard_cache[cache_key] = data
        return data

    @staticmethod
    def _ensure_combined(data: dict, unknown_token: int | None = None) -> dict:
        """Idempotently add canonical delimiter-free action arrays to a shard."""
        if "p1_actions_combined" in data:
            return data  # already combined
        for key in [
            "p1_actions",
            "p1_opponent_actions",
            "p2_actions",
            "p2_opponent_actions",
        ]:
            # .npz keys use singular "action": p1_action_offsets, p1_action_lengths
            offs_key = key[:-1] + "_offsets"   # p1_actions → p1_action_offsets
            lens_key = key[:-1] + "_lengths"    # p1_actions → p1_action_lengths
            combined, offs, lens = PairedJEPADataset._canonicalize_actions(
                data[key], data[offs_key], data[lens_key], unknown_token
            )
            data[f"{key}_combined"] = combined
            data[f"{key}_combined_offsets"] = offs
            data[f"{key}_combined_lengths"] = lens
        return data

    @staticmethod
    def _slice_view(flat: np.ndarray, offsets: np.ndarray,
                    lengths: np.ndarray, start: int, end: int) -> list[np.ndarray]:
        """Return list of views (no copy) into flat for blocks [start, end)."""
        end = min(end, len(lengths))  # guard against off-by-one at battle boundaries
        out: list[np.ndarray] = []
        for i in range(start, end):
            off = int(offsets[i])
            length = int(lengths[i])
            out.append(flat[off : off + length])
        return out

    @staticmethod
    def _slice_state_window(
        flat: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        battle_start: int,
        state_start: int,
        state_end: int,
    ) -> list[np.ndarray]:
        """Return a state history, always retaining the team header."""
        sv = PairedJEPADataset._slice_view
        if state_start <= battle_start:
            return sv(flat, offsets, lengths, battle_start, state_end)
        return (
            sv(flat, offsets, lengths, battle_start, battle_start + 1)
            + sv(flat, offsets, lengths, state_start, state_end)
        )

    @staticmethod
    def _iter_shard(data: dict, unknown_token: int | None,
                    shuffle_transitions: bool, max_hist: int,
                    include_simple_world_model_fields: bool = False) -> Iterator[dict]:
        p1_target_state_idx = np.asarray(data["p1_target_state_idx"])
        if p1_target_state_idx.ndim == 1:
            p1_target_state_idx = p1_target_state_idx[:, None]
        p1_next_state_idx = np.asarray(data["p1_next_state_idx"])
        if p1_next_state_idx.ndim == 1:
            p1_next_state_idx = p1_next_state_idx[:, None]
        p1_action_idx = np.asarray(data["p1_action_idx"])
        if p1_action_idx.ndim == 1:
            p1_action_idx = p1_action_idx[:, None]
        p2_target_state_idx = np.asarray(data["p2_target_state_idx"])
        if p2_target_state_idx.ndim == 1:
            p2_target_state_idx = p2_target_state_idx[:, None]
        p2_next_state_idx = np.asarray(data["p2_next_state_idx"])
        if p2_next_state_idx.ndim == 1:
            p2_next_state_idx = p2_next_state_idx[:, None]
        p2_action_idx = np.asarray(data["p2_action_idx"])
        if p2_action_idx.ndim == 1:
            p2_action_idx = p2_action_idx[:, None]
        p1_next_terminal_class = np.asarray(
            data["p1_next_terminal_class"]
            if "p1_next_terminal_class" in data
            else np.zeros_like(p1_target_state_idx, dtype=np.int16)
        )
        if p1_next_terminal_class.ndim == 1:
            p1_next_terminal_class = p1_next_terminal_class[:, None]

        n, rollout_len = p1_target_state_idx.shape
        for name, arr in [
            ("p1_next_state_idx", p1_next_state_idx),
            ("p1_action_idx", p1_action_idx),
            ("p1_next_terminal_class", p1_next_terminal_class),
            ("p2_target_state_idx", p2_target_state_idx),
            ("p2_next_state_idx", p2_next_state_idx),
            ("p2_action_idx", p2_action_idx),
        ]:
            if arr.shape != (n, rollout_len):
                raise ValueError(f"{name} shape {arr.shape} does not match p1_target_state_idx {(n, rollout_len)}")

        order = np.arange(n)
        if shuffle_transitions:
            np.random.default_rng().shuffle(order)

        # Pre-combine action blocks once
        data = PairedJEPADataset._ensure_combined(data, unknown_token)

        for row in order:
            battle_id = int(data["battle_id"][row])
            p1_bs = int(data["p1_battle_start"][battle_id])
            p2_bs = int(data["p2_battle_start"][battle_id])
            p1_as = int(data["p1_battle_action_start"][battle_id])
            p2_as = int(data["p2_battle_action_start"][battle_id])

            w = PairedJEPADataset._resolve_window
            sv = PairedJEPADataset._slice_view
            ssv = PairedJEPADataset._slice_state_window

            sample: dict[str, list] = {
                key: []
                for key in (
                    *BLOCK_KEYS,
                    *SINGLE_BLOCK_KEYS,
                    *ACTION_KEYS,
                )
            }
            sample["battle_id"] = battle_id
            sample["raw_battle_key"] = str(
                data["raw_battle_key"][battle_id]
                if "raw_battle_key" in data else battle_id
            )
            sample["turn_idx"] = np.asarray(data["turn_idx"][row], dtype=np.int32).copy()
            sample["turn_number"] = np.asarray(data["turn_number"][row], dtype=np.int32).copy()
            sample["subturn_idx"] = np.asarray(data["subturn_idx"][row], dtype=np.int32).copy()
            sample["p1_target_state_idx_meta"] = p1_target_state_idx[row].astype(np.int32, copy=True)
            sample["p1_next_state_idx_meta"] = p1_next_state_idx[row].astype(np.int32, copy=True)
            sample["p1_action_idx_meta"] = p1_action_idx[row].astype(np.int32, copy=True)
            sample["p2_target_state_idx_meta"] = p2_target_state_idx[row].astype(np.int32, copy=True)
            sample["p2_next_state_idx_meta"] = p2_next_state_idx[row].astype(np.int32, copy=True)
            sample["p2_action_idx_meta"] = p2_action_idx[row].astype(np.int32, copy=True)
            if include_simple_world_model_fields:
                sample["p1_legal_actions"] = []
                sample["p1_legal_action_mask"] = []
                sample["p1_chosen_legal_action_idx"] = []
                sample["p1_next_terminal_class"] = p1_next_terminal_class[row].astype(np.int16, copy=True)

            for step in range(rollout_len):
                p1_si = int(p1_target_state_idx[row, step])
                p1_nsi = int(p1_next_state_idx[row, step])
                p1_ai = int(p1_action_idx[row, step])
                p2_si = int(p2_target_state_idx[row, step])
                p2_nsi = int(p2_next_state_idx[row, step])
                p2_ai = int(p2_action_idx[row, step])

                p1_hist_s, p1_aT_s, p1_aT_e = w(p1_bs, p1_si, p1_as, max_hist)
                p2_hist_s, p2_aT_s, p2_aT_e = w(p2_bs, p2_si, p2_as, max_hist)

                sample["p1_history_T"].append(ssv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_bs, p1_hist_s, p1_si))
                sample["p2_history_T"].append(ssv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_bs, p2_hist_s, p2_si))
                sample["p1_target_state_T"].append(sv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_si, p1_si + 1)[0])
                sample["p1_next_state_T1"].append(sv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_nsi, p1_nsi + 1)[0])
                sample["p2_target_state_T"].append(sv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_si, p2_si + 1)[0])
                sample["p2_next_state_T1"].append(sv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_nsi, p2_nsi + 1)[0])
                sample["p1_player_hist_T"].append(sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_aT_s, p1_aT_e))
                sample["p1_opponent_hist_T"].append(sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_aT_s, p1_aT_e))
                sample["p2_player_hist_T"].append(sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_aT_s, p2_aT_e))
                sample["p2_opponent_hist_T"].append(sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_aT_s, p2_aT_e))
                sample["p1_action"].append(sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_ai, p1_ai + 1)[0])
                sample["p2_action"].append(sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_ai, p2_ai + 1)[0])
                sample["actual_p2_action_from_p1_perspective"].append(sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_ai, p1_ai + 1)[0])
                sample["actual_p1_action_from_p2_perspective"].append(sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_ai, p2_ai + 1)[0])
                if include_simple_world_model_fields:
                    if "p1_legal_actions" in data:
                        sample["p1_legal_actions"].append(np.asarray(data["p1_legal_actions"][p1_ai], dtype=np.int16))
                        sample["p1_legal_action_mask"].append(np.asarray(data["p1_legal_action_mask"][p1_ai], dtype=np.bool_))
                        chosen_idx = int(np.asarray(data["p1_chosen_legal_action_idx"])[p1_ai])
                    else:
                        # Backward-compatible fallback for older shards: make the
                        # replay action the only legal candidate.
                        sample["p1_legal_actions"].append(np.asarray([sample["p1_action"][-1]], dtype=np.int16))
                        sample["p1_legal_action_mask"].append(np.asarray([True], dtype=np.bool_))
                        chosen_idx = 0
                    sample["p1_chosen_legal_action_idx"].append(chosen_idx)
            yield sample

    def __iter__(self) -> Iterator[dict[str, object]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        unknown_token = self.structural.get("unknown")

        for path in paths:
            # Load once per worker (OS page cache makes repeated loads fast)
            data = self._get_shard(path)
            yield from self._iter_shard(
                data, unknown_token,
                self.shuffle_transitions, self.max_history_blocks,
                self.include_simple_world_model_fields,
            )


BLOCK_KEYS = (
    "p1_history_T",
    "p1_player_hist_T",
    "p1_opponent_hist_T",
    "p2_history_T",
    "p2_player_hist_T",
    "p2_opponent_hist_T",
)
SINGLE_BLOCK_KEYS = (
    "p1_target_state_T",
    "p1_next_state_T1",
    "p2_target_state_T",
    "p2_next_state_T1",
)
ACTION_KEYS = (
    "p1_action",
    "p2_action",
    "actual_p2_action_from_p1_perspective",
    "actual_p1_action_from_p2_perspective",
)
ROLLOUT_METADATA_KEYS = (
    "turn_idx",
    "turn_number",
    "subturn_idx",
    "p1_target_state_idx_meta",
    "p1_next_state_idx_meta",
    "p1_action_idx_meta",
    "p2_target_state_idx_meta",
    "p2_next_state_idx_meta",
    "p2_action_idx_meta",
)


def collate_paired_fn(
    batch: list[dict[str, object]],
    pad_id: int,
) -> dict[str, object]:
    rollout_lengths = {
        len(item["p1_history_T"])  # type: ignore[arg-type,index]
        for item in batch
    }
    if len(rollout_lengths) != 1:
        raise ValueError(f"Mixed rollout lengths in one batch: {sorted(rollout_lengths)}")
    rollout_len = next(iter(rollout_lengths))

    def pad_block_rollouts(
        block_rollouts: list[list[list[np.ndarray]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_blocks = max(
            (len(blocks) for rollout in block_rollouts for blocks in rollout),
            default=0,
        )
        max_tokens = max(
            (len(block) for rollout in block_rollouts for blocks in rollout for block in blocks),
            default=1,
        )
        padded = torch.full(
            (len(block_rollouts), rollout_len, max_blocks, max_tokens),
            pad_id,
            dtype=torch.int32,
        )
        valid = torch.zeros((len(block_rollouts), rollout_len, max_blocks), dtype=torch.bool)
        for batch_idx, rollout in enumerate(block_rollouts):
            for step_idx, blocks in enumerate(rollout):
                for block_idx, block in enumerate(blocks):
                    tokens = torch.from_numpy(block.astype(np.int32, copy=False))
                    padded[batch_idx, step_idx, block_idx, :len(tokens)] = tokens
                    valid[batch_idx, step_idx, block_idx] = True
        return padded, valid

    def pad_action_rollouts(actions: list[list[np.ndarray]]) -> torch.Tensor:
        max_tokens = max(
            (len(action) for rollout in actions for action in rollout),
            default=1,
        )
        padded = torch.full((len(actions), rollout_len, max_tokens), pad_id, dtype=torch.int32)
        for batch_idx, rollout in enumerate(actions):
            for step_idx, action in enumerate(rollout):
                tokens = torch.from_numpy(action.astype(np.int32, copy=False))
                padded[batch_idx, step_idx, :len(tokens)] = tokens
        return padded

    def pad_legal_action_rollouts(
        legal_rollouts: list[list[np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_legal = max(
            (int(candidates.shape[0]) for rollout in legal_rollouts for candidates in rollout),
            default=1,
        )
        max_tokens = max(
            (
                int(candidates.shape[1])
                for rollout in legal_rollouts
                for candidates in rollout
                if candidates.ndim == 2
            ),
            default=1,
        )
        padded = torch.full(
            (len(legal_rollouts), rollout_len, max_legal, max_tokens),
            pad_id,
            dtype=torch.int32,
        )
        mask = torch.zeros((len(legal_rollouts), rollout_len, max_legal), dtype=torch.bool)
        for batch_idx, rollout in enumerate(legal_rollouts):
            for step_idx, candidates in enumerate(rollout):
                arr = np.asarray(candidates, dtype=np.int32)
                if arr.ndim != 2:
                    raise ValueError(f"Expected legal action candidates to be rank-2, got {arr.shape}")
                l_count = min(arr.shape[0], max_legal)
                tok_count = min(arr.shape[1], max_tokens)
                padded[batch_idx, step_idx, :l_count, :tok_count] = torch.from_numpy(arr[:l_count, :tok_count])
                mask[batch_idx, step_idx, :l_count] = True
        return padded, mask

    out: dict[str, object] = {}
    for key in BLOCK_KEYS:
        blocks, valid = pad_block_rollouts([item[key] for item in batch])  # type: ignore[index]
        out[key] = blocks
        out[f"{key}_valid"] = valid
    for key in (*SINGLE_BLOCK_KEYS, *ACTION_KEYS):
        out[key] = pad_action_rollouts([item[key] for item in batch])  # type: ignore[index]
    if "p1_legal_actions" in batch[0]:
        legal_actions, inferred_mask = pad_legal_action_rollouts([item["p1_legal_actions"] for item in batch])  # type: ignore[index]
        out["p1_legal_actions"] = legal_actions
        if "p1_legal_action_mask" in batch[0]:
            legal_masks = [item["p1_legal_action_mask"] for item in batch]  # type: ignore[index]
            explicit_mask = torch.zeros_like(inferred_mask)
            for batch_idx, rollout in enumerate(legal_masks):
                for step_idx, mask_arr in enumerate(rollout):
                    mask_tensor = torch.from_numpy(np.asarray(mask_arr, dtype=np.bool_))
                    count = min(mask_tensor.numel(), explicit_mask.shape[-1])
                    explicit_mask[batch_idx, step_idx, :count] = mask_tensor[:count]
            out["p1_legal_action_mask"] = inferred_mask & explicit_mask
        else:
            out["p1_legal_action_mask"] = inferred_mask
        out["p1_chosen_legal_action_idx"] = torch.tensor(
            np.stack([np.asarray(item["p1_chosen_legal_action_idx"], dtype=np.int64) for item in batch], axis=0),  # type: ignore[index]
            dtype=torch.long,
        )
        out["p1_next_terminal_class"] = torch.tensor(
            np.stack([np.asarray(item["p1_next_terminal_class"], dtype=np.int64) for item in batch], axis=0),  # type: ignore[index]
            dtype=torch.long,
        )
    out["battle_id"] = torch.tensor([int(item["battle_id"]) for item in batch], dtype=torch.int32)
    out["raw_battle_key"] = [str(item["raw_battle_key"]) for item in batch]  # type: ignore[assignment]
    for key in ROLLOUT_METADATA_KEYS:
        out[key] = torch.tensor(np.stack([item[key] for item in batch], axis=0), dtype=torch.int32)  # type: ignore[index]

    return out


def _batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _debug_dump_tensor_mapping(
    title: str,
    tensors: dict[str, object],
    *,
    pad_id: int,
    max_values: int,
    tokenizer: PokemonTokenizer | None = None,
) -> None:
    print(f"\n[train tensor debug] {title}", flush=True)
    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            print(f"  {key}: {value}", flush=True)
            continue
        key_tokenizer = tokenizer if key in (*BLOCK_KEYS, *SINGLE_BLOCK_KEYS, *ACTION_KEYS) else None
        print(
            "  " + format_tensor_debug(
                key,
                value,
                max_values=max_values,
                pad_id=pad_id,
                tokenizer=key_tokenizer,
            ),
            flush=True,
        )


def _ids_and_text(
    tokens: torch.Tensor,
    *,
    tokenizer: PokemonTokenizer,
    pad_id: int,
    max_values: int,
) -> tuple[list[int], str, bool]:
    ids = [
        int(v)
        for v in tokens.detach().cpu().reshape(-1).tolist()
        if int(v) != pad_id
    ]
    truncated = len(ids) > max_values
    preview = ids[:max_values]
    return preview, " ".join(tokenizer.detokenize(preview)), truncated


def _debug_dump_detokenized_batch_inputs(
    title: str,
    batch: dict[str, object],
    *,
    tokenizer: PokemonTokenizer,
    pad_id: int,
    max_values: int,
    max_samples: int,
) -> None:
    print(f"\n[train tensor debug] {title}", flush=True)
    raw_keys = batch.get("raw_battle_key", [])
    battle_ids = batch.get("battle_id")
    turn_numbers = batch.get("turn_number")
    subturn_idx = batch.get("subturn_idx")
    turn_idx = batch.get("turn_idx")

    if not isinstance(battle_ids, torch.Tensor):
        return
    batch_size = int(battle_ids.shape[0])
    rollout_len = int(turn_numbers.shape[1]) if isinstance(turn_numbers, torch.Tensor) and turn_numbers.ndim == 2 else 0
    max_b = min(batch_size, max_samples)

    for b in range(max_b):
        raw_key = raw_keys[b] if isinstance(raw_keys, list) and b < len(raw_keys) else "<unknown>"
        print(
            f"  sample[{b}] raw_battle_key={raw_key} battle_id={int(battle_ids[b].item())} rollout_len={rollout_len}",
            flush=True,
        )
        for k in range(rollout_len):
            parts = []
            if isinstance(turn_numbers, torch.Tensor):
                parts.append(f"turn={int(turn_numbers[b, k].item())}")
            if isinstance(subturn_idx, torch.Tensor):
                parts.append(f"subturn={int(subturn_idx[b, k].item())}")
            if isinstance(turn_idx, torch.Tensor):
                parts.append(f"turn_idx={int(turn_idx[b, k].item())}")
            for key in (
                "p1_target_state_idx_meta",
                "p1_next_state_idx_meta",
                "p1_action_idx_meta",
                "p2_target_state_idx_meta",
                "p2_next_state_idx_meta",
                "p2_action_idx_meta",
            ):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    parts.append(f"{key.removesuffix('_meta')}={int(value[b, k].item())}")
            print(f"    step[{k}] " + " ".join(parts), flush=True)

            for key in BLOCK_KEYS:
                tokens = batch.get(key)
                valid = batch.get(f"{key}_valid")
                if not isinstance(tokens, torch.Tensor) or not isinstance(valid, torch.Tensor):
                    continue
                valid_blocks = valid[b, k].detach().cpu().nonzero(as_tuple=True)[0].tolist()
                print(f"      {key}: blocks={len(valid_blocks)}", flush=True)
                for block_idx in valid_blocks:
                    ids, text, truncated = _ids_and_text(
                        tokens[b, k, int(block_idx)],
                        tokenizer=tokenizer,
                        pad_id=pad_id,
                        max_values=max_values,
                    )
                    suffix = " ..." if truncated else ""
                    print(
                        f"        block[{int(block_idx)}] ids={ids}{suffix} text={text!r}{suffix}",
                        flush=True,
                    )

            for key in (*SINGLE_BLOCK_KEYS, *ACTION_KEYS):
                tokens = batch.get(key)
                if not isinstance(tokens, torch.Tensor):
                    continue
                ids, text, truncated = _ids_and_text(
                    tokens[b, k],
                    tokenizer=tokenizer,
                    pad_id=pad_id,
                    max_values=max_values,
                )
                suffix = " ..." if truncated else ""
                print(f"      {key}: ids={ids}{suffix} text={text!r}{suffix}", flush=True)


def _debug_dump_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n[train tensor debug] {title}", flush=True)
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.8g}", flush=True)


def _load_compatible_checkpoint(
    model: PairedJEPAModel,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Load a checkpoint as a warm start, skipping incompatible tensors.

    ``--checkpoint`` is also the "save best here" path.  After architecture
    changes, that file may exist but contain old predictor heads.  Loading only
    matching keys lets the encoder/action encoder warm-start while new or
    resized Gaussian heads train from initialization.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_state = ckpt["model_state_dict"]
    cleaned = {key.replace("_orig_mod.", ""): value for key, value in raw_state.items()}

    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...] | None]] = []
    for key, value in cleaned.items():
        target = current.get(key)
        if target is None:
            skipped.append((key, tuple(value.shape), None))
            continue
        if tuple(value.shape) != tuple(target.shape):
            skipped.append((key, tuple(value.shape), tuple(target.shape)))
            continue
        compatible[key] = value

    model.load_state_dict(compatible, strict=False)

    missing = [key for key in current if key not in compatible]
    print(
        f"Loaded checkpoint warm start: {checkpoint_path} "
        f"({len(compatible)}/{len(current)} tensors matched)"
    )
    if skipped:
        print(f"  skipped {len(skipped)} incompatible checkpoint tensors:")
        for key, old_shape, new_shape in skipped[:12]:
            if new_shape is None:
                print(f"    {key}: checkpoint shape {old_shape}, not present in current model")
            else:
                print(f"    {key}: checkpoint shape {old_shape}, current shape {new_shape}")
        if len(skipped) > 12:
            print(f"    ... {len(skipped) - 12} more")
    if missing:
        print(f"  initialized {len(missing)} current-model tensors from scratch")
    return ckpt


def _forward_paired(
    model: PairedJEPAModel,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    outputs = model(
        p1_history_T=batch["p1_history_T"],
        p1_history_T_valid=batch["p1_history_T_valid"],
        p1_player_hist_T=batch["p1_player_hist_T"],
        p1_player_hist_T_valid=batch["p1_player_hist_T_valid"],
        p1_opponent_hist_T=batch["p1_opponent_hist_T"],
        p1_opponent_hist_T_valid=batch["p1_opponent_hist_T_valid"],
        p1_target_state_T=batch["p1_target_state_T"],
        p1_next_state_T1=batch["p1_next_state_T1"],
        p1_action_tokens=batch["p1_action"],
        actual_p2_action_from_p1_perspective_tokens=batch["actual_p2_action_from_p1_perspective"],
        p2_history_T=batch["p2_history_T"],
        p2_history_T_valid=batch["p2_history_T_valid"],
        p2_player_hist_T=batch["p2_player_hist_T"],
        p2_player_hist_T_valid=batch["p2_player_hist_T_valid"],
        p2_opponent_hist_T=batch["p2_opponent_hist_T"],
        p2_opponent_hist_T_valid=batch["p2_opponent_hist_T_valid"],
        p2_target_state_T=batch["p2_target_state_T"],
        p2_next_state_T1=batch["p2_next_state_T1"],
        p2_action_tokens=batch["p2_action"],
        actual_p1_action_from_p2_perspective_tokens=batch["actual_p1_action_from_p2_perspective"],
    )
    return outputs




def _make_loader(
    dataset: PairedJEPADataset,
    batch_size: int,
    pad_id: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=functools.partial(collate_paired_fn, pad_id=pad_id),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _build_dataloaders(
    data_root: str,
    formats: list[str],
    structural_ids: dict[str, int],
    max_history_blocks: int,
    batch_size: int,
    pad_id: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader | None,
        list[str],
        list[str],
    ]:
    """Discover shards, build datasets and return train/val DataLoaders."""
    train_shards = PairedJEPADataset.discover(data_root, formats, "train")
    val_shards = PairedJEPADataset.discover(
        data_root, formats, "val", required=False
    )
    train_dataset = PairedJEPADataset(
        train_shards, structural_ids, shuffle_shards=True,
        max_history_blocks=max_history_blocks,
    )
    val_dataset = PairedJEPADataset(
        val_shards, structural_ids, shuffle_shards=False,
        max_history_blocks=max_history_blocks,
    )
    val_loader: torch.utils.data.DataLoader | None = None
    if val_shards:
        val_loader = _make_loader(
            val_dataset,
            batch_size,
            pad_id,
            max(0, num_workers // 2),
            prefetch_factor,
            device.type == "cuda",
        )
    train_loader = _make_loader(
        train_dataset,
        batch_size,
        pad_id,
        num_workers,
        prefetch_factor,
        device.type == "cuda",
    )
    return train_loader, val_loader, train_shards, val_shards


def _auto_detect_device() -> torch.device:
    """Detect the best available torch device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def _load_rollout_len(data_root: str, shard_paths: list[str]) -> int:
    """Return K from generated metadata, falling back to the first shard."""
    metadata_path = os.path.join(data_root, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        rollout_len = int(metadata.get("rollout_len", 1))
        return max(1, rollout_len)

    if shard_paths:
        with np.load(shard_paths[0]) as data:
            if "rollout_len" in data:
                return max(1, int(data["rollout_len"]))
            idx = data["p1_target_state_idx"]
            return int(idx.shape[1]) if idx.ndim == 2 else 1

    return 1


def train(args: argparse.Namespace) -> None:
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.encoder_chunk_tokens < 0:
        raise ValueError("--encoder_chunk_tokens must be >= 0")
    if args.belief_batch_size < 0:
        raise ValueError("--belief_batch_size must be >= 0")

    device = _auto_detect_device()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]
    structural_ids = {
        "unknown": tokenizer["unknown"],
    }

    latent_dim = model_cfg.get("latent_dim", LATENT_DIM)
    lambda_sigreg_state = (
        float(args.lambda_sigreg_state)
        if args.lambda_sigreg_state is not None
        else float(model_cfg.get("lambda_sigreg_state", model_cfg.get("lambda_sigreg", 0.0)))
    )
    model_cfg["lambda_sigreg_state"] = lambda_sigreg_state
    sigreg_num_slices = int(model_cfg.get("sigreg_num_slices", SIGREG_NUM_SLICES))
    sigreg_num_points = int(model_cfg.get("sigreg_num_points", SIGREG_NUM_POINTS))
    sigreg_domain = float(model_cfg.get("sigreg_domain", SIGREG_DOMAIN))

    # ── max_seq_len: source of truth is sequence_stats.json ────────
    # Values from the stats file already include a 1.2× safety multiplier.
    stats_path = os.path.join(args.data_root, "sequence_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"sequence_stats.json not found at {stats_path}. "
            "Run scripts/generate_world_model_data.py first."
        )
    with open(stats_path, "r") as f:
        seq_stats = json.load(f)
    state_block_max = seq_stats["state_block_len"]["max"]
    temporal_max = seq_stats["temporal_sequence_len"]["max"]

    model_cfg.setdefault("encoder", {})["max_seq_len"] = state_block_max
    print(f"Encoder max_seq_len: {state_block_max} [from sequence_stats.json]")

    # ── Cap temporal max_seq_len by max_history_blocks ─────────────
    temporal_capped = False
    if args.max_history_blocks > 0:
        # TODO: double check the calculation below is consistent with self belief encoder
        max_temporal_from_history = 3 * args.max_history_blocks + 2
        if temporal_max > max_temporal_from_history:
            temporal_max = max_temporal_from_history
            temporal_capped = True

    model_cfg.setdefault("self_belief_encoder", {})["max_seq_len"] = temporal_max
    model_cfg.setdefault("opponent_belief_predictor", {})["max_seq_len"] = temporal_max
    cap_note = (
        f" (capped by --max_history_blocks={args.max_history_blocks})"
        if temporal_capped else ""
    )
    print(f"Temporal max_seq_len: {temporal_max} [from sequence_stats.json{cap_note}]")

    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: bos={bos_id} eos={eos_id} pad={pad_id}")
    print(f"Structural token IDs: {structural_ids}")
    print(f"Latent dim: {latent_dim}")

    train_loader, val_loader, train_shards, val_shards = _build_dataloaders(
        args.data_root, args.formats, structural_ids, args.max_history_blocks,
        args.batch_size, pad_id, args.num_workers, args.prefetch_factor, device,
    )
    rollout_len = _load_rollout_len(args.data_root, train_shards)

    model = PairedJEPAModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        latent_dim=latent_dim,
        encoder_cfg=model_cfg.get("encoder", {}),
        self_belief_encoder_cfg=model_cfg.get("self_belief_encoder", {}),
        opponent_belief_predictor_cfg=model_cfg.get("opponent_belief_predictor", {}),
        opponent_policy_belief_cfg=model_cfg.get("opponent_policy_belief", {}),
        next_state_predictor_cfg=model_cfg.get("next_state_predictor", {}),
        encoder_chunk_tokens=args.encoder_chunk_tokens,
        belief_batch_size=args.belief_batch_size,
    ).to(device)

    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── torch.compile submodules individually (CUDA only) ─────────
    # With max_history_blocks=100, temporal sequences are ≤302 positions
    # so the belief encoder, opponent belief predictor is small enough that it doesn't
    # need compilation.  The heavy encoder still benefits.
    if args.compile and device.type == "cuda":
        torch._dynamo.config.capture_scalar_outputs = True
        compiled_any = False
        for name in ["encoder"]:
            module = getattr(model, name, None)
            if module is None:
                continue
            try:
                compiled = torch.compile(module, dynamic=True)
                setattr(model, name, compiled)
                compiled_any = True
            except Exception as e:
                print(f"  [{name}] torch.compile failed: {e}")
        if compiled_any:
            print("torch.compile enabled on: encoder")

    loaded_ckpt = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        loaded_ckpt = _load_compatible_checkpoint(model, args.checkpoint, device)

    model.set_debug_tensor_logging(
        args.debug_tensors,
        max_steps=args.debug_tensor_steps,
        max_values=args.debug_tensor_values,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_transitions = PairedJEPADataset.count_transitions(train_shards)
    val_transitions = PairedJEPADataset.count_transitions(val_shards) if val_shards else 0
    print(
        f"Params: {n_params:,}  Shards: {len(train_shards)} train + {len(val_shards)} val"
    )
    print(
        f"Transitions: {train_transitions:,} train  {val_transitions:,} val  "
        f"batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"rollout_len={rollout_len} "
        f"microbatch_transitions={args.batch_size * rollout_len} "
        f"effective_transitions={args.batch_size * rollout_len * args.grad_accum_steps} "
        f"max_history_blocks={args.max_history_blocks} "
        f"encoder_chunk_tokens={args.encoder_chunk_tokens} "
        f"belief_batch_size={args.belief_batch_size} "
        f"lambda_sigreg_state={lambda_sigreg_state:g}"
    )

    # ---- wandb init ----
    wandb_run = None
    if args.wandb and _wandb_available:
        wandb_init_kwargs: dict = dict(
            project=args.wandb_project or "metamon-jepa-" + "-".join(args.formats),
        )
        if args.wandb_name:
            wandb_init_kwargs["name"] = args.wandb_name
        wandb_run = wandb.init(
            **wandb_init_kwargs,
            config={
                **model_cfg,
                "vocab_size": vocab_size,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "rollout_len": rollout_len,
                "microbatch_transitions": args.batch_size * rollout_len,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "effective_transition_batch_size": args.batch_size * rollout_len * args.grad_accum_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "grad_clip": args.grad_clip,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "compile": args.compile,
                "config_path": args.config,
                "val_interval": args.val_interval,
                "val_max_batches": args.val_max_batches,
                "print_interval": args.print_interval,
                "log_interval": args.log_interval,
                "debug_tensors": args.debug_tensors,
                "debug_tensor_steps": args.debug_tensor_steps,
                "debug_tensor_values": args.debug_tensor_values,
                "debug_tensor_samples": args.debug_tensor_samples,
                "encoder_chunk_tokens": args.encoder_chunk_tokens,
                "belief_batch_size": args.belief_batch_size,
                "n_params": n_params,
                "n_train_transitions": train_transitions,
                "n_val_transitions": val_transitions,
                "state_block_max_seq_len": state_block_max,
                "temporal_max_seq_len": temporal_max,
                "max_history_blocks": args.max_history_blocks,
                "lambda_self_state": args.lambda_self_state,
                "lambda_opponent_state": args.lambda_opponent_state,
                "lambda_action": args.lambda_action,
                "lambda_next_state": args.lambda_next_state,
                "lambda_sigreg_state": lambda_sigreg_state,
                "sigreg_num_slices": sigreg_num_slices,
                "sigreg_num_points": sigreg_num_points,
                "sigreg_domain": sigreg_domain,
                "checkpoint": args.checkpoint,
            },
        )
    elif args.wandb and not _wandb_available:
        print("WARNING: --wandb enabled but wandb not installed (pip install wandb)")

    def loss_from_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_paired_losses(
            outputs,
            lambda_self_state=args.lambda_self_state,
            lambda_opponent_state=args.lambda_opponent_state,
            lambda_action=args.lambda_action,
            lambda_next_state=args.lambda_next_state,
            lambda_sigreg_state=lambda_sigreg_state,
            sigreg_num_slices=sigreg_num_slices,
            sigreg_num_points=sigreg_num_points,
            sigreg_domain=sigreg_domain,
        )

    @torch.no_grad()
    def validate(max_batches: int) -> dict[str, float]:
        if val_loader is None:
            return {}
        model.eval()
        totals: dict[str, float] = {}
        steps = 0
        for batch_idx, batch in enumerate(val_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = _batch_to_device(batch, device)
            outputs = _forward_paired(
                model,
                batch
            )
            _, metrics = loss_from_outputs(outputs)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        model.train()
        return {f"val_{key}": value / max(steps, 1) for key, value in totals.items()}

    global_step = 0
    best_val_loss = float("inf")
    best_val_epoch: int | None = None
    best_val_global_step: int | None = None
    best_val_metrics: dict[str, float] | None = None
    if loaded_ckpt is not None:
        ckpt_best = loaded_ckpt.get("best_val_loss")
        if ckpt_best is not None:
            best_val_loss = float(ckpt_best)
            best_val_epoch = loaded_ckpt.get("best_val_epoch")
            best_val_global_step = loaded_ckpt.get("best_val_global_step")
            best_val_metrics = loaded_ckpt.get("best_val_metrics")
            print(
                f"  restored best val from checkpoint: loss={best_val_loss:.4f}"
                f" epoch={best_val_epoch} step={best_val_global_step}"
            )

    def update_best_val(
        epoch_idx: int,
        step_idx: int,
        val_metrics: dict[str, float],
    ) -> bool:
        nonlocal best_val_loss, best_val_epoch, best_val_global_step, best_val_metrics
        val_loss = val_metrics.get("val_loss")
        if val_loss is None or val_loss >= best_val_loss:
            return False
        best_val_loss = val_loss
        best_val_epoch = epoch_idx
        best_val_global_step = step_idx
        best_val_metrics = dict(val_metrics)
        return True

    def checkpoint_val_metadata(
        last_val_metrics: dict[str, float] | None = None,
    ) -> dict[str, object]:
        return {
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_val_epoch": best_val_epoch,
            "best_val_global_step": best_val_global_step,
            "best_val_metrics": best_val_metrics,
            "last_val_metrics": dict(last_val_metrics) if last_val_metrics else None,
        }

    optimizer.zero_grad(set_to_none=True)
    done = False
    t_last_print = time.time()
    tokens_since_print = 0
    t_last_wandb = time.time()
    tokens_since_wandb = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        model.train()
        epoch_totals: dict[str, float] = {}
        epoch_steps = 0
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            debug_this_step = args.debug_tensors and global_step < args.debug_tensor_steps
            if debug_this_step:
                _debug_dump_tensor_mapping(
                    f"train step {global_step + 1} batch inputs after collation/device move",
                    batch,
                    pad_id=pad_id,
                    max_values=args.debug_tensor_values,
                    tokenizer=tokenizer,
                )
                _debug_dump_detokenized_batch_inputs(
                    f"train step {global_step + 1} detokenized rollout inputs",
                    batch,
                    tokenizer=tokenizer,
                    pad_id=pad_id,
                    max_values=args.debug_tensor_values,
                    max_samples=args.debug_tensor_samples,
                )
            outputs = _forward_paired(
                model,
                batch
            )
            loss, metrics = loss_from_outputs(outputs)
            if debug_this_step:
                _debug_dump_tensor_mapping(
                    f"train step {global_step + 1} model outputs",
                    outputs,
                    pad_id=pad_id,
                    max_values=args.debug_tensor_values,
                )
                _debug_dump_metrics(
                    f"train step {global_step + 1} loss metrics",
                    metrics,
                )
            try:
                (loss / args.grad_accum_steps).backward()
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    print(
                        "CUDA OOM during backward. "
                        f"allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB "
                        f"batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
                        f"rollout_len={rollout_len} "
                        f"microbatch_transitions={args.batch_size * rollout_len} "
                        f"max_history_blocks={args.max_history_blocks} "
                        f"encoder_chunk_tokens={args.encoder_chunk_tokens} "
                        f"belief_batch_size={args.belief_batch_size}. "
                        "Reduce JEPA_PAIRED_BATCH_SIZE or JEPA_MAX_HISTORY; "
                        "increase JEPA_PAIRED_GRAD_ACCUM_STEPS to keep the same effective batch; "
                        "or lower --belief_batch_size / --encoder_chunk_tokens."
                    )
                raise

            for key, value in metrics.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + value
            epoch_steps += 1
            global_step += 1

            if global_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Count tokens processed (non-pad).  State blocks dominate;
            # count all 4 state tensors + single-action tokens.
            batch_tokens = 0
            for key in (
                "p1_history_T", "p2_history_T",
                "p1_target_state_T", "p2_target_state_T",
                "p1_next_state_T1", "p2_next_state_T1",
                "p1_player_hist_T", "p2_player_hist_T",
                "p1_opponent_hist_T", "p2_opponent_hist_T",
                "p1_action", "p2_action",
                "actual_p2_action_from_p1_perspective",
                "actual_p1_action_from_p2_perspective",
            ):
                batch_tokens += int((batch[key] != pad_id).sum().item())
            tokens_since_print += batch_tokens
            tokens_since_wandb += batch_tokens

            log_step = args.log_interval if args.log_interval > 0 else args.print_interval
            if wandb_run and log_step > 0 and global_step % log_step == 0:
                now = time.time()
                elapsed = now - t_last_wandb
                tok_per_sec_wandb = tokens_since_wandb / elapsed if elapsed > 0 else 0
                t_last_wandb = now
                tokens_since_wandb = 0
                wandb_payload = {
                    "train/tok_per_sec": tok_per_sec_wandb,
                    "train/loss": metrics["loss"],
                    "train/self_state_loss": metrics["self_state_loss"],
                    "train/self_state_loss_p1": metrics["self_state_loss_p1"],
                    "train/self_state_loss_p2": metrics["self_state_loss_p2"],
                    "train/opponent_state_loss": metrics["opponent_state_loss"],
                    "train/opponent_state_loss_p1_to_p2": metrics["opponent_state_loss_p1_to_p2"],
                    "train/opponent_state_loss_p2_to_p1": metrics["opponent_state_loss_p2_to_p1"],
                    "train/action_loss": metrics["action_loss"],
                    "train/action_loss_p1_to_p2": metrics["action_loss_p1_to_p2"],
                    "train/action_loss_p2_to_p1": metrics["action_loss_p2_to_p1"],
                    "train/next_state_loss": metrics["next_state_loss"],
                    "train/next_state_loss_p1": metrics["next_state_loss_p1"],
                    "train/next_state_loss_p2": metrics["next_state_loss_p2"],
                    "train/sigreg_state_loss": metrics["sigreg_state_loss"],
                    "train/sigreg_state_weighted": metrics["sigreg_state_weighted"],
                    "train/target_latent_norm": metrics["target_latent_norm"],
                    "train/target_latent_std_per_dim": metrics["target_latent_std_per_dim"],
                    "train/target_pairwise_distance": metrics["target_pairwise_distance"],
                    "train/self_state_logvar_p1": metrics["self_state_logvar_p1"],
                    "train/self_state_logvar_p2": metrics["self_state_logvar_p2"],
                    "train/opponent_state_logvar_p1_to_p2": metrics["opponent_state_logvar_p1_to_p2"],
                    "train/opponent_state_logvar_p2_to_p1": metrics["opponent_state_logvar_p2_to_p1"],
                    "train/action_logvar_p1_to_p2": metrics["action_logvar_p1_to_p2"],
                    "train/action_logvar_p2_to_p1": metrics["action_logvar_p2_to_p1"],
                    "train/next_state_logvar_p1": metrics["next_state_logvar_p1"],
                    "train/next_state_logvar_p2": metrics["next_state_logvar_p2"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "samples_seen": args.batch_size * global_step,
                }
                if device.type == "cuda":
                    wandb_payload.update({
                        "mem/cuda_reserved_gib": torch.cuda.memory_reserved() / 1024 ** 3,
                        "mem/cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                    })
                wandb_run.log(wandb_payload)

            if args.print_interval > 0 and global_step % args.print_interval == 0:
                now = time.time()
                elapsed = now - t_last_print
                tok_per_sec = tokens_since_print / elapsed if elapsed > 0 else 0
                t_last_print = now
                tokens_since_print = 0
                cuda_mem = ""
                if device.type == "cuda":
                    cuda_mem = (
                        " | cuda_mem "
                        f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f}G reserved/"
                        f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}G peak"
                    )
                print(
                    f"  epoch {epoch:3d} | step {global_step:6d} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"self_state {metrics['self_state_loss']:.4f} "
                    f"[p1 {metrics['self_state_loss_p1']:.4f}, "
                    f"p2 {metrics['self_state_loss_p2']:.4f}] | "
                    f"opp_state {metrics['opponent_state_loss']:.4f} "
                    f"[p1->p2 {metrics['opponent_state_loss_p1_to_p2']:.4f}, "
                    f"p2->p1 {metrics['opponent_state_loss_p2_to_p1']:.4f}] | "
                    f"action {metrics['action_loss']:.4f} "
                    f"[p1->p2 {metrics['action_loss_p1_to_p2']:.4f}, "
                    f"p2->p1 {metrics['action_loss_p2_to_p1']:.4f}] | "
                    f"next {metrics['next_state_loss']:.4f} "
                    f"[p1 {metrics['next_state_loss_p1']:.4f}, "
                    f"p2 {metrics['next_state_loss_p2']:.4f}] | "
                    f"sigreg_state {metrics['sigreg_state_loss']:.4f} | "
                    f"z {metrics['target_latent_norm']:.2f}/"
                    f"{metrics['target_latent_std_per_dim']:.3f}/"
                    f"{metrics['target_pairwise_distance']:.2f} | "
                    f"logvar_next {metrics['next_state_logvar_p1']:.3f}/{metrics['next_state_logvar_p2']:.3f}"
                    f"{cuda_mem}"
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

            if args.val_interval > 0 and global_step % args.val_interval == 0:
                val_metrics = validate(args.val_max_batches)
                if val_metrics:
                    print(
                        f"  val @ step {global_step:6d} | "
                        f"loss {val_metrics['val_loss']:.4f} | "
                        f"self_state {val_metrics['val_self_state_loss']:.4f} | "
                        f"opp_state {val_metrics['val_opponent_state_loss']:.4f} | "
                        f"action {val_metrics['val_action_loss']:.4f} | "
                        f"next {val_metrics['val_next_state_loss']:.4f} | "
                        f"sigreg_state {val_metrics['val_sigreg_state_loss']:.4f} | "
                        f"z {val_metrics['val_target_latent_norm']:.2f}/"
                        f"{val_metrics['val_target_latent_std_per_dim']:.3f}/"
                        f"{val_metrics['val_target_pairwise_distance']:.2f}"
                    )
                    if wandb_run:
                        wandb_run.log({
                            "val/loss": val_metrics["val_loss"],
                            "val/self_state_loss": val_metrics["val_self_state_loss"],
                            "val/opponent_state_loss": val_metrics["val_opponent_state_loss"],
                            "val/action_loss": val_metrics["val_action_loss"],
                            "val/next_state_loss": val_metrics["val_next_state_loss"],
                            "val/sigreg_state_loss": val_metrics["val_sigreg_state_loss"],
                            "val/sigreg_state_weighted": val_metrics["val_sigreg_state_weighted"],
                            "val/target_latent_norm": val_metrics["val_target_latent_norm"],
                            "val/target_latent_std_per_dim": val_metrics["val_target_latent_std_per_dim"],
                            "val/target_pairwise_distance": val_metrics["val_target_pairwise_distance"],
                            "epoch": epoch,
                            "global_step": global_step,
                            "samples_seen": args.batch_size * global_step,
                        })
                    val_improved = update_best_val(epoch, global_step, val_metrics)
                    if val_improved and args.checkpoint:
                        save_paired_jepa_checkpoint(
                            model,
                            args.checkpoint,
                            epoch=epoch,
                            global_step=global_step,
                            config=model_cfg,
                            vocab_size=vocab_size,
                            max_history_blocks=args.max_history_blocks,
                            tokenizer=tokenizer,
                            **checkpoint_val_metadata(val_metrics),
                        )
                        print(f"  best checkpoint -> {args.checkpoint}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break

        if epoch_steps > 0 and global_step % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg = {key: value / max(epoch_steps, 1) for key, value in epoch_totals.items()}
        val_metrics = validate(args.val_max_batches)
        epoch_val_improved = update_best_val(epoch, global_step, val_metrics) if val_metrics else False
        msg = (
            f"=== epoch {epoch:3d} done | train loss {avg.get('loss', 0.0):.4f} | "
            f"self_state {avg.get('self_state_loss', 0.0):.4f} | "
            f"opp_state {avg.get('opponent_state_loss', 0.0):.4f} | "
            f"action {avg.get('action_loss', 0.0):.4f} | "
            f"next {avg.get('next_state_loss', 0.0):.4f} | "
            f"sigreg_state {avg.get('sigreg_state_loss', 0.0):.4f}"
        )
        if val_metrics:
            msg += (
                f" | val loss {val_metrics.get('val_loss', 0.0):.4f}"
                f" | val self_state {val_metrics.get('val_self_state_loss', 0.0):.4f}"
                f" | val opp_state {val_metrics.get('val_opponent_state_loss', 0.0):.4f}"
                f" | val action {val_metrics.get('val_action_loss', 0.0):.4f}"
                f" | val next {val_metrics.get('val_next_state_loss', 0.0):.4f}"
                f" | val sigreg_state {val_metrics.get('val_sigreg_state_loss', 0.0):.4f}"
                f" | val z {val_metrics.get('val_target_latent_norm', 0.0):.2f}/"
                f"{val_metrics.get('val_target_latent_std_per_dim', 0.0):.3f}/"
                f"{val_metrics.get('val_target_pairwise_distance', 0.0):.2f}"
            )
        print(msg + " ===")

        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": avg.get("loss", 0.0),
                "epoch/train_self_state_loss": avg.get("self_state_loss", 0.0),
                "epoch/train_opponent_state_loss": avg.get("opponent_state_loss", 0.0),
                "epoch/train_action_loss": avg.get("action_loss", 0.0),
                "epoch/train_next_state_loss": avg.get("next_state_loss", 0.0),
                "epoch/train_sigreg_state_loss": avg.get("sigreg_state_loss", 0.0),
                "epoch/train_target_latent_norm": avg.get("target_latent_norm", 0.0),
                "epoch/train_target_latent_std_per_dim": avg.get("target_latent_std_per_dim", 0.0),
                "epoch/train_target_pairwise_distance": avg.get("target_pairwise_distance", 0.0),
                "epoch/val_loss": val_metrics.get("val_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_self_state_loss": val_metrics.get("val_self_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_opponent_state_loss": val_metrics.get("val_opponent_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_action_loss": val_metrics.get("val_action_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_next_state_loss": val_metrics.get("val_next_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_sigreg_state_loss": val_metrics.get("val_sigreg_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_target_latent_norm": val_metrics.get("val_target_latent_norm", 0.0) if val_metrics else 0.0,
                "epoch/val_target_latent_std_per_dim": val_metrics.get("val_target_latent_std_per_dim", 0.0) if val_metrics else 0.0,
                "epoch/val_target_pairwise_distance": val_metrics.get("val_target_pairwise_distance", 0.0) if val_metrics else 0.0,
                "epoch": epoch,
                "samples_seen": args.batch_size * global_step,
            })

        latest_path = save_dir / "paired_latest.pt"
        save_paired_jepa_checkpoint(
            model,
            str(latest_path),
            epoch=epoch,
            global_step=global_step,
            config=model_cfg,
            vocab_size=vocab_size,
            max_history_blocks=args.max_history_blocks,
            tokenizer=tokenizer,
            **checkpoint_val_metadata(val_metrics if val_metrics else None),
        )
        if args.checkpoint and (not val_metrics or epoch_val_improved):
            save_paired_jepa_checkpoint(
                model,
                args.checkpoint,
                epoch=epoch,
                global_step=global_step,
                config=model_cfg,
                vocab_size=vocab_size,
                max_history_blocks=args.max_history_blocks,
                tokenizer=tokenizer,
                **checkpoint_val_metadata(val_metrics if val_metrics else None),
            )
        if done:
            break

    if wandb_run:
        wandb_run.finish()

    print(f"Training complete. Checkpoints: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train paired-POV JEPA.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--val_max_batches", type=int, default=10)
    parser.add_argument("--lambda_self_state", type=float, default=1.0)
    parser.add_argument("--lambda_opponent_state", type=float, default=1.0)
    parser.add_argument("--lambda_action", type=float, default=1.0)
    parser.add_argument("--lambda_next_state", type=float, default=1.0)
    parser.add_argument("--lambda_sigreg_state", type=float, default=None,
                        help="Weight for SIGReg on all p1/p2 state embeddings. "
                             "Default: use model config lambda_sigreg_state.")
    parser.add_argument("--print_interval", type=int, default=10, help="Print training progress to stdout at this cadence")
    parser.add_argument("--log_interval", type=int, default=0,
                        help="Log every N training steps to wandb (0 = same as print_interval).")
    parser.add_argument("--debug_tensors", action="store_true",
                        help="Dump train batch tensors, model submodule handoff tensors, model outputs, "
                             "and loss metrics for the first --debug_tensor_steps training steps.")
    parser.add_argument("--debug_tensor_steps", type=int, default=1,
                        help="Number of initial training steps to dump when --debug_tensors is set.")
    parser.add_argument("--debug_tensor_values", type=int, default=16,
                        help="Number of flattened values and detokenized IDs to preview per dumped tensor.")
    parser.add_argument("--debug_tensor_samples", type=int, default=2,
                        help="Number of batch samples to expand in the structured detokenized debug dump.")
    parser.add_argument("--encoder_chunk_tokens", type=int, default=65536,
                        help="Maximum padded token slots per JEPAEncoder call when encoding history blocks "
                             "(0 = no chunking). Lower uses less peak CUDA memory.")
    parser.add_argument("--belief_batch_size", type=int, default=128,
                        help="Maximum flattened rollout items per state-belief transformer call "
                             "(0 = no chunking). Lower uses less peak CUDA memory.")
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction,
                        help="Enable Weights & Biases logging (default: True). Use --no-wandb to disable.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (default: metamon-jepa-<format>).")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name.")
    parser.add_argument("--max_history_blocks", type=int, default=100,
                        help="Maximum non-header history state blocks per sample (0 = unlimited). "
                             "The team header is always retained. Lower = faster data loading + shorter "
                             "temporal sequences. Default: 0 (unlimited)")
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable torch.compile (default: False). Use --compile to enable.")
    train(parser.parse_args())
