"""Train paired-POV JEPA on synchronized battle perspectives.

This trainer consumes ``paired_shard_*.npz`` files produced by:

    uv run python scripts/generate_world_model_data.py --rollout_len K ...

For each K-step rollout sample, the dataset provides both player perspectives
through state T and T+1 at every rollout step. The model learns:

1. history context + visible POV latent -> hidden opponent POV latent
2. the same shared opponent-belief backbone -> opponent action latent
3. visible POV latent + predicted opponent POV/action + own action -> next POV latent
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
    ACTION_LATENT_DIM,
    CONTEXT_LENGTH,
    LATENT_DIM,
    SIGREG_DOMAIN,
    SIGREG_NUM_POINTS,
    SIGREG_NUM_SLICES,
    PairedJEPAModel,
    compute_paired_losses,
)


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
        max_history_blocks: int = 0,
    ):
        super().__init__()
        if not shard_paths:
            raise ValueError("No paired shard paths provided")
        self.shard_paths = list(shard_paths)
        self.structural = structural_token_ids
        self.shuffle_shards = shuffle_shards
        self.shuffle_transitions = shuffle_shards
        self.max_history_blocks = max_history_blocks  # 0 = unlimited

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

        Supports two layouts:

        1. **Flat / interleaved** (multi-format generation):
           ``data_root/{split}/paired_shard_*.npz``

        2. **Per-format** (single-format / legacy):
           ``data_root/{fmt}/{split}/paired_shard_*.npz``

        Both are searched; results are merged and sorted.
        """
        shard_paths: list[str] = []

        # Layout 1: flat (interleaved, multi-format)
        flat_dir = os.path.join(data_root, split)
        if os.path.isdir(flat_dir):
            for name in sorted(os.listdir(flat_dir)):
                if name.startswith("paired_shard_") and name.endswith(".npz"):
                    shard_paths.append(os.path.join(flat_dir, name))

        # Layout 2: per-format (single-format / legacy)
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
            idx = data["state_idx"]
            if idx.ndim == 1:
                total += int(len(idx))
            else:
                total += int(idx.shape[0] * idx.shape[1])
        return total

    @staticmethod
    def _resolve_window(
        battle_start: int,
        state_end: int,
        action_base: int,
        max_hist: int,
    ) -> tuple[int, int, int]:
        state_start = battle_start
        if max_hist > 0:
            state_start = max(battle_start + 1, state_end - max_hist)
        # State index 0 within each battle is the team header, so action i
        # connects state i+1 -> state i+2. Keep only actions between retained
        # state blocks; the current transition appears in the T1 window, not T.
        action_start = action_base + max(0, state_start - battle_start - 1)
        action_end = action_base + max(0, state_end - battle_start - 2)
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
    def _rollout_index_matrix(data: dict, key: str, fallback: str | None = None) -> np.ndarray:
        arr = data[key] if key in data else data[fallback]  # type: ignore[index]
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{key} must be a 1D legacy array or 2D rollout matrix, got {arr.shape}")
        return arr

    @staticmethod
    def _iter_shard(data: dict, unknown_token: int | None,
                    shuffle_transitions: bool, max_hist: int) -> Iterator[dict]:
        p1_state_idx = PairedJEPADataset._rollout_index_matrix(data, "p1_state_idx", "state_idx")
        p1_next_state_idx = PairedJEPADataset._rollout_index_matrix(data, "p1_next_state_idx", "next_state_idx")
        p1_action_idx = PairedJEPADataset._rollout_index_matrix(data, "p1_action_idx", "action_idx")
        p2_state_idx = PairedJEPADataset._rollout_index_matrix(data, "p2_state_idx", "state_idx")
        p2_next_state_idx = PairedJEPADataset._rollout_index_matrix(data, "p2_next_state_idx", "next_state_idx")
        p2_action_idx = PairedJEPADataset._rollout_index_matrix(data, "p2_action_idx", "action_idx")

        n, rollout_len = p1_state_idx.shape
        for name, arr in [
            ("p1_next_state_idx", p1_next_state_idx),
            ("p1_action_idx", p1_action_idx),
            ("p2_state_idx", p2_state_idx),
            ("p2_next_state_idx", p2_next_state_idx),
            ("p2_action_idx", p2_action_idx),
        ]:
            if arr.shape != (n, rollout_len):
                raise ValueError(f"{name} shape {arr.shape} does not match p1_state_idx {(n, rollout_len)}")

        order = np.arange(n)
        if shuffle_transitions:
            np.random.default_rng().shuffle(order)

        # Pre-combine action blocks once
        data = PairedJEPADataset._ensure_combined(data, unknown_token)

        for row in order:
            battle_id = int(data["battle_id"][row])
            p1_bs = int(data["p1_battle_start"][battle_id]) if "p1_battle_start" in data else int(data["battle_start"][battle_id])
            p2_bs = int(data["p2_battle_start"][battle_id]) if "p2_battle_start" in data else int(data["battle_start"][battle_id])
            p1_as = int(data["p1_battle_action_start"][battle_id]) if "p1_battle_action_start" in data else int(data["battle_action_start"][battle_id])
            p2_as = int(data["p2_battle_action_start"][battle_id]) if "p2_battle_action_start" in data else int(data["battle_action_start"][battle_id])

            w = PairedJEPADataset._resolve_window
            sv = PairedJEPADataset._slice_view
            ssv = PairedJEPADataset._slice_state_window

            sample: dict[str, list] = {
                key: []
                for key in (
                    *BLOCK_KEYS,
                    *ACTION_KEYS,
                    *LEGAL_ACTION_KEYS,
                    *NEXT_LEGAL_ACTION_KEYS,
                    *LEGAL_MASK_KEYS,
                    *NEXT_LEGAL_MASK_KEYS,
                    *LEGAL_INDEX_KEYS,
                    *SCALAR_KEYS,
                )
            }
            p1_won = bool(data["p1_won"][battle_id])
            p2_won = bool(data["p2_won"][battle_id])
            rank_valid = (
                bool(data["rank_valid"][battle_id])
                if "rank_valid" in data
                else p1_won != p2_won
            )

            for step in range(rollout_len):
                p1_si = int(p1_state_idx[row, step])
                p1_nsi = int(p1_next_state_idx[row, step])
                p1_ai = int(p1_action_idx[row, step])
                p2_si = int(p2_state_idx[row, step])
                p2_nsi = int(p2_next_state_idx[row, step])
                p2_ai = int(p2_action_idx[row, step])
                p1_action_end = (
                    int(data["p1_battle_action_start"][battle_id + 1])
                    if "p1_battle_action_start" in data
                    else int(data["battle_action_start"][battle_id + 1])
                )
                p2_action_end = (
                    int(data["p2_battle_action_start"][battle_id + 1])
                    if "p2_battle_action_start" in data
                    else int(data["battle_action_start"][battle_id + 1])
                )
                p1_terminal = p1_ai + 1 >= p1_action_end
                p2_terminal = p2_ai + 1 >= p2_action_end

                p1_sT_s, p1_aT_s, p1_aT_e = w(p1_bs, p1_si + 1, p1_as, max_hist)
                p1_sT1_s, p1_aT1_s, p1_aT1_e = w(p1_bs, p1_nsi + 1, p1_as, max_hist)
                p2_sT_s, p2_aT_s, p2_aT_e = w(p2_bs, p2_si + 1, p2_as, max_hist)
                p2_sT1_s, p2_aT1_s, p2_aT1_e = w(p2_bs, p2_nsi + 1, p2_as, max_hist)

                sample["p1_state_T"].append(ssv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_bs, p1_sT_s, p1_si + 1))
                sample["p1_state_T1"].append(ssv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_bs, p1_sT1_s, p1_nsi + 1))
                sample["p2_state_T"].append(ssv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_bs, p2_sT_s, p2_si + 1))
                sample["p2_state_T1"].append(ssv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_bs, p2_sT1_s, p2_nsi + 1))
                sample["p1_player_hist_T"].append(sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_aT_s, p1_aT_e))
                sample["p1_opponent_hist_T"].append(sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_aT_s, p1_aT_e))
                sample["p1_player_hist_T1"].append(sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_aT1_s, p1_aT1_e))
                sample["p1_opponent_hist_T1"].append(sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_aT1_s, p1_aT1_e))
                sample["p2_player_hist_T"].append(sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_aT_s, p2_aT_e))
                sample["p2_opponent_hist_T"].append(sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_aT_s, p2_aT_e))
                sample["p2_player_hist_T1"].append(sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_aT1_s, p2_aT1_e))
                sample["p2_opponent_hist_T1"].append(sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_aT1_s, p2_aT1_e))
                sample["p1_action"].append(sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_ai, p1_ai + 1)[0])
                sample["p2_action"].append(sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_ai, p2_ai + 1)[0])
                sample["actual_p2_action_from_p1_perspective"].append(sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_ai, p1_ai + 1)[0])
                sample["actual_p1_action_from_p2_perspective"].append(sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_ai, p2_ai + 1)[0])
                if "p1_legal_actions" in data:
                    sample["p1_legal_actions"].append(data["p1_legal_actions"][p1_ai])
                    sample["p1_legal_action_mask"].append(data["p1_legal_action_mask"][p1_ai])
                    sample["p1_chosen_legal_action_idx"].append(int(data["p1_chosen_legal_action_idx"][p1_ai]))
                    if not p1_terminal:
                        sample["p1_next_legal_actions"].append(data["p1_legal_actions"][p1_ai + 1])
                        sample["p1_next_legal_action_mask"].append(data["p1_legal_action_mask"][p1_ai + 1])
                    else:
                        sample["p1_next_legal_actions"].append(np.zeros((0, 1), dtype=np.int16))
                        sample["p1_next_legal_action_mask"].append(np.zeros((0,), dtype=bool))
                else:
                    sample["p1_legal_actions"].append(sample["p1_action"][-1][None, :])
                    sample["p1_legal_action_mask"].append(np.array([True], dtype=bool))
                    sample["p1_chosen_legal_action_idx"].append(0)
                    if not p1_terminal:
                        next_action = sv(
                            data["p1_actions_combined"],
                            data["p1_actions_combined_offsets"],
                            data["p1_actions_combined_lengths"],
                            p1_ai + 1,
                            p1_ai + 2,
                        )[0]
                        sample["p1_next_legal_actions"].append(next_action[None, :])
                        sample["p1_next_legal_action_mask"].append(np.array([True], dtype=bool))
                    else:
                        sample["p1_next_legal_actions"].append(np.zeros((0, 1), dtype=np.int16))
                        sample["p1_next_legal_action_mask"].append(np.zeros((0,), dtype=bool))
                if "p2_legal_actions" in data:
                    sample["p2_legal_actions"].append(data["p2_legal_actions"][p2_ai])
                    sample["p2_legal_action_mask"].append(data["p2_legal_action_mask"][p2_ai])
                    sample["p2_chosen_legal_action_idx"].append(int(data["p2_chosen_legal_action_idx"][p2_ai]))
                    if not p2_terminal:
                        sample["p2_next_legal_actions"].append(data["p2_legal_actions"][p2_ai + 1])
                        sample["p2_next_legal_action_mask"].append(data["p2_legal_action_mask"][p2_ai + 1])
                    else:
                        sample["p2_next_legal_actions"].append(np.zeros((0, 1), dtype=np.int16))
                        sample["p2_next_legal_action_mask"].append(np.zeros((0,), dtype=bool))
                else:
                    sample["p2_legal_actions"].append(sample["p2_action"][-1][None, :])
                    sample["p2_legal_action_mask"].append(np.array([True], dtype=bool))
                    sample["p2_chosen_legal_action_idx"].append(0)
                    if not p2_terminal:
                        next_action = sv(
                            data["p2_actions_combined"],
                            data["p2_actions_combined_offsets"],
                            data["p2_actions_combined_lengths"],
                            p2_ai + 1,
                            p2_ai + 2,
                        )[0]
                        sample["p2_next_legal_actions"].append(next_action[None, :])
                        sample["p2_next_legal_action_mask"].append(np.array([True], dtype=bool))
                    else:
                        sample["p2_next_legal_actions"].append(np.zeros((0, 1), dtype=np.int16))
                        sample["p2_next_legal_action_mask"].append(np.zeros((0,), dtype=bool))
                sample["p1_won"].append(p1_won)
                sample["p2_won"].append(p2_won)
                sample["rank_valid"].append(rank_valid)
                sample["p1_is_terminal"].append(p1_terminal)
                sample["p2_is_terminal"].append(p2_terminal)
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
            )


BLOCK_KEYS = (
    "p1_state_T",
    "p1_state_T1",
    "p1_player_hist_T",
    "p1_opponent_hist_T",
    "p1_player_hist_T1",
    "p1_opponent_hist_T1",
    "p2_state_T",
    "p2_state_T1",
    "p2_player_hist_T",
    "p2_opponent_hist_T",
    "p2_player_hist_T1",
    "p2_opponent_hist_T1",
)
ACTION_KEYS = (
    "p1_action",
    "p2_action",
    "actual_p2_action_from_p1_perspective",
    "actual_p1_action_from_p2_perspective",
)
LEGAL_ACTION_KEYS = (
    "p1_legal_actions",
    "p2_legal_actions",
)
NEXT_LEGAL_ACTION_KEYS = (
    "p1_next_legal_actions",
    "p2_next_legal_actions",
)
LEGAL_MASK_KEYS = (
    "p1_legal_action_mask",
    "p2_legal_action_mask",
)
NEXT_LEGAL_MASK_KEYS = (
    "p1_next_legal_action_mask",
    "p2_next_legal_action_mask",
)
LEGAL_INDEX_KEYS = (
    "p1_chosen_legal_action_idx",
    "p2_chosen_legal_action_idx",
)
SCALAR_KEYS = (
    "p1_won",
    "p2_won",
    "rank_valid",
    "p1_is_terminal",
    "p2_is_terminal",
)


def collate_paired_fn(
    batch: list[dict[str, object]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    rollout_lengths = {
        len(item["p1_state_T"])  # type: ignore[arg-type,index]
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
            dtype=torch.long,
        )
        valid = torch.zeros((len(block_rollouts), rollout_len, max_blocks), dtype=torch.bool)
        for batch_idx, rollout in enumerate(block_rollouts):
            for step_idx, blocks in enumerate(rollout):
                for block_idx, block in enumerate(blocks):
                    tokens = torch.from_numpy(block.astype(np.int64, copy=False))
                    padded[batch_idx, step_idx, block_idx, :len(tokens)] = tokens
                    valid[batch_idx, step_idx, block_idx] = True
        return padded, valid

    def pad_action_rollouts(actions: list[list[np.ndarray]]) -> torch.Tensor:
        max_tokens = max(
            (len(action) for rollout in actions for action in rollout),
            default=1,
        )
        padded = torch.full((len(actions), rollout_len, max_tokens), pad_id, dtype=torch.long)
        for batch_idx, rollout in enumerate(actions):
            for step_idx, action in enumerate(rollout):
                tokens = torch.from_numpy(action.astype(np.int64, copy=False))
                padded[batch_idx, step_idx, :len(tokens)] = tokens
        return padded

    def pad_legal_action_rollouts(
        actions: list[list[np.ndarray]],
        masks: list[list[np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_candidates = max(
            1,
            max(
                (legal.shape[0] for rollout in actions for legal in rollout),
                default=1,
            ),
        )
        max_tokens = max(
            1,
            max(
                (legal.shape[1] for rollout in actions for legal in rollout),
                default=1,
            ),
        )
        padded = torch.full(
            (len(actions), rollout_len, max_candidates, max_tokens),
            pad_id,
            dtype=torch.long,
        )
        out_mask = torch.zeros(
            (len(actions), rollout_len, max_candidates),
            dtype=torch.bool,
        )
        for batch_idx, (rollout, rollout_masks) in enumerate(zip(actions, masks)):
            for step_idx, (legal, legal_mask) in enumerate(zip(rollout, rollout_masks)):
                candidate_count = int(legal.shape[0])
                token_count = int(legal.shape[1]) if legal.ndim == 2 else 0
                if candidate_count <= 0 or token_count <= 0:
                    continue
                tokens = torch.from_numpy(legal.astype(np.int64, copy=False))
                mask = torch.from_numpy(legal_mask.astype(bool, copy=False))
                padded[batch_idx, step_idx, :candidate_count, :token_count] = tokens
                out_mask[batch_idx, step_idx, :candidate_count] = mask[:candidate_count]
        return padded, out_mask

    out: dict[str, torch.Tensor] = {}
    for key in BLOCK_KEYS:
        blocks, valid = pad_block_rollouts([item[key] for item in batch])  # type: ignore[index]
        out[key] = blocks
        out[f"{key}_valid"] = valid
    for key in ACTION_KEYS:
        out[key] = pad_action_rollouts([item[key] for item in batch])  # type: ignore[index]
    for action_key, mask_key in zip(LEGAL_ACTION_KEYS, LEGAL_MASK_KEYS):
        legal, mask = pad_legal_action_rollouts(
            [item[action_key] for item in batch],  # type: ignore[index]
            [item[mask_key] for item in batch],  # type: ignore[index]
        )
        out[action_key] = legal
        out[mask_key] = mask
    for action_key, mask_key in zip(NEXT_LEGAL_ACTION_KEYS, NEXT_LEGAL_MASK_KEYS):
        legal, mask = pad_legal_action_rollouts(
            [item[action_key] for item in batch],  # type: ignore[index]
            [item[mask_key] for item in batch],  # type: ignore[index]
        )
        out[action_key] = legal
        out[mask_key] = mask
    for key in LEGAL_INDEX_KEYS:
        out[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
    for key in SCALAR_KEYS:
        out[key] = torch.tensor([item[key] for item in batch], dtype=torch.bool)

    # Legacy/defensive terminal detection: an empty T+1 state is terminal.
    # Rollout boundaries are not terminals; the model bootstraps from T+1
    # directly for those steps.
    for pov in ("p1", "p2"):
        t1_valid = out.get(f"{pov}_state_T1_valid")
        if t1_valid is not None:
            empty_state = ~t1_valid.any(dim=-1)  # [B, K]
            out[f"{pov}_is_terminal"] = out[f"{pov}_is_terminal"] | empty_state

    return out


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _load_compatible_checkpoint(
    model: PairedJEPAModel,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """Load a checkpoint as a warm start, skipping incompatible tensors.

    ``--checkpoint`` is also the "save best here" path.  After architecture
    changes, that file may exist but contain old predictor heads.  Loading only
    matching keys lets the encoder/action encoder warm-start while new or
    resized Gaussian/ranking heads train from initialization.
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


def _paired_outcome_stats(shard_paths: list[str]) -> dict[str, int]:
    stats = {
        "battles": 0,
        "p1_won": 0,
        "p2_won": 0,
        "both_won": 0,
        "both_lost": 0,
        "rank_valid": 0,
        "missing": 0,
    }
    for path in shard_paths:
        data = np.load(path)
        if "p1_won" not in data or "p2_won" not in data:
            stats["missing"] += 1
            continue
        p1 = data["p1_won"].astype(bool, copy=False)
        p2 = data["p2_won"].astype(bool, copy=False)
        n = min(len(p1), len(p2))
        if len(p1) != len(p2):
            stats["missing"] += 1
            p1 = p1[:n]
            p2 = p2[:n]
        stats["battles"] += n
        stats["p1_won"] += int(p1.sum())
        stats["p2_won"] += int(p2.sum())
        stats["both_won"] += int((p1 & p2).sum())
        stats["both_lost"] += int((~p1 & ~p2).sum())
        if "rank_valid" in data:
            stats["rank_valid"] += int(data["rank_valid"][:n].astype(bool, copy=False).sum())
        else:
            stats["rank_valid"] += int((p1 ^ p2).sum())
    return stats


def _validate_paired_outcomes(shard_paths: list[str], outcome_loss_weight: float) -> dict[str, int]:
    stats = _paired_outcome_stats(shard_paths)
    if stats["battles"] <= 0:
        return stats
    invalid = stats["both_won"] + stats["both_lost"]
    print(
        "Paired outcome labels: "
        f"battles={stats['battles']:,} "
        f"p1_won={stats['p1_won']:,} "
        f"p2_won={stats['p2_won']:,} "
        f"outcome_valid={stats['rank_valid']:,} "
        f"both_lost={stats['both_lost']:,} "
        f"both_won={stats['both_won']:,}"
    )
    invalid_fraction = invalid / max(stats["battles"], 1)
    if outcome_loss_weight > 0 and stats["missing"] > 0:
        raise RuntimeError(
            "Paired JEPA shards are missing p1_won/p2_won labels for value/Q training. "
            "Regenerate paired shards with the patched scripts/generate_world_model_data.py, "
            "or temporarily disable outcome heads via --lambda_value 0 --lambda_q_value 0 "
            "--lambda_policy 0 --lambda_value_teacher 0 --lambda_q_teacher 0."
        )
    if invalid > 0:
        print(
            "WARNING: Some paired outcome labels are invalid for value/Q supervision "
            f"({invalid_fraction:.2%}); these battles will be ignored by outcome losses."
        )
    return stats


def _forward_paired(
    model: PairedJEPAModel,
    batch: dict[str, torch.Tensor],
    *,
    compute_td_bootstrap: bool = True,
) -> dict[str, torch.Tensor]:
    outputs = model(
        batch["p1_state_T"], batch["p1_state_T_valid"],
        batch["p1_state_T1"], batch["p1_state_T1_valid"],
        batch["p1_player_hist_T"], batch["p1_player_hist_T_valid"],
        batch["p1_opponent_hist_T"], batch["p1_opponent_hist_T_valid"],
        batch["p1_player_hist_T1"], batch["p1_player_hist_T1_valid"],
        batch["p1_opponent_hist_T1"], batch["p1_opponent_hist_T1_valid"],
        batch["p2_state_T"], batch["p2_state_T_valid"],
        batch["p2_state_T1"], batch["p2_state_T1_valid"],
        batch["p2_player_hist_T"], batch["p2_player_hist_T_valid"],
        batch["p2_opponent_hist_T"], batch["p2_opponent_hist_T_valid"],
        batch["p2_player_hist_T1"], batch["p2_player_hist_T1_valid"],
        batch["p2_opponent_hist_T1"], batch["p2_opponent_hist_T1_valid"],
        batch["p1_action"],
        batch["p2_action"],
        batch["actual_p2_action_from_p1_perspective"],
        batch["actual_p1_action_from_p2_perspective"],
        batch["p1_legal_actions"],
        batch["p1_legal_action_mask"],
        batch["p1_chosen_legal_action_idx"],
        batch["p2_legal_actions"],
        batch["p2_legal_action_mask"],
        batch["p2_chosen_legal_action_idx"],
        p1_next_legal_action_tokens=batch.get("p1_next_legal_actions"),
        p1_next_legal_action_mask=batch.get("p1_next_legal_action_mask"),
        p2_next_legal_action_tokens=batch.get("p2_next_legal_actions"),
        p2_next_legal_action_mask=batch.get("p2_next_legal_action_mask"),
        compute_td_bootstrap=compute_td_bootstrap,
    )
    outputs["p1_won"] = batch["p1_won"]
    outputs["p2_won"] = batch["p2_won"]
    outputs["rank_valid"] = batch["rank_valid"]
    outputs["p1_is_terminal"] = batch.get("p1_is_terminal")
    outputs["p2_is_terminal"] = batch.get("p2_is_terminal")
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


def train(args: argparse.Namespace) -> None:
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    from metamon.tokenizer import PokemonTokenizer

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    vocab_size = model_cfg.get("vocab_size") or len(tokenizer)
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]
    structural_ids = {
        "unknown": tokenizer["unknown"],
    }

    latent_dim = model_cfg.get("latent_dim", LATENT_DIM)
    action_latent_dim = model_cfg.get("action_latent_dim", ACTION_LATENT_DIM)
    lambda_sigreg = args.lambda_sigreg
    if lambda_sigreg is None:
        lambda_sigreg = model_cfg.get("lambda_sigreg", 0.1)
    lambda_sigreg_state = args.lambda_sigreg_state
    if lambda_sigreg_state is None:
        lambda_sigreg_state = model_cfg.get("lambda_sigreg_state")
        if lambda_sigreg_state is None:
            lambda_sigreg_state = lambda_sigreg  # backward-compat fallback
    lambda_sigreg_action = args.lambda_sigreg_action
    if lambda_sigreg_action is None:
        lambda_sigreg_action = model_cfg.get("lambda_sigreg_action")
        if lambda_sigreg_action is None:
            lambda_sigreg_action = lambda_sigreg  # backward-compat fallback
    config_lambda_rank = model_cfg.get("lambda_rank", 0.0)
    if args.lambda_rank is not None or float(config_lambda_rank or 0.0) != 0.0:
        print("WARNING: lambda_rank is deprecated and ignored; actor-critic value/Q losses are used instead.")
    lambda_rank = 0.0
    lambda_value = args.lambda_value
    if lambda_value is None:
        lambda_value = model_cfg.get("lambda_value", 1.0)
    lambda_q_value = args.lambda_q_value
    if lambda_q_value is None:
        lambda_q_value = model_cfg.get("lambda_q_value", 1.0)
    lambda_policy = args.lambda_policy
    if lambda_policy is None:
        lambda_policy = model_cfg.get("lambda_policy", 1.0)
    lambda_value_teacher = args.lambda_value_teacher
    if lambda_value_teacher is None:
        lambda_value_teacher = model_cfg.get("lambda_value_teacher", 0.25)
    lambda_q_teacher = args.lambda_q_teacher
    if lambda_q_teacher is None:
        lambda_q_teacher = model_cfg.get("lambda_q_teacher", 0.25)
    advantage_temperature = args.advantage_temperature
    if advantage_temperature is None:
        advantage_temperature = model_cfg.get("advantage_temperature", None)
    advantage_weight_min = model_cfg.get("advantage_weight_min", 0.1)
    advantage_weight_max = model_cfg.get("advantage_weight_max", 10.0)
    gamma = args.gamma
    if gamma is None:
        gamma = model_cfg.get("gamma", 1.0)
    sigreg_num_slices = model_cfg.get("sigreg_num_slices", SIGREG_NUM_SLICES)
    sigreg_num_points = model_cfg.get("sigreg_num_points", SIGREG_NUM_POINTS)
    sigreg_domain = model_cfg.get("sigreg_domain", SIGREG_DOMAIN)
    context_length = model_cfg.get("encoder", {}).get("max_seq_len", CONTEXT_LENGTH)
    temporal_max_seq_len = model_cfg.get("temporal_encoder", {}).get("max_seq_len", 6144)
    # Defaults if config has null (match previous default.yaml concrete values)
    if context_length is None:
        context_length = 256
    if temporal_max_seq_len is None:
        temporal_max_seq_len = 6144

    # ── auto-detect max_seq_len from dataset sequence_stats.json ──
    # The stats file already includes a 1.2× safety multiplier on max values.
    auto_state_block_max = None
    auto_temporal_max = None
    auto_safety_multiplier = None
    for fmt in args.formats:
        stats_path = os.path.join(args.data_root, fmt, "sequence_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                seq_stats = json.load(f)
            sl = seq_stats["state_block_len"]["max"]
            tl = seq_stats["temporal_sequence_len"]["max"]
            if auto_state_block_max is None or sl > auto_state_block_max:
                auto_state_block_max = sl
            if auto_temporal_max is None or tl > auto_temporal_max:
                auto_temporal_max = tl
            if auto_safety_multiplier is None:
                auto_safety_multiplier = seq_stats.get("safety_multiplier")
    if auto_state_block_max is not None:
        # The inflated max already includes safety margin; use it directly.
        if auto_state_block_max > context_length:
            context_length = auto_state_block_max
        model_cfg.setdefault("encoder", {})["max_seq_len"] = context_length
    if auto_temporal_max is not None:
        if auto_temporal_max > temporal_max_seq_len:
            temporal_max_seq_len = auto_temporal_max
        model_cfg.setdefault("temporal_encoder", {})["max_seq_len"] = temporal_max_seq_len
    if auto_safety_multiplier is not None:
        print(f"Auto-detected max_seq_len from sequence_stats.json (safety ×{auto_safety_multiplier})")

    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: bos={bos_id} eos={eos_id} pad={pad_id}")
    print(f"Structural token IDs: {structural_ids}")
    print(f"Latent dim: {latent_dim}  action_latent_dim: {action_latent_dim}")
    print(f"CONTEXT_LENGTH (encoder max_seq_len): {context_length}")
    print(f"Temporal max_seq_len: {temporal_max_seq_len}")

    train_shards = PairedJEPADataset.discover(args.data_root, args.formats, "train")
    val_shards = PairedJEPADataset.discover(
        args.data_root, args.formats, "val", required=False
    )
    outcome_loss_weight = (
        float(lambda_value)
        + float(lambda_q_value)
        + float(lambda_policy)
        + float(lambda_value_teacher)
        + float(lambda_q_teacher)
    )
    _validate_paired_outcomes(train_shards, outcome_loss_weight)
    train_dataset = PairedJEPADataset(
        train_shards, structural_ids, shuffle_shards=True,
        max_history_blocks=args.max_history_blocks,
    )
    val_loader = None
    if val_shards:
        val_dataset = PairedJEPADataset(
            val_shards, structural_ids, shuffle_shards=False,
            max_history_blocks=args.max_history_blocks,
        )
        val_loader = _make_loader(
            val_dataset,
            args.batch_size,
            pad_id,
            max(0, args.num_workers // 2),
            args.prefetch_factor,
            device.type == "cuda",
        )

    train_loader = _make_loader(
        train_dataset,
        args.batch_size,
        pad_id,
        args.num_workers,
        args.prefetch_factor,
        device.type == "cuda",
    )

    model = PairedJEPAModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        latent_dim=latent_dim,
        action_latent_dim=action_latent_dim,
        encoder_cfg=model_cfg.get("encoder", {}),
        temporal_encoder_cfg=model_cfg.get("temporal_encoder", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        opponent_belief_predictor_cfg=model_cfg.get(
            "opponent_belief_predictor",
            model_cfg.get("opponent_state_predictor", {})
        ),
        next_state_predictor_cfg=model_cfg.get("next_state_predictor", {}),
        rank_head_cfg=model_cfg.get("rank_head", {}),
        decision_state_encoder_cfg=model_cfg.get("decision_state_encoder", {}),
        value_head_cfg=model_cfg.get("value_head", {}),
        action_projector_cfg=model_cfg.get("action_projector", {}),
        action_value_head_cfg=model_cfg.get("action_value_head", {}),
    ).to(device)

    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── torch.compile submodules individually (CUDA only) ─────────
    # Compiling the full model or temporal_encoder hits a PyTorch 2.12
    # inductor partitioner bug (in-place tensor writes in the temporal
    # encoder interleave logic).  Compile only the heavy encoder + action
    # encoder; the temporal encoder and MLPs stay eager (they're small).
    if args.compile and device.type == "cuda":
        torch._dynamo.config.capture_scalar_outputs = True
        compiled_any = False
        for name in ["encoder", "action_encoder"]:
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
            print("torch.compile enabled on: encoder, action_encoder")

    if args.checkpoint and os.path.exists(args.checkpoint):
        _load_compatible_checkpoint(model, args.checkpoint, device)

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
        f"effective={args.batch_size * args.grad_accum_steps} "
        f"max_history_blocks={args.max_history_blocks} gamma={gamma}"
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
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
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
                "n_params": n_params,
                "n_train_transitions": train_transitions,
                "n_val_transitions": val_transitions,
                "context_length": context_length,
                "max_history_blocks": args.max_history_blocks,
                "lambda_sigreg": lambda_sigreg,
                "lambda_sigreg_state": lambda_sigreg_state,
                "lambda_sigreg_action": lambda_sigreg_action,
                "lambda_opponent_state": args.lambda_opponent_state,
                "lambda_action": args.lambda_action,
                "lambda_next_state": args.lambda_next_state,
                "lambda_value": lambda_value,
                "lambda_q_value": lambda_q_value,
                "lambda_policy": lambda_policy,
                "lambda_value_teacher": lambda_value_teacher,
                "lambda_q_teacher": lambda_q_teacher,
                "advantage_temperature": advantage_temperature,
                "gamma": gamma,
                "checkpoint": args.checkpoint,
            },
        )
    elif args.wandb and not _wandb_available:
        print("WARNING: --wandb enabled but wandb not installed (pip install wandb)")

    def loss_from_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_paired_losses(
            outputs,
            lambda_sigreg_state=lambda_sigreg_state,
            lambda_sigreg_action=lambda_sigreg_action,
            lambda_opponent_state=args.lambda_opponent_state,
            lambda_action=args.lambda_action,
            lambda_next_state=args.lambda_next_state,
            lambda_rank=lambda_rank,
            lambda_value=lambda_value,
            lambda_q_value=lambda_q_value,
            lambda_policy=lambda_policy,
            lambda_value_teacher=lambda_value_teacher,
            lambda_q_teacher=lambda_q_teacher,
            advantage_temperature=advantage_temperature,
            advantage_weight_min=advantage_weight_min,
            advantage_weight_max=advantage_weight_max,
            gamma=gamma,
            sigreg_num_slices=sigreg_num_slices,
            sigreg_num_points=sigreg_num_points,
            sigreg_domain=sigreg_domain,
        )
    use_td_bootstrap = abs(float(gamma) - 1.0) > 1e-8

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
                batch,
                compute_td_bootstrap=use_td_bootstrap,
            )
            _, metrics = loss_from_outputs(outputs)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        model.train()
        return {f"val_{key}": value / max(steps, 1) for key, value in totals.items()}

    global_step = 0
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    done = False
    t_last_print = time.time()
    tokens_since_print = 0
    t_last_wandb = time.time()
    tokens_since_wandb = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_totals: dict[str, float] = {}
        epoch_steps = 0
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            outputs = _forward_paired(
                model,
                batch,
                compute_td_bootstrap=use_td_bootstrap,
            )
            loss, metrics = loss_from_outputs(outputs)
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
                        f"max_history_blocks={args.max_history_blocks}. "
                        "Reduce JEPA_PAIRED_BATCH_SIZE or JEPA_MAX_HISTORY; "
                        "increase JEPA_PAIRED_GRAD_ACCUM_STEPS to keep the same effective batch."
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
            for key in ("p1_state_T", "p2_state_T", "p1_state_T1", "p2_state_T1",
                        "p1_action", "p2_action", "actual_p2_action_from_p1_perspective",
                        "actual_p1_action_from_p2_perspective",
                        "p1_legal_actions", "p2_legal_actions"):
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
                wandb_run.log({
                    "train/tok_per_sec": tok_per_sec_wandb,
                    "train/loss": metrics["loss"],
                    "train/opponent_state_loss": metrics["opponent_state_loss"],
                    "train/opponent_state_loss_p1_to_p2": metrics["opponent_state_loss_p1_to_p2"],
                    "train/opponent_state_loss_p2_to_p1": metrics["opponent_state_loss_p2_to_p1"],
                    "train/action_loss": metrics["action_loss"],
                    "train/action_loss_p1_to_p2": metrics["action_loss_p1_to_p2"],
                    "train/action_loss_p2_to_p1": metrics["action_loss_p2_to_p1"],
                    "train/next_state_loss": metrics["next_state_loss"],
                    "train/next_state_loss_p1": metrics["next_state_loss_p1"],
                    "train/next_state_loss_p2": metrics["next_state_loss_p2"],
                    "train/value_loss": metrics["value_loss"],
                    "train/q_value_loss": metrics["q_value_loss"],
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_teacher_loss": metrics["value_teacher_loss"],
                    "train/q_teacher_loss": metrics["q_teacher_loss"],
                    "train/sigreg_loss": metrics["sigreg_loss"],
                    "train/sigreg_state_loss": metrics["sigreg_state_loss"],
                    "train/sigreg_action_loss": metrics["sigreg_action_loss"],
                    "train/sigreg_current": metrics["sigreg_current"],
                    "train/sigreg_next_true": metrics["sigreg_next_true"],
                    "train/sigreg_context": metrics["sigreg_context"],
                    "train/sigreg_action_own": metrics["sigreg_action_own"],
                    "train/sigreg_action_opponent": metrics["sigreg_action_opponent"],
                    "train/next_state_logvar_p1": metrics["next_state_logvar_p1"],
                    "train/next_state_logvar_p2": metrics["next_state_logvar_p2"],
                    "train/p1_terminal_fraction": metrics["p1_terminal_fraction"],
                    "train/p2_terminal_fraction": metrics["p2_terminal_fraction"],
                    "train/p1_value_td_fraction": metrics["p1_value_td_fraction"],
                    "train/p2_value_td_fraction": metrics["p2_value_td_fraction"],
                    "train/p1_q_td_fraction": metrics["p1_q_td_fraction"],
                    "train/p2_q_td_fraction": metrics["p2_q_td_fraction"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "samples_seen": args.batch_size * global_step,
                })

            if args.print_interval > 0 and global_step % args.print_interval == 0:
                now = time.time()
                elapsed = now - t_last_print
                tok_per_sec = tokens_since_print / elapsed if elapsed > 0 else 0
                t_last_print = now
                tokens_since_print = 0
                print(
                    f"  epoch {epoch:3d} | step {global_step:6d} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"opp_state {metrics['opponent_state_loss']:.4f} "
                    f"[p1->p2 {metrics['opponent_state_loss_p1_to_p2']:.4f}, "
                    f"p2->p1 {metrics['opponent_state_loss_p2_to_p1']:.4f}] | "
                    f"action {metrics['action_loss']:.4f} "
                    f"[p1->p2 {metrics['action_loss_p1_to_p2']:.4f}, "
                    f"p2->p1 {metrics['action_loss_p2_to_p1']:.4f}] | "
                    f"next {metrics['next_state_loss']:.4f} "
                    f"[p1 {metrics['next_state_loss_p1']:.4f}, "
                    f"p2 {metrics['next_state_loss_p2']:.4f}] | "
                    f"value {metrics['value_loss']:.4f} | "
                    f"q {metrics['q_value_loss']:.4f} | "
                    f"policy {metrics['policy_loss']:.4f} | "
                    f"teacher {metrics['value_teacher_loss']:.4f}/{metrics['q_teacher_loss']:.4f} | "
                    f"sigreg_state {metrics['sigreg_state_loss']:.4f} "
                    f"[cur {metrics['sigreg_current']:.4f}, "
                    f"next {metrics['sigreg_next_true']:.4f}, "
                    f"ctx {metrics['sigreg_context']:.4f}] | "
                    f"sigreg_action {metrics['sigreg_action_loss']:.4f} "
                    f"[own {metrics['sigreg_action_own']:.4f}, "
                    f"opp {metrics['sigreg_action_opponent']:.4f}] | "
                    f"td_v {metrics['p1_value_td_fraction']:.2f}/{metrics['p2_value_td_fraction']:.2f} "
                    f"td_q {metrics['p1_q_td_fraction']:.2f}/{metrics['p2_q_td_fraction']:.2f} | "
                    f"logvar_next {metrics['next_state_logvar_p1']:.3f}/{metrics['next_state_logvar_p2']:.3f}"
                )

            if args.val_interval > 0 and global_step % args.val_interval == 0:
                val_metrics = validate(args.val_max_batches)
                if val_metrics:
                    print(
                        f"  val @ step {global_step:6d} | "
                        f"loss {val_metrics['val_loss']:.4f} | "
                        f"opp_state {val_metrics['val_opponent_state_loss']:.4f} | "
                        f"action {val_metrics['val_action_loss']:.4f} | "
                        f"next {val_metrics['val_next_state_loss']:.4f} | "
                        f"value {val_metrics.get('val_value_loss', 0.0):.4f} | "
                        f"q {val_metrics.get('val_q_value_loss', 0.0):.4f} | "
                        f"policy {val_metrics.get('val_policy_loss', 0.0):.4f} | "
                        f"sigreg_state {val_metrics.get('val_sigreg_state_loss', 0.0):.4f} | "
                        f"sigreg_action {val_metrics.get('val_sigreg_action_loss', 0.0):.4f}"
                    )
                    if wandb_run:
                        wandb_run.log({
                            "val/loss": val_metrics["val_loss"],
                            "val/opponent_state_loss": val_metrics["val_opponent_state_loss"],
                            "val/action_loss": val_metrics["val_action_loss"],
                            "val/next_state_loss": val_metrics["val_next_state_loss"],
                            "val/value_loss": val_metrics.get("val_value_loss", 0.0),
                            "val/q_value_loss": val_metrics.get("val_q_value_loss", 0.0),
                            "val/policy_loss": val_metrics.get("val_policy_loss", 0.0),
                            "val/sigreg_loss": val_metrics.get("val_sigreg_loss", 0.0),
                            "val/sigreg_state_loss": val_metrics.get("val_sigreg_state_loss", 0.0),
                            "val/sigreg_action_loss": val_metrics.get("val_sigreg_action_loss", 0.0),
                            "epoch": epoch,
                            "global_step": global_step,
                            "samples_seen": args.batch_size * global_step,
                        })
                    if val_metrics["val_loss"] < best_val_loss and args.checkpoint:
                        best_val_loss = val_metrics["val_loss"]
                        save_paired_jepa_checkpoint(
                            model,
                            args.checkpoint,
                            epoch=epoch,
                            global_step=global_step,
                            config=model_cfg,
                            vocab_size=vocab_size,
                            max_history_blocks=args.max_history_blocks,
                            tokenizer=tokenizer,
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
        msg = (
            f"=== epoch {epoch:3d} done | train loss {avg.get('loss', 0.0):.4f} | "
            f"opp_state {avg.get('opponent_state_loss', 0.0):.4f} | "
            f"action {avg.get('action_loss', 0.0):.4f} | "
            f"next {avg.get('next_state_loss', 0.0):.4f} | "
            f"value {avg.get('value_loss', 0.0):.4f} | "
            f"q {avg.get('q_value_loss', 0.0):.4f} | "
            f"policy {avg.get('policy_loss', 0.0):.4f} | "
            f"sigreg_state {avg.get('sigreg_state_loss', 0.0):.4f} | "
            f"sigreg_action {avg.get('sigreg_action_loss', 0.0):.4f}"
        )
        if val_metrics:
            msg += (
                f" | val loss {val_metrics.get('val_loss', 0.0):.4f}"
                f" | val sigreg_state {val_metrics.get('val_sigreg_state_loss', 0.0):.4f}"
                f" | val sigreg_action {val_metrics.get('val_sigreg_action_loss', 0.0):.4f}"
            )
        print(msg + " ===")

        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": avg.get("loss", 0.0),
                "epoch/train_opponent_state_loss": avg.get("opponent_state_loss", 0.0),
                "epoch/train_action_loss": avg.get("action_loss", 0.0),
                "epoch/train_next_state_loss": avg.get("next_state_loss", 0.0),
                "epoch/train_value_loss": avg.get("value_loss", 0.0),
                "epoch/train_q_value_loss": avg.get("q_value_loss", 0.0),
                "epoch/train_policy_loss": avg.get("policy_loss", 0.0),
                "epoch/train_sigreg_state_loss": avg.get("sigreg_state_loss", 0.0),
                "epoch/train_sigreg_action_loss": avg.get("sigreg_action_loss", 0.0),
                "epoch/val_loss": val_metrics.get("val_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_opponent_state_loss": val_metrics.get("val_opponent_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_action_loss": val_metrics.get("val_action_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_next_state_loss": val_metrics.get("val_next_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_value_loss": val_metrics.get("val_value_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_q_value_loss": val_metrics.get("val_q_value_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_policy_loss": val_metrics.get("val_policy_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_sigreg_loss": val_metrics.get("val_sigreg_loss", 0.0) if val_metrics else 0.0,
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
        )
        if args.checkpoint and (not val_metrics or val_metrics.get("val_loss", float("inf")) < best_val_loss):
            best_val_loss = val_metrics.get("val_loss", avg.get("loss", best_val_loss))
            save_paired_jepa_checkpoint(
                model,
                args.checkpoint,
                epoch=epoch,
                global_step=global_step,
                config=model_cfg,
                vocab_size=vocab_size,
                max_history_blocks=args.max_history_blocks,
                tokenizer=tokenizer,
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
    parser.add_argument("--lambda_sigreg", type=float, default=None,
                        help="Deprecated: backward-compat fallback for lambda_sigreg_state and lambda_sigreg_action.")
    parser.add_argument("--lambda_sigreg_state", type=float, default=None,
                        help="SIGReg weight on state latents (current encoder outputs, true next-state targets, context; not predicted next-state latents). Default from config or 0.1.")
    parser.add_argument("--lambda_sigreg_action", type=float, default=None,
                        help="SIGReg weight on true action encoder outputs (own and opponent actions from both POVs). Default from config or 0.0 (off).")
    parser.add_argument("--lambda_opponent_state", type=float, default=1.0)
    parser.add_argument("--lambda_action", type=float, default=1.0)
    parser.add_argument("--lambda_next_state", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=None,
                        help="Deprecated/no-op: ranking loss has been replaced by value/Q losses.")
    parser.add_argument("--lambda_value", type=float, default=None)
    parser.add_argument("--lambda_q_value", type=float, default=None)
    parser.add_argument("--lambda_policy", type=float, default=None)
    parser.add_argument("--lambda_value_teacher", type=float, default=None)
    parser.add_argument("--lambda_q_teacher", type=float, default=None)
    parser.add_argument("--advantage_temperature", type=float, default=None,
                        help="Optional advantage-weighting temperature for Q/policy losses. Disabled when unset.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="TD discount factor for V and Q heads. Non-terminal rollout steps use "
                             "the furthest valid in-window gamma^n * V/Q bootstrap target; true "
                             "terminals stop early on the discounted outcome. Set to 1.0 to disable "
                             "TD bootstrapping and use MC outcome supervision. Typical: 0.95-0.99.")
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=0,
                        help="Log every N training steps to wandb (0 = same as print_interval).")
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction,
                        help="Enable Weights & Biases logging (default: True). Use --no-wandb to disable.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (default: metamon-jepa-<format>).")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name.")
    parser.add_argument("--max_history_blocks", type=int, default=0,
                        help="Maximum non-header history state blocks per sample (0 = unlimited). "
                             "The team header is always retained. Lower = faster data loading + shorter "
                             "temporal sequences. Default: 0 (unlimited)")
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable torch.compile (default: False). Use --compile to enable.")
    train(parser.parse_args())
