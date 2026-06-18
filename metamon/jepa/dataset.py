"""Dataset classes for JEPA training (shared by paired and legacy single-POV)."""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch


class JEPADataset(torch.utils.data.IterableDataset):
    """Iterable dataset over legacy (single-POV) transition-table shards.

    Reads ``shard_*.npz`` files produced by
    ``scripts/generate_world_model_data.py`` (without ``--paired_pov``).
    """

    def __init__(
        self,
        shard_paths: list[str],
        structural_token_ids: dict[str, int],
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.structural = structural_token_ids
        self.shuffle_shards = shuffle_shards
        self.shuffle_transitions = shuffle_shards

    @staticmethod
    def from_formats(
        data_root: str,
        formats: list[str],
        split: str,
        structural_token_ids: dict[str, int],
        *,
        required: bool = True,
    ) -> "JEPADataset":
        """Discover shard paths and return a configured dataset."""
        shard_paths: list[str] = []
        for fmt in formats:
            split_dir = os.path.join(data_root, fmt, split)
            if not os.path.isdir(split_dir):
                continue
            for name in sorted(os.listdir(split_dir)):
                if (
                    (name.startswith("shard_") or name.startswith("seq_shard_"))
                    and name.endswith(".npz")
                ):
                    shard_paths.append(os.path.join(split_dir, name))
        if required and not shard_paths:
            raise FileNotFoundError(
                f"No {split!r} shards found under {data_root} for {formats}"
            )
        return JEPADataset(shard_paths, structural_token_ids)

    @staticmethod
    def count_transitions(shard_paths: list[str]) -> int:
        total = 0
        for path in shard_paths:
            data = np.load(path)
            key = "state_idx" if "state_idx" in data else "prev_state_idx"
            total += int(len(data[key]))
        return total

    @staticmethod
    def _slice_blocks(
        flat: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        start_idx: int,
        end_idx_exclusive: int,
    ) -> list[np.ndarray]:
        blocks: list[np.ndarray] = []
        for idx in range(start_idx, end_idx_exclusive):
            off = int(offsets[idx])
            length = int(lengths[idx])
            blocks.append(flat[off:off + length].astype(np.int16, copy=False))
        return blocks

    @staticmethod
    def _slice_action_blocks(
        flat: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        start_idx: int,
        end_idx_exclusive: int,
        start_token_id: int,
        end_token_id: int,
    ) -> list[np.ndarray]:
        blocks: list[np.ndarray] = []
        prefix = np.array([start_token_id], dtype=np.int16)
        suffix = np.array([end_token_id], dtype=np.int16)
        for idx in range(start_idx, end_idx_exclusive):
            off = int(offsets[idx])
            length = int(lengths[idx])
            content = flat[off:off + length].astype(np.int16, copy=False)
            blocks.append(np.concatenate([prefix, content, suffix]))
        return blocks

    def _iter_shard(self, path: str) -> Iterator[tuple[object, ...]]:
        data = np.load(path)
        cm_id = self.structural["chosen_move"]
        ecm_id = self.structural["end_chosen_move"]
        ocm_id = self.structural["opponent_chosen_move"]
        eocm_id = self.structural["end_opponent_chosen_move"]

        state_idx_arr = data["state_idx"] if "state_idx" in data else data["prev_state_idx"]
        next_state_idx_arr = data["next_state_idx"]
        action_idx_arr = data["action_idx"] if "action_idx" in data else np.arange(len(state_idx_arr))
        player_actions_key = "actions" if "actions" in data else "player_actions"
        player_action_offsets_key = "action_offsets" if "action_offsets" in data else "player_action_offsets"
        player_action_lengths_key = "action_lengths" if "action_lengths" in data else "player_action_lengths"
        opponent_actions_key = "opponent_actions"
        opponent_action_offsets_key = "opponent_action_offsets"
        opponent_action_lengths_key = "opponent_action_lengths"

        n_transitions = len(state_idx_arr)
        order = np.arange(n_transitions)
        if self.shuffle_transitions:
            np.random.default_rng().shuffle(order)

        for row in order:
            battle_id = int(data["battle_id"][row])
            state_idx = int(state_idx_arr[row])
            next_state_idx = int(next_state_idx_arr[row])
            action_idx = int(action_idx_arr[row])

            battle_start = int(data["battle_start"][battle_id])
            action_start = int(data["battle_action_start"][battle_id])

            state_blocks_N = self._slice_blocks(
                data["states"], data["state_offsets"], data["state_lengths"],
                battle_start, state_idx + 1,
            )
            state_blocks_N1 = self._slice_blocks(
                data["states"], data["state_offsets"], data["state_lengths"],
                battle_start, next_state_idx + 1,
            )
            pa_hist_N = self._slice_action_blocks(
                data[player_actions_key], data[player_action_offsets_key], data[player_action_lengths_key],
                action_start, action_idx, cm_id, ecm_id,
            )
            oa_hist_N = self._slice_action_blocks(
                data[opponent_actions_key], data[opponent_action_offsets_key], data[opponent_action_lengths_key],
                action_start, action_idx, ocm_id, eocm_id,
            )
            pa_hist_N1 = self._slice_action_blocks(
                data[player_actions_key], data[player_action_offsets_key], data[player_action_lengths_key],
                action_start, action_idx + 1, cm_id, ecm_id,
            )
            oa_hist_N1 = self._slice_action_blocks(
                data[opponent_actions_key], data[opponent_action_offsets_key], data[opponent_action_lengths_key],
                action_start, action_idx + 1, ocm_id, eocm_id,
            )
            pa_tokens = self._slice_action_blocks(
                data[player_actions_key], data[player_action_offsets_key], data[player_action_lengths_key],
                action_idx, action_idx + 1, cm_id, ecm_id,
            )[0]
            oa_tokens = self._slice_action_blocks(
                data[opponent_actions_key], data[opponent_action_offsets_key], data[opponent_action_lengths_key],
                action_idx, action_idx + 1, ocm_id, eocm_id,
            )[0]

            yield (
                state_blocks_N,
                state_blocks_N1,
                pa_hist_N,
                oa_hist_N,
                pa_hist_N1,
                oa_hist_N1,
                pa_tokens,
                oa_tokens,
            )

    def __iter__(self) -> Iterator[tuple[object, ...]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]
        for path in paths:
            yield from self._iter_shard(path)
