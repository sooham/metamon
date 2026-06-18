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
                if name.startswith("shard_") and name.endswith(".npz"):
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
            total += int(len(data["state_idx"]))
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

    def _iter_shard(self, path: str) -> Iterator[dict[str, object]]:
        data = np.load(path)
        cm_id = self.structural["chosen_move"]
        ecm_id = self.structural["end_chosen_move"]
        ocm_id = self.structural["opponent_chosen_move"]
        eocm_id = self.structural["end_opponent_chosen_move"]

        n_transitions = len(data["state_idx"])
        order = np.arange(n_transitions)
        if self.shuffle_transitions:
            np.random.default_rng().shuffle(order)

        for row in order:
            battle_id = int(data["battle_id"][row])
            state_idx = int(data["state_idx"][row])
            next_state_idx = int(data["next_state_idx"][row])
            action_idx = int(data["action_idx"][row])

            battle_start = int(data["battle_start"][battle_id])
            action_start = int(data["battle_action_start"][battle_id])

            yield {
                "state_N_tokens": self._slice_blocks(
                    data["states"], data["state_offsets"], data["state_lengths"],
                    battle_start, state_idx + 1,
                ),
                "state_N1_tokens": self._slice_blocks(
                    data["states"], data["state_offsets"], data["state_lengths"],
                    battle_start, next_state_idx + 1,
                ),
                "player_hist_N_tokens": self._slice_action_blocks(
                    data["actions"], data["action_offsets"], data["action_lengths"],
                    action_start, action_idx, cm_id, ecm_id,
                ),
                "opponent_hist_N_tokens": self._slice_action_blocks(
                    data["opponent_actions"],
                    data["opponent_action_offsets"],
                    data["opponent_action_lengths"],
                    action_start, action_idx, ocm_id, eocm_id,
                ),
                "player_hist_N1_tokens": self._slice_action_blocks(
                    data["actions"], data["action_offsets"], data["action_lengths"],
                    action_start, action_idx + 1, cm_id, ecm_id,
                ),
                "opponent_hist_N1_tokens": self._slice_action_blocks(
                    data["opponent_actions"],
                    data["opponent_action_offsets"],
                    data["opponent_action_lengths"],
                    action_start, action_idx + 1, ocm_id, eocm_id,
                ),
                "player_action_tokens": self._slice_action_blocks(
                    data["actions"], data["action_offsets"], data["action_lengths"],
                    action_idx, action_idx + 1, cm_id, ecm_id,
                )[0],
                "opponent_action_tokens": self._slice_action_blocks(
                    data["opponent_actions"],
                    data["opponent_action_offsets"],
                    data["opponent_action_lengths"],
                    action_idx, action_idx + 1, ocm_id, eocm_id,
                )[0],
            }

    def __iter__(self) -> Iterator[dict[str, object]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]
        for path in paths:
            yield from self._iter_shard(path)
