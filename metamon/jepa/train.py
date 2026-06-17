"""Train a LeJEPA v2 model on world-model battle-prefix transitions.

Loads sharded .npz files produced by scripts/generate_world_model_data.py
and trains a LeJEPA (Latent-Euclidean Joint Embedding Predictive Architecture)
model that:

1. Encodes each team-header/state block separately, encodes historical player
   and opponent actions separately, then runs a temporal transformer over
   interleaved block embeddings up to state N to produce *enc_N*.
2. Encodes current action texts (player / opponent) into action embeddings via a
   smaller bidirectional transformer sharing the main encoder's token embeddings.
3. Predicts the opponent's next action embedding from *enc_N* via a small MLP
   (JEPAActionPredictor). The player's chosen action is encoded directly.
4. Predicts the next-prefix embedding *enc_{N+1}* from
   (*enc_N*, player_action_embedding, predicted_opponent_action_embedding)
   via an AdaLN-zero conditional transformer
   (JEPAPredictor).
5. Regularises embeddings toward an isotropic Gaussian via SIGReg.

Usage:
    uv run python -m metamon.jepa.train \\
        --data_root $METAMON_CACHE_DIR/world-model-samples \\
        --formats gen1ou \\
        --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \\
        --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \\
        --batch_size 16 --lr 5e-5 --epochs 100

    # With wandb + CSV logging
    uv run python -m metamon.jepa.train \\
        --data_root $METAMON_CACHE_DIR/world-model-samples \\
        --formats gen1ou \\
        --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \\
        --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \\
        --batch_size 16 --lr 5e-5 --epochs 100 \\
        --wandb --wandb_project metamon-jepa --wandb_name run-01 \\
        --log --log_interval 100
"""

import argparse
import functools
import math
import os
import time
import sys
from pathlib import Path
from typing import Iterator, Optional

# Mitigate CUDA memory fragmentation — the allocator can expand existing
# segments instead of creating new ones, which helps when the compiled
# encoder/predictor create many CUDA graphs for different input shapes.
# Must be set before any CUDA API call (i.e. before importing torch).
if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{existing + ',' if existing else ''}expandable_segments:True"
    )

import numpy as np
import torch
import yaml

from metamon.jepa.model import (
    JEPAModel,
    compute_losses,
    LATENT_DIM,
    ACTION_LATENT_DIM,
    CONTEXT_LENGTH,
    SIGREG_NUM_SLICES,
    SIGREG_NUM_POINTS,
    SIGREG_DOMAIN,
)

# Optional wandb import
_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    pass


# ── Embedding stats helpers ──────────────────────────────────────────────

def _new_embedding_stats(device: torch.device, latent_dim: int) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "sum": torch.zeros(latent_dim, device=device, dtype=torch.float64),
        "sum_sq": torch.zeros(latent_dim, device=device, dtype=torch.float64),
        "outer": torch.zeros(latent_dim, latent_dim, device=device, dtype=torch.float64),
    }


def _update_embedding_stats(
    stats: dict[str, torch.Tensor | int],
    embeddings: torch.Tensor,
) -> None:
    z = embeddings.detach().to(dtype=torch.float64)
    stats["count"] = int(stats["count"]) + z.shape[0]
    stats["sum"] = stats["sum"] + z.sum(dim=0)
    stats["sum_sq"] = stats["sum_sq"] + z.square().sum(dim=0)
    stats["outer"] = stats["outer"] + z.T @ z


def _finalize_embedding_stats(
    stats: dict[str, torch.Tensor | int],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    count = int(stats["count"])
    if count <= 0:
        empty = np.array([], dtype=np.float32)
        return {
            "enc_dim_mean_mean": 0.0,
            "enc_dim_mean_abs_mean": 0.0,
            "enc_dim_mean_abs_max": 0.0,
            "enc_dim_std_mean": 0.0,
            "enc_dim_std_min": 0.0,
            "enc_dim_std_max": 0.0,
            "enc_cov_eff_rank": 0.0,
        }, {"enc_dim_mean": empty, "enc_dim_std": empty}

    z_sum = stats["sum"]
    z_sum_sq = stats["sum_sq"]
    mean = z_sum / count
    var = (z_sum_sq / count - mean.square()).clamp_min(0.0)
    std = var.sqrt()

    if count > 1:
        cov = (stats["outer"] - count * torch.outer(mean, mean)) / (count - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        eig_sum = eigvals.sum()
        if eig_sum > 0:
            probs = eigvals / eig_sum
            eff_rank = torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())
        else:
            eff_rank = torch.tensor(0.0, device=mean.device, dtype=mean.dtype)
    else:
        eff_rank = torch.tensor(0.0, device=mean.device, dtype=mean.dtype)

    scalar_metrics = {
        "enc_dim_mean_mean": mean.mean().item(),
        "enc_dim_mean_abs_mean": mean.abs().mean().item(),
        "enc_dim_mean_abs_max": mean.abs().max().item(),
        "enc_dim_std_mean": std.mean().item(),
        "enc_dim_std_min": std.min().item(),
        "enc_dim_std_max": std.max().item(),
        "enc_cov_eff_rank": eff_rank.item(),
    }
    arrays = {
        "enc_dim_mean": mean.detach().cpu().to(dtype=torch.float32).numpy(),
        "enc_dim_std": std.detach().cpu().to(dtype=torch.float32).numpy(),
    }
    return scalar_metrics, arrays


def _wandb_validation_payload(
    metrics: dict[str, float],
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "val/loss": metrics.get("val_loss", 0.0),
        "val/state_loss": metrics.get("val_state_loss", 0.0),
        "val/player_action_loss": metrics.get("val_player_action_loss", 0.0),
        "val/opponent_action_loss": metrics.get("val_opponent_action_loss", 0.0),
        "val/pred_loss": metrics.get("val_pred_loss", 0.0),
        "val/sigreg_enc": metrics.get("val_sigreg_enc", 0.0),
        "val/sigreg_act": metrics.get("val_sigreg_act", 0.0),
        "val/sigreg_loss": metrics.get("val_sigreg_loss", 0.0),
        "val/enc_dim_mean_mean": metrics.get("val_enc_dim_mean_mean", 0.0),
        "val/enc_dim_mean_abs_mean": metrics.get("val_enc_dim_mean_abs_mean", 0.0),
        "val/enc_dim_mean_abs_max": metrics.get("val_enc_dim_mean_abs_max", 0.0),
        "val/enc_dim_std_mean": metrics.get("val_enc_dim_std_mean", 0.0),
        "val/enc_dim_std_min": metrics.get("val_enc_dim_std_min", 0.0),
        "val/enc_dim_std_max": metrics.get("val_enc_dim_std_max", 0.0),
        "val/enc_cov_eff_rank": metrics.get("val_enc_cov_eff_rank", 0.0),
    }
    if _wandb_available and arrays["enc_dim_mean"].size > 0:
        payload["val/enc_dim_mean_hist"] = wandb.Histogram(arrays["enc_dim_mean"])
        payload["val/enc_dim_std_hist"] = wandb.Histogram(arrays["enc_dim_std"])
    return payload


def _validation_csv_fields(metrics: dict[str, float]) -> str:
    keys = (
        "val_loss",
        "val_state_loss",
        "val_player_action_loss",
        "val_opponent_action_loss",
        "val_pred_loss",
        "val_sigreg_enc",
        "val_sigreg_act",
        "val_sigreg_loss",
        "val_enc_dim_mean_mean",
        "val_enc_dim_mean_abs_mean",
        "val_enc_dim_mean_abs_max",
        "val_enc_dim_std_mean",
        "val_enc_dim_std_min",
        "val_enc_dim_std_max",
        "val_enc_cov_eff_rank",
    )
    return ",".join(f"{metrics.get(k, 0.0):.6f}" for k in keys)


# ── Dataset ─────────────────────────────────────────────────────────────

class JEPADataset(torch.utils.data.IterableDataset):
    """Iterable over sharded .npz files, yielding block-level transitions.

    Each transition yields:
      - state_blocks_N / state_blocks_N1: header + states through N / N+1
      - pa_hist_N / oa_hist_N: historical player/opponent action blocks before N
      - pa_hist_N1 / oa_hist_N1: historical action blocks through N+1
      - pa_tokens / oa_tokens: current transition action blocks

    Parameters
    ----------
    shard_paths : list[str]
        Paths to .npz shard files.
    structural_token_ids : dict[str, int]
        Token IDs for structural tokens needed to reconstruct action texts.
    shuffle_shards : bool
        Whether to shuffle shard order each epoch.
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

        if not self.shard_paths:
            raise ValueError("No shard paths provided")

    @staticmethod
    def count_transitions(shard_paths: list[str]) -> int:
        """Quickly count total transition pairs across shards (reads metadata only)."""
        total = 0
        for path in shard_paths:
            data = np.load(path)
            if "prev_state_idx" in data:
                total += int(len(data["prev_state_idx"]))
            else:
                # Legacy fallback
                battle_start = data["battle_start"]
                num_battles = len(battle_start) - 1
                for b in range(num_battles):
                    n_states = int(battle_start[b + 1]) - int(battle_start[b])
                    if n_states >= 2:
                        total += n_states - 1
        return total

    @classmethod
    def from_formats(
        cls,
        data_root: str,
        formats: list[str],
        split: str,
        structural_token_ids: dict[str, int],
        shuffle_shards: bool = True,
    ) -> "JEPADataset":
        """Discover .npz shards under *data_root*/*format*/*split*."""
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        shard_paths: list[str] = []
        for fmt in formats:
            fmt_dir = os.path.join(data_root, fmt, split)
            if not os.path.isdir(fmt_dir):
                continue
            for f in sorted(os.listdir(fmt_dir)):
                if f.endswith(".npz"):
                    shard_paths.append(os.path.join(fmt_dir, f))

        if not shard_paths:
            raise FileNotFoundError(
                f"No {split!r} .npz shards found under {data_root} for formats {formats}"
            )
        return cls(shard_paths, structural_token_ids, shuffle_shards)

    def _slice_state_blocks(
        self,
        states: np.ndarray,
        state_offsets: np.ndarray,
        state_lengths: np.ndarray,
        start_idx: int,
        end_idx_exclusive: int,
    ) -> list[np.ndarray]:
        """Return state/header blocks for local shard indices [start, end)."""
        blocks: list[np.ndarray] = []
        for idx in range(start_idx, end_idx_exclusive):
            off = int(state_offsets[idx])
            length = int(state_lengths[idx])
            blocks.append(states[off:off + length].astype(np.int16, copy=False))
        return blocks

    def _slice_action_blocks(
        self,
        actions: np.ndarray,
        action_offsets: np.ndarray,
        action_lengths: np.ndarray,
        start_idx: int,
        end_idx_exclusive: int,
        start_token_id: int,
        end_token_id: int,
    ) -> list[np.ndarray]:
        """Return action blocks with side-specific delimiters attached."""
        blocks: list[np.ndarray] = []
        for idx in range(start_idx, end_idx_exclusive):
            off = int(action_offsets[idx])
            length = int(action_lengths[idx])
            content = actions[off:off + length].astype(np.int16, copy=False)
            blocks.append(np.concatenate([
                np.array([start_token_id], dtype=np.int16),
                content,
                np.array([end_token_id], dtype=np.int16),
            ]))
        return blocks

    def _iter_shard(
        self, path: str
    ) -> Iterator[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]]:
        """Yield block-level transition samples for a single shard.

        For a transition from state N to state N+1:
          - state_blocks_N contains header + states through N.
          - *_hist_N contains actions before state N.
          - state_blocks_N1 contains header + states through N+1.
          - *_hist_N1 contains actions through the action that produced N+1.
          - pa_tokens / oa_tokens are the current transition's action blocks.
        """
        data = np.load(path)
        states = data["states"]
        state_offsets = data["state_offsets"]
        state_lengths = data["state_lengths"]
        prev_state_idx = data["prev_state_idx"]
        next_state_idx = data["next_state_idx"]
        battle_id = data["battle_id"]
        battle_start = data["battle_start"]
        battle_action_start = data["battle_action_start"]
        player_actions = data["player_actions"]
        player_action_offsets = data["player_action_offsets"]
        player_action_lengths = data["player_action_lengths"]
        opponent_actions = data["opponent_actions"]
        opponent_action_offsets = data["opponent_action_offsets"]
        opponent_action_lengths = data["opponent_action_lengths"]

        cm_id = self.structural["chosen_move"]
        ecm_id = self.structural["end_chosen_move"]
        ocm_id = self.structural["opponent_chosen_move"]
        eocm_id = self.structural["end_opponent_chosen_move"]

        n_trans = len(prev_state_idx)
        order = np.arange(n_trans)
        if self.shuffle_transitions:
            rng = np.random.default_rng()
            rng.shuffle(order)

        for row in order:
            b = int(battle_id[row])
            prev_s = int(prev_state_idx[row])
            next_s = int(next_state_idx[row])

            b_start = int(battle_start[b])
            a_start = int(battle_action_start[b])

            # Local step within battle (0 = header, 1 = state_0, ...)
            prev_step = prev_s - b_start
            next_step = next_s - b_start

            state_blocks_N = self._slice_state_blocks(
                states, state_offsets, state_lengths, b_start, prev_s + 1
            )
            state_blocks_N1 = self._slice_state_blocks(
                states, state_offsets, state_lengths, b_start, next_s + 1
            )

            # Historical actions before state_N: action indices [0, prev_step - 1).
            # Historical actions through state_N1: [0, next_step - 1).
            pa_hist_N = self._slice_action_blocks(
                player_actions, player_action_offsets, player_action_lengths,
                a_start, a_start + max(prev_step - 1, 0), cm_id, ecm_id,
            )
            oa_hist_N = self._slice_action_blocks(
                opponent_actions, opponent_action_offsets, opponent_action_lengths,
                a_start, a_start + max(prev_step - 1, 0), ocm_id, eocm_id,
            )
            pa_hist_N1 = self._slice_action_blocks(
                player_actions, player_action_offsets, player_action_lengths,
                a_start, a_start + max(next_step - 1, 0), cm_id, ecm_id,
            )
            oa_hist_N1 = self._slice_action_blocks(
                opponent_actions, opponent_action_offsets, opponent_action_lengths,
                a_start, a_start + max(next_step - 1, 0), ocm_id, eocm_id,
            )

            current_action_idx = a_start + prev_step - 1
            pa_tokens = self._slice_action_blocks(
                player_actions, player_action_offsets, player_action_lengths,
                current_action_idx, current_action_idx + 1, cm_id, ecm_id,
            )[0]
            oa_tokens = self._slice_action_blocks(
                opponent_actions, opponent_action_offsets, opponent_action_lengths,
                current_action_idx, current_action_idx + 1, ocm_id, eocm_id,
            )[0]

            yield (
                state_blocks_N, state_blocks_N1,
                pa_hist_N, oa_hist_N,
                pa_hist_N1, oa_hist_N1,
                pa_tokens, oa_tokens,
            )

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            yield from self._iter_shard(path)


# ── Collate ──────────────────────────────────────────────────────────────

def collate_fn(
    batch: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]],
    pad_id: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Collate variable-length block-level transitions into padded tensors.

    Args:
        batch: list of block-level samples from :class:`JEPADataset`.
        pad_id: token ID for padding.

    Returns:
        Padded state/history block tensors plus current action tensors.
    """
    def pad_block_lists(
        block_lists: list[list[np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_blocks = max((len(blocks) for blocks in block_lists), default=0)
        max_tokens = max(
            (len(block) for blocks in block_lists for block in blocks),
            default=1,
        )
        padded = torch.full(
            (len(block_lists), max_blocks, max_tokens),
            pad_id,
            dtype=torch.long,
        )
        valid = torch.zeros((len(block_lists), max_blocks), dtype=torch.bool)
        for batch_idx, blocks in enumerate(block_lists):
            for block_idx, block in enumerate(blocks):
                tokens = torch.from_numpy(block.astype(np.int64))
                padded[batch_idx, block_idx, :len(tokens)] = tokens
                valid[batch_idx, block_idx] = True
        return padded, valid

    def pad_actions(actions: list[np.ndarray]) -> torch.Tensor:
        max_tokens = max((len(action) for action in actions), default=1)
        padded = torch.full((len(actions), max_tokens), pad_id, dtype=torch.long)
        for batch_idx, action in enumerate(actions):
            tokens = torch.from_numpy(action.astype(np.int64))
            padded[batch_idx, :len(tokens)] = tokens
        return padded

    state_N, state_N_valid = pad_block_lists([item[0] for item in batch])
    state_N1, state_N1_valid = pad_block_lists([item[1] for item in batch])
    pa_hist_N, pa_hist_N_valid = pad_block_lists([item[2] for item in batch])
    oa_hist_N, oa_hist_N_valid = pad_block_lists([item[3] for item in batch])
    pa_hist_N1, pa_hist_N1_valid = pad_block_lists([item[4] for item in batch])
    oa_hist_N1, oa_hist_N1_valid = pad_block_lists([item[5] for item in batch])
    pa_tokens = pad_actions([item[6] for item in batch])
    oa_tokens = pad_actions([item[7] for item in batch])

    return (
        state_N, state_N_valid,
        state_N1, state_N1_valid,
        pa_hist_N, pa_hist_N_valid,
        oa_hist_N, oa_hist_N_valid,
        pa_hist_N1, pa_hist_N1_valid,
        oa_hist_N1, oa_hist_N1_valid,
        pa_tokens, oa_tokens,
    )


# ── Training loop ───────────────────────────────────────────────────────

def train(args):
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")

    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.print_interval > 0:
        print(f"Using device: {device}")

    # ---- config ----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]

    # ---- tokenizer (for vocab size + special token IDs + structural IDs) ----
    from metamon.tokenizer import PokemonTokenizer

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    vocab_size = model_cfg.get("vocab_size") or len(tokenizer)

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]

    # Structural token IDs for prefix/action reconstruction
    structural_ids = {
        "boa": tokenizer["<boa>"],
        "eoa": tokenizer["<eoa>"],
        "chosen_move": tokenizer["<chosen_move>"],
        "end_chosen_move": tokenizer["<end_chosen_move>"],
        "opponent_chosen_move": tokenizer["<opponent_chosen_move>"],
        "end_opponent_chosen_move": tokenizer["<end_opponent_chosen_move>"],
    }

    if args.print_interval > 0:
        print(f"Vocabulary size: {vocab_size}")
        print(f"Special tokens: bos={bos_id} eos={eos_id} pad={pad_id}")
        print(f"Structural token IDs: {structural_ids}")

    # ---- model hyperparameters ----
    latent_dim = model_cfg.get("latent_dim", LATENT_DIM)
    action_latent_dim = model_cfg.get("action_latent_dim", ACTION_LATENT_DIM)
    lambda_sigreg = model_cfg.get("lambda_sigreg", 0.1)
    if args.lambda_sigreg is not None:
        lambda_sigreg = args.lambda_sigreg
    sigreg_num_slices = model_cfg.get("sigreg_num_slices", SIGREG_NUM_SLICES)
    sigreg_num_points = model_cfg.get("sigreg_num_points", SIGREG_NUM_POINTS)
    sigreg_domain = model_cfg.get("sigreg_domain", SIGREG_DOMAIN)

    context_length = model_cfg.get("encoder", {}).get("max_seq_len", CONTEXT_LENGTH)

    if args.print_interval > 0:
        print(f"Latent dim: {latent_dim}  action_latent_dim: {action_latent_dim}")
        print(f"λ_sigreg={lambda_sigreg}  SIGReg slices={sigreg_num_slices} points={sigreg_num_points} domain={sigreg_domain}")
        print(f"CONTEXT_LENGTH (encoder max_seq_len): {context_length}")

    # ---- datasets (train / val split generated at raw-battle-group level) ----
    train_shards = JEPADataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        split="train",
        structural_token_ids=structural_ids,
        shuffle_shards=False,
    ).shard_paths
    val_shards = JEPADataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        split="val",
        structural_token_ids=structural_ids,
        shuffle_shards=False,
    ).shard_paths

    train_dataset = JEPADataset(
        train_shards, structural_ids, shuffle_shards=True,
    )
    val_dataset = JEPADataset(
        val_shards, structural_ids, shuffle_shards=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate_fn, pad_id=pad_id),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate_fn, pad_id=pad_id),
        num_workers=max(1, args.num_workers // 2),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=False,
    )

    # ---- model ----
    model = JEPAModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        latent_dim=latent_dim,
        action_latent_dim=action_latent_dim,
        encoder_cfg=model_cfg.get("encoder", {}),
        temporal_encoder_cfg=model_cfg.get("temporal_encoder", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        action_predictor_cfg=model_cfg.get("action_predictor", {}),
        predictor_cfg=model_cfg.get("predictor", {}),
    ).to(device)

    # BF16 + TF32 for GPU training
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.cache_size_limit = 512

    # Compile is opt-in. reduce-overhead CUDA graphs can reuse output storage
    # across micro-batches and break gradient accumulation.
    if args.compile and device.type == "cuda":
        try:
            model.predictor = torch.compile(
                model.predictor, dynamic=True
            )
        except Exception:
            pass
        try:
            model.action_predictor = torch.compile(
                model.action_predictor, dynamic=True
            )
        except Exception:
            pass

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=True if device.type == "cuda" else False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ---- checkpoint dir ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb ----
    wandb_run: Optional["wandb"] = None
    if args.wandb and _wandb_available:
        wandb_init_kwargs = dict(
            project=args.wandb_project or "metamon-jepa",
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
                "n_params": n_params,
                "lambda_sigreg": lambda_sigreg,
                "sigreg_num_slices": sigreg_num_slices,
                "sigreg_num_points": sigreg_num_points,
                "sigreg_domain": sigreg_domain,
                "context_length": context_length,
            },
        )
    elif args.wandb and not _wandb_available:
        print("WARNING: --wandb set but wandb not installed (pip install wandb)")

    # ---- CSV logging ----
    log_file = None
    if args.log:
        log_path = save_dir / "metrics.csv"
        log_file = open(log_path, "w")
        log_file.write(
            "epoch,step,loss,state_loss,player_action_loss,opponent_action_loss,"
            "pred_loss,sigreg_enc,sigreg_act,sigreg_loss,lr,"
            "enc_tok_s,"
            "val_loss,val_state_loss,val_player_action_loss,val_opponent_action_loss,"
            "val_pred_loss,val_sigreg_enc,val_sigreg_act,val_sigreg_loss,"
            "val_enc_dim_mean_mean,val_enc_dim_mean_abs_mean,val_enc_dim_mean_abs_max,"
            "val_enc_dim_std_mean,val_enc_dim_std_min,val_enc_dim_std_max,"
            "val_enc_cov_eff_rank\n"
        )

    # ---- print header ----
    if args.print_interval > 0:
        train_transitions = JEPADataset.count_transitions(train_shards)
        train_batches = math.ceil(train_transitions / args.batch_size)
        val_transitions = JEPADataset.count_transitions(val_shards)
        val_batches = math.ceil(val_transitions / args.batch_size)
        print(f"Params: {n_params:,}  "
              f"Shards: {len(train_shards)} train + {len(val_shards)} val "
              f"= {len(train_shards) + len(val_shards)} total")
        print(f"Transitions: {train_transitions:,} train  {val_transitions:,} val  "
              f"→ {train_batches:,} train batches/epoch  {val_batches:,} val batches")
        print(
            f"Batch size: {args.batch_size}  grad_accum_steps: {args.grad_accum_steps}  "
            f"effective batch: {args.batch_size * args.grad_accum_steps}  "
            f"CONTEXT_LENGTH: {context_length}"
        )

    # ---- validation function ----
    @torch.no_grad()
    def run_validation(max_batches: int | None = None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        model.eval()
        total_metrics: dict[str, float] = {}
        total_steps = 0
        embedding_stats = _new_embedding_stats(device, latent_dim)

        compiled_predictor = model.predictor
        if hasattr(compiled_predictor, "_orig_mod"):
            model.predictor = compiled_predictor._orig_mod
        compiled_action_pred = model.action_predictor
        if hasattr(compiled_action_pred, "_orig_mod"):
            model.action_predictor = compiled_action_pred._orig_mod
        try:
            for batch_idx, batch_tuple in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                (
                    state_N, state_N_valid,
                    state_N1, state_N1_valid,
                    pa_hist_N, pa_hist_N_valid,
                    oa_hist_N, oa_hist_N_valid,
                    pa_hist_N1, pa_hist_N1_valid,
                    oa_hist_N1, oa_hist_N1_valid,
                    pa_tokens, oa_tokens,
                ) = (tensor.to(device) for tensor in batch_tuple)

                outputs = model(
                    state_N, state_N_valid,
                    state_N1, state_N1_valid,
                    pa_hist_N, pa_hist_N_valid,
                    oa_hist_N, oa_hist_N_valid,
                    pa_hist_N1, pa_hist_N1_valid,
                    oa_hist_N1, oa_hist_N1_valid,
                    pa_tokens, oa_tokens,
                )
                _update_embedding_stats(
                    embedding_stats,
                    torch.cat([outputs["enc_N"], outputs["enc_N1"]], dim=0),
                )
                _, metrics = compute_losses(
                    outputs,
                    lambda_sigreg=lambda_sigreg,
                    sigreg_num_slices=sigreg_num_slices,
                    sigreg_num_points=sigreg_num_points,
                    sigreg_domain=sigreg_domain,
                )
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                total_steps += 1
        finally:
            model.predictor = compiled_predictor
            model.action_predictor = compiled_action_pred
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

        val_metrics = {
            f"val_{k}": total_metrics[k] / max(total_steps, 1)
            for k in total_metrics
        }
        enc_metrics, enc_arrays = _finalize_embedding_stats(embedding_stats)
        val_metrics.update({f"val_{k}": v for k, v in enc_metrics.items()})
        return val_metrics, enc_arrays

    # ---- training ----
    global_step = 0
    t_start = time.time()
    best_val_loss = float("inf")

    # ── throughput tracking (CUDA events for accurate GPU timing) ──
    enc_tokens_total = 0       # cumulative non-pad tokens through encoder
    enc_time_total = 0.0       # cumulative encoder wall time (seconds)
    has_cuda = (device.type == "cuda")
    pending_cuda_events = []

    for epoch in range(args.epochs):
        model.train()
        epoch_metrics: dict[str, float] = {}
        epoch_steps = 0
        t_epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_tuple in train_loader:
            (
                state_N, state_N_valid,
                state_N1, state_N1_valid,
                pa_hist_N, pa_hist_N_valid,
                oa_hist_N, oa_hist_N_valid,
                pa_hist_N1, pa_hist_N1_valid,
                oa_hist_N1, oa_hist_N1_valid,
                pa_tokens, oa_tokens,
            ) = (tensor.to(device) for tensor in batch_tuple)

            # ── Encoder forward (both prefixes) ──
            enc_tokens = int(
                (state_N != pad_id).sum().item()
                + (state_N1 != pad_id).sum().item()
                + (pa_hist_N != pad_id).sum().item()
                + (oa_hist_N != pad_id).sum().item()
                + (pa_hist_N1 != pad_id).sum().item()
                + (oa_hist_N1 != pad_id).sum().item()
            )
            if has_cuda:
                enc_start = torch.cuda.Event(enable_timing=True)
                enc_end = torch.cuda.Event(enable_timing=True)
                enc_start.record()
            else:
                enc_start_time = time.perf_counter()

            outputs = model(
                state_N, state_N_valid,
                state_N1, state_N1_valid,
                pa_hist_N, pa_hist_N_valid,
                oa_hist_N, oa_hist_N_valid,
                pa_hist_N1, pa_hist_N1_valid,
                oa_hist_N1, oa_hist_N1_valid,
                pa_tokens, oa_tokens,
            )

            if has_cuda:
                enc_end.record()
                pending_cuda_events.append((enc_start, enc_end))
            else:
                enc_time_total += time.perf_counter() - enc_start_time
            enc_tokens_total += enc_tokens

            loss, metrics = compute_losses(
                outputs,
                lambda_sigreg=lambda_sigreg,
                sigreg_num_slices=sigreg_num_slices,
                sigreg_num_points=sigreg_num_points,
                sigreg_domain=sigreg_domain,
            )

            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            epoch_steps += 1
            global_step += 1

            if global_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # ---- mid-epoch validation (fast: limited batches) ----
            if args.val_interval > 0 and global_step % args.val_interval == 0:
                _mb = args.val_max_batches if args.val_max_batches > 0 else None
                mid_val, mid_val_arrays = run_validation(max_batches=_mb)
                model.train()  # restore training mode (gradient checkpointing)
                if args.print_interval > 0:
                    print(
                        f"  val @ step {global_step:7d} | "
                        f"val loss {mid_val['val_loss']:.4f} | "
                        f"state {mid_val.get('val_state_loss', 0):.4f} | "
                        f"p_act {mid_val.get('val_player_action_loss', 0):.4f} | "
                        f"o_act {mid_val.get('val_opponent_action_loss', 0):.4f} | "
                        f"sigreg {mid_val.get('val_sigreg_loss', 0):.4f} | "
                        f"enc std {mid_val.get('val_enc_dim_std_mean', 0):.4f} | "
                        f"enc rank {mid_val.get('val_enc_cov_eff_rank', 0):.1f}"
                    )
                if wandb_run:
                    wandb_run.log({
                        **_wandb_validation_payload(mid_val, mid_val_arrays),
                        "global_step": global_step,
                        "epoch": epoch,
                    })
                if log_file:
                    log_file.write(",".join([
                        str(epoch),
                        str(global_step),
                        *([""] * 10),
                        *_validation_csv_fields(mid_val).split(","),
                    ]) + "\n")
                    log_file.flush()
                # Update best checkpoint if improved
                if args.checkpoint and mid_val["val_loss"] < best_val_loss:
                    best_val_loss = mid_val["val_loss"]
                    model.save_checkpoint(
                        args.checkpoint,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        config=model_cfg,
                        vocab_size=vocab_size,
                    )
                    if args.print_interval > 0:
                        print(f"  ✓ Best checkpoint (val_loss={best_val_loss:.4f}) → {args.checkpoint}")

            # ---- per-step logging ----
            if global_step % args.log_interval == 0:
                if has_cuda and pending_cuda_events:
                    pending_cuda_events[-1][1].synchronize()
                    enc_time_total += sum(
                        start.elapsed_time(end) / 1000.0
                        for start, end in pending_cuda_events
                    )
                    pending_cuda_events.clear()
                enc_tok_s = enc_tokens_total / enc_time_total if enc_time_total > 0 else 0.0

                if log_file:
                    log_file.write(",".join([
                        str(epoch),
                        str(global_step),
                        f"{metrics['loss']:.6f}",
                        f"{metrics['state_loss']:.6f}",
                        f"{metrics['player_action_loss']:.6f}",
                        f"{metrics['opponent_action_loss']:.6f}",
                        f"{metrics['pred_loss']:.6f}",
                        f"{metrics['sigreg_enc']:.6f}",
                        f"{metrics['sigreg_act']:.6f}",
                        f"{metrics['sigreg_loss']:.6f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                        f"{enc_tok_s:.1f}",
                        *([""] * 12),
                    ]) + "\n")
                    log_file.flush()

                if wandb_run:
                    wandb_run.log({
                        "train/loss": metrics["loss"],
                        "train/state_loss": metrics["state_loss"],
                        "train/player_action_loss": metrics["player_action_loss"],
                        "train/opponent_action_loss": metrics["opponent_action_loss"],
                        "train/pred_loss": metrics["pred_loss"],
                        "train/sigreg_enc": metrics["sigreg_enc"],
                        "train/sigreg_act": metrics["sigreg_act"],
                        "train/sigreg_loss": metrics["sigreg_loss"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/enc_tok_s": enc_tok_s,
                        "epoch": epoch,
                        "global_step": global_step,
                    })

                if args.print_interval > 0 and global_step % args.print_interval == 0:
                    print(
                        f"  epoch {epoch:3d} | step {global_step:7d} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"state {metrics['state_loss']:.4f} | "
                        f"p_act {metrics['player_action_loss']:.4f} | "
                        f"o_act {metrics['opponent_action_loss']:.4f} | "
                        f"sigreg {metrics['sigreg_loss']:.4f} | "
                        f"enc {enc_tok_s:,.0f} tok/s | "
                        f"lr {optimizer.param_groups[0]['lr']:.2e}"
                    )

                # ── Reset throughput accumulators for next interval ──
                enc_tokens_total = 0
                enc_time_total = 0.0

        if epoch_steps > 0 and global_step % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ---- epoch-end validation ----
        val_metrics, val_arrays = run_validation(max_batches=None)

        # ---- epoch-end metrics ----
        avg_metrics = {k: v / max(epoch_steps, 1) for k, v in epoch_metrics.items()}
        t_epoch = time.time() - t_epoch_start

        print(
            f"=== epoch {epoch:3d} done | "
            f"train loss {avg_metrics['loss']:.4f} | "
            f"state {avg_metrics.get('state_loss', 0):.4f} | "
            f"p_act {avg_metrics.get('player_action_loss', 0):.4f} | "
            f"o_act {avg_metrics.get('opponent_action_loss', 0):.4f} | "
            f"sigreg {avg_metrics.get('sigreg_loss', 0):.4f} | "
            f"val loss {val_metrics.get('val_loss', 0):.4f} | "
            f"val state {val_metrics.get('val_state_loss', 0):.4f} | "
            f"enc std {val_metrics.get('val_enc_dim_std_mean', 0):.4f} | "
            f"enc rank {val_metrics.get('val_enc_cov_eff_rank', 0):.1f} | "
            f"time {t_epoch:.0f}s ==="
        )

        if wandb_run:
            wandb_run.log({
                **_wandb_validation_payload(val_metrics, val_arrays),
                "epoch/train_loss": avg_metrics["loss"],
                "epoch/train_state_loss": avg_metrics.get("state_loss", 0),
                "epoch/train_player_action_loss": avg_metrics.get("player_action_loss", 0),
                "epoch/train_opponent_action_loss": avg_metrics.get("opponent_action_loss", 0),
                "epoch/val_loss": val_metrics.get("val_loss", 0),
                "epoch/val_state_loss": val_metrics.get("val_state_loss", 0),
                "epoch/val_sigreg_loss": val_metrics.get("val_sigreg_loss", 0),
                "epoch/val_enc_dim_std_mean": val_metrics.get("val_enc_dim_std_mean", 0),
                "epoch/val_enc_cov_eff_rank": val_metrics.get("val_enc_cov_eff_rank", 0),
                "epoch/time_s": t_epoch,
                "epoch": epoch,
            })

        # ---- checkpoint ----
        if args.checkpoint:
            latest_path = os.path.join(
                os.path.dirname(args.checkpoint), "latest_checkpoint.pt"
            )
            model.save_checkpoint(
                latest_path,
                epoch=epoch,
                global_step=global_step,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                config=model_cfg,
                vocab_size=vocab_size,
            )
            current_val_loss = val_metrics.get("val_loss", float("inf"))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                model.save_checkpoint(
                    args.checkpoint,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    config=model_cfg,
                    vocab_size=vocab_size,
                )
                if args.print_interval > 0:
                    print(f"  ✓ Best checkpoint (val_loss={best_val_loss:.4f}) → {args.checkpoint}")

    if log_file:
        log_file.close()

    if wandb_run:
        wandb_run.finish()

    print(f"Training complete.  Checkpoints: {save_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LeJEPA v2 model on battle-prefix transitions."
    )
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing world-model-samples/{format}/*.npz.")
    parser.add_argument("--formats", type=str, nargs="+", required=True,
                        help="Format names (e.g. gen1ou).")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to WorldModelObservationSpace tokenizer JSON.")
    # Model config
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    # Training
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Accumulate gradients over this many micro-batches before optimizer.step().")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save best checkpoint. Also saves latest_checkpoint.pt alongside.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--compile", action="store_true",
                        help="Opt into torch.compile for small predictor modules. Disabled by default to avoid CUDA graph storage reuse with gradient accumulation.")
    # Validation
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Run validation every N training steps (0 = only at epoch end).")
    parser.add_argument("--val_max_batches", type=int, default=100,
                        help="Limit mid-epoch validation to this many batches. "
                             "Epoch-end validation always does a full pass.")
    # Loss
    parser.add_argument("--lambda_sigreg", type=float, default=None,
                        help="SIGReg weight λ. L = L_pred + λ · L_sigreg. "
                             "Default: from model config (0.1).")
    # Logging
    parser.add_argument("--log", action="store_true",
                        help="Write per-step metrics to metrics.csv in save_dir.")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=100)
    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()
    train(args)
