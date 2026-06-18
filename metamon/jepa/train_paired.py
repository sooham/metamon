"""Train paired-POV JEPA on synchronized battle perspectives.

This trainer consumes ``paired_shard_*.npz`` files produced by:

    uv run python scripts/generate_world_model_data.py --paired_pov ...

For each transition, the dataset provides both player perspectives through
state T and T+1. The model learns:

1. visible POV latent -> hidden opponent POV latent
2. visible POV latent + predicted opponent POV latent -> opponent action latent
3. visible POV latent + predicted opponent POV/action + own action -> next POV latent
"""

from __future__ import annotations

import argparse
import functools
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

from metamon.jepa.model import (
    ACTION_LATENT_DIM,
    CONTEXT_LENGTH,
    LATENT_DIM,
    SIGREG_DOMAIN,
    SIGREG_NUM_POINTS,
    SIGREG_NUM_SLICES,
    PairedJEPAModel,
    compute_paired_losses,
    sigreg,
)


class PairedJEPADataset(torch.utils.data.IterableDataset):
    """Iterable paired-POV transition dataset with pre-computed action blocks.

    On first access each shard is loaded into RAM and action blocks are
    pre-concatenated with their delimiters.  The hot iterator path then
    yields raw numpy arrays with zero allocations — collation handles
    padding and dtype conversion.
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
            total += int(len(data["state_idx"]))
        return total

    @staticmethod
    def _resolve_window(
        battle_start: int,
        state_end: int,
        action_base: int,
        max_hist: int,
    ) -> tuple[int, int, int]:
        state_start = battle_start
        if max_hist > 0 and (state_end - state_start) > max_hist:
            state_start = state_end - max_hist
        n_states = state_end - state_start
        n_actions = n_states - 1
        action_start = action_base + (state_start - battle_start)
        action_end = action_start + n_actions
        return state_start, action_start, action_end

    @staticmethod
    def _preprocess_actions(
        flat: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        start_token: int,
        end_token: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_flat, combined_offsets, combined_lengths)
        with delimiter tokens baked in.  The combined_flat array holds
        [start, block..., end] for each action block contiguously."""
        n = len(offsets)
        new_lengths = lengths + 2  # prefix + suffix
        new_offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(new_lengths, out=new_offsets[1:])
        total = int(new_offsets[-1])
        combined = np.empty(total, dtype=np.int16)
        # Fill in bulk using vectorized indexing where possible
        combined[:] = 0  # safety
        for i in range(n):
            off = int(offsets[i])
            length = int(lengths[i])
            dest = int(new_offsets[i])
            combined[dest] = start_token
            if length > 0:
                combined[dest + 1 : dest + 1 + length] = flat[off : off + length]
            combined[dest + 1 + length] = end_token
        return combined, new_offsets, new_lengths.astype(np.int32, copy=False)

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
    def _ensure_combined(data: dict, cm: int, ecm: int, ocm: int, eocm: int) -> dict:
        """Idempotently add pre-combined action arrays to a loaded shard."""
        if "p1_actions_combined" in data:
            return data  # already combined
        for key, sid, eid in [
            ("p1_actions", cm, ecm),
            ("p1_opponent_actions", ocm, eocm),
            ("p2_actions", cm, ecm),
            ("p2_opponent_actions", ocm, eocm),
        ]:
            # .npz keys use singular "action": p1_action_offsets, p1_action_lengths
            offs_key = key[:-1] + "_offsets"   # p1_actions → p1_action_offsets
            lens_key = key[:-1] + "_lengths"    # p1_actions → p1_action_lengths
            combined, offs, lens = PairedJEPADataset._preprocess_actions(
                data[key], data[offs_key], data[lens_key], sid, eid
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
    def _iter_shard(data: dict, cm: int, ecm: int, ocm: int, eocm: int,
                    shuffle_transitions: bool, max_hist: int) -> Iterator[dict]:
        n = len(data["state_idx"])
        order = np.arange(n)
        if shuffle_transitions:
            np.random.default_rng().shuffle(order)

        # Pre-combine action blocks once
        data = PairedJEPADataset._ensure_combined(data, cm, ecm, ocm, eocm)

        for row in order:
            battle_id = int(data["battle_id"][row])
            p1_si = int(data["p1_state_idx"][row]) if "p1_state_idx" in data else int(data["state_idx"][row])
            p1_nsi = int(data["p1_next_state_idx"][row]) if "p1_next_state_idx" in data else int(data["next_state_idx"][row])
            p1_ai = int(data["p1_action_idx"][row]) if "p1_action_idx" in data else int(data["action_idx"][row])
            p2_si = int(data["p2_state_idx"][row]) if "p2_state_idx" in data else int(data["state_idx"][row])
            p2_nsi = int(data["p2_next_state_idx"][row]) if "p2_next_state_idx" in data else int(data["next_state_idx"][row])
            p2_ai = int(data["p2_action_idx"][row]) if "p2_action_idx" in data else int(data["action_idx"][row])

            p1_bs = int(data["p1_battle_start"][battle_id]) if "p1_battle_start" in data else int(data["battle_start"][battle_id])
            p2_bs = int(data["p2_battle_start"][battle_id]) if "p2_battle_start" in data else int(data["battle_start"][battle_id])
            p1_as = int(data["p1_battle_action_start"][battle_id]) if "p1_battle_action_start" in data else int(data["battle_action_start"][battle_id])
            p2_as = int(data["p2_battle_action_start"][battle_id]) if "p2_battle_action_start" in data else int(data["battle_action_start"][battle_id])

            w = PairedJEPADataset._resolve_window
            p1_sT_s, p1_aT_s, p1_aT_e = w(p1_bs, p1_si + 1, p1_as, max_hist)
            p1_sT1_s, p1_aT1_s, p1_aT1_e = w(p1_bs, p1_nsi + 1, p1_as, max_hist)
            p2_sT_s, p2_aT_s, p2_aT_e = w(p2_bs, p2_si + 1, p2_as, max_hist)
            p2_sT1_s, p2_aT1_s, p2_aT1_e = w(p2_bs, p2_nsi + 1, p2_as, max_hist)

            sv = PairedJEPADataset._slice_view

            sample: dict = {
                "p1_state_T": sv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_sT_s, p1_si + 1),
                "p1_state_T1": sv(data["p1_states"], data["p1_state_offsets"], data["p1_state_lengths"], p1_sT1_s, p1_nsi + 1),
                "p2_state_T": sv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_sT_s, p2_si + 1),
                "p2_state_T1": sv(data["p2_states"], data["p2_state_offsets"], data["p2_state_lengths"], p2_sT1_s, p2_nsi + 1),
                "p1_player_hist_T": sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_aT_s, p1_aT_e),
                "p1_opponent_hist_T": sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_aT_s, p1_aT_e),
                "p1_player_hist_T1": sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_aT1_s, p1_aT1_e),
                "p1_opponent_hist_T1": sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_aT1_s, p1_aT1_e),
                "p2_player_hist_T": sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_aT_s, p2_aT_e),
                "p2_opponent_hist_T": sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_aT_s, p2_aT_e),
                "p2_player_hist_T1": sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_aT1_s, p2_aT1_e),
                "p2_opponent_hist_T1": sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_aT1_s, p2_aT1_e),
                "p1_action": sv(data["p1_actions_combined"], data["p1_actions_combined_offsets"], data["p1_actions_combined_lengths"], p1_ai, p1_ai + 1)[0],
                "p2_action": sv(data["p2_actions_combined"], data["p2_actions_combined_offsets"], data["p2_actions_combined_lengths"], p2_ai, p2_ai + 1)[0],
                "p1_action_as_opponent": sv(data["p1_opponent_actions_combined"], data["p1_opponent_actions_combined_offsets"], data["p1_opponent_actions_combined_lengths"], p1_ai, p1_ai + 1)[0],
                "p2_action_as_opponent": sv(data["p2_opponent_actions_combined"], data["p2_opponent_actions_combined_offsets"], data["p2_opponent_actions_combined_lengths"], p2_ai, p2_ai + 1)[0],
            }
            yield sample

    def __iter__(self) -> Iterator[dict[str, object]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        cm = self.structural["chosen_move"]
        ecm = self.structural["end_chosen_move"]
        ocm = self.structural["opponent_chosen_move"]
        eocm = self.structural["end_opponent_chosen_move"]

        for path in paths:
            # Load once per worker (OS page cache makes repeated loads fast)
            data = self._get_shard(path)
            yield from self._iter_shard(
                data, cm, ecm, ocm, eocm,
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
    "p1_action_as_opponent",
    "p2_action_as_opponent",
)


def collate_paired_fn(
    batch: list[dict[str, object]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    def pad_block_lists(block_lists: list[list[np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
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
                tokens = torch.from_numpy(block.astype(np.int64, copy=False))
                padded[batch_idx, block_idx, :len(tokens)] = tokens
                valid[batch_idx, block_idx] = True
        return padded, valid

    def pad_actions(actions: list[np.ndarray]) -> torch.Tensor:
        max_tokens = max((len(action) for action in actions), default=1)
        padded = torch.full((len(actions), max_tokens), pad_id, dtype=torch.long)
        for batch_idx, action in enumerate(actions):
            tokens = torch.from_numpy(action.astype(np.int64, copy=False))
            padded[batch_idx, :len(tokens)] = tokens
        return padded

    out: dict[str, torch.Tensor] = {}
    for key in BLOCK_KEYS:
        blocks, valid = pad_block_lists([item[key] for item in batch])  # type: ignore[index]
        out[key] = blocks
        out[f"{key}_valid"] = valid
    for key in ACTION_KEYS:
        out[key] = pad_actions([item[key] for item in batch])  # type: ignore[index]
    return out


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward_paired(model: PairedJEPAModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
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
        batch["p1_action_as_opponent"],
        batch["p2_action_as_opponent"],
    )


def _paired_sigreg_breakdown(
    outputs: dict[str, torch.Tensor],
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_domain: float,
) -> dict[str, float]:
    state_current = (
        sigreg(outputs["enc_p1_T"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["enc_p2_T"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    state_next = (
        sigreg(outputs["enc_p1_T1"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["enc_p2_T1"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    action_own = (
        sigreg(outputs["p1_action"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["p2_action"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    action_opponent = (
        sigreg(outputs["p1_action_as_opponent"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["p2_action_as_opponent"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    return {
        "sigreg_state_current": state_current.item(),
        "sigreg_state_next": state_next.item(),
        "sigreg_action_own": action_own.item(),
        "sigreg_action_opponent": action_opponent.item(),
        "sigreg_state_loss": (state_current + state_next).item(),
        "sigreg_action_loss": (action_own + action_opponent).item(),
        "sigreg_total_detail": (state_current + state_next + action_own + action_opponent).item(),
    }


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
        "chosen_move": tokenizer["<chosen_move>"],
        "end_chosen_move": tokenizer["<end_chosen_move>"],
        "opponent_chosen_move": tokenizer["<opponent_chosen_move>"],
        "end_opponent_chosen_move": tokenizer["<end_opponent_chosen_move>"],
    }

    latent_dim = model_cfg.get("latent_dim", LATENT_DIM)
    action_latent_dim = model_cfg.get("action_latent_dim", ACTION_LATENT_DIM)
    lambda_sigreg = args.lambda_sigreg
    if lambda_sigreg is None:
        lambda_sigreg = model_cfg.get("lambda_sigreg", 0.1)
    sigreg_num_slices = model_cfg.get("sigreg_num_slices", SIGREG_NUM_SLICES)
    sigreg_num_points = model_cfg.get("sigreg_num_points", SIGREG_NUM_POINTS)
    sigreg_domain = model_cfg.get("sigreg_domain", SIGREG_DOMAIN)
    context_length = model_cfg.get("encoder", {}).get("max_seq_len", CONTEXT_LENGTH)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: bos={bos_id} eos={eos_id} pad={pad_id}")
    print(f"Structural token IDs: {structural_ids}")
    print(f"Latent dim: {latent_dim}  action_latent_dim: {action_latent_dim}")
    print(f"CONTEXT_LENGTH (encoder max_seq_len): {context_length}")

    train_shards = PairedJEPADataset.discover(args.data_root, args.formats, "train")
    val_shards = PairedJEPADataset.discover(
        args.data_root, args.formats, "val", required=False
    )
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
        opponent_state_predictor_cfg=model_cfg.get("opponent_state_predictor", {}),
        action_predictor_cfg=model_cfg.get("paired_action_predictor", model_cfg.get("action_predictor", {})),
        next_state_predictor_cfg=model_cfg.get("next_state_predictor", {}),
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
        ckpt = torch.load(args.checkpoint, map_location=device)
        # Strip torch.compile _orig_mod. prefix if present (checkpoint may
        # have been saved while submodule compile was active).
        state_dict = ckpt["model_state_dict"]
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("_orig_mod.", "")] = v
        model.load_state_dict(cleaned)
        print(f"Loaded checkpoint: {args.checkpoint}")

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
        f"effective={args.batch_size * args.grad_accum_steps}"
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
                "n_params": n_params,
                "n_train_transitions": train_transitions,
                "n_val_transitions": val_transitions,
                "context_length": context_length,
                "lambda_sigreg": lambda_sigreg,
                "lambda_opponent_state": args.lambda_opponent_state,
                "lambda_action": args.lambda_action,
                "lambda_next_state": args.lambda_next_state,
            },
        )
    elif args.wandb and not _wandb_available:
        print("WARNING: --wandb enabled but wandb not installed (pip install wandb)")

    def loss_from_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_paired_losses(
            outputs,
            lambda_sigreg=lambda_sigreg,
            lambda_opponent_state=args.lambda_opponent_state,
            lambda_action=args.lambda_action,
            lambda_next_state=args.lambda_next_state,
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
            outputs = _forward_paired(model, batch)
            _, metrics = loss_from_outputs(outputs)
            diagnostics = _paired_sigreg_breakdown(
                outputs,
                sigreg_num_slices,
                sigreg_num_points,
                sigreg_domain,
            )
            metrics.update(diagnostics)
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

    for epoch in range(args.epochs):
        model.train()
        epoch_totals: dict[str, float] = {}
        epoch_steps = 0
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            outputs = _forward_paired(model, batch)
            loss, metrics = loss_from_outputs(outputs)
            (loss / args.grad_accum_steps).backward()

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
                        "p1_action", "p2_action", "p1_action_as_opponent", "p2_action_as_opponent"):
                batch_tokens += int((batch[key] != pad_id).sum().item())
            tokens_since_print += batch_tokens

            log_step = args.log_interval if args.log_interval > 0 else args.print_interval
            if wandb_run and log_step > 0 and global_step % log_step == 0:
                diagnostics = _paired_sigreg_breakdown(
                    outputs,
                    sigreg_num_slices,
                    sigreg_num_points,
                    sigreg_domain,
                )
                wandb_run.log({
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
                    "train/sigreg_loss": metrics["sigreg_loss"],
                    "train/sigreg_state_current": diagnostics["sigreg_state_current"],
                    "train/sigreg_state_next": diagnostics["sigreg_state_next"],
                    "train/sigreg_action_own": diagnostics["sigreg_action_own"],
                    "train/sigreg_action_opponent": diagnostics["sigreg_action_opponent"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                })

            if args.print_interval > 0 and global_step % args.print_interval == 0:
                now = time.time()
                elapsed = now - t_last_print
                tok_per_sec = tokens_since_print / elapsed if elapsed > 0 else 0
                t_last_print = now
                tokens_since_print = 0
                diagnostics = _paired_sigreg_breakdown(
                    outputs,
                    sigreg_num_slices,
                    sigreg_num_points,
                    sigreg_domain,
                )
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
                    f"sigreg {metrics['sigreg_loss']:.4f} "
                    f"[state_cur {diagnostics['sigreg_state_current']:.4f}, "
                    f"state_next {diagnostics['sigreg_state_next']:.4f}, "
                    f"action_own {diagnostics['sigreg_action_own']:.4f}, "
                    f"action_opp {diagnostics['sigreg_action_opponent']:.4f}]"
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
                        f"sigreg {val_metrics.get('val_sigreg_loss', 0.0):.4f} | "
                        f"state_cur {val_metrics.get('val_sigreg_state_current', 0.0):.4f} | "
                        f"state_next {val_metrics.get('val_sigreg_state_next', 0.0):.4f} | "
                        f"action_own {val_metrics.get('val_sigreg_action_own', 0.0):.4f} | "
                        f"action_opp {val_metrics.get('val_sigreg_action_opponent', 0.0):.4f}"
                    )
                    if wandb_run:
                        wandb_run.log({
                            "val/loss": val_metrics["val_loss"],
                            "val/opponent_state_loss": val_metrics["val_opponent_state_loss"],
                            "val/action_loss": val_metrics["val_action_loss"],
                            "val/next_state_loss": val_metrics["val_next_state_loss"],
                            "val/sigreg_loss": val_metrics.get("val_sigreg_loss", 0.0),
                            "val/sigreg_state_current": val_metrics.get("val_sigreg_state_current", 0.0),
                            "val/sigreg_state_next": val_metrics.get("val_sigreg_state_next", 0.0),
                            "val/sigreg_action_own": val_metrics.get("val_sigreg_action_own", 0.0),
                            "val/sigreg_action_opponent": val_metrics.get("val_sigreg_action_opponent", 0.0),
                            "epoch": epoch,
                            "global_step": global_step,
                        })
                    if val_metrics["val_loss"] < best_val_loss and args.checkpoint:
                        best_val_loss = val_metrics["val_loss"]
                        model.save_checkpoint(
                            args.checkpoint,
                            epoch=epoch,
                            global_step=global_step,
                            config=model_cfg,
                            vocab_size=vocab_size,
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
            f"sigreg {avg.get('sigreg_loss', 0.0):.4f}"
        )
        if val_metrics:
            msg += (
                f" | val loss {val_metrics.get('val_loss', 0.0):.4f}"
                f" | val sigreg {val_metrics.get('val_sigreg_loss', 0.0):.4f}"
            )
        print(msg + " ===")

        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": avg.get("loss", 0.0),
                "epoch/train_opponent_state_loss": avg.get("opponent_state_loss", 0.0),
                "epoch/train_action_loss": avg.get("action_loss", 0.0),
                "epoch/train_next_state_loss": avg.get("next_state_loss", 0.0),
                "epoch/train_sigreg_loss": avg.get("sigreg_loss", 0.0),
                "epoch/val_loss": val_metrics.get("val_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_opponent_state_loss": val_metrics.get("val_opponent_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_action_loss": val_metrics.get("val_action_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_next_state_loss": val_metrics.get("val_next_state_loss", 0.0) if val_metrics else 0.0,
                "epoch/val_sigreg_loss": val_metrics.get("val_sigreg_loss", 0.0) if val_metrics else 0.0,
                "epoch": epoch,
            })

        latest_path = save_dir / "paired_latest.pt"
        model.save_checkpoint(
            str(latest_path),
            epoch=epoch,
            global_step=global_step,
            config=model_cfg,
            vocab_size=vocab_size,
        )
        if args.checkpoint and (not val_metrics or val_metrics.get("val_loss", float("inf")) < best_val_loss):
            best_val_loss = val_metrics.get("val_loss", avg.get("loss", best_val_loss))
            model.save_checkpoint(
                args.checkpoint,
                epoch=epoch,
                global_step=global_step,
                config=model_cfg,
                vocab_size=vocab_size,
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
    parser.add_argument("--lambda_sigreg", type=float, default=None)
    parser.add_argument("--lambda_opponent_state", type=float, default=1.0)
    parser.add_argument("--lambda_action", type=float, default=1.0)
    parser.add_argument("--lambda_next_state", type=float, default=1.0)
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
                        help="Maximum history state blocks per sample (0 = unlimited). "
                             "Lower = faster data loading + shorter temporal sequences. Default: 0 (unlimited)")
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable torch.compile (default: False). Use --compile to enable.")
    train(parser.parse_args())
