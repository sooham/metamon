"""Train a LeJEPA model on world-model state transitions.

Loads sharded .npz files produced by scripts/generate_world_model_data.py
and trains a LeJEPA (Latent-Euclidean Joint Embedding Predictive Architecture)
model that:

1. Encodes each battle state into a deterministic embedding *e* via a
   bidirectional transformer encoder.
2. Predicts the next state's embedding from the previous state's embedding
   conditioned on the action, via a small causal transformer predictor.
3. Regularises embeddings toward an isotropic Gaussian distribution via
   SIGReg (Sketched Isotropic Gaussian Regularization).

No VAE decoder, no stop-gradient, no teacher-student.  A single
hyperparameter λ (lambda_sigreg) balances prediction vs. regularization.

Newer shards store every state as ``<bos> ... <eos>``.  The JEPA loader uses
stored state tokens as-is; it does not add boundary tokens.

Usage:
    uv run python -m metamon.jepa.train \\
        --data_root $METAMON_CACHE_DIR/world-model-samples \\
        --formats gen1ou gen9ou \\
        --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \\
        --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \\
        --batch_size 256 --lr 3e-4 --epochs 100

    # With wandb + CSV logging
    uv run python -m metamon.jepa.train \\
        --data_root $METAMON_CACHE_DIR/world-model-samples \\
        --formats gen1ou gen9ou \\
        --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \\
        --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \\
        --batch_size 256 --lr 3e-4 --epochs 100 \\
        --wandb --wandb_project metamon-jepa --wandb_name run-01 \\
        --log --log_interval 100
"""

import argparse
import functools
import math
import os
import time
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
    MAX_STATE_LENGTH,
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
        "val/loss": metrics["val_loss"],
        "val/jepa_loss": metrics["val_jepa_loss"],
        "val/sigreg_prev": metrics["val_sigreg_prev"],
        "val/sigreg_next": metrics["val_sigreg_next"],
        "val/sigreg_loss": metrics["val_sigreg_loss"],
        "val/enc_dim_mean_mean": metrics["val_enc_dim_mean_mean"],
        "val/enc_dim_mean_abs_mean": metrics["val_enc_dim_mean_abs_mean"],
        "val/enc_dim_mean_abs_max": metrics["val_enc_dim_mean_abs_max"],
        "val/enc_dim_std_mean": metrics["val_enc_dim_std_mean"],
        "val/enc_dim_std_min": metrics["val_enc_dim_std_min"],
        "val/enc_dim_std_max": metrics["val_enc_dim_std_max"],
        "val/enc_cov_eff_rank": metrics["val_enc_cov_eff_rank"],
    }
    if _wandb_available and arrays["enc_dim_mean"].size > 0:
        payload["val/enc_dim_mean_hist"] = wandb.Histogram(arrays["enc_dim_mean"])
        payload["val/enc_dim_std_hist"] = wandb.Histogram(arrays["enc_dim_std"])
    return payload


def _validation_csv_fields(metrics: dict[str, float]) -> str:
    keys = (
        "val_loss",
        "val_jepa_loss",
        "val_sigreg_prev",
        "val_sigreg_next",
        "val_sigreg_loss",
        "val_enc_dim_mean_mean",
        "val_enc_dim_mean_abs_mean",
        "val_enc_dim_mean_abs_max",
        "val_enc_dim_std_mean",
        "val_enc_dim_std_min",
        "val_enc_dim_std_max",
        "val_enc_cov_eff_rank",
    )
    return ",".join(f"{metrics[k]:.6f}" for k in keys)


# ── Dataset ─────────────────────────────────────────────────────────────

class JEPADataset(torch.utils.data.IterableDataset):
    """Iterable over sharded .npz files, yielding (prev, next, action) pairs.

    New shards contain tokenized states plus an explicit transition table
    (prev_state_idx, next_state_idx, actions).  Transition rows are shuffled
    within each training shard so batches mix battles and game phases.

    Legacy trajectory shards are still supported.  For those, each battle
    yields N-1 real transition pairs:

        (S[t], S[t+1], action[t])

    States are yielded **unpadded** exactly as stored in the shard.
    For transition-table shards, transition rows are shuffled before yielding.

    Parameters
    ----------
    shard_paths : list[str]
        Paths to .npz shard files.
    shuffle_shards : bool
        Whether to shuffle shard order each epoch.
    """

    def __init__(
        self,
        shard_paths: list[str],
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = shard_paths
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
                continue
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
        return cls(shard_paths, shuffle_shards)

    def _iter_shard(
        self, path: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """Yield (prev_tokens, next_tokens, action_idx) pairs for a single shard.

        Yields only real transitions: (S[t], S[t+1], action[t])
        for t = 0 .. N-2 within each battle.
        """
        data = np.load(path)
        states = data["states"]              # flat 1-D array of all token IDs
        state_lengths = data["state_lengths"]  # (N,) actual token counts
        state_offsets = data["state_offsets"]  # (N,) start index per state
        actions = data["actions"]            # (total_actions,) — one per transition

        if "prev_state_idx" in data:
            prev_state_idx = data["prev_state_idx"]
            next_state_idx = data["next_state_idx"]
            order = np.arange(len(actions))
            if self.shuffle_transitions:
                rng = np.random.default_rng()
                rng.shuffle(order)

            for row in order:
                prev_idx = int(prev_state_idx[row])
                next_idx = int(next_state_idx[row])
                prev_len = int(state_lengths[prev_idx])
                next_len = int(state_lengths[next_idx])
                prev_off = state_offsets[prev_idx]
                next_off = state_offsets[next_idx]
                prev_tokens = states[prev_off : prev_off + prev_len]
                next_tokens = states[next_off : next_off + next_len]
                yield prev_tokens.astype(np.int16, copy=False), next_tokens.astype(np.int16, copy=False), int(actions[row])
            return

        battle_start = data["battle_start"]  # (B+1,) cumulative state indices
        num_battles = len(battle_start) - 1

        for b in range(num_battles):
            s_start = battle_start[b]
            s_end = battle_start[b + 1]
            n_states = s_end - s_start

            if n_states < 2:
                continue

            # Extract all real states for this battle as stored.
            battle_states: list[np.ndarray] = []
            for i in range(n_states):
                idx = s_start + i
                length = int(state_lengths[idx])
                offset = state_offsets[idx]
                raw = states[offset : offset + length]
                battle_states.append(raw.astype(np.int16, copy=False))

            # Real transitions: S[t] → S[t+1]
            pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
            for t in range(n_states - 1):
                action_idx = int(actions[s_start + t - b])
                pairs.append((battle_states[t], battle_states[t + 1], action_idx))

            # Shuffle pairs within this shard.
            rng = np.random.default_rng()
            rng.shuffle(pairs)

            yield from pairs

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            yield from self._iter_shard(path)


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray, int]],
    pad_id: int,
    max_state_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length state pairs into padded tensors.

    Args:
        batch: list of (prev_tokens, next_tokens, action_idx) — unpadded.
        pad_id: token ID for padding.
        max_state_len: if set, ALWAYS pad to this fixed length (eliminates
            dynamic shape variants in CUDA graphs, preventing OOMs).

    Returns:
        prev_padded:  (B, max_prev) int64
        next_padded:  (B, max_next) int64
        prev_lengths: (B,) int64
        next_lengths: (B,) int64
        actions:      (B,) int64 — action indices (-1..12)
    """
    prev_lengths = torch.tensor([len(item[0]) for item in batch], dtype=torch.long)
    next_lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    actions = torch.tensor([item[2] for item in batch], dtype=torch.long)

    if max_state_len is not None:
        # Always pad to the fixed max_state_len to eliminate dynamic
        # sequence-length shapes that cause CUDA graph proliferation.
        max_prev = max_state_len
        max_next = max_state_len
    else:
        max_prev = int(prev_lengths.max().item())
        max_next = int(next_lengths.max().item())

    prev_padded = torch.full((len(batch), max_prev), pad_id, dtype=torch.long)
    next_padded = torch.full((len(batch), max_next), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        prev_tokens = item[0][:max_prev]
        next_tokens = item[1][:max_next]
        prev_padded[i, :len(prev_tokens)] = torch.from_numpy(prev_tokens.astype(np.int64))
        next_padded[i, :len(next_tokens)] = torch.from_numpy(next_tokens.astype(np.int64))

    return prev_padded, next_padded, prev_lengths, next_lengths, actions


# ── Training loop ───────────────────────────────────────────────────────

def train(args):
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

    # ---- tokenizer (for vocab size + special token IDs) ----
    from metamon.tokenizer import PokemonTokenizer

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    vocab_size = model_cfg.get("vocab_size") or len(tokenizer)

    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]
    pad_id = tokenizer.pad_token_id

    if args.print_interval > 0:
        print(f"Vocabulary size: {vocab_size}")
        print(f"Special tokens: bos={bos_id} eos={eos_id} pad={pad_id}")

    # ---- model hyperparameters ----
    latent_dim = model_cfg.get("latent_dim", LATENT_DIM)
    lambda_sigreg = model_cfg.get("lambda_sigreg", 0.05)
    if args.lambda_sigreg is not None:
        lambda_sigreg = args.lambda_sigreg
    sigreg_num_slices = model_cfg.get("sigreg_num_slices", SIGREG_NUM_SLICES)
    sigreg_num_points = model_cfg.get("sigreg_num_points", SIGREG_NUM_POINTS)
    sigreg_domain = model_cfg.get("sigreg_domain", SIGREG_DOMAIN)

    if args.print_interval > 0:
        print(f"Latent dim: {latent_dim}  λ_sigreg={lambda_sigreg}  "
              f"SIGReg slices={sigreg_num_slices} points={sigreg_num_points} domain={sigreg_domain}")

    # ---- datasets (train / val split generated at raw-battle-group level) ----
    train_shards = JEPADataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        split="train",
        shuffle_shards=False,
    ).shard_paths
    val_shards = JEPADataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        split="val",
        shuffle_shards=False,
    ).shard_paths

    train_dataset = JEPADataset(
        train_shards, shuffle_shards=True,
    )
    val_dataset = JEPADataset(
        val_shards, shuffle_shards=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate_fn, pad_id=pad_id, max_state_len=MAX_STATE_LENGTH),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=functools.partial(collate_fn, pad_id=pad_id, max_state_len=MAX_STATE_LENGTH),
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
        encoder_cfg=model_cfg.get("encoder", {}),
        predictor_cfg=model_cfg.get("predictor", {}),
    ).to(device)

    # BF16 + TF32 for GPU training
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.capture_scalar_outputs = True

    # Compile (CUDA only — MPS does not support torch.compile).
    #
    # Compile only the predictor. The encoder now runs eager by default;
    # predictor compilation still removes launch overhead without the long
    # Triton autotuning phase from max-autotune.
    if device.type == "cuda":
        try:
            model.predictor = torch.compile(
                model.predictor, dynamic=True, mode="reduce-overhead"
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
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "n_params": n_params,
                "lambda_sigreg": lambda_sigreg,
                "sigreg_num_slices": sigreg_num_slices,
                "sigreg_num_points": sigreg_num_points,
                "sigreg_domain": sigreg_domain,
                "max_state_length": MAX_STATE_LENGTH,
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
            "epoch,step,loss,jepa_loss,sigreg_prev,sigreg_next,sigreg_loss,lr,"
            "enc_tok_s,pred_tok_s,"
            "val_loss,val_jepa_loss,val_sigreg_prev,val_sigreg_next,val_sigreg_loss,"
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
        print(f"Batch size: {args.batch_size}  "
              f"MAX_STATE_LENGTH: {MAX_STATE_LENGTH}")

    # ---- validation function ----
    @torch.no_grad()
    def run_validation(max_batches: int | None = None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Run a pass over the val loader, return average metrics.

        Validation runs in eager mode so the compiled training workspace
        is never polluted by eval-mode allocations.  Without this the
        eval forward pass (no gradient checkpointing, self.training=False)
        triggers a separate compilation whose workspace is sized for all
        layers' intermediates at once, pushing the GPU over the edge when
        training resumes.
        """
        model.eval()
        total_metrics: dict[str, float] = {}
        total_steps = 0
        embedding_stats = _new_embedding_stats(device, latent_dim)
        # Temporarily disable dynamo so compiled wrappers run their
        # original (eager) code.  This avoids allocating eval-mode
        # workspace that competes with training memory.
        old_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            for batch_idx, (prev, next_, prev_lens, next_lens, actions) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                prev = prev.to(device)
                next_ = next_.to(device)
                actions = actions.to(device)

                outputs = model(prev, next_, actions)
                _update_embedding_stats(
                    embedding_stats,
                    torch.cat([outputs["e_prev"], outputs["e_next"]], dim=0),
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
            torch._dynamo.config.disable = old_disable
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
    pred_tokens_total = 0      # cumulative tokens through predictor (2 per transition)
    enc_time_total = 0.0       # cumulative encoder wall time (seconds)
    pred_time_total = 0.0      # cumulative predictor wall time (seconds)
    has_cuda = (device.type == "cuda")

    for epoch in range(args.epochs):
        model.train()
        epoch_metrics: dict[str, float] = {}
        epoch_steps = 0
        t_epoch_start = time.time()

        for prev, next_, prev_lens, next_lens, actions in train_loader:
            prev = prev.to(device)
            next_ = next_.to(device)
            actions = actions.to(device)

            # ── Encoder forward (prev + next states) ──
            enc_tokens = int((prev_lens + next_lens).sum().item())
            if has_cuda:
                enc_start = torch.cuda.Event(enable_timing=True)
                enc_end = torch.cuda.Event(enable_timing=True)
                enc_start.record()
            e_prev = model.encoder(prev)
            e_next = model.encoder(next_)
            if has_cuda:
                enc_end.record()

            # ── Predictor forward ──
            compact_idx = actions + 1  # -1..12 → 0..13
            if has_cuda:
                pred_start = torch.cuda.Event(enable_timing=True)
                pred_end = torch.cuda.Event(enable_timing=True)
                pred_start.record()
            predicted_next = model.predictor(e_prev, compact_idx)
            if has_cuda:
                pred_end.record()

            # ── Assemble outputs ──
            outputs = {"e_prev": e_prev, "e_next": e_next, "predicted_next": predicted_next}

            loss, metrics = compute_losses(
                outputs,
                lambda_sigreg=lambda_sigreg,
                sigreg_num_slices=sigreg_num_slices,
                sigreg_num_points=sigreg_num_points,
                sigreg_domain=sigreg_domain,
            )

            # ── Accumulate throughput (events ready after compute_losses .item() syncs GPU) ──
            if has_cuda:
                enc_time_total += enc_start.elapsed_time(enc_end) / 1000.0
                pred_time_total += pred_start.elapsed_time(pred_end) / 1000.0
            enc_tokens_total += enc_tokens
            pred_tokens_total += actions.size(0) * 2  # 2 causal tokens per transition

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            epoch_steps += 1
            global_step += 1

            # ---- mid-epoch validation (fast: limited batches) ----
            if args.val_interval > 0 and global_step % args.val_interval == 0:
                _mb = args.val_max_batches if args.val_max_batches > 0 else None
                mid_val, mid_val_arrays = run_validation(max_batches=_mb)
                model.train()  # restore training mode (gradient checkpointing)
                if args.print_interval > 0:
                    print(
                        f"  val @ step {global_step:7d} | "
                        f"val loss {mid_val['val_loss']:.4f} | "
                        f"val jepa {mid_val['val_jepa_loss']:.4f} | "
                        f"val sigreg {mid_val['val_sigreg_loss']:.4f} | "
                        f"enc std {mid_val['val_enc_dim_std_mean']:.4f} | "
                        f"enc rank {mid_val['val_enc_cov_eff_rank']:.1f}"
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
                        *([""] * 8),
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
                # ── Compute throughput over the last log_interval steps ──
                enc_tok_s = enc_tokens_total / enc_time_total if enc_time_total > 0 else 0.0
                pred_tok_s = pred_tokens_total / pred_time_total if pred_time_total > 0 else 0.0

                if log_file:
                    log_file.write(",".join([
                        str(epoch),
                        str(global_step),
                        f"{metrics['loss']:.6f}",
                        f"{metrics['jepa_loss']:.6f}",
                        f"{metrics['sigreg_prev']:.6f}",
                        f"{metrics['sigreg_next']:.6f}",
                        f"{metrics['sigreg_loss']:.6f}",
                        f"{optimizer.param_groups[0]['lr']:.2e}",
                        f"{enc_tok_s:.1f}",
                        f"{pred_tok_s:.1f}",
                        *([""] * 12),
                    ]) + "\n")
                    log_file.flush()

                if wandb_run:
                    wandb_run.log({
                        "train/loss": metrics["loss"],
                        "train/jepa_loss": metrics["jepa_loss"],
                        "train/sigreg_prev": metrics["sigreg_prev"],
                        "train/sigreg_next": metrics["sigreg_next"],
                        "train/sigreg_loss": metrics["sigreg_loss"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/enc_tok_s": enc_tok_s,
                        "train/pred_tok_s": pred_tok_s,
                        "epoch": epoch,
                        "global_step": global_step,
                    })

                if args.print_interval > 0 and global_step % args.print_interval == 0:
                    print(
                        f"  epoch {epoch:3d} | step {global_step:7d} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"jepa {metrics['jepa_loss']:.4f} | "
                        f"sigreg_prev {metrics['sigreg_prev']:.4f} | "
                        f"sigreg_next {metrics['sigreg_next']:.4f} | "
                        f"enc {enc_tok_s:,.0f} tok/s | "
                        f"pred {pred_tok_s:,.0f} tok/s | "
                        f"lr {optimizer.param_groups[0]['lr']:.2e}"
                    )

                # ── Reset throughput accumulators for next interval ──
                enc_tokens_total = 0
                pred_tokens_total = 0
                enc_time_total = 0.0
                pred_time_total = 0.0

        scheduler.step()

        # ---- validation ----
        val_metrics, val_arrays = run_validation(max_batches=None)

        # ---- epoch-end metrics ----
        avg_metrics = {k: v / max(epoch_steps, 1) for k, v in epoch_metrics.items()}
        t_epoch = time.time() - t_epoch_start

        print(
            f"=== epoch {epoch:3d} done | "
            f"train loss {avg_metrics['loss']:.4f} | "
            f"jepa {avg_metrics['jepa_loss']:.4f} | "
            f"sigreg {avg_metrics['sigreg_loss']:.4f} | "
            f"val loss {val_metrics.get('val_loss', 0):.4f} | "
            f"val jepa {val_metrics.get('val_jepa_loss', 0):.4f} | "
            f"enc std {val_metrics.get('val_enc_dim_std_mean', 0):.4f} | "
            f"enc rank {val_metrics.get('val_enc_cov_eff_rank', 0):.1f} | "
            f"time {t_epoch:.0f}s ==="
        )

        if wandb_run:
            wandb_run.log({
                **_wandb_validation_payload(val_metrics, val_arrays),
                "epoch/train_loss": avg_metrics["loss"],
                "epoch/train_jepa_loss": avg_metrics["jepa_loss"],
                "epoch/train_sigreg_prev": avg_metrics["sigreg_prev"],
                "epoch/train_sigreg_next": avg_metrics["sigreg_next"],
                "epoch/train_sigreg_loss": avg_metrics["sigreg_loss"],
                "epoch/val_loss": val_metrics.get("val_loss", 0),
                "epoch/val_jepa_loss": val_metrics.get("val_jepa_loss", 0),
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
        description="Train a LeJEPA model on world-model state transitions."
    )
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing world-model-samples/{format}/*.npz.")
    parser.add_argument("--formats", type=str, nargs="+", required=True,
                        help="Format names (e.g. gen1ou gen9ou).")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to WorldModelObservationSpace tokenizer JSON.")
    # Model config
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    # Training
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save best checkpoint. Also saves latest_checkpoint.pt alongside.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    # Validation
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Run validation every N training steps (0 = only at epoch end).")
    parser.add_argument("--val_max_batches", type=int, default=100,
                        help="Limit mid-epoch validation to this many batches. "
                             "Epoch-end validation always does a full pass.")
    # Loss
    parser.add_argument("--lambda_sigreg", type=float, default=None,
                        help="SIGReg weight λ. L = L_jepa + λ · L_sigreg. "
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
