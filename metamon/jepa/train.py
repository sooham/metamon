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

All states end with <eos>.

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
import json
import math
import os
import time
from pathlib import Path
from typing import Iterator, Optional

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


# ── Dataset ─────────────────────────────────────────────────────────────

class JEPADataset(torch.utils.data.IterableDataset):
    """Iterable over sharded .npz files, yielding (prev, next, action) pairs.

    Each .npz shard contains concatenated battles.  For each battle we yield
    N-1 real transition pairs:

        (S[t]+<eos>, S[t+1]+<eos>, action[t])

    States are yielded **unpadded** (variable-length with <eos> appended).
    Pairs within each shard are shuffled before yielding.

    Parameters
    ----------
    shard_paths : list[str]
        Paths to .npz shard files.
    eos_id : int
        Token ID for <eos> (appended to every state).
    shuffle_shards : bool
        Whether to shuffle shard order each epoch.
    """

    def __init__(
        self,
        shard_paths: list[str],
        eos_id: int,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.eos_id = eos_id
        self.shuffle_shards = shuffle_shards

        if not self.shard_paths:
            raise ValueError("No shard paths provided")

    @classmethod
    def from_formats(
        cls,
        data_root: str,
        formats: list[str],
        eos_id: int,
        shuffle_shards: bool = True,
    ) -> "JEPADataset":
        """Discover all .npz shards under *data_root* for the given *formats*."""
        shard_paths: list[str] = []
        for fmt in formats:
            fmt_dir = os.path.join(data_root, fmt)
            if not os.path.isdir(fmt_dir):
                continue
            for f in sorted(os.listdir(fmt_dir)):
                if f.endswith(".npz"):
                    shard_paths.append(os.path.join(fmt_dir, f))

        if not shard_paths:
            raise FileNotFoundError(
                f"No .npz shards found under {data_root} for formats {formats}"
            )
        return cls(shard_paths, eos_id, shuffle_shards)

    def _iter_shard(
        self, path: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """Yield (prev_tokens, next_tokens, action_idx) pairs for a single shard.

        Yields only real transitions: (S[t]+<eos>, S[t+1]+<eos>, action[t])
        for t = 0 .. N-2 within each battle.
        """
        data = np.load(path)
        states = data["states"]              # flat 1-D array of all token IDs
        state_lengths = data["state_lengths"]  # (N,) actual token counts
        state_offsets = data["state_offsets"]  # (N,) start index per state
        actions = data["actions"]            # (total_actions,) — one per transition
        battle_start = data["battle_start"]  # (B+1,) cumulative state indices

        num_battles = len(battle_start) - 1
        eos = self.eos_id

        for b in range(num_battles):
            s_start = battle_start[b]
            s_end = battle_start[b + 1]
            n_states = s_end - s_start

            if n_states < 2:
                continue

            # Extract all real states for this battle (with <eos> appended).
            battle_states: list[np.ndarray] = []
            for i in range(n_states):
                idx = s_start + i
                length = int(state_lengths[idx])
                offset = state_offsets[idx]
                raw = states[offset : offset + length]
                # Append <eos>
                state_with_eos = np.append(raw, eos).astype(np.int16)
                battle_states.append(state_with_eos)

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
        max_state_len: if set, cap state lengths to this value.

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

    max_prev = int(prev_lengths.max().item())
    max_next = int(next_lengths.max().item())
    if max_state_len is not None:
        max_prev = min(max_prev, max_state_len)
        max_next = min(max_next, max_state_len)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    sigreg_num_slices = model_cfg.get("sigreg_num_slices", SIGREG_NUM_SLICES)
    sigreg_num_points = model_cfg.get("sigreg_num_points", SIGREG_NUM_POINTS)
    sigreg_domain = model_cfg.get("sigreg_domain", SIGREG_DOMAIN)

    if args.print_interval > 0:
        print(f"Latent dim: {latent_dim}  λ_sigreg={lambda_sigreg}  "
              f"SIGReg slices={sigreg_num_slices} points={sigreg_num_points} domain={sigreg_domain}")

    # ---- datasets (train / val split at shard level) ----
    all_shards = JEPADataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        eos_id=eos_id,
        shuffle_shards=False,  # we shuffle manually before splitting
    ).shard_paths

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_shards))
    all_shards = [all_shards[i] for i in perm]

    n_val = max(1, int(len(all_shards) * args.val_split))
    n_train = len(all_shards) - n_val
    train_shards = all_shards[:n_train]
    val_shards = all_shards[n_train:]

    train_dataset = JEPADataset(
        train_shards, eos_id, shuffle_shards=True,
    )
    val_dataset = JEPADataset(
        val_shards, eos_id, shuffle_shards=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id, max_state_len=MAX_STATE_LENGTH),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id, max_state_len=MAX_STATE_LENGTH),
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

    # Compile
    if device.type == "cuda":
        try:
            model = torch.compile(model, dynamic=True, mode="max-autotune")
        except Exception:
            if args.print_interval > 0:
                print("torch.compile max-autotune failed, trying default mode")
            try:
                model = torch.compile(model, dynamic=True)
            except Exception:
                if args.print_interval > 0:
                    print("torch.compile failed, falling back to eager mode")

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
                "val_split": args.val_split,
                "seed": args.seed,
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
            "val_loss,val_jepa_loss,val_sigreg_prev,val_sigreg_next,val_sigreg_loss\n"
        )

    # ---- print header ----
    if args.print_interval > 0:
        print(f"Params: {n_params:,}  "
              f"Shards: {len(train_shards)} train + {len(val_shards)} val "
              f"= {len(all_shards)} total")
        print(f"Batch size: {args.batch_size}  "
              f"MAX_STATE_LENGTH: {MAX_STATE_LENGTH}")

    # ---- validation function ----
    @torch.no_grad()
    def run_validation() -> dict[str, float]:
        model.eval()
        total_metrics: dict[str, float] = {}
        total_steps = 0
        for prev, next_, prev_lens, next_lens, actions in val_loader:
            prev = prev.to(device)
            next_ = next_.to(device)
            actions = actions.to(device)

            outputs = model(prev, next_, actions)
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
        return {
            f"val_{k}": total_metrics[k] / max(total_steps, 1)
            for k in total_metrics
        }

    # ---- training ----
    global_step = 0
    t_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_metrics: dict[str, float] = {}
        epoch_steps = 0
        t_epoch_start = time.time()

        for prev, next_, prev_lens, next_lens, actions in train_loader:
            prev = prev.to(device)
            next_ = next_.to(device)
            actions = actions.to(device)

            outputs = model(prev, next_, actions)
            loss, metrics = compute_losses(
                outputs,
                lambda_sigreg=lambda_sigreg,
                sigreg_num_slices=sigreg_num_slices,
                sigreg_num_points=sigreg_num_points,
                sigreg_domain=sigreg_domain,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            epoch_steps += 1
            global_step += 1

            # ---- per-step logging ----
            if global_step % args.log_interval == 0:
                if log_file:
                    log_file.write(
                        f"{epoch},{global_step},{metrics['loss']:.6f},"
                        f"{metrics['jepa_loss']:.6f},"
                        f"{metrics['sigreg_prev']:.6f},"
                        f"{metrics['sigreg_next']:.6f},"
                        f"{metrics['sigreg_loss']:.6f},"
                        f"{optimizer.param_groups[0]['lr']:.2e},"
                        f",,,,,,,\n"
                    )
                    log_file.flush()

                if wandb_run:
                    wandb_run.log({
                        "train/loss": metrics["loss"],
                        "train/jepa_loss": metrics["jepa_loss"],
                        "train/sigreg_prev": metrics["sigreg_prev"],
                        "train/sigreg_next": metrics["sigreg_next"],
                        "train/sigreg_loss": metrics["sigreg_loss"],
                        "train/lr": optimizer.param_groups[0]["lr"],
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
                        f"lr {optimizer.param_groups[0]['lr']:.2e}"
                    )

        scheduler.step()

        # ---- validation ----
        val_metrics = run_validation()

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
            f"time {t_epoch:.0f}s ==="
        )

        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": avg_metrics["loss"],
                "epoch/train_jepa_loss": avg_metrics["jepa_loss"],
                "epoch/train_sigreg_prev": avg_metrics["sigreg_prev"],
                "epoch/train_sigreg_next": avg_metrics["sigreg_next"],
                "epoch/train_sigreg_loss": avg_metrics["sigreg_loss"],
                "epoch/val_loss": val_metrics.get("val_loss", 0),
                "epoch/val_jepa_loss": val_metrics.get("val_jepa_loss", 0),
                "epoch/val_sigreg_loss": val_metrics.get("val_sigreg_loss", 0),
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save best checkpoint. Also saves latest_checkpoint.pt alongside.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
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
