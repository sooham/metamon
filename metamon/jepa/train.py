"""Train a JEPA model on raw Pokémon showdown replays.

Loads raw replay JSON files, tokenizes them with a BPE tokenizer learned
from the replay corpus, randomly masks spans, and trains:

1. A JEPA (Joint Embedding Predictive Architecture) that learns
   representations by predicting the unmasked-view latent from a
   masked-view latent.
2. A β-VAE autoencoder that reconstructs the replay from the latent,
   regularised toward an isotropic Gaussian prior.

Usage:
    # Minimal — only epoch summaries printed
    uv run python -m metamon.jepa.train \\
        --data_root $METAMON_CACHE_DIR/raw-replays \\
        --formats gen1ou gen9ou \\
        --bpe_vocab_size 16384 \\
        --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \\
        --batch_size 16 --lr 3e-4 --epochs 10

    # With wandb + CSV logging
    uv run python -m metamon.jepa.train \\
        --data_root $METAMON_CACHE_DIR/raw-replays \\
        --formats gen1ou gen9ou \\
        --bpe_vocab_size 16384 \\
        --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \\
        --batch_size 16 --lr 3e-4 --epochs 10 \\
        --wandb --wandb_project metamon-jepa --wandb_name run-01 \\
        --log --print_interval 100
"""

import argparse
import json
import math
import os
import random
import sys
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
    MAX_SEQ_LENGTH,
)

# Optional wandb import
_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    pass


# ── Masking utilities ───────────────────────────────────────────────────


def random_span_mask(
    token_ids: torch.Tensor,
    mask_id: int,
    pad_id: int,
    mask_ratio: float = 0.3,
    span_lambda: float = 3.0,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Randomly mask spans of tokens with *mask_id*.

    Implements span-based masking similar to BERT / Wav2Vec 2.0 / data2vec:
    spans are sampled from a geometric distribution (mean = span_lambda)
    until the target *mask_ratio* of non-pad tokens is covered.

    Args:
        token_ids:  (S,) int — single sequence of BPE token IDs.
        mask_id:    token ID to use for <mask>.
        pad_id:     token ID for <pad>.
        mask_ratio: fraction of non-pad tokens to mask (0.0–1.0).
        span_lambda: mean span length for geometric distribution.
        rng:        optional random.Random instance (for reproducibility).

    Returns:
        masked: (S,) int — copy of *token_ids* with some spans replaced by
                *mask_id*.
    """
    if rng is None:
        rng = random.Random()

    masked = token_ids.clone()
    valid_mask = token_ids != pad_id
    valid_positions = valid_mask.nonzero(as_tuple=True)[0].tolist()

    if len(valid_positions) == 0:
        return masked

    target_count = int(len(valid_positions) * mask_ratio)
    masked_count = 0

    while masked_count < target_count and len(valid_positions) > 0:
        # Sample span length from geometric distribution
        span_len = 0
        while span_len < 1:
            span_len = int(rng.expovariate(1.0 / span_lambda)) + 1

        # Pick a random valid position as the start of the span
        start_idx = rng.randrange(len(valid_positions))
        start = valid_positions[start_idx]

        # Mask up to span_len tokens starting from start
        for offset in range(span_len):
            pos = start + offset
            if pos >= len(token_ids):
                break
            if token_ids[pos] == pad_id:
                break
            if masked[pos] != mask_id:
                masked[pos] = mask_id
                masked_count += 1

        if masked_count >= target_count:
            break

    return masked


def collate_and_mask(
    batch: list[torch.Tensor],
    mask_id: int,
    pad_id: int,
    mask_ratio: float = 0.3,
    span_lambda: float = 3.0,
    max_seq_len: int = MAX_SEQ_LENGTH,
    rng: Optional[random.Random] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length token sequences into padded tensors and
    produce masked versions for the JEPA context encoder.

    Args:
        batch:         list of (S_i,) int tensors — one per replay.
        mask_id:       <mask> token ID.
        pad_id:        <pad> token ID.
        mask_ratio:    fraction of non-pad tokens to mask.
        span_lambda:   mean span length for masking.
        max_seq_len:   maximum sequence length (truncation).
        rng:           random.Random for reproducible masking.

    Returns:
        token_ids:        (B, S) int — padded (and truncated) original tokens.
        masked_token_ids: (B, S) int — same but with random span masks.
        lengths:          (B,)  int — actual (non-pad) token count per example.
    """
    if rng is None:
        rng = random.Random()

    B = len(batch)
    # Truncate and find max length
    truncated = []
    lengths = []
    for t in batch:
        t = t[:max_seq_len]
        truncated.append(t)
        lengths.append(len(t))

    max_len = max(lengths) if lengths else 1
    S = max_len

    token_ids = torch.full((B, S), pad_id, dtype=torch.long)
    masked_token_ids = torch.full((B, S), pad_id, dtype=torch.long)

    for i, t in enumerate(truncated):
        L = len(t)
        token_ids[i, :L] = t
        masked = random_span_mask(
            t, mask_id=mask_id, pad_id=pad_id,
            mask_ratio=mask_ratio, span_lambda=span_lambda, rng=rng,
        )
        masked_token_ids[i, :L] = masked

    return token_ids, masked_token_ids, torch.tensor(lengths, dtype=torch.long)


# ── Dataset — loads raw replay strings from disk ────────────────────────
#
# NOTE: This is scaffolding.  Currently it loads raw replay JSON files
# and extracts the "log" field as plain text.  In a real implementation
# you will:
#   1. Build a BPE tokenizer over the raw-replay corpus (sentencepiece,
#      tokenizers, or tiktoken).
#   2. Serialise the BPE model to disk.
#   3. Replace the dummy_tokenize() stub below with real tokenization.
#   4. Optionally pre-tokenize and cache the token sequences to avoid
#      re-tokenizing every epoch.
# ────────────────────────────────────────────────────────────────────────


def dummy_tokenize(text: str, vocab_size: int) -> torch.Tensor:
    """Stub BPE tokenizer — replace with real implementation.

    Currently does a trivial character-level tokenization capped to
    *vocab_size* — this is NOT a real BPE tokenizer and is only for
    scaffolding / smoke-testing the training loop.
    """
    # Character-level fallback: every byte becomes a token ID in [1, 255].
    # IDs 0, 256+ are reserved for special tokens.
    byte_ids = text.encode("utf-8", errors="replace")
    # Map to 1-based token IDs, clamped to vocab_size-1
    tokens = []
    for b in byte_ids:
        tok = min(b + 1, vocab_size - 1)
        tokens.append(tok)
    return torch.tensor(tokens, dtype=torch.long)


class RawReplayDataset(torch.utils.data.IterableDataset):
    """Iterable dataset over raw replay JSON files.

    Each file is loaded, its ``"log"`` field is extracted, and the
    resulting text is tokenized (currently with a dummy tokenizer).

    Parameters
    ----------
    file_paths : list[str]
        Paths to raw replay JSON files.
    vocab_size : int
        BPE vocabulary size (used by dummy tokenizer for now).
    shuffle : bool
        Whether to shuffle file order each epoch.
    """

    def __init__(self, file_paths: list[str], vocab_size: int, shuffle: bool = True):
        super().__init__()
        self.file_paths = file_paths
        self.vocab_size = vocab_size
        self.shuffle = shuffle

        if not self.file_paths:
            raise ValueError("No replay file paths provided")

    @classmethod
    def from_formats(
        cls,
        data_root: str,
        formats: list[str],
        vocab_size: int,
        max_files: int = 0,
        shuffle: bool = True,
    ) -> "RawReplayDataset":
        """Discover replay JSON files under *data_root* for the given *formats*.

        Directory layout: ``{data_root}/{format}/*.json``

        Args:
            data_root: root of raw-replays directory.
            formats: list of format names (e.g. ``["gen1ou", "gen9ou"]``).
            vocab_size: BPE vocabulary size.
            max_files: if > 0, limit to this many files total (for debugging).
            shuffle: shuffle file order.
        """
        file_paths: list[str] = []
        for fmt in formats:
            fmt_dir = os.path.join(data_root, fmt)
            if not os.path.isdir(fmt_dir):
                print(f"WARNING: format directory not found: {fmt_dir}")
                continue
            # Use ls -f to avoid listing millions of files —
            # we sample a subset of inode entries.
            try:
                entries = os.listdir(fmt_dir)
            except Exception as e:
                print(f"WARNING: could not list {fmt_dir}: {e}")
                continue
            for fname in entries:
                if fname.endswith(".json"):
                    file_paths.append(os.path.join(fmt_dir, fname))
                    if max_files > 0 and len(file_paths) >= max_files:
                        break
            if max_files > 0 and len(file_paths) >= max_files:
                break

        if not file_paths:
            raise FileNotFoundError(
                f"No replay JSON files found under {data_root} for formats {formats}"
            )
        if shuffle:
            random.shuffle(file_paths)
        return cls(file_paths, vocab_size, shuffle=shuffle)

    def __iter__(self) -> Iterator[torch.Tensor]:
        paths = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                log_text = data.get("log", "")
                if not log_text:
                    continue
            except Exception:
                continue

            # ── TODO: replace dummy_tokenize with real BPE tokenizer ──
            tokens = dummy_tokenize(log_text, self.vocab_size)
            yield tokens


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

    # ── TODO: load real BPE tokenizer here ─────────────────────────────
    # For now, use a stub vocabulary:
    #   Special tokens:
    #     0 = <pad>
    #     1 = <mask>
    #     2 = <bos>
    #     3 = <eos>
    #     4–255 = byte-level tokens
    #     256+ = reserved for learned BPE merges
    # ────────────────────────────────────────────────────────────────────
    vocab_size = model_cfg.get("vocab_size") or args.bpe_vocab_size
    pad_id = 0
    mask_id = 1
    bos_id = 2
    eos_id = 3

    latent_dim = model_cfg.get("latent_dim", LATENT_DIM)
    beta_recon = model_cfg.get("beta_recon", 1.0)
    beta_kl = model_cfg.get("beta_kl", 0.001)

    if args.print_interval > 0:
        print(f"Vocab size: {vocab_size}  Latent dim: {latent_dim}  "
              f"Special tokens: pad={pad_id} mask={mask_id} bos={bos_id} eos={eos_id}")
        print(f"β_recon={beta_recon}  β_kl={beta_kl}")

    # ---- datasets ----
    all_files = RawReplayDataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        vocab_size=vocab_size,
        max_files=args.max_files,
        shuffle=False,  # we shuffle manually before splitting
    ).file_paths

    rng = random.Random(args.seed)
    rng.shuffle(all_files)

    n_val = max(1, int(len(all_files) * args.val_split))
    n_train = len(all_files) - n_val
    train_files = all_files[:n_train]
    val_files = all_files[n_train:]

    train_dataset = RawReplayDataset(train_files, vocab_size, shuffle=True)
    val_dataset = RawReplayDataset(val_files, vocab_size, shuffle=False)

    # Collate wrapper that handles masking
    _mask_rng = random.Random(args.seed)

    def train_collate(batch):
        return collate_and_mask(
            batch,
            mask_id=mask_id,
            pad_id=pad_id,
            mask_ratio=args.mask_ratio,
            span_lambda=args.span_lambda,
            max_seq_len=MAX_SEQ_LENGTH,
            rng=_mask_rng,
        )

    def val_collate(batch):
        return collate_and_mask(
            batch,
            mask_id=mask_id,
            pad_id=pad_id,
            mask_ratio=args.mask_ratio,
            span_lambda=args.span_lambda,
            max_seq_len=MAX_SEQ_LENGTH,
            rng=_mask_rng,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=val_collate,
        num_workers=max(1, args.num_workers // 2),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=False,
    )

    # ---- model ----
    model = JEPAModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        mask_id=mask_id,
        bos_id=bos_id,
        eos_id=eos_id,
        latent_dim=latent_dim,
        encoder_cfg=model_cfg.get("encoder", {}),
        predictor_cfg=model_cfg.get("predictor", {}),
        decoder_cfg=model_cfg.get("decoder", {}),
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
                "mask_ratio": args.mask_ratio,
                "span_lambda": args.span_lambda,
                "n_params": n_params,
                "beta_recon": beta_recon,
                "beta_kl": beta_kl,
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
            "epoch,step,loss,jepa_loss,recon_loss,recon_acc,kl_loss,lr,"
            "val_loss,val_jepa_loss,val_recon_loss,val_recon_acc,val_kl_loss\n"
        )

    # ---- print header ----
    if args.print_interval > 0:
        print(f"Vocab: {vocab_size}  Params: {n_params:,}  "
              f"Files: {len(train_files)} train + {len(val_files)} val "
              f"= {len(all_files)} total")
        print(f"Batch size: {args.batch_size}  "
              f"Mask ratio: {args.mask_ratio}  Span lambda: {args.span_lambda}")
        print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

    # ---- validation function ----
    @torch.no_grad()
    def run_validation() -> dict[str, float]:
        model.eval()
        total_metrics = {
            "loss": 0.0, "jepa_loss": 0.0, "recon_loss": 0.0,
            "recon_acc": 0.0, "kl_loss": 0.0,
        }
        total_steps = 0
        for token_ids, masked_token_ids, lengths in val_loader:
            token_ids = token_ids.to(device)
            masked_token_ids = masked_token_ids.to(device)

            outputs = model(token_ids, masked_token_ids, target_ids=token_ids)
            _, metrics = compute_losses(
                outputs,
                target_ids=token_ids,
                mask_id=mask_id,
                pad_id=pad_id,
                beta_recon=beta_recon,
                beta_kl=beta_kl,
            )
            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0.0)
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
        epoch_metrics = {
            "loss": 0.0, "jepa_loss": 0.0, "recon_loss": 0.0,
            "recon_acc": 0.0, "kl_loss": 0.0,
        }
        epoch_steps = 0
        t_epoch_start = time.time()

        for token_ids, masked_token_ids, lengths in train_loader:
            token_ids = token_ids.to(device)
            masked_token_ids = masked_token_ids.to(device)

            outputs = model(token_ids, masked_token_ids, target_ids=token_ids)
            loss, metrics = compute_losses(
                outputs,
                target_ids=token_ids,
                mask_id=mask_id,
                pad_id=pad_id,
                beta_recon=beta_recon,
                beta_kl=beta_kl,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k in epoch_metrics:
                epoch_metrics[k] += metrics.get(k, 0.0)
            epoch_steps += 1
            global_step += 1

            # ---- per-step logging ----
            if global_step % args.log_interval == 0:
                elapsed = time.time() - t_start

                if log_file:
                    log_file.write(
                        f"{epoch},{global_step},{metrics['loss']:.6f},"
                        f"{metrics['jepa_loss']:.6f},"
                        f"{metrics['recon_loss']:.6f},"
                        f"{metrics['recon_accuracy']:.4f},"
                        f"{metrics['kl_loss']:.6f},"
                        f"{optimizer.param_groups[0]['lr']:.2e},"
                        f",,,,,,\n"
                    )
                    log_file.flush()

                if wandb_run:
                    wandb_run.log({
                        "train/loss": metrics["loss"],
                        "train/jepa_loss": metrics["jepa_loss"],
                        "train/recon_loss": metrics["recon_loss"],
                        "train/recon_accuracy": metrics["recon_accuracy"],
                        "train/kl_loss": metrics["kl_loss"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "global_step": global_step,
                    })

                if args.print_interval > 0 and global_step % args.print_interval == 0:
                    print(
                        f"  epoch {epoch:3d} | step {global_step:7d} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"jepa {metrics['jepa_loss']:.4f} | "
                        f"recon {metrics['recon_loss']:.4f} (acc {metrics['recon_accuracy']:.3f}) | "
                        f"kl {metrics['kl_loss']:.4f} | "
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
            f"recon {avg_metrics['recon_loss']:.4f} (acc {avg_metrics['recon_acc']:.3f}) | "
            f"kl {avg_metrics['kl_loss']:.4f} | "
            f"val loss {val_metrics['val_loss']:.4f} | "
            f"val jepa {val_metrics['val_jepa_loss']:.4f} | "
            f"time {t_epoch:.0f}s ==="
        )

        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": avg_metrics["loss"],
                "epoch/train_jepa_loss": avg_metrics["jepa_loss"],
                "epoch/train_recon_loss": avg_metrics["recon_loss"],
                "epoch/train_recon_accuracy": avg_metrics["recon_acc"],
                "epoch/train_kl_loss": avg_metrics["kl_loss"],
                "epoch/val_loss": val_metrics["val_loss"],
                "epoch/val_jepa_loss": val_metrics["val_jepa_loss"],
                "epoch/val_recon_loss": val_metrics["val_recon_loss"],
                "epoch/val_recon_accuracy": val_metrics["val_recon_acc"],
                "epoch/val_kl_loss": val_metrics["val_kl_loss"],
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
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
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
        description="Train a JEPA model on raw Pokémon showdown replays."
    )
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing raw-replays/{format}/*.json files.")
    parser.add_argument("--formats", type=str, nargs="+", required=True,
                        help="Format names (e.g. gen1ou gen9ou).")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Cap the total number of replay files (0 = no limit; for debugging).")
    # BPE tokenizer (stub for now)
    parser.add_argument("--bpe_vocab_size", type=int, default=16384,
                        help="BPE vocabulary size (used by dummy tokenizer for now).")
    # Model config
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    # Training
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # Masking
    parser.add_argument("--mask_ratio", type=float, default=0.3,
                        help="Fraction of non-pad tokens to mask (default: 0.3).")
    parser.add_argument("--span_lambda", type=float, default=3.0,
                        help="Mean span length for geometric masking (default: 3.0).")
    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=100)
    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()
    train(args)
