"""Train a WorldModelTransformer on next-state prediction.

Loads sharded .npz files produced by scripts/generate_world_model_data.py
and trains an autoregressive transformer to predict state[t+1] tokens
conditioned on state[t] and action[t].

Prompt structure:
    <bos>  state[t]  <eos>  <boa>  action  <eoa>  <bos>  state[t+1]  <eos>

Loss is computed only on the state[t+1] region (including its <eos>).

Usage:
    # Minimal — only epoch summaries printed
    uv run python -m metamon.sl.train \
        --data_root ~/Repositories/poke-datasets/world-model-samples \
        --formats gen1ou \
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \
        --save_dir ~/metamon_sl_checkpoints \
        --batch_size 32 --lr 3e-4 --epochs 10

    # With wandb + CSV logging, prints every 500 steps
    uv run python -m metamon.sl.train \
        --data_root ~/Repositories/poke-datasets/world-model-samples \
        --formats gen1ou \
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \
        --save_dir ~/metamon_sl_checkpoints \
        --batch_size 32 --lr 3e-4 --epochs 10 \
        --wandb --wandb_project my-project --wandb_name my-run \
        --log --print_interval 200
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import yaml

from metamon.sl.model import WorldModelTransformer, compute_loss
from metamon.tokenizer.tokenizer import PokemonTokenizer

# Optional wandb import
_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass


# ── Dataset ──────────────────────────────────────────────────────────────

class WorldModelDataset(torch.utils.data.IterableDataset):
    """Iterable over sharded .npz files, yielding (state_t, state_next, action).

    Each .npz shard contains concatenated battles.  We stream through all
    shards, emitting every valid transition as a training example.
    """

    def __init__(
        self,
        data_root: str,
        formats: list[str],
        max_state_tokens: int = 336,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.max_state_tokens = max_state_tokens
        self.shuffle_shards = shuffle_shards

        self.shard_paths: list[str] = []
        for fmt in formats:
            fmt_dir = os.path.join(data_root, fmt)
            if not os.path.isdir(fmt_dir):
                continue
            for f in sorted(os.listdir(fmt_dir)):
                if f.endswith(".npz"):
                    self.shard_paths.append(os.path.join(fmt_dir, f))

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No .npz shards found under {data_root} for formats {formats}"
            )

    def _pad_or_truncate(self, arr: np.ndarray, length: int) -> np.ndarray:
        """Pad or truncate a 1-D token array to *length* with 0 (UNKNOWN_TOKEN)."""
        out = np.full(length, 0, dtype=np.int32)
        n = min(len(arr), length)
        out[:n] = arr[:n].astype(np.int32)
        return out

    def _iter_shard(self, path: str) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (state_t, state_next, action) tuples from one .npz shard."""
        data = np.load(path)
        states = data["states"]       # (N, S_raw) int16
        actions = data["actions"]     # (N-1,) int16
        n_states = states.shape[0]

        for t in range(n_states - 1):
            state_t = self._pad_or_truncate(states[t], self.max_state_tokens)
            state_next = self._pad_or_truncate(states[t + 1], self.max_state_tokens)
            action = int(actions[t])
            yield state_t, state_next, action

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            for st, sn, act in self._iter_shard(path):
                yield (
                    torch.from_numpy(st),
                    torch.from_numpy(sn),
                    torch.tensor(act, dtype=torch.long),
                )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of (state_t, state_next, action) into batched tensors."""
    states_t = torch.stack([item[0] for item in batch])
    states_next = torch.stack([item[1] for item in batch])
    actions = torch.stack([item[2] for item in batch])
    return states_t, states_next, actions


# ── Vocabulary size helpers ──────────────────────────────────────────────

def detect_vocab_size_from_data(data_root: str, formats: list[str]) -> int:
    """Scan the first few shards to find the maximum token ID."""
    max_id = 0
    count = 0
    for fmt in formats:
        fmt_dir = os.path.join(data_root, fmt)
        if not os.path.isdir(fmt_dir):
            continue
        for f in sorted(os.listdir(fmt_dir)):
            if not f.endswith(".npz"):
                continue
            data = np.load(os.path.join(fmt_dir, f))
            max_id = max(max_id, int(data["states"].max()))
            count += 1
            if count >= 3:
                break
        if count >= 3:
            break
    return max_id  # UNKNOWN_TOKEN = 0, so max_id = vocab_size


def detect_vocab_size_from_tokenizer(tokenizer_path: str) -> int:
    """Load a tokenizer JSON and return its vocabulary size."""
    with open(tokenizer_path, "rb") as f:
        tokens = json.loads(f.read())
    return len(tokens)


# ── Training loop ────────────────────────────────────────────────────────

def train(args):
    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.print_interval > 0:
        print(f"Using device: {device}")

    # ---- config ----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    max_state_tokens = model_cfg.get("max_state_tokens", 336)

    # ---- tokenizer (for vocab size + special token IDs) ----
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    vocab_size = model_cfg.get("vocab_size") or len(tokenizer)

    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]
    boa_id = tokenizer["<boa>"]
    eoa_id = tokenizer["<eoa>"]
    assert bos_id != 0, "<bos> not found in tokenizer — rebuild the tokenizer"
    assert eos_id != 0, "<eos> not found in tokenizer — rebuild the tokenizer"
    assert boa_id != 0, "<boa> not found in tokenizer — rebuild the tokenizer"
    assert eoa_id != 0, "<eoa> not found in tokenizer — rebuild the tokenizer"

    # ---- dataset ----
    dataset = WorldModelDataset(
        data_root=args.data_root,
        formats=args.formats,
        max_state_tokens=max_state_tokens,
        shuffle_shards=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # ---- model ----
    d_model = model_cfg["d_model"]
    n_heads = model_cfg["n_heads"]
    n_layers = model_cfg["n_layers"]
    d_ff = model_cfg["d_ff"]
    dropout = model_cfg.get("dropout", 0.1)
    max_seq_len = model_cfg.get("max_seq_len", 680)

    model = WorldModelTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        theta=model_cfg.get("theta", 10000.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs estimate (forward pass, per token, no backward):
    #   2 × (attention + FFN)  [multiply-add = 2 ops]
    #   attention ≈ 4 × d_model² + 2 × d_model × seq_len (dot products, rough)
    #   SwiGLU FFN ≈ 3 × 2 × d_model × d_ff  (w1, w2, out projections)
    # We report total FLOPs/s including backward (~3× fwd for training).
    attn_flops_per_token = 4 * d_model * d_model                         # QKV + out projections
    ffn_flops_per_token = 6 * d_model * d_ff                             # gate, up, out (each 2 ops)
    flops_per_token_fwd = n_layers * (attn_flops_per_token + ffn_flops_per_token)
    # training: forward + backward ≈ 3×
    flops_per_token_total = flops_per_token_fwd * 3

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ---- checkpoint dir ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb init ----
    wandb_run: Optional["wandb"] = None
    if args.wandb and _wandb_available:
        wandb_run = wandb.init(
            project=args.wandb_project or "metamon-" + "-".join(args.formats),
            name=args.wandb_name or save_dir.name,
            config={
                **model_cfg,
                "vocab_size": vocab_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "n_params": n_params,
                "flops_per_token_fwd": flops_per_token_fwd,
                "max_seq_len": max_seq_len,
                "max_state_tokens": max_state_tokens,
            },
        )
    elif args.wandb and not _wandb_available:
        print("WARNING: --wandb set but wandb not installed (pip install wandb)")

    # ---- CSV logging ----
    log_file = None
    if args.log:
        log_path = save_dir / "metrics.csv"
        log_file = open(log_path, "w")
        log_file.write("epoch,step,loss,nll,token_accuracy,lr,tokens_per_s,mflops,elapsed_s\n")

    # ---- print header ----
    if args.print_interval > 0:
        print(f"Vocab: {vocab_size}  Params: {n_params:,}  Shards: {len(dataset.shard_paths)}")
        print(f"FLOPs/token (fwd): {flops_per_token_fwd/1e6:.1f}M  "
              f"FLOPs/token (train): {flops_per_token_total/1e6:.1f}M")

    # ---- training ----
    global_step = 0
    t_start = time.time()
    token_count = 0  # total tokens processed (for throughput)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        for state_t, state_next, actions in dataloader:
            state_t = state_t.to(device)
            state_next = state_next.to(device)
            actions = actions.to(device)

            logits, targets, loss_mask = model(
                state_t, state_next, actions,
                bos_id=bos_id, eos_id=eos_id, boa_id=boa_id, eoa_id=eoa_id,
            )
            loss, metrics = compute_loss(logits, targets, loss_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            B = state_t.shape[0]
            T = logits.shape[1]  # prompt length - 1
            tokens_this_step = B * T
            token_count += tokens_this_step

            epoch_loss += metrics["loss"]
            epoch_nll += metrics["loss"]  # cross-entropy IS negative log-likelihood (natural log)
            epoch_acc += metrics["token_accuracy"]
            epoch_steps += 1
            global_step += 1

            # ---- per-step logging ----
            if global_step % args.log_interval == 0:
                elapsed = time.time() - t_start
                tokens_per_s = token_count / elapsed if elapsed > 0 else 0.0
                flops_est = token_count * flops_per_token_total
                mflops = flops_est / (elapsed * 1e6) if elapsed > 0 else 0.0

                if log_file:
                    log_file.write(
                        f"{epoch},{global_step},{metrics['loss']:.6f},"
                        f"{metrics['loss']:.6f},"
                        f"{metrics['token_accuracy']:.4f},"
                        f"{optimizer.param_groups[0]['lr']:.2e},"
                        f"{tokens_per_s:.0f},{mflops:.1f},{elapsed:.1f}\n"
                    )
                    log_file.flush()

                if wandb_run:
                    wandb_run.log({
                        "train/loss": metrics["loss"],
                        "train/nll": metrics["loss"],
                        "train/token_accuracy": metrics["token_accuracy"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/tokens_per_s": tokens_per_s,
                        "train/mflops": mflops,
                        "epoch": epoch,
                        "global_step": global_step,
                    })

                if args.print_interval > 0 and global_step % args.print_interval == 0:
                    print(
                        f"  epoch {epoch:3d} | step {global_step:7d} | "
                        f"loss {metrics['loss']:.4f} | nll {metrics['loss']:.4f} | "
                        f"acc {metrics['token_accuracy']:.3f} | "
                        f"tok/s {tokens_per_s:,.0f} | "
                        f"MFLOPS {mflops:.0f} | "
                        f"lr {optimizer.param_groups[0]['lr']:.2e}"
                    )

        scheduler.step()

        # ---- epoch-end metrics ----
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_nll = epoch_nll / max(epoch_steps, 1)
        avg_acc = epoch_acc / max(epoch_steps, 1)
        t_epoch = time.time() - t_epoch_start
        elapsed = time.time() - t_start
        tokens_per_s = token_count / elapsed if elapsed > 0 else 0.0
        flops_est = token_count * flops_per_token_total
        mflops = flops_est / (elapsed * 1e6) if elapsed > 0 else 0.0

        print(
            f"=== epoch {epoch:3d} done | "
            f"loss {avg_loss:.4f} | nll {avg_nll:.4f} | acc {avg_acc:.3f} | "
            f"time {t_epoch:.0f}s | "
            f"tok/s {tokens_per_s:,.0f} | "
            f"MFLOPS {mflops:.0f} ==="
        )

        if wandb_run:
            wandb_run.log({
                "epoch/avg_loss": avg_loss,
                "epoch/avg_nll": avg_nll,
                "epoch/avg_token_accuracy": avg_acc,
                "epoch/time_s": t_epoch,
                "epoch/tokens_per_s": tokens_per_s,
                "epoch/mflops": mflops,
                "epoch": epoch,
            })

        # ---- checkpoint ----
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model_cfg,
            "vocab_size": vocab_size,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "boa_id": boa_id,
            "eoa_id": eoa_id,
        }
        ckpt_path = save_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(ckpt, ckpt_path)
        if args.print_interval > 0:
            print(f"  Saved checkpoint to {ckpt_path}")

    if log_file:
        log_file.close()

    if wandb_run:
        wandb_run.finish()

    print(f"Training complete.  Checkpoints: {save_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a WorldModelTransformer on next-state prediction."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--formats", type=str, nargs="+", required=True)
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to WorldModel tokenizer JSON (must contain <bos>/<eos>/<boa>/<eoa>).")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    # Logging
    parser.add_argument("--log", action="store_true",
                        help="Write per-step metrics to metrics.csv in save_dir.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log every N training steps (CSV + wandb).")
    parser.add_argument("--print_interval", type=int, default=500,
                        help="Print to console every N steps (0 = only epoch summaries).")
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (default: metamon-<format>).")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name (default: save_dir basename).")
    args = parser.parse_args()

    train(args)
