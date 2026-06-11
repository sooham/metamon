"""Train a WorldModelTransformer on next-state prediction.

Loads sharded .npz files produced by scripts/generate_world_model_data.py
and trains an autoregressive transformer to predict state[t+1] tokens
conditioned on state[t] and action[t].

States are variable-length (unpadded in storage).  Batches are padded to
the per-batch maximum and the model uses <eos> to find state boundaries.

Prompt structure:
    <bos>  state[t]  <eos>  <boa>  action  <eoa>  <bos>  state[t+1]  <eos>

Loss is computed only on the state[t+1] region (including its <eos>).

Usage:
    # Minimal — only epoch summaries printed
    uv run python -m metamon.sl.train \\
        --data_root ~/Repositories/poke-datasets/world-model-samples \\
        --formats gen1ou \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \\
        --save_dir ~/metamon_sl_checkpoints \\
        --batch_size 32 --lr 3e-4 --epochs 10

    # With wandb + CSV logging, prints every 500 steps
    uv run python -m metamon.sl.train \\
        --data_root ~/Repositories/poke-datasets/world-model-samples \\
        --formats gen1ou \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \\
        --save_dir ~/metamon_sl_checkpoints \\
        --batch_size 32 --lr 3e-4 --epochs 10 \\
        --wandb --wandb_project my-project --wandb_name my-run \\
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

from metamon.sl.model import (
    WorldModelTransformer,
    compute_loss,
    MAX_CONTEXT_LENGTH,
    SAFETY_FACTOR,
)

# Optional wandb import
_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass


# ── Dataset ──────────────────────────────────────────────────────────────

class WorldModelDataset(torch.utils.data.IterableDataset):
    """Iterable over sharded .npz files, yielding variable-length transitions.

    Each .npz shard contains concatenated battles with ``states`` (flat array),
    ``state_lengths`` (per-state token count), ``actions``, and
    ``battle_start`` arrays.  Transitions are yielded **within** battles only
    — cross-battle boundaries are skipped.

    States are yielded **unpadded**; padding to batch-max happens in the
    collate function.
    """

    def __init__(
        self,
        data_root: str,
        formats: list[str],
        shuffle_shards: bool = True,
    ):
        super().__init__()
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

    def _iter_shard(
        self, path: str
    ) -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]:
        """Yield (state_t_len, state_next_len, action, state_t, state_next) within battles.

        Uses ``battle_start`` to honour battle boundaries, ensuring transitions
        never cross between different battles.  States are sliced from a flat
        1-D token array using ``state_offsets`` and ``state_lengths``.
        """
        data = np.load(path)
        states = data["states"]              # flat 1-D array of all token IDs
        state_lengths = data["state_lengths"]  # (N,) actual token counts
        state_offsets = data["state_offsets"]  # (N,) start index of each state in *states*
        actions = data["actions"]            # (total_actions,) — one per valid transition
        battle_start = data["battle_start"]  # (num_battles+1,) cumulative state indices

        num_battles = len(battle_start) - 1

        # Iterate battles, then transitions within each battle
        for b in range(num_battles):
            s_start = battle_start[b]
            s_end = battle_start[b + 1]
            n_states = s_end - s_start  # states in this battle

            for t in range(n_states - 1):
                idx_t = s_start + t
                idx_next = idx_t + 1

                # Find action: actions[idx_t - b] connects state_t → state_next.
                # (idx_t is a state-space index; the actions array has fewer entries
                #  than states — one fewer per battle — so we subtract the battle offset.)
                action = int(actions[idx_t - b])

                st_len = int(state_lengths[idx_t])
                sn_len = int(state_lengths[idx_next])

                st_off = state_offsets[idx_t]
                sn_off = state_offsets[idx_next]

                state_t = states[st_off : st_off + st_len]
                state_next = states[sn_off : sn_off + sn_len]

                yield st_len, sn_len, action, state_t, state_next

    def __iter__(
        self,
    ) -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]:
        paths = self.shard_paths.copy()
        if self.shuffle_shards:
            np.random.shuffle(paths)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        for path in paths:
            yield from self._iter_shard(path)


def collate_fn(
    batch: list[tuple[int, int, int, np.ndarray, np.ndarray]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length transitions into padded tensors.

    Input: list of (state_t_len, state_next_len, action, state_t, state_next)
    Output: (state_t, state_next, actions, state_t_lengths, state_next_lengths)
      where state_t/state_next are padded to the batch maximum with *pad_id*.
    """
    state_t_lengths = torch.tensor([item[0] for item in batch], dtype=torch.long)
    state_next_lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    actions = torch.tensor([item[2] for item in batch], dtype=torch.long)

    max_st = int(state_t_lengths.max().item())
    max_sn = int(state_next_lengths.max().item())

    state_t_padded = torch.full((len(batch), max_st), pad_id, dtype=torch.long)
    state_next_padded = torch.full((len(batch), max_sn), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        st = item[3]
        sn = item[4]
        st_len = item[0]
        sn_len = item[1]
        state_t_padded[i, :st_len] = torch.from_numpy(st[:st_len].astype(np.int64))
        state_next_padded[i, :sn_len] = torch.from_numpy(sn[:sn_len].astype(np.int64))

    return state_t_padded, state_next_padded, actions, state_t_lengths, state_next_lengths


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
            if "states" in data:
                max_id = max(max_id, int(data["states"].max()))
            count += 1
            if count >= 3:
                break
        if count >= 3:
            break
    return max_id  # tokens are 1-based, 0 is unused → max_id == vocab_size


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

    # ---- tokenizer (for vocab size + special token IDs) ----
    from metamon.tokenizer import PokemonTokenizer

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

    # Action-index → token-ID mapping (world model uses action tokens
    # as regular vocabulary entries instead of a separate embedding table).
    action_to_token_id = {
        i: tokenizer.get_action_token_id(i) for i in range(-1, 13)
    }

    # Build the IGNORE_LOSS_TOKENS set.
    # <pad> tokens are structural padding — never learn to predict them.
    # <bos>, <boa>, <eoa> are fixed structural markers — not informative.
    # <unk> is NOT ignored — if a genuinely unknown token appears mid-sequence,
    # the model should learn to represent it from context.
    pad_id = tokenizer.pad_token_id
    ignore_loss_tokens: set[int] = {pad_id}
    if args.print_interval > 0:
        print(f"Ignoring loss for token IDs: {sorted(ignore_loss_tokens)} "
              f"(<pad>)")

    # ---- dataset ----
    dataset = WorldModelDataset(
        data_root=args.data_root,
        formats=args.formats,
        shuffle_shards=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # ---- model ----
    d_model = model_cfg["d_model"]
    n_heads = model_cfg["n_heads"]
    n_layers = model_cfg["n_layers"]
    d_ff = model_cfg["d_ff"]
    dropout = model_cfg.get("dropout", 0.1)
    # RoPE cache must cover at least MAX_CONTEXT_LENGTH
    max_seq_len = max(model_cfg.get("max_seq_len", 1024), MAX_CONTEXT_LENGTH)
    safety_factor = model_cfg.get("safety_factor", SAFETY_FACTOR)

    model = WorldModelTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        action_to_token_id=action_to_token_id,
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
    attn_flops_per_token = 4 * d_model * d_model
    ffn_flops_per_token = 6 * d_model * d_ff
    flops_per_token_fwd = n_layers * (attn_flops_per_token + ffn_flops_per_token)
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
                "max_context_length": MAX_CONTEXT_LENGTH,
                "safety_factor": safety_factor,
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
        print(f"MAX_CONTEXT_LENGTH: {MAX_CONTEXT_LENGTH}  "
              f"SAFETY_FACTOR: {safety_factor}")
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

        for state_t, state_next, actions, st_lens, sn_lens in dataloader:
            state_t = state_t.to(device)
            state_next = state_next.to(device)
            actions = actions.to(device)
            st_lens = st_lens.to(device)
            sn_lens = sn_lens.to(device)

            logits, targets, loss_mask = model(
                state_t, state_next, actions,
                state_t_lengths=st_lens,
                state_next_lengths=sn_lens,
                bos_id=bos_id, eos_id=eos_id, boa_id=boa_id, eoa_id=eoa_id,
                ignore_loss_tokens=ignore_loss_tokens,
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
            epoch_nll += metrics["loss"]
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
                    avg_st_len = st_lens.float().mean().item()
                    avg_sn_len = sn_lens.float().mean().item()
                    print(
                        f"  epoch {epoch:3d} | step {global_step:7d} | "
                        f"loss {metrics['loss']:.4f} | nll {metrics['loss']:.4f} | "
                        f"acc {metrics['token_accuracy']:.3f} | "
                        f"st_len {avg_st_len:.0f} | sn_len {avg_sn_len:.0f} | "
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

        # ---- checkpoint (every 10 epochs, overwrite single file) ----
        if args.checkpoint is not None and (epoch + 1) % 10 == 0:
            model.save_checkpoint(
                args.checkpoint,
                epoch=epoch,
                global_step=global_step,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                config=model_cfg,
                vocab_size=vocab_size,
                bos_id=bos_id,
                eos_id=eos_id,
                boa_id=boa_id,
                eoa_id=eoa_id,
            )
            if args.print_interval > 0:
                print(f"  Saved checkpoint to {args.checkpoint}")

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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save checkpoint every 10 epochs (overwrites). If absent, no checkpointing.")
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
