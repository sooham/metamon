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

    # With wandb + CSV logging, prints every 1000 steps
    uv run python -m metamon.sl.train \\
        --data_root ~/Repositories/poke-datasets/world-model-samples \\
        --formats gen1ou \\
        --tokenizer_path ~/Repositories/poke-datasets/tokenizers/WorldModelObservationSpace-v0.json \\
        --save_dir ~/metamon_sl_checkpoints \\
        --batch_size 32 --lr 3e-4 --epochs 10 \\
        --wandb --wandb_project my-project --wandb_name my-run \\
        --log --print_interval 1000
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
    MAX_STATE_LENGTH,
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

    Parameters
    ----------
    shard_paths : list[str]
        Explicit list of .npz shard paths to iterate over (caller handles
        train/val partitioning).
    shuffle_shards : bool
        Whether to shuffle the shard order each epoch.
    """

    def __init__(
        self,
        shard_paths: list[str],
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.shuffle_shards = shuffle_shards

        if not self.shard_paths:
            raise ValueError("No shard paths provided")

    @classmethod
    def from_formats(
        cls,
        data_root: str,
        formats: list[str],
        shuffle_shards: bool = True,
    ) -> "WorldModelDataset":
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
        return cls(shard_paths, shuffle_shards=shuffle_shards)

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
    max_state_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length transitions into padded tensors.

    Input: list of (state_t_len, state_next_len, action, state_t, state_next)
    Output: (state_t, state_next, actions, state_t_lengths, state_next_lengths)
      where state_t/state_next are padded to *max_state_len* if given,
      otherwise to the batch maximum.
    """
    state_t_lengths = torch.tensor([item[0] for item in batch], dtype=torch.long)
    state_next_lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    actions = torch.tensor([item[2] for item in batch], dtype=torch.long)

    max_st = max_state_len if max_state_len is not None else int(state_t_lengths.max().item())
    max_sn = max_state_len if max_state_len is not None else int(state_next_lengths.max().item())

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

    # ── Diagnostic: print detokenized states for high-loss examples ──
    _diag_counter = [0]  # mutable counter to rate-limit diagnostic prints

    def diagnose_batch(
        per_example_loss: list[float],
        per_example_acc: list[float],
        prompt_ids: torch.Tensor,
        st_lens: torch.Tensor,
        sn_lens: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """When batch loss exceeds threshold, print detokenized states
        for the worst examples in the batch."""
        threshold = args.diag_loss_threshold
        batch_loss = metrics["loss"]
        if batch_loss <= threshold:
            return

        _diag_counter[0] += 1
        # Rate-limit: only print every N spikes to avoid flooding
        if _diag_counter[0] % args.diag_rate_limit != 1:
            return

        B = len(per_example_loss)
        top_k = min(args.diag_top_k, B)
        # Find top-k worst examples by per-example loss
        ranked = sorted(
            range(B), key=lambda i: per_example_loss[i], reverse=True
        )[:top_k]

        print(f"\n{'='*70}")
        print(f"⚠  HIGH LOSS BATCH  loss={batch_loss:.4f}  "
              f"step={global_step}  epoch={epoch}")
        print(f"{'='*70}")

        for rank, idx in enumerate(ranked):
            ex_loss = per_example_loss[idx]
            ex_acc = per_example_acc[idx]
            st_len = st_lens[idx].item()
            sn_len = sn_lens[idx].item()
            action_idx = actions[idx].item()
            action_str = tokenizer.detokenize(
                [tokenizer.get_action_token_id(action_idx)]
            )[0]

            print(f"\n  ── Example {rank+1}/{top_k} (idx={idx}) "
                  f"loss={ex_loss:.4f} acc={ex_acc:.3f} "
                  f"|state_t|={st_len} |state_next|={sn_len} "
                  f"action={action_str} ──")

            # ── state_t ──
            # state_t starts at column 1 in prompt, length st_len
            st_ids = prompt_ids[idx, 1 : 1 + st_len].tolist()
            st_tokens = tokenizer.detokenize(st_ids)
            print(f"\n  [state_t] ({st_len} tokens)")
            print(f"    {' '.join(st_tokens)}")

            # ── state_next (target) ──
            # state_next starts at column st_len + 6 in prompt, length sn_len
            sn_start_col = st_len + 6
            sn_ids = prompt_ids[idx, sn_start_col : sn_start_col + sn_len].tolist()
            sn_tokens = tokenizer.detokenize(sn_ids)
            print(f"\n  [state_next — TARGET] ({sn_len} tokens)")
            print(f"    {' '.join(sn_tokens)}")

            # ── Raw token comparison for the state_next region ──
            # Show a compact diff: predicted vs actual for problematic tokens
            preds = logits[idx].argmax(dim=-1)  # (T-1,)
            tgt = targets[idx]  # (T-1,)
            mask = loss_mask[idx]  # (T-1,)
            # sn_start/sn_end in targets space is st_len+5 to st_len+5+sn_len
            t_sn_start = st_len + 5
            t_sn_end = t_sn_start + sn_len
            # Find positions where prediction was wrong
            wrong_positions = []
            for t in range(t_sn_start, min(t_sn_end, len(mask))):
                if mask[t] and preds[t].item() != tgt[t].item():
                    wrong_positions.append(t)
            if wrong_positions:
                print(f"\n  [Mis-predicted tokens in state_next] "
                      f"({len(wrong_positions)}/{sn_len} wrong)")
                for t in wrong_positions[:20]:  # limit output
                    pred_tok = tokenizer.detokenize([preds[t].item()])[0]
                    true_tok = tokenizer.detokenize([tgt[t].item()])[0]
                    print(f"    pos {t - t_sn_start:3d}: "
                          f"pred={pred_tok:<20} true={true_tok}")
            print()

        print(f"{'='*70}\n")

    # ---- datasets (train / val split at shard level) ----
    # Discover all shards, shuffle, then partition by shard index.
    # This guarantees no battle appears in both splits.
    all_shards = WorldModelDataset.from_formats(
        data_root=args.data_root,
        formats=args.formats,
        shuffle_shards=False,  # we shuffle manually before splitting
    ).shard_paths

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(all_shards))
    all_shards = [all_shards[i] for i in perm]

    n_val = max(1, int(len(all_shards) * args.val_split))
    n_train = len(all_shards) - n_val
    train_shards = all_shards[:n_train]
    val_shards = all_shards[n_train:]

    train_dataset = WorldModelDataset(train_shards, shuffle_shards=True)
    val_dataset = WorldModelDataset(val_shards, shuffle_shards=False)

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
    d_model = model_cfg["d_model"]
    n_heads = model_cfg["n_heads"]
    n_layers = model_cfg["n_layers"]
    d_ff = model_cfg["d_ff"]
    dropout = model_cfg.get("dropout", 0.1)
    # RoPE cache must cover at least MAX_CONTEXT_LENGTH
    max_seq_len = max(model_cfg.get("max_seq_len", 1024), MAX_CONTEXT_LENGTH)
    safety_factor = model_cfg.get("safety_factor", SAFETY_FACTOR)
    ffn_activation = model_cfg.get("ffn_activation", "gelu")

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
        ffn_activation=ffn_activation,
    ).to(device)

    # Use bfloat16 for model weights — halves memory, enables tensor-core
    # ops on Ampere+ GPUs.  bf16 does NOT need GradScaler (same exponent
    # range as fp32).  RoPE cos/sin buffers autoconvert to bf16 at runtime.
    # Compile AFTER dtype conversion so the graph captures the right dtypes.
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        # TF32 matmuls — faster on Ampere+ with negligible precision loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Capture .item() calls inside torch.compile regions so that
        # scalar tensor reads (e.g. max().item()) don't cause graph breaks.
        torch._dynamo.config.capture_scalar_outputs = True

    # Compile the model for faster training (CUDA only — MPS does not support torch.compile).
    # dynamic=True handles variable-length sequences (prompt length T varies
    # per batch because we pad to the batch maximum, not a fixed size).
    # mode="max-autotune" picks the best CUDA kernels for the given shapes
    # (takes longer on first compilation but faster thereafter).
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
        fused=True if device.type == "cuda" else False,
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
        wandb_init_kwargs = dict(
            project=args.wandb_project or "metamon-" + "-".join(args.formats),
        )
        if args.wandb_name:
            wandb_init_kwargs["name"] = args.wandb_name
        wandb_run = wandb.init(**wandb_init_kwargs,
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
        log_file.write(
            "epoch,step,loss,token_accuracy,lr,tokens_per_s,mflops,elapsed_s,"
            "val_loss,val_token_accuracy\n"
        )

    # ---- count transitions from metadata (approximate, for display) ----
    total_transitions = 0
    for fmt in args.formats:
        fmt_dir = os.path.join(args.data_root, fmt)
        meta_path = os.path.join(fmt_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            total_transitions += meta.get("total_actions", 0)
    train_est = int(total_transitions * (1.0 - args.val_split))
    val_est = total_transitions - train_est
    batches_per_epoch = math.ceil(train_est / args.batch_size) if train_est > 0 else "?"

    # ---- print header ----
    if args.print_interval > 0:
        print(f"Vocab: {vocab_size}  Params: {n_params:,}  "
              f"Shards: {len(train_shards)} train + {len(val_shards)} val "
              f"= {len(all_shards)} total")
        print(f"Batch size: {args.batch_size}  "
              f"Transitions: ~{train_est:,} train + ~{val_est:,} val  "
              f"Batches/epoch: {batches_per_epoch}")
        print(f"MAX_CONTEXT_LENGTH: {MAX_CONTEXT_LENGTH}  "
              f"SAFETY_FACTOR: {safety_factor}")
        print(f"FLOPs/token (fwd): {flops_per_token_fwd/1e6:.1f}M  "
              f"FLOPs/token (train): {flops_per_token_total/1e6:.1f}M")

    # ---- validation function ----
    @torch.no_grad()
    def run_validation(max_batches: int | None = None) -> dict[str, float]:
        """Run a pass over the val loader, return average metrics.

        Parameters
        ----------
        max_batches : int | None
            If set, stop after this many batches (for fast mid-epoch checks).
            If None, iterate the entire val dataset (for epoch-end evaluation).
        """
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_steps = 0
        for batch_idx, (state_t, state_next, actions, st_lens, sn_lens) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            state_t = state_t.to(device)
            state_next = state_next.to(device)
            actions = actions.to(device)
            st_lens = st_lens.to(device)
            sn_lens = sn_lens.to(device)

            logits, targets, loss_mask, _prompt_ids = model(
                state_t, state_next, actions,
                state_t_lengths=st_lens,
                state_next_lengths=sn_lens,
                bos_id=bos_id, eos_id=eos_id, boa_id=boa_id, eoa_id=eoa_id,
                ignore_loss_tokens=ignore_loss_tokens,
            )
            _, metrics = compute_loss(logits, targets, loss_mask)
            total_loss += metrics["loss"]
            total_acc += metrics["token_accuracy"]
            total_steps += 1
        return {
            "val_loss": total_loss / max(total_steps, 1),
            "val_acc": total_acc / max(total_steps, 1),
        }

    # ---- training ----
    global_step = 0
    t_start = time.time()
    token_count = 0  # total tokens processed (for throughput)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        for state_t, state_next, actions, st_lens, sn_lens in train_loader:
            state_t = state_t.to(device)
            state_next = state_next.to(device)
            actions = actions.to(device)
            st_lens = st_lens.to(device)
            sn_lens = sn_lens.to(device)

            logits, targets, loss_mask, prompt_ids = model(
                state_t, state_next, actions,
                state_t_lengths=st_lens,
                state_next_lengths=sn_lens,
                bos_id=bos_id, eos_id=eos_id, boa_id=boa_id, eoa_id=eoa_id,
                ignore_loss_tokens=ignore_loss_tokens,
            )
            # First compute loss without per-example overhead
            loss, metrics = compute_loss(logits, targets, loss_mask)

            # ── Diagnostic: re-compute with per-example detail on spikes ──
            _diag_enabled = args.diag_loss_threshold is not None
            if _diag_enabled and metrics["loss"] > args.diag_loss_threshold:
                _, metrics_diag = compute_loss(
                    logits, targets, loss_mask,
                    return_per_example=True,
                )
                diagnose_batch(
                    per_example_loss=metrics_diag["per_example_loss"],
                    per_example_acc=metrics_diag["per_example_acc"],
                    prompt_ids=prompt_ids,
                    st_lens=st_lens,
                    sn_lens=sn_lens,
                    actions=actions,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            B = state_t.shape[0]
            T = logits.shape[1]  # prompt length - 1
            tokens_this_step = B * T
            token_count += tokens_this_step

            epoch_loss += metrics["loss"]
            epoch_acc += metrics["token_accuracy"]
            epoch_steps += 1
            global_step += 1

            # ── Track per-batch stats & overflow check ────────────
            _max_st = int(st_lens.max().item())
            _max_sn = int(sn_lens.max().item())
            # Warn if any state exceeds MAX_STATE_LENGTH (the raw limit
            # before the safety factor).  Such states will eat into the
            # safety margin and may cause truncation at MAX_CONTEXT_LENGTH.
            _max_prompt = _max_st + _max_sn + 7
            if _max_st > MAX_STATE_LENGTH or _max_sn > MAX_STATE_LENGTH:
                overflow_idx = int((st_lens + sn_lens).argmax().item())
                print(
                    f"\n⚠  STATE OVERFLOW  step={global_step}  "
                    f"max|st|={_max_st}  max|sn|={_max_sn}  "
                    f"(MAX_STATE_LENGTH={MAX_STATE_LENGTH})\n"
                    f"   example {overflow_idx}: "
                    f"|state_t|={st_lens[overflow_idx].item()}  "
                    f"|state_next|={sn_lens[overflow_idx].item()}  "
                    f"prompt_total={_max_prompt}"
                )
                # Show the overflowing state
                if st_lens[overflow_idx].item() > MAX_STATE_LENGTH:
                    which = "state_t"
                    st_overflow = prompt_ids[
                        overflow_idx, 1 : 1 + st_lens[overflow_idx].item()
                    ].tolist()
                else:
                    which = "state_next"
                    st_overflow = prompt_ids[
                        overflow_idx,
                        st_lens[overflow_idx].item() + 6 : st_lens[overflow_idx].item() + 6 + sn_lens[overflow_idx].item()
                    ].tolist()
                print(f"   {which}: {' '.join(tokenizer.detokenize(st_overflow))}")

            # ---- per-step logging ----
            if global_step % args.log_interval == 0:
                elapsed = time.time() - t_start
                tokens_per_s = token_count / elapsed if elapsed > 0 else 0.0
                flops_est = token_count * flops_per_token_total
                mflops = flops_est / (elapsed * 1e6) if elapsed > 0 else 0.0

                if log_file:
                    log_file.write(
                        f"{epoch},{global_step},{metrics['loss']:.6f},"
                        f"{metrics['token_accuracy']:.4f},"
                        f"{optimizer.param_groups[0]['lr']:.2e},"
                        f"{tokens_per_s:.0f},{mflops:.1f},{elapsed:.1f},"
                        f",\n"
                    )
                    log_file.flush()

                if wandb_run:
                    wandb_run.log({
                        "train/loss": metrics["loss"],
                        "train/token_accuracy": metrics["token_accuracy"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/tokens_per_s": tokens_per_s,
                        "train/mflops": mflops,
                        "train/max_st_len": _max_st,
                        "train/max_sn_len": _max_sn,
                        "epoch": epoch,
                        "global_step": global_step,
                    })

                if args.print_interval > 0 and global_step % args.print_interval == 0:
                    print(
                        f"  epoch {epoch:3d} | step {global_step:7d} | "
                        f"loss {metrics['loss']:.4f} | "
                        f"acc {metrics['token_accuracy']:.3f} | "
                        f"tok/s {tokens_per_s:,.0f} | "
                        f"MFLOPS {mflops:.0f} | "
                        f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                        f"max|st|={_max_st} max|sn|={_max_sn}"
                    )

            # ---- mid-epoch validation (fast: limited batches) ----
            if args.val_interval > 0 and global_step % args.val_interval == 0:
                _mb = args.val_max_batches if args.val_max_batches > 0 else None
                mid_val = run_validation(max_batches=_mb)
                if args.print_interval > 0:
                    print(
                        f"  val @ step {global_step:7d} | "
                        f"val loss {mid_val['val_loss']:.4f} | "
                        f"val acc {mid_val['val_acc']:.3f}"
                    )
                if wandb_run:
                    wandb_run.log({
                        "val/loss": mid_val["val_loss"],
                        "val/token_accuracy": mid_val["val_acc"],
                        "global_step": global_step,
                        "epoch": epoch,
                    })
                if log_file:
                    log_file.write(
                        f"{epoch},{global_step},,,,,,,"
                        f"{mid_val['val_loss']:.6f},{mid_val['val_acc']:.4f}\n"
                    )
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
                        bos_id=bos_id,
                        eos_id=eos_id,
                        boa_id=boa_id,
                        eoa_id=eoa_id,
                    )
                    if args.print_interval > 0:
                        print(f"  ✓ Best checkpoint (val_loss={best_val_loss:.4f}) → {args.checkpoint}")

        scheduler.step()

        # ---- epoch-end validation (full pass) ----
        val_metrics = run_validation(max_batches=None)

        # ---- epoch-end metrics ----
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_acc = epoch_acc / max(epoch_steps, 1)
        t_epoch = time.time() - t_epoch_start
        elapsed = time.time() - t_start
        tokens_per_s = token_count / elapsed if elapsed > 0 else 0.0
        flops_est = token_count * flops_per_token_total
        mflops = flops_est / (elapsed * 1e6) if elapsed > 0 else 0.0

        print(
            f"=== epoch {epoch:3d} done | "
            f"train loss {avg_loss:.4f} | acc {avg_acc:.3f} | "
            f"val loss {val_metrics['val_loss']:.4f} | acc {val_metrics['val_acc']:.3f} | "
            f"time {t_epoch:.0f}s | "
            f"tok/s {tokens_per_s:,.0f} | "
            f"MFLOPS {mflops:.0f} ==="
        )

        if wandb_run:
            wandb_run.log({
                "epoch/train_loss": avg_loss,
                "epoch/train_token_accuracy": avg_acc,
                "epoch/val_loss": val_metrics["val_loss"],
                "epoch/val_token_accuracy": val_metrics["val_acc"],
                "epoch/time_s": t_epoch,
                "epoch/tokens_per_s": tokens_per_s,
                "epoch/mflops": mflops,
                "epoch": epoch,
            })

        # ---- checkpoint ----
        if args.checkpoint:
            # Always save latest
            latest_path = os.path.join(
                os.path.dirname(args.checkpoint),
                f"latest_checkpoint.pt",
            )
            model.save_checkpoint(
                latest_path,
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
            # Save best if val loss improved
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
                    bos_id=bos_id,
                    eos_id=eos_id,
                    boa_id=boa_id,
                    eoa_id=eoa_id,
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
        description="Train a WorldModelTransformer on next-state prediction."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--formats", type=str, nargs="+", required=True)
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "conservative.yaml"))
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to WorldModel tokenizer JSON (must contain <bos>/<eos>/<boa>/<eoa>).")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save best-val-loss checkpoint (overwrites). "
                             "Also saves latest_checkpoint.pt alongside it every epoch. "
                             "If absent, no checkpointing.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Fraction of shards to hold out for validation (default: 0.1).")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Run validation every N training steps (0 = only at epoch end).")
    parser.add_argument("--val_max_batches", type=int, default=100,
                        help="Limit mid-epoch validation to this many batches (default: 200). "
                             "Set to 0 or a very large number to do a full pass every time. "
                             "Epoch-end validation always does a full pass.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shard partition (default: 42).")
    # Logging
    parser.add_argument("--log", action="store_true",
                        help="Write per-step metrics to metrics.csv in save_dir.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log every N training steps (CSV + wandb).")
    parser.add_argument("--print_interval", type=int, default=1000,
                        help="Print to console every N steps (0 = only epoch summaries).")
    # Diagnostics
    parser.add_argument("--diag_loss_threshold", type=float, default=0.8,
                        help="When batch loss exceeds this, print detokenized states "
                             "for the worst examples (e.g. 1.5). None = disabled.")
    parser.add_argument("--diag_top_k", type=int, default=5,
                        help="Number of worst examples to show per spike (default: 2).")
    parser.add_argument("--diag_rate_limit", type=int, default=10,
                        help="Only print diagnostics every N spikes (default: 10).")
    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (default: metamon-<format>).")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name (default: save_dir basename).")
    args = parser.parse_args()

    train(args)
