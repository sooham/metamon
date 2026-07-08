"""Train the simple p1-only V/M/C world model."""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    import wandb
    _wandb_available = True
except ImportError:  # pragma: no cover - depends on optional install
    wandb = None
    _wandb_available = False

from metamon.jepa.model import format_tensor_debug
from metamon.jepa.train_paired import (
    PairedJEPADataset,
    _auto_detect_device,
    _batch_to_device,
    _load_rollout_len,
    collate_paired_fn,
)
from metamon.simple_world_model.checkpointing import (
    load_matching_weights,
    resume_training_state,
    save_simple_world_model_checkpoint,
)
from metamon.simple_world_model.model import (
    SimpleWorldModel,
    compute_simple_world_model_losses,
)
from metamon.tokenizer import PokemonTokenizer


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


def _build_dataloaders(
    data_root: str,
    formats: list[str],
    structural_ids: dict[str, int],
    max_history_blocks: int,
    batch_size: int,
    pad_id: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None, list[str], list[str]]:
    train_shards = PairedJEPADataset.discover(data_root, formats, "train")
    val_shards = PairedJEPADataset.discover(data_root, formats, "val", required=False)
    train_dataset = PairedJEPADataset(
        train_shards,
        structural_ids,
        shuffle_shards=True,
        max_history_blocks=max_history_blocks,
        include_simple_world_model_fields=True,
    )
    val_dataset = PairedJEPADataset(
        val_shards,
        structural_ids,
        shuffle_shards=False,
        max_history_blocks=max_history_blocks,
        include_simple_world_model_fields=True,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size,
        pad_id,
        num_workers,
        prefetch_factor,
        device.type == "cuda",
    )
    val_loader = None
    if val_shards:
        val_loader = _make_loader(
            val_dataset,
            batch_size,
            pad_id,
            max(0, num_workers // 2),
            prefetch_factor,
            device.type == "cuda",
        )
    return train_loader, val_loader, train_shards, val_shards


def _load_sequence_stats(data_root: str) -> dict[str, Any]:
    path = os.path.join(data_root, "sequence_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _max_action_len_from_shards(shard_paths: list[str]) -> int:
    max_len = 1
    for path in shard_paths[:4]:
        with np.load(path) as data:
            if "p1_action_lengths" in data and len(data["p1_action_lengths"]):
                max_len = max(max_len, int(np.max(data["p1_action_lengths"])))
            if "p1_legal_actions" in data:
                max_len = max(max_len, int(data["p1_legal_actions"].shape[-1]))
    return max_len


def _loss_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    loss_cfg = dict(cfg.get("loss", {}))
    for key in (
        "lambda_recon",
        "beta_kl",
        "lambda_mdn",
        "lambda_terminal",
        "lambda_controller_bc",
    ):
        value = getattr(args, key)
        if value is not None:
            loss_cfg[key] = float(value)
    return {
        "lambda_recon": float(loss_cfg.get("lambda_recon", 1.0)),
        "beta_kl": float(loss_cfg.get("beta_kl", 0.01)),
        "lambda_mdn": float(loss_cfg.get("lambda_mdn", 1.0)),
        "lambda_terminal": float(loss_cfg.get("lambda_terminal", 0.25)),
        "lambda_controller_bc": float(loss_cfg.get("lambda_controller_bc", 1.0)),
    }


def _gather_valid_blocks(tokens: torch.Tensor, valid: torch.Tensor, pad_id: int) -> list[torch.Tensor]:
    """Extract the non-pad 1-D token vectors for the ``True`` blocks.

    ``tokens`` is ``(max_blocks, max_tokens)`` and ``valid`` is ``(max_blocks,)``.
    Returns a list (one per valid block) of ``[non-pad tokens]`` long tensors.
    """
    blocks: list[torch.Tensor] = []
    for b in range(tokens.shape[0]):
        if bool(valid[b]):
            row = tokens[b]
            blocks.append(row[row != pad_id])
    return blocks


def _interleave_history(
    history_states: list[torch.Tensor],
    player_actions: list[torch.Tensor],
    opponent_actions: list[torch.Tensor],
    current_state: torch.Tensor,
) -> torch.Tensor:
    """Build the full seen-history token sequence for the POV player.

    Layout (matching ``AGENTS.md``): block 0 of ``history_states`` is the team
    header, then per step ``[state_i, p_action_i, o_action_i]``, and finally
    the current state ``state_T`` as the last block::

        team_header, state_0, p_act_0, o_act_0, state_1, ..., state_{T-1},
        p_act_{T-1}, o_act_{T-1}, state_T
    """
    seq: list[torch.Tensor] = []
    if history_states:
        seq.append(history_states[0])  # team header (always retained)
    n_steps = min(max(len(history_states) - 1, 0), len(player_actions), len(opponent_actions))
    for i in range(n_steps):
        seq.append(history_states[i + 1])
        seq.append(player_actions[i])
        seq.append(opponent_actions[i])
    seq.append(current_state)
    return torch.cat(seq, dim=0)


def _pad_to_batch(seqs: list[torch.Tensor], pad_id: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    max_len = max(int(t.shape[-1]) for t in seqs) if seqs else 1
    max_len = max(max_len, 1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=dtype, device=device)
    for i, t in enumerate(seqs):
        n = int(t.shape[-1])
        out[i, :n] = t
    return out


def _prepare_batch(batch: dict[str, object], pad_id: int) -> dict[str, torch.Tensor]:
    """Build the interleaved seen-history token sequences for V.

    Consumes the collated paired-shard fields and produces, per battle:

    * ``history_tokens``      — ``[team_header, prior states, prior p/o actions,
      ..., current state_T]`` (V encodes this into ``z_T``).
    * ``next_history_tokens`` — the above extended with the current player
      action, opponent action, and next state ``state_{T+1}`` (V encodes it
      into the MDN target ``z_{T+1}``).
    * ``state_tokens``        — the current state ``state_T`` (recon target).
    * ``action_tokens``       — the current POV action.
    """
    rollout_axis = 0
    hist = batch["p1_history_T"][:, rollout_axis]
    hist_valid = batch["p1_history_T_valid"][:, rollout_axis]
    player_hist = batch["p1_player_hist_T"][:, rollout_axis]
    player_valid = batch["p1_player_hist_T_valid"][:, rollout_axis]
    opponent_hist = batch["p1_opponent_hist_T"][:, rollout_axis]
    opponent_valid = batch["p1_opponent_hist_T_valid"][:, rollout_axis]
    target_state = batch["p1_target_state_T"][:, rollout_axis].long()
    next_state = batch["p1_next_state_T1"][:, rollout_axis].long()
    cur_action = batch["p1_action"][:, rollout_axis].long()
    # Opponent's current action for this transition, from p1's perspective.
    opp_cur_key = "actual_p2_action_from_p1_perspective"
    opp_cur_action = batch[opp_cur_key][:, rollout_axis].long() if opp_cur_key in batch else cur_action

    history_seqs: list[torch.Tensor] = []
    next_history_seqs: list[torch.Tensor] = []
    device = hist.device
    for b in range(hist.shape[0]):
        h_states = _gather_valid_blocks(hist[b], hist_valid[b], pad_id)
        p_acts = _gather_valid_blocks(player_hist[b], player_valid[b], pad_id)
        o_acts = _gather_valid_blocks(opponent_hist[b], opponent_valid[b], pad_id)
        cur_st = target_state[b][target_state[b] != pad_id]
        if cur_st.numel() == 0:
            cur_st = target_state[b][:1].clone()
        history_tokens = _interleave_history(h_states, p_acts, o_acts, cur_st)
        # Extend to the next history: history_T || p_act_T || o_act_T || state_{T+1}
        p_T = cur_action[b][cur_action[b] != pad_id]
        o_T = opp_cur_action[b][opp_cur_action[b] != pad_id]
        ns = next_state[b][next_state[b] != pad_id]
        ext = [p_T if p_T.numel() else cur_action[b][:1].clone(),
               o_T if o_T.numel() else cur_action[b][:1].clone(),
               ns if ns.numel() else next_state[b][:1].clone()]
        next_history_tokens = torch.cat([history_tokens, *ext], dim=0)
        history_seqs.append(history_tokens)
        next_history_seqs.append(next_history_tokens)

    history_tokens = _pad_to_batch(history_seqs, pad_id, torch.long, device)
    next_history_tokens = _pad_to_batch(next_history_seqs, pad_id, torch.long, device)

    out: dict[str, torch.Tensor] = {
        "history_tokens": history_tokens,
        "next_history_tokens": next_history_tokens,
        "state_tokens": target_state,
        "action_tokens": cur_action,
        "terminal_class": batch["p1_next_terminal_class"][:, 0],
    }
    if "p1_legal_actions" in batch:
        out.update({
            "legal_action_tokens": batch["p1_legal_actions"][:, 0].long(),
            "legal_action_mask": batch["p1_legal_action_mask"][:, 0],
            "chosen_legal_action_idx": batch["p1_chosen_legal_action_idx"][:, 0],
        })
    return out


def _count_processed_tokens(
    prepared: dict[str, torch.Tensor],
    components: str,
    pad_id: int,
) -> int:
    """Count non-pad tokens the model actually consumes, weighted by pass count.

    V now encodes the full seen history as one sequence:

    * vm: ``history_tokens`` (V encode, grad) + ``history_tokens`` again
      (V decode reconstructing the FULL history, grad) + ``next_history_tokens``
      (V encode, no grad, for the MDN target) + ``action_tokens``
      (action encoder).
    * c/all: additionally ``history_tokens`` (V encode, no grad) +
      ``legal_action_tokens`` (action encoder). In ``vm`` the legal-action
      fields are prepared but unused, so they are not counted.
    """
    def _nonpad(t: torch.Tensor) -> int:
        return int((t != pad_id).sum().item())

    n = 0
    if components in {"vm", "all"}:
        n += _nonpad(prepared["history_tokens"])        # V encode (full history, grad)
        n += _nonpad(prepared["history_tokens"])        # V decode (reconstruct full history, grad)
        n += _nonpad(prepared["next_history_tokens"])   # V encode (no grad, MDN target)
        n += _nonpad(prepared["action_tokens"])         # action encoder
    if components in {"c", "all"} and "legal_action_tokens" in prepared:
        n += _nonpad(prepared["history_tokens"])          # V encode (controller)
        n += _nonpad(prepared["legal_action_tokens"])    # action encoder (controller)
    return n


def _forward(
    model: SimpleWorldModel,
    prepared: dict[str, torch.Tensor],
    components: str,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    if components in {"vm", "all"}:
        outputs.update(model.forward_vm(
            prepared["history_tokens"],
            prepared["next_history_tokens"],
            prepared["action_tokens"],
        ))
    if components in {"c", "all"}:
        outputs.update(model.forward_controller(
            prepared["history_tokens"],
            prepared["legal_action_tokens"],
        ))
    return outputs


def _debug_dump(title: str, tensors: dict[str, object], pad_id: int, max_values: int) -> None:
    print(f"\n[simple-world-model tensor debug] {title}", flush=True)
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            print("  " + format_tensor_debug(key, value, pad_id=pad_id, max_values=max_values), flush=True)
        else:
            print(f"  {key}: {value}", flush=True)


def _freeze_for_controller(model: SimpleWorldModel) -> None:
    for module in (model.v, model.m, model.action_encoder):
        for param in module.parameters():
            param.requires_grad_(False)


def _save_checkpoint(
    path: str,
    *,
    model: SimpleWorldModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    model_cfg: dict[str, Any],
    vocab_size: int,
    pad_id: int,
    tokenizer: PokemonTokenizer,
    components: str,
    max_history_blocks: int,
    best_val_loss: float,
    best_val_epoch: int | None,
    best_val_global_step: int | None,
    best_val_metrics: dict[str, float] | None,
    last_val_metrics: dict[str, float] | None,
) -> None:
    save_simple_world_model_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        model_config=model_cfg,
        vocab_size=vocab_size,
        pad_id=pad_id,
        tokenizer=tokenizer,
        components=components,
        max_history_blocks=max_history_blocks,
        best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
        best_val_epoch=best_val_epoch,
        best_val_global_step=best_val_global_step,
        best_val_metrics=best_val_metrics,
        last_val_metrics=last_val_metrics,
    )


def train(args: argparse.Namespace) -> None:
    if args.components not in {"vm", "c", "all"}:
        raise ValueError("--components must be one of: vm, c, all")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")

    device = _auto_detect_device()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = dict(cfg["model"])
    loss_cfg = _loss_cfg(cfg, args)

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    structural_ids = {"unknown": tokenizer["unknown"]}

    train_shards_probe = PairedJEPADataset.discover(args.data_root, args.formats, "train")
    rollout_len = _load_rollout_len(args.data_root, train_shards_probe)
    if rollout_len != 1:
        raise ValueError(
            f"simple-world-model currently requires rollout_len=1, got {rollout_len}. "
            "Regenerate data with make wm-dataset WM_ROLLOUT_LEN=1."
        )

    seq_stats = _load_sequence_stats(args.data_root)
    state_block_max = int(seq_stats.get("state_block_len", {}).get("max", model_cfg.get("v", {}).get("max_seq_len", 1024)))
    configured_v_max = int(model_cfg.get("v", {}).get("max_seq_len", 0) or 0)
    action_max = _max_action_len_from_shards(train_shards_probe)
    # V now encodes the full interleaved seen history as one sequence. Bound
    # the sequence length by the block counts implied by --max_history_blocks:
    #   prior states = N -> blocks = 1 team header + N prior states + 1 current
    #   state (+ for next history: +1 player act +1 opponent act +1 next state).
    nh = args.max_history_blocks
    if nh <= 0:
        nh = int(seq_stats.get("temporal_sequence_len", {}).get("max", 256)) // 3
    nh = max(nh, 1)
    v_state_blocks = nh + 3              # header + N prior + current (+ next)
    v_action_blocks = 2 * nh + 2         # N prior player + N prior opp + cur player + cur opp
    v_bound = v_state_blocks * state_block_max + v_action_blocks * action_max + 8
    model_cfg.setdefault("v", {})["max_seq_len"] = max(configured_v_max, v_bound)
    model_cfg.setdefault("action_encoder", {})["max_seq_len"] = max(
        int(model_cfg.get("action_encoder", {}).get("max_seq_len", 32)),
        action_max,
    )

    train_loader, val_loader, train_shards, val_shards = _build_dataloaders(
        args.data_root,
        args.formats,
        structural_ids,
        args.max_history_blocks,
        args.batch_size,
        pad_id,
        args.num_workers,
        args.prefetch_factor,
        device,
    )

    model = SimpleWorldModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        latent_dim=int(model_cfg.get("latent_dim", 1024)),
        v_cfg=model_cfg.get("v", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        m_cfg=model_cfg.get("m", {}),
        controller_cfg=model_cfg.get("controller", {}),
    ).to(device)

    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.resume is None and args.checkpoint and os.path.exists(args.checkpoint):
        load_matching_weights(model, args.checkpoint, device)
    if args.components == "c" and not args.finetune_vm_for_c:
        _freeze_for_controller(model)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters selected")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_val_epoch: int | None = None
    best_val_global_step: int | None = None
    best_val_metrics: dict[str, float] | None = None
    if args.resume:
        ckpt = resume_training_state(
            model=model,
            optimizer=optimizer,
            checkpoint_path=args.resume,
            device=device,
            model_config=model_cfg,
            vocab_size=vocab_size,
            pad_id=pad_id,
        )
        if ckpt.get("components") != args.components:
            raise ValueError(
                f"Cannot --resume checkpoint trained with components={ckpt.get('components')!r} "
                f"as components={args.components!r}"
            )
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss") or float("inf"))
        best_val_epoch = ckpt.get("best_val_epoch")
        best_val_global_step = ckpt.get("best_val_global_step")
        best_val_metrics = ckpt.get("best_val_metrics")
    elif args.components == "c" and not args.checkpoint:
        raise ValueError("--components c requires --checkpoint with trained V/M weights unless --resume is used")

    if args.compile and device.type == "cuda":
        for name in ("v", "action_encoder", "m"):
            try:
                setattr(model, name, torch.compile(getattr(model, name), dynamic=True))
                print(f"torch.compile enabled on: {name}")
            except Exception as exc:
                print(f"  [{name}] torch.compile failed: {exc}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_transitions = PairedJEPADataset.count_transitions(train_shards)
    val_transitions = PairedJEPADataset.count_transitions(val_shards) if val_shards else 0
    print(
        f"Params: {n_params:,} total / {n_trainable:,} trainable | "
        f"shards {len(train_shards)} train + {len(val_shards)} val | "
        f"transitions {train_transitions:,} train / {val_transitions:,} val"
    )
    print(
        f"components={args.components} batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"lr={args.lr:g} max_history_blocks={args.max_history_blocks} "
        f"state_block_max={state_block_max} v_max_seq_len={model_cfg['v']['max_seq_len']} "
        f"action_max_seq_len={model_cfg['action_encoder']['max_seq_len']}"
    )

    wandb_run = None
    if args.wandb and _wandb_available:
        wandb_run = wandb.init(
            project=args.wandb_project or "metamon-simple-world-model-" + "-".join(args.formats),
            name=args.wandb_name,
            config={
                **model_cfg,
                **loss_cfg,
                "components": args.components,
                "vocab_size": vocab_size,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "grad_clip": args.grad_clip,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "compile": args.compile,
                "config_path": args.config,
                "val_interval": args.val_interval,
                "val_max_batches": args.val_max_batches,
                "print_interval": args.print_interval,
                "log_interval": args.log_interval,
                "n_params": n_params,
                "n_trainable_params": n_trainable,
                "n_train_transitions": train_transitions,
                "n_val_transitions": val_transitions,
                "checkpoint": args.checkpoint,
                "resume": args.resume,
            },
        )
    elif args.wandb and not _wandb_available:
        print("WARNING: --wandb enabled but wandb is not installed")

    def loss_for_batch(batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor], int]:
        prepared = _prepare_batch(batch, pad_id)
        outputs = _forward(model, prepared, args.components)
        loss, metrics = compute_simple_world_model_losses(
            outputs,
            prepared,
            pad_id=pad_id,
            components=args.components,
            **loss_cfg,
        )
        batch_tokens = _count_processed_tokens(prepared, args.components, pad_id)
        return loss, metrics, {**prepared, **outputs}, batch_tokens

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
            _, metrics, _, _ = loss_for_batch(batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
        model.train()
        return {f"val_{key}": value / max(steps, 1) for key, value in totals.items()}

    def update_best_val(epoch: int, step: int, val_metrics: dict[str, float]) -> bool:
        nonlocal best_val_loss, best_val_epoch, best_val_global_step, best_val_metrics
        val_loss = val_metrics.get("val_loss")
        if val_loss is None or val_loss >= best_val_loss:
            return False
        best_val_loss = val_loss
        best_val_epoch = epoch
        best_val_global_step = step
        best_val_metrics = dict(val_metrics)
        return True

    optimizer.zero_grad(set_to_none=True)
    done = False
    t_last_print = time.time()
    t_last_wandb = time.time()
    tokens_since_print = 0
    tokens_since_wandb = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_totals: dict[str, float] = {}
        epoch_counts: dict[str, int] = {}
        epoch_steps = 0
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            debug_this_step = args.debug_tensors and global_step < args.debug_tensor_steps
            loss, metrics, debug_tensors, batch_tokens = loss_for_batch(batch)
            if debug_this_step:
                _debug_dump(
                    f"train step {global_step + 1}",
                    debug_tensors,
                    pad_id=pad_id,
                    max_values=args.debug_tensor_values,
                )
            (loss / args.grad_accum_steps).backward()
            grad_norm_value = 0.0
            grad_norm_logged = False
            if (global_step + 1) % args.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                grad_norm_value = float(grad_norm.detach().float().cpu().item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_norm_logged = True
            # Only record grad_norm on actual optimizer steps so per-step wandb
            # logging and epoch averaging aren't diluted by zeros (which would
            # otherwise happen on non-accumulation steps).
            if grad_norm_logged:
                metrics["grad_norm"] = grad_norm_value
            elif "grad_norm" in metrics:
                del metrics["grad_norm"]
            global_step += 1
            epoch_steps += 1
            for key, value in metrics.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value)
                epoch_counts[key] = epoch_counts.get(key, 0) + 1

            tokens_since_print += batch_tokens
            tokens_since_wandb += batch_tokens

            log_step = args.log_interval if args.log_interval > 0 else args.print_interval
            if wandb_run and log_step > 0 and global_step % log_step == 0:
                now = time.time()
                tok_per_sec = tokens_since_wandb / max(now - t_last_wandb, 1e-6)
                t_last_wandb = now
                tokens_since_wandb = 0
                payload = {
                    **{f"train/{key}": value for key, value in metrics.items()},
                    "train/tok_per_sec": tok_per_sec,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "samples_seen": args.batch_size * global_step,
                }
                if device.type == "cuda":
                    payload.update({
                        "mem/cuda_reserved_gib": torch.cuda.memory_reserved() / 1024 ** 3,
                        "mem/cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                    })
                wandb_run.log(payload)

            if args.print_interval > 0 and global_step % args.print_interval == 0:
                now = time.time()
                tok_per_sec = tokens_since_print / max(now - t_last_print, 1e-6)
                t_last_print = now
                tokens_since_print = 0
                z_diag = (
                    f"z {metrics.get('z_mu_norm', 0.0):.2f}/"
                    f"{metrics.get('z_mu_std_per_dim', 0.0):.3f}/"
                    f"{metrics.get('z_mu_pairwise_distance', 0.0):.2f}"
                )
                print(
                    f"  epoch {epoch:3d} | step {global_step:6d} | tok/s {tok_per_sec:,.0f} | "
                    f"loss {metrics.get('loss', 0.0):.4f} | "
                    f"recon {metrics.get('recon_ce', 0.0):.4f} | "
                    f"kl {metrics.get('kl', 0.0):.2f} | "
                    f"mdn {metrics.get('mdn_nll', 0.0):.4f} | "
                    f"term_acc {metrics.get('terminal_acc', 0.0):.3f} | "
                    f"ctrl_acc {metrics.get('controller_acc', 0.0):.3f} | {z_diag}"
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

            if args.val_interval > 0 and global_step % args.val_interval == 0:
                val_metrics = validate(args.val_max_batches)
                if val_metrics:
                    print(
                        f"  val @ step {global_step:6d} | "
                        f"loss {val_metrics.get('val_loss', 0.0):.4f} | "
                        f"recon {val_metrics.get('val_recon_ce', 0.0):.4f} | "
                        f"mdn {val_metrics.get('val_mdn_nll', 0.0):.4f} | "
                        f"term_acc {val_metrics.get('val_terminal_acc', 0.0):.3f} | "
                        f"ctrl_acc {val_metrics.get('val_controller_acc', 0.0):.3f} | "
                        f"z {val_metrics.get('val_z_mu_norm', 0.0):.2f}/"
                        f"{val_metrics.get('val_z_mu_std_per_dim', 0.0):.3f}/"
                        f"{val_metrics.get('val_z_mu_pairwise_distance', 0.0):.2f}"
                    )
                    if wandb_run:
                        wandb_run.log({
                            **{f"val/{key.removeprefix('val_')}": value for key, value in val_metrics.items()},
                            "epoch": epoch,
                            "global_step": global_step,
                        })
                    improved = update_best_val(epoch, global_step, val_metrics)
                    latest_path = save_dir / "simple_world_model_latest.pt"
                    _save_checkpoint(
                        str(latest_path),
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        model_cfg=model_cfg,
                        vocab_size=vocab_size,
                        pad_id=pad_id,
                        tokenizer=tokenizer,
                        components=args.components,
                        max_history_blocks=args.max_history_blocks,
                        best_val_loss=best_val_loss,
                        best_val_epoch=best_val_epoch,
                        best_val_global_step=best_val_global_step,
                        best_val_metrics=best_val_metrics,
                        last_val_metrics=val_metrics,
                    )
                    if improved and args.checkpoint:
                        _save_checkpoint(
                            args.checkpoint,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            model_cfg=model_cfg,
                            vocab_size=vocab_size,
                            pad_id=pad_id,
                            tokenizer=tokenizer,
                            components=args.components,
                            max_history_blocks=args.max_history_blocks,
                            best_val_loss=best_val_loss,
                            best_val_epoch=best_val_epoch,
                            best_val_global_step=best_val_global_step,
                            best_val_metrics=best_val_metrics,
                            last_val_metrics=val_metrics,
                        )
                        print(f"  best checkpoint -> {args.checkpoint}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break

        if epoch_steps > 0 and global_step % args.grad_accum_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_totals["grad_norm"] = epoch_totals.get("grad_norm", 0.0) + float(grad_norm.detach().float().cpu().item())
            epoch_counts.setdefault("grad_norm", 0)
            epoch_counts["grad_norm"] += 1

        # Average each metric over the number of steps that actually produced it
        # (grad_norm is only recorded on optimizer steps under grad accumulation).
        avg = {key: value / max(epoch_counts.get(key, epoch_steps), 1) for key, value in epoch_totals.items()}
        val_metrics = validate(args.val_max_batches)
        if val_metrics:
            update_best_val(epoch, global_step, val_metrics)
        print(
            f"=== epoch {epoch:3d} done | train loss {avg.get('loss', 0.0):.4f} | "
            f"recon {avg.get('recon_ce', 0.0):.4f} | mdn {avg.get('mdn_nll', 0.0):.4f} | "
            f"term_acc {avg.get('terminal_acc', 0.0):.3f} | ctrl_acc {avg.get('controller_acc', 0.0):.3f} | "
            f"val loss {val_metrics.get('val_loss', 0.0) if val_metrics else 0.0:.4f} ==="
        )
        if wandb_run:
            wandb_run.log({
                **{f"epoch/train_{key}": value for key, value in avg.items()},
                **{f"epoch/{key}": value for key, value in val_metrics.items()},
                "epoch": epoch,
                "global_step": global_step,
            })

        latest_path = save_dir / "simple_world_model_latest.pt"
        _save_checkpoint(
            str(latest_path),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            model_cfg=model_cfg,
            vocab_size=vocab_size,
            pad_id=pad_id,
            tokenizer=tokenizer,
            components=args.components,
            max_history_blocks=args.max_history_blocks,
            best_val_loss=best_val_loss,
            best_val_epoch=best_val_epoch,
            best_val_global_step=best_val_global_step,
            best_val_metrics=best_val_metrics,
            last_val_metrics=val_metrics if val_metrics else None,
        )
        if args.checkpoint and (not val_metrics or val_metrics.get("val_loss") == best_val_loss):
            _save_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                model_cfg=model_cfg,
                vocab_size=vocab_size,
                pad_id=pad_id,
                tokenizer=tokenizer,
                components=args.components,
                max_history_blocks=args.max_history_blocks,
                best_val_loss=best_val_loss,
                best_val_epoch=best_val_epoch,
                best_val_global_step=best_val_global_step,
                best_val_metrics=best_val_metrics,
                last_val_metrics=val_metrics if val_metrics else None,
            )
        if done:
            break

    if wandb_run:
        wandb_run.finish()
    print(f"Training complete. Checkpoints: {save_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train simple-world-model.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--components", choices=["vm", "c", "all"], default="vm")
    parser.add_argument("--batch_size", type=int, default=32)
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
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--lambda_recon", type=float, default=None)
    parser.add_argument("--beta_kl", type=float, default=None)
    parser.add_argument("--lambda_mdn", type=float, default=None)
    parser.add_argument("--lambda_terminal", type=float, default=None)
    parser.add_argument("--lambda_controller_bc", type=float, default=None)
    parser.add_argument("--debug_tensors", action="store_true")
    parser.add_argument("--debug_tensor_steps", type=int, default=1)
    parser.add_argument("--debug_tensor_values", type=int, default=16)
    parser.add_argument("--debug_tensor_samples", type=int, default=2)
    parser.add_argument("--encoder_chunk_tokens", type=int, default=65536,
                        help="Accepted for JEPA CLI parity; simple-world-model v1 encodes state batches directly.")
    parser.add_argument("--max_history_blocks", type=int, default=64,
                        help="Number of prior state blocks V sees (windowed by the dataset, matching the JEPA trainer). 0 = unlimited/full battle history. Default 64.")
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--finetune_vm_for_c", action="store_true",
                        help="When training --components c, also allow gradients through V/M/action encoder.")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
