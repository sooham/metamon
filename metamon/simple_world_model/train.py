"""Staged trainer for the simple V/M/C world model.

The entry point deliberately has four stages instead of the old combined
``--components`` switch:

``v`` -> current-state posterior, ``cache`` -> frozen posterior sidecars,
``m`` -> causal latent dynamics, ``c`` -> legal-action behavior cloning.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import itertools
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:  # Keep local training usable if an optional environment lacks wandb.
    import wandb as _wandb
except ImportError:  # pragma: no cover - wandb is a declared dependency.
    _wandb = None

from metamon.jepa.model import sigreg
from metamon.simple_world_model.action_vocab import ActionVocabulary
from metamon.simple_world_model.cache_latents import build_cache
from metamon.simple_world_model.checkpointing import (
    load_stage_checkpoint,
    load_stage_weights,
    save_simple_world_model_checkpoint,
)
from metamon.simple_world_model.data import (
    BalancedFormatBatchSampler,
    CompactFormatBatchSampler,
    LatentTransitionDataset,
    VStateDataset,
    assert_matching_cache,
    collate_latent,
    collate_v,
    dataset_manifest_hash,
    discover_source_shards,
    format_id_to_name,
    load_cache_manifest,
    move_batch_to_device,
)
from metamon.simple_world_model.model import (
    NUM_OUTCOME_CLASSES,
    SimpleWorldModel,
    StateVAE,
    aggregate_posterior_sigreg,
    c_losses,
    mdn_nll,
    m_losses,
    vae_losses,
)
from metamon.tokenizer import PokemonTokenizer


DEFAULT_UPDATES = {"v": 200_000, "m": 100_000, "c": 50_000}


def _warmup_cosine_lr(
    update: int,
    *,
    peak_lr: float,
    min_lr: float,
    warmup_updates: int,
    schedule_updates: int,
) -> float:
    """Learning rate for a one-indexed optimizer update.

    V warms linearly to the peak LR and then follows a cosine to the minimum.
    Updates after the configured schedule hold the minimum, which also makes
    the schedule deterministic across checkpoint resumes without serializing
    scheduler state.
    """
    if peak_lr <= 0.0:
        raise ValueError("peak learning rate must be positive")
    if not 0.0 <= min_lr <= peak_lr:
        raise ValueError("minimum learning rate must be between zero and the peak")
    if warmup_updates < 0:
        raise ValueError("LR warmup updates must be non-negative")
    if schedule_updates <= warmup_updates:
        raise ValueError("LR schedule updates must be greater than LR warmup updates")
    step = max(int(update), 1)
    if warmup_updates and step <= warmup_updates:
        return float(peak_lr) * step / warmup_updates
    progress = min(
        max((step - warmup_updates) / (schedule_updates - warmup_updates), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(peak_lr) - float(min_lr)) * cosine


def _mask_non_structural_tokens(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    structural_token_lookup: torch.Tensor,
    mask_token_id: int,
    probability: float,
) -> tuple[torch.Tensor, int, int]:
    """Replace a fraction of valid non-structural encoder tokens with UNK."""
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("encoder token mask probability must be between zero and one")
    valid = valid_mask.bool()
    structural = structural_token_lookup.to(tokens.device)[tokens.long()]
    eligible = valid & ~structural
    eligible_count = int(eligible.sum().item())
    if probability <= 0.0 or eligible_count == 0:
        return tokens, 0, eligible_count
    selected = eligible & torch.rand(tokens.shape, device=tokens.device).lt(float(probability))
    selected_count = int(selected.sum().item())
    return tokens.masked_fill(selected, int(mask_token_id)), selected_count, eligible_count


def _resolve_updates_budget(args: argparse.Namespace, stage: str, *, start_update: int = 0) -> int:
    additional = int(getattr(args, "additional_updates", 0) or 0)
    absolute = int(args.max_updates or args.max_steps or 0)
    if additional < 0:
        raise ValueError("--additional_updates must be non-negative")
    if additional and absolute:
        raise ValueError("Use either --additional_updates or --max_updates, not both")
    if additional:
        return int(start_update) + additional
    return absolute or DEFAULT_UPDATES[stage]


class _WandbLogger:
    """Failure-tolerant W&B run wrapper for the staged trainers."""

    def __init__(self, run: Any):
        self.run = run
        self._disabled_after_error = False

    def log(self, payload: Mapping[str, Any], *, step: int) -> None:
        if self._disabled_after_error:
            return
        try:
            self.run.log(dict(payload), step=int(step))
        except Exception as exc:  # Network/auth failures must not stop a long run.
            self._disabled_after_error = True
            print(f"WARNING: W&B logging failed; continuing local-only ({type(exc).__name__}: {exc})", flush=True)

    def finish(self) -> None:
        try:
            self.run.finish()
        except Exception as exc:  # pragma: no cover - defensive cleanup only.
            print(f"WARNING: W&B finish failed ({type(exc).__name__}: {exc})", flush=True)


def _numeric_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    """Namespace scalar metrics without pushing batch tensors to W&B."""
    return {
        f"{prefix}/{key}": float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def _start_wandb(
    args: argparse.Namespace,
    *,
    stage: str,
    model: SimpleWorldModel,
    model_config: Mapping[str, Any],
    source_hash: str,
    cache_hash: str | None,
    device: torch.device,
    start_update: int = 0,
) -> _WandbLogger | None:
    """Create one stage-scoped W&B run, or retain local logging."""
    if not getattr(args, "wandb", False):
        return None
    if _wandb is None:
        print("WARNING: W&B requested but the wandb package is unavailable; continuing local-only.", flush=True)
        return None
    project = getattr(args, "wandb_project", None) or "metamon-simple-world-model"
    config = {
        "stage": stage,
        "formats": list(args.formats),
        "model_config": dict(model_config),
        "dataset_manifest_hash": source_hash,
        "latent_cache_manifest_hash": cache_hash,
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "start_update": int(start_update),
        "additional_updates": int(getattr(args, "additional_updates", 0) or 0),
        "max_updates": _resolve_updates_budget(args, stage, start_update=start_update),
        "val_interval": int(args.val_interval),
        "val_samples": int(args.val_samples),
        "val_mc_samples": int(args.val_mc_samples),
        "train_eval_samples": int(args.train_eval_samples),
        "train_metric_window": int(args.train_metric_window),
        "wandb_log_interval": int(args.wandb_log_interval),
        "balanced_formats": bool(args.balanced_formats),
        "v_battle_sampling_alpha": float(args.v_battle_sampling_alpha),
        "lr_warmup_updates": int(args.lr_warmup_updates),
        "min_lr": float(args.min_lr),
        "lr_schedule_updates": int(args.lr_schedule_updates),
        "early_stop_patience": int(args.early_stop_patience),
        "encoder_token_mask_prob": float(args.encoder_token_mask_prob),
        "mean_recon_weight": float(args.mean_recon_weight),
        "lambda_mu_sigreg": float(getattr(args, "lambda_mu_sigreg", 0.0)),
        "mu_sigreg_warmup_updates": int(getattr(args, "mu_sigreg_warmup_updates", 0)),
        "lambda_sampled_sigreg": float(getattr(args, "lambda_sampled_sigreg", 0.0)),
        "sampled_sigreg_warmup_updates": int(
            getattr(args, "sampled_sigreg_warmup_updates", 0)
        ),
        "lambda_aggregate_sigreg": float(getattr(args, "lambda_aggregate_sigreg", 0.0)),
        "aggregate_sigreg_warmup_updates": int(
            getattr(args, "aggregate_sigreg_warmup_updates", 0)
        ),
        "lambda_team_aggregate_sigreg": float(
            getattr(args, "lambda_team_aggregate_sigreg", 0.0)
        ),
        "team_aggregate_sigreg_warmup_updates": int(
            getattr(args, "team_aggregate_sigreg_warmup_updates", 0)
        ),
        "team_sigreg_batch_size": int(getattr(args, "team_sigreg_batch_size", 32)),
        "team_sigreg_battle_sampling_alpha": 0.0,
        "sigreg_num_slices": int(getattr(args, "sigreg_num_slices", 128)),
        "sigreg_num_points": int(getattr(args, "sigreg_num_points", 17)),
        "sigreg_domain": float(getattr(args, "sigreg_domain", 3.0)),
        "posterior_std_target": getattr(args, "posterior_std_target", None),
        "posterior_std_weight": float(getattr(args, "posterior_std_weight", 0.0)),
        "posterior_std_warmup_updates": int(
            getattr(args, "posterior_std_warmup_updates", 0)
        ),
        "kl_warmup_updates": int(args.kl_warmup_updates),
        "beta_kl": None if args.beta_kl is None else float(args.beta_kl),
        "free_bits": float(args.free_bits),
        "kl_capacity": float(args.kl_capacity),
        "kl_capacity_weight": float(args.kl_capacity_weight),
        "grad_clip_fraction_window": int(args.grad_clip_fraction_window),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "warm_start_checkpoint": (
            str(args.warm_start_checkpoint) if args.warm_start_checkpoint else None
        ),
        "max_context_transitions": int(args.max_context_transitions),
        "compile": bool(args.compile),
        "seed": int(args.seed),
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "config_path": str(args.config),
    }
    init_kwargs: dict[str, Any] = {
        "project": project,
        "job_type": stage,
        "tags": ["simple-world-model", stage, *map(str, args.formats)],
        "config": config,
    }
    if getattr(args, "wandb_name", None):
        init_kwargs["name"] = args.wandb_name
    try:
        run = _wandb.init(**init_kwargs)
        if hasattr(run, "define_metric"):
            run.define_metric("global_step")
            run.define_metric("train/*", step_metric="global_step")
            run.define_metric("val/*", step_metric="global_step")
            run.define_metric("checkpoint/*", step_metric="global_step")
    except Exception as exc:
        print(f"WARNING: W&B initialization failed; continuing local-only ({type(exc).__name__}: {exc})", flush=True)
        return None
    print(f"[{stage}] W&B enabled: project={project}", flush=True)
    return _WandbLogger(run)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_tokenizer(path: str) -> PokemonTokenizer:
    return PokemonTokenizer().load_tokens_from_disk(path)


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "model" not in config:
        raise ValueError(f"Config {path} does not have a top-level model mapping")
    return config


def _apply_loss_config(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Let YAML supply stage defaults while explicit CLI flags still win."""
    values = dict(config.get("loss", {}))
    defaults = {
        "beta_kl": 0.01,
        "lambda_opponent": 1.0,
        "lambda_mdn": 1.0,
        "lambda_done": 0.25,
        "lambda_value": 0.25,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, float(values.get(name, fallback)))


def _model_from_config(
    config: Mapping[str, Any],
    *,
    tokenizer: PokemonTokenizer,
    action_vocabulary: ActionVocabulary,
    max_context_transitions: int,
    device: torch.device,
) -> SimpleWorldModel:
    model_cfg = dict(config["model"])
    m_cfg = dict(model_cfg.get("m", {}))
    m_cfg["max_context_transitions"] = int(max_context_transitions)
    return SimpleWorldModel(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        action_vocab_size=len(action_vocabulary),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        v_cfg=model_cfg.get("v", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        m_cfg=m_cfg,
        controller_cfg=model_cfg.get("controller", {}),
    ).to(device)


def _full_model_config(config: Mapping[str, Any], max_context_transitions: int) -> dict[str, Any]:
    model_cfg = dict(config["model"])
    m_cfg = dict(model_cfg.get("m", {}))
    m_cfg["max_context_transitions"] = int(max_context_transitions)
    model_cfg["m"] = m_cfg
    return model_cfg


def _freeze(module: torch.nn.Module, frozen: bool = True) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(not frozen)


def _checkpoint_manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save(
    path: str | Path,
    *,
    model: SimpleWorldModel,
    optimizer: torch.optim.Optimizer | None,
    stage: str,
    update: int,
    config: Mapping[str, Any],
    tokenizer: PokemonTokenizer,
    action_vocabulary: ActionVocabulary,
    source_hash: str,
    cache_hash: str | None,
    max_context_transitions: int,
    metrics: Mapping[str, float] | None,
    training_config: Mapping[str, Any] | None = None,
) -> None:
    save_simple_world_model_checkpoint(
        str(path),
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=update,
        model_config=_full_model_config(config, max_context_transitions),
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
        stage=stage,
        action_vocabulary=action_vocabulary.to_state(),
        dataset_manifest_hash=source_hash,
        latent_cache_manifest_hash=cache_hash,
        max_context_transitions=max_context_transitions,
        best_val_loss=None if metrics is None else metrics.get("selection_score"),
        best_val_metrics=None if metrics is None else dict(metrics),
        training_config=None if training_config is None else dict(training_config),
    )


def _fixed_subset(dataset: Any, max_samples: int) -> Any:
    if max_samples > 0 and hasattr(dataset, "fixed_subset"):
        return dataset.fixed_subset(max_samples)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    # Deterministic evenly spaced refs; source train/val is already battle
    # disjoint, so this remains a fixed battle-disjoint validation set.
    step = len(dataset) / max_samples
    indices = [min(int(index * step), len(dataset) - 1) for index in range(max_samples)]
    return Subset(dataset, indices)


def _dataset_formats_and_lengths(dataset: Any) -> tuple[list[str], list[int]]:
    refs = getattr(dataset, "refs", None)
    if refs is not None:
        return [ref.fmt for ref in refs], [ref.length for ref in refs]
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base))
    refs = base.refs
    return [refs[index].fmt for index in indices], [refs[index].length for index in indices]


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    balanced_formats: bool,
    shuffle: bool,
    seed: int,
    collate: Callable[[Sequence[Any]], dict[str, Any]],
    num_workers: int,
    battle_sampling_alpha: float = 1.0,
) -> tuple[DataLoader, Any]:
    if hasattr(dataset, "draw_ref") and hasattr(dataset, "total_by_format"):
        sampler: Any = CompactFormatBatchSampler(
            dataset, batch_size=batch_size, balanced=balanced_formats, shuffle=shuffle, seed=seed,
            battle_sampling_alpha=battle_sampling_alpha,
        )
        return (
            DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            ),
            sampler,
        )
    formats, lengths = _dataset_formats_and_lengths(dataset)
    sampler = BalancedFormatBatchSampler(
        formats, lengths, batch_size=batch_size, balanced=balanced_formats, shuffle=shuffle, seed=seed
    )
    # Subset sampler indices are relative to the subset, which is exactly what
    # DataLoader expects; its lengths remain accessible through the base refs.
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ),
        sampler,
    )


def _mean_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if value != value:  # NaN (for undefined AUROC) is not averaged.
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: value / max(counts[key], 1) for key, value in sums.items()}


def _format_mask(action_vocabulary: ActionVocabulary, formats: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.stack(
        [torch.as_tensor(action_vocabulary.format_mask(fmt), dtype=torch.bool, device=device) for fmt in formats]
    )


def _sample_posterior(mu: torch.Tensor, logvar: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
    return StateVAE.sample(mu, logvar, deterministic=deterministic)


def _v_uses_decoder_header_conditioning(vae: Any) -> bool:
    vae = getattr(vae, "_orig_mod", vae)
    return getattr(vae, "decoder_header_conditioning", "none") == "cross_attention"


def _decoder_header_kwargs(vae: Any, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Supply raw header memory only to decoders whose config enables it."""
    if not _v_uses_decoder_header_conditioning(vae):
        return {}
    return {
        "header_tokens": batch["header_tokens"],
        "header_valid_mask": batch["header_mask"],
    }


def _balanced_format_row_indices(
    formats: Sequence[str],
    max_examples: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Select a deterministic, approximately format-balanced row subset.

    Training batches are shuffled, so taking rows round-robin within each
    format rotates the selected raw battles across optimizer steps without an
    additional RNG stream.  Per-format counts differ by at most one whenever
    each group has enough rows.
    """
    if max_examples < 1:
        raise ValueError("team_sigreg_batch_size must be positive")
    if not formats:
        return torch.empty(0, dtype=torch.long, device=device)
    if max_examples >= len(formats):
        return torch.arange(len(formats), dtype=torch.long, device=device)
    grouped: dict[str, list[int]] = {}
    for row, fmt in enumerate(formats):
        grouped.setdefault(str(fmt), []).append(row)
    selected: list[int] = []
    offsets = {fmt: 0 for fmt in grouped}
    while len(selected) < max_examples:
        progressed = False
        for fmt in sorted(grouped):
            offset = offsets[fmt]
            rows = grouped[fmt]
            if offset < len(rows):
                selected.append(rows[offset])
                offsets[fmt] = offset + 1
                progressed = True
                if len(selected) == max_examples:
                    break
        if not progressed:
            break
    return torch.as_tensor(selected, dtype=torch.long, device=device)


def _cuda_bf16(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _v_format_metrics(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, float]:
    """Compact per-format V reconstruction metrics for validation logs."""
    result: dict[str, float] = {}
    logits = outputs["logits"].detach()
    targets = batch["state_tokens"]
    mask = batch["state_mask"]
    for fmt in sorted(set(batch["formats"])):
        rows = torch.tensor([value == fmt for value in batch["formats"]], device=logits.device, dtype=torch.bool)
        valid = mask[rows]
        if not bool(valid.any()):
            continue
        ce = torch.nn.functional.cross_entropy(
            logits[rows].reshape(-1, logits.shape[-1]), targets[rows].reshape(-1), reduction="none"
        )
        ce = ce[valid.reshape(-1)]
        correct = logits[rows].argmax(dim=-1)[valid].eq(targets[rows][valid])
        base = f"v_{fmt}_token"
        result[f"{base}_ce"] = float(ce.mean())
        result[f"{base}_acc"] = float(correct.float().mean())
        # Private sufficient statistics let validation combine uneven batches
        # by tokens rather than taking a mean of per-batch means.
        result[f"_count/{base}"] = float(correct.numel())
        result[f"_ce_sum/{base}"] = float(ce.double().sum())
        result[f"_correct/{base}"] = float(correct.sum())
    return result


def _v_token_category_metrics(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any], tokenizer: PokemonTokenizer,
) -> dict[str, float]:
    """Report the requested interpretable V token groups without extra passes."""
    targets = batch["state_tokens"]
    valid = batch["state_mask"]
    lookup: list[int] = [0] * (len(tokenizer) + 1)
    statuses_effects = {
        "par", "slp", "frz", "brn", "psn", "tox", "nostatus", "noboosts", "noeffect",
        "noweather", "none", "unknown", "fnt",
    }
    terminal_words = {"won", "lost", "tie", "forfeit_won", "forfeit_lost", "<terminal>", "<end_terminal>"}
    # 0 structural; 1 species/move/open vocabulary; 2 hp/status/effect; 3 terminal.
    for token_id, word in enumerate(tokenizer.detokenize(list(range(len(tokenizer) + 1)))):
        if word in terminal_words:
            lookup[token_id] = 3
        elif word.startswith("<") and word.endswith(">"):
            lookup[token_id] = 0
        elif word in statuses_effects or word.replace(".", "", 1).isdigit():
            lookup[token_id] = 2
        else:
            lookup[token_id] = 1
    categories = torch.as_tensor(lookup, device=targets.device)[targets]
    ce = torch.nn.functional.cross_entropy(
        outputs["logits"].detach().reshape(-1, outputs["logits"].shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    pred = outputs["logits"].detach().argmax(dim=-1)
    names = ("structural", "species_move", "hp_status_effect", "terminal")
    result: dict[str, float] = {}
    for category, name in enumerate(names):
        mask = valid & categories.eq(category)
        if bool(mask.any()):
            values = ce[mask]
            correct = pred[mask].eq(targets[mask])
            base = f"v_{name}_token"
            result[f"{base}_ce"] = float(values.mean())
            result[f"{base}_acc"] = float(correct.float().mean())
            result[f"_count/{base}"] = float(correct.numel())
            result[f"_ce_sum/{base}"] = float(values.double().sum())
            result[f"_correct/{base}"] = float(correct.sum())
        else:
            result[f"v_{name}_token_ce"] = float("nan")
            result[f"v_{name}_token_acc"] = float("nan")
    return result


def _reconstruction_totals(
    outputs: Mapping[str, torch.Tensor], targets: torch.Tensor,
) -> tuple[float, int, int]:
    """Return CE sum, correct-token count, and valid-token count."""
    valid = outputs["state_valid_mask"].bool()
    if not bool(valid.any()):
        return 0.0, 0, 0
    logits = outputs["logits"].detach()
    ce = torch.nn.functional.cross_entropy(logits[valid].float(), targets[valid], reduction="sum")
    correct = logits.argmax(dim=-1)[valid].eq(targets[valid]).sum()
    return float(ce.double()), int(correct), int(valid.sum())


def _accumulate_group_totals(
    totals: dict[str, list[float]], metrics: Mapping[str, float],
) -> dict[str, float]:
    """Consume private group sufficient statistics and return public metrics."""
    public = {key: value for key, value in metrics.items() if not key.startswith("_")}
    for key, count in metrics.items():
        if not key.startswith("_count/"):
            continue
        base = key.removeprefix("_count/")
        row = totals.setdefault(base, [0.0, 0.0, 0.0])
        row[0] += float(metrics[f"_ce_sum/{base}"])
        row[1] += float(metrics[f"_correct/{base}"])
        row[2] += float(count)
    return public


def _aggregate_gaussian_metrics(
    first_moment_sum: torch.Tensor,
    second_moment_sum: torch.Tensor,
    count: int,
    *,
    prefix: str,
) -> dict[str, float]:
    """Summarize a population from accumulated first and second moments."""
    mean = first_moment_sum.double().cpu() / count
    second_moment = second_moment_sum.double().cpu() / count
    covariance = second_moment - mean.outer(mean)
    covariance = 0.5 * (covariance + covariance.T)
    variance = covariance.diagonal().clamp_min(1e-12)
    std = variance.sqrt()
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    latent_dim = int(mean.numel())
    # A singular Gaussian has infinite KL to a full-rank standard normal.
    # Work with covariance eigenvalues so the trace and log-determinant use
    # exactly the same matrix; adding jitter only to logdet can manufacture a
    # small negative "KL" even for an identity covariance.
    eigenvalues = torch.linalg.eigvalsh(covariance)
    positive_definite = bool(torch.isfinite(eigenvalues).all() and eigenvalues.min() > 1e-10)
    if positive_definite:
        gaussian_kl = 0.5 * (
            mean.square().sum()
            + (eigenvalues - eigenvalues.log() - 1.0).sum()
        )
        # Guard only roundoff at the exact identity; the eigenvalue expression
        # is non-negative analytically.
        gaussian_kl = gaussian_kl.clamp_min(0.0)
    else:
        gaussian_kl = mean.new_tensor(float("inf"))
    return {
        f"{prefix}_mean_rms": float(mean.square().mean().sqrt()),
        f"{prefix}_mean_abs": float(mean.abs().mean()),
        f"{prefix}_std_mean": float(std.mean()),
        f"{prefix}_std_min": float(std.min()),
        f"{prefix}_std_max": float(std.max()),
        f"{prefix}_variance_mae_from_one": float((variance - 1.0).abs().mean()),
        f"{prefix}_cov_offdiag_rms": float(off_diagonal.square().mean().sqrt()),
        f"{prefix}_gaussian_kl": float(gaussian_kl),
        f"{prefix}_gaussian_kl_per_dim": (
            float(gaussian_kl / latent_dim)
        ),
    }


def _new_posterior_audit() -> dict[str, Any]:
    """Create bounded-moment plus full-row storage for one posterior family."""
    return {
        "count": 0,
        "mu_sum": None,
        "second_moment_sum": None,
        "mu_second_moment_sum": None,
        "mu_rows": [],
        "logvar_rows": [],
    }


def _accumulate_posterior_audit(
    audit: dict[str, Any], mu: torch.Tensor, logvar: torch.Tensor,
) -> None:
    """Accumulate exact aggregate moments for diagonal Gaussian posteriors."""
    mu = mu.detach().float().reshape(-1, mu.shape[-1])
    logvar = logvar.detach().float().reshape_as(mu)
    variance = logvar.exp()
    audit["mu_rows"].append(mu)
    audit["logvar_rows"].append(logvar)
    if audit["mu_sum"] is None:
        latent_dim = int(mu.shape[-1])
        audit["mu_sum"] = torch.zeros(
            latent_dim, device=mu.device, dtype=torch.float32,
        )
        audit["second_moment_sum"] = torch.zeros(
            latent_dim, latent_dim, device=mu.device, dtype=torch.float32,
        )
        audit["mu_second_moment_sum"] = torch.zeros_like(
            audit["second_moment_sum"],
        )
    audit["count"] += int(mu.shape[0])
    audit["mu_sum"].add_(mu.sum(dim=0))
    mu_second_moment = mu.transpose(0, 1).matmul(mu)
    audit["second_moment_sum"].add_(mu_second_moment)
    audit["second_moment_sum"].diagonal().add_(variance.sum(dim=0))
    audit["mu_second_moment_sum"].add_(mu_second_moment)


def _posterior_audit_metrics(
    audit: Mapping[str, Any],
    *,
    prefix: str,
    fixed_directions: torch.Tensor,
    num_points: int,
    domain: float,
) -> dict[str, float]:
    """Finalize explicitly named moment and characteristic-function audits."""
    count = int(audit["count"])
    if not count:
        return {}
    result = _aggregate_gaussian_metrics(
        audit["mu_sum"], audit["second_moment_sum"], count,
        prefix=f"{prefix}_aggregate",
    )
    result.update(_aggregate_gaussian_metrics(
        audit["mu_sum"], audit["mu_second_moment_sum"], count,
        prefix=f"{prefix}_aggregate_mu",
    ))
    all_mu = torch.cat(audit["mu_rows"], dim=0)
    all_logvar = torch.cat(audit["logvar_rows"], dim=0)
    sigreg_kwargs = {
        "num_slices": int(fixed_directions.shape[1]),
        "num_points": int(num_points),
        "domain": float(domain),
        "directions": fixed_directions,
    }
    result[f"{prefix}_aggregate_sigreg"] = float(
        aggregate_posterior_sigreg(all_mu, all_logvar, **sigreg_kwargs)
    )
    deterministic_logvar = torch.full_like(all_mu, -30.0)
    result[f"{prefix}_aggregate_mu_sigreg"] = float(
        aggregate_posterior_sigreg(all_mu, deterministic_logvar, **sigreg_kwargs)
    )
    return result


def _run_v_batch(
    model: SimpleWorldModel,
    batch: Mapping[str, Any],
    *,
    beta: float,
    args: argparse.Namespace,
    team_batch: Mapping[str, Any] | None = None,
    structural_token_lookup: torch.Tensor | None = None,
    mask_token_id: int | None = None,
    mu_sigreg_scale: float = 1.0,
    sampled_sigreg_scale: float = 1.0,
    aggregate_sigreg_scale: float = 1.0,
    team_aggregate_sigreg_scale: float = 1.0,
    posterior_std_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    header_tokens = batch["header_tokens"]
    state_tokens = batch["state_tokens"]
    masked_count = eligible_count = 0
    probability = float(args.encoder_token_mask_prob)
    if probability > 0.0:
        if structural_token_lookup is None or mask_token_id is None:
            raise ValueError("V encoder masking requires structural-token lookup and mask token")
        header_tokens, selected, eligible = _mask_non_structural_tokens(
            header_tokens,
            batch["header_mask"],
            structural_token_lookup=structural_token_lookup,
            mask_token_id=mask_token_id,
            probability=probability,
        )
        masked_count += selected
        eligible_count += eligible
        state_tokens, selected, eligible = _mask_non_structural_tokens(
            state_tokens,
            batch["state_mask"],
            structural_token_lookup=structural_token_lookup,
            mask_token_id=mask_token_id,
            probability=probability,
        )
        masked_count += selected
        eligible_count += eligible

    mean_recon_weight = float(args.mean_recon_weight)
    # With a pure mean-path objective, decoding a posterior sample builds and
    # retains an entire second decoder graph whose loss is multiplied by zero.
    # Skip it unless sampled-z SIGReg explicitly needs the reparameterized z.
    mean_only_forward = (
        mean_recon_weight == 1.0
        and not float(getattr(args, "lambda_sampled_sigreg", 0.0))
    )
    with _cuda_bf16(batch["header_tokens"].device):
        decoder_forward_kwargs: dict[str, torch.Tensor] = {}
        if _v_uses_decoder_header_conditioning(model.v):
            decoder_forward_kwargs = {
                "decoder_header_tokens": batch["header_tokens"],
                "decoder_header_valid_mask": batch["header_mask"],
            }
        outputs = model.encode_state(
            header_tokens, state_tokens,
            header_valid_mask=batch["header_mask"], state_valid_mask=batch["state_mask"],
            # Encoder masking is an augmentation; the conditional decoder
            # receives the unmodified, deployment-available team header.
            deterministic=mean_only_forward,
            **decoder_forward_kwargs,
        )
        loss, metrics = vae_losses(
            outputs, batch["state_tokens"], beta_kl=beta, free_bits=args.free_bits,
            capacity=args.kl_capacity, capacity_weight=args.kl_capacity_weight,
        )
        if mean_recon_weight > 0.0 and not mean_only_forward:
            # VAE training normally teaches the decoder only on posterior
            # samples.  A small mean-path blend keeps ``z = mu`` on the
            # decoder's training manifold, which matters because cached and
            # online deterministic inference consume the posterior mean.  By
            # interpolating complete losses, the reconstruction and KL scales
            # remain unchanged as the blend changes.
            vae = getattr(model.v, "_orig_mod", model.v)
            mean_outputs = {
                "logits": vae.decode(
                    outputs["mu"],
                    outputs["state_valid_mask"],
                    **_decoder_header_kwargs(vae, batch),
                ),
                "mu": outputs["mu"],
                "logvar": outputs["logvar"],
                "state_valid_mask": outputs["state_valid_mask"],
            }
            mean_loss, mean_metrics = vae_losses(
                mean_outputs, batch["state_tokens"], beta_kl=beta,
                free_bits=args.free_bits, capacity=args.kl_capacity,
                capacity_weight=args.kl_capacity_weight,
            )
            loss = (1.0 - mean_recon_weight) * loss + mean_recon_weight * mean_loss
            metrics["loss"] = float(loss.detach())
            metrics["objective_recon_ce"] = (
                (1.0 - mean_recon_weight) * metrics["recon_ce"]
                + mean_recon_weight * mean_metrics["recon_ce"]
            )
            metrics["mean_recon_ce"] = mean_metrics["recon_ce"]
            metrics["mean_recon_token_acc"] = mean_metrics["recon_token_acc"]
            metrics["mean_recon_weight"] = mean_recon_weight
        elif mean_only_forward:
            metrics["objective_recon_ce"] = metrics["recon_ce"]
            metrics["mean_recon_ce"] = metrics["recon_ce"]
            metrics["mean_recon_token_acc"] = metrics["recon_token_acc"]
            metrics["mean_recon_weight"] = 1.0
    # Pointwise VAE KL controls each q(z|x), but it also taxes mutual
    # information.  An optional aggregate-code term instead forces the
    # deterministic representation used by cache/player inference (mu) toward
    # N(0,I) without requiring it to forget the input.  Keep this outside the
    # bf16 autocast block: SIGReg's characteristic-function calculation is
    # intentionally fp32.
    configured_sigreg_weight = float(getattr(args, "lambda_mu_sigreg", 0.0))
    effective_sigreg_weight = configured_sigreg_weight * float(mu_sigreg_scale)
    if effective_sigreg_weight:
        mu_sigreg_loss = sigreg(
            outputs["mu"],
            num_slices=int(getattr(args, "sigreg_num_slices", 128)),
            num_points=int(getattr(args, "sigreg_num_points", 17)),
            domain=float(getattr(args, "sigreg_domain", 3.0)),
        )
        loss = loss + effective_sigreg_weight * mu_sigreg_loss
        metrics["loss"] = float(loss.detach())
        metrics["mu_sigreg_loss"] = float(mu_sigreg_loss.detach())
        metrics["mu_sigreg_weight"] = effective_sigreg_weight
        metrics["mu_sigreg_weighted"] = float(
            (effective_sigreg_weight * mu_sigreg_loss).detach()
        )
    else:
        metrics["mu_sigreg_loss"] = 0.0
        metrics["mu_sigreg_weight"] = 0.0
        metrics["mu_sigreg_weighted"] = 0.0
    # Aggregate posterior matching on the actual reparameterized draw is the
    # InfoVAE/WAE-style complement to the pointwise KL.  Unlike applying this
    # term only to mu, it can preserve q(z) ~= N(0,I) while reconstruction
    # moves information from conditional noise into between-example means.
    configured_sampled_weight = float(getattr(args, "lambda_sampled_sigreg", 0.0))
    effective_sampled_weight = configured_sampled_weight * float(sampled_sigreg_scale)
    if effective_sampled_weight:
        sampled_sigreg_loss = sigreg(
            outputs["z"],
            num_slices=int(getattr(args, "sigreg_num_slices", 128)),
            num_points=int(getattr(args, "sigreg_num_points", 17)),
            domain=float(getattr(args, "sigreg_domain", 3.0)),
        )
        loss = loss + effective_sampled_weight * sampled_sigreg_loss
        metrics["loss"] = float(loss.detach())
        metrics["sampled_sigreg_loss"] = float(sampled_sigreg_loss.detach())
        metrics["sampled_sigreg_weight"] = effective_sampled_weight
        metrics["sampled_sigreg_weighted"] = float(
            (effective_sampled_weight * sampled_sigreg_loss).detach()
        )
    else:
        metrics["sampled_sigreg_loss"] = 0.0
        metrics["sampled_sigreg_weight"] = 0.0
        metrics["sampled_sigreg_weighted"] = 0.0
    configured_aggregate_weight = float(getattr(args, "lambda_aggregate_sigreg", 0.0))
    effective_aggregate_weight = configured_aggregate_weight * float(aggregate_sigreg_scale)
    if effective_aggregate_weight:
        aggregate_sigreg_loss = aggregate_posterior_sigreg(
            outputs["mu"], outputs["logvar"],
            num_slices=int(getattr(args, "sigreg_num_slices", 128)),
            num_points=int(getattr(args, "sigreg_num_points", 17)),
            domain=float(getattr(args, "sigreg_domain", 3.0)),
        )
        loss = loss + effective_aggregate_weight * aggregate_sigreg_loss
        metrics["loss"] = float(loss.detach())
        metrics["aggregate_sigreg_loss"] = float(aggregate_sigreg_loss.detach())
        metrics["aggregate_sigreg_weight"] = effective_aggregate_weight
        metrics["aggregate_sigreg_weighted"] = float(
            (effective_aggregate_weight * aggregate_sigreg_loss).detach()
        )
    else:
        metrics["aggregate_sigreg_loss"] = 0.0
        metrics["aggregate_sigreg_weight"] = 0.0
        metrics["aggregate_sigreg_weighted"] = 0.0
    # The permanent team/header latent is a different deployed posterior
    # family from header+state q(z).  Regularize it independently on a small,
    # format-balanced raw-header subset so one family cannot hide the other's
    # mismatch and B384 training stays within GPU memory.
    configured_team_weight = float(
        getattr(args, "lambda_team_aggregate_sigreg", 0.0)
    )
    effective_team_weight = configured_team_weight * float(team_aggregate_sigreg_scale)
    if effective_team_weight:
        team_source = team_batch if team_batch is not None else batch
        team_rows = _balanced_format_row_indices(
            team_source.get(
                "formats", [""] * int(team_source["header_tokens"].shape[0])
            ),
            int(getattr(args, "team_sigreg_batch_size", 32)),
            device=team_source["header_tokens"].device,
        )
        vae = getattr(model.v, "_orig_mod", model.v)
        with _cuda_bf16(team_source["header_tokens"].device):
            team_mu, team_logvar = vae.encode(
                team_source["header_tokens"].index_select(0, team_rows),
                header_valid_mask=team_source["header_mask"].index_select(0, team_rows),
            )
        team_aggregate_sigreg_loss = aggregate_posterior_sigreg(
            team_mu,
            team_logvar,
            num_slices=int(getattr(args, "sigreg_num_slices", 128)),
            num_points=int(getattr(args, "sigreg_num_points", 17)),
            domain=float(getattr(args, "sigreg_domain", 3.0)),
        )
        loss = loss + effective_team_weight * team_aggregate_sigreg_loss
        metrics["loss"] = float(loss.detach())
        metrics["team_aggregate_sigreg_loss"] = float(
            team_aggregate_sigreg_loss.detach()
        )
        metrics["team_aggregate_sigreg_weight"] = effective_team_weight
        metrics["team_aggregate_sigreg_weighted"] = float(
            (effective_team_weight * team_aggregate_sigreg_loss).detach()
        )
        metrics["team_sigreg_examples"] = float(team_rows.numel())
    else:
        metrics["team_aggregate_sigreg_loss"] = 0.0
        metrics["team_aggregate_sigreg_weight"] = 0.0
        metrics["team_aggregate_sigreg_weighted"] = 0.0
        metrics["team_sigreg_examples"] = 0.0
    configured_std_weight = float(getattr(args, "posterior_std_weight", 0.0))
    effective_std_weight = configured_std_weight * float(posterior_std_scale)
    std_target = getattr(args, "posterior_std_target", None)
    if effective_std_weight and std_target is not None:
        posterior_std = outputs["logvar"].float().mul(0.5).exp()
        posterior_std_loss = (posterior_std - float(std_target)).square().mean()
        loss = loss + effective_std_weight * posterior_std_loss
        metrics["loss"] = float(loss.detach())
        metrics["posterior_std_target"] = float(std_target)
        metrics["posterior_std_target_loss"] = float(posterior_std_loss.detach())
        metrics["posterior_std_target_weight"] = effective_std_weight
        metrics["posterior_std_target_weighted"] = float(
            (effective_std_weight * posterior_std_loss).detach()
        )
    else:
        metrics["posterior_std_target"] = (
            float(std_target) if std_target is not None else float("nan")
        )
        metrics["posterior_std_target_loss"] = 0.0
        metrics["posterior_std_target_weight"] = 0.0
        metrics["posterior_std_target_weighted"] = 0.0
    metrics["encoder_mask_fraction"] = masked_count / max(eligible_count, 1)
    metrics["encoder_masked_tokens"] = float(masked_count)
    metrics["encoder_mask_eligible_tokens"] = float(eligible_count)
    vae = getattr(model.v, "_orig_mod", model.v)
    decoder_header_gate = getattr(vae, "decoder_header_gate", None)
    if decoder_header_gate is not None:
        gate = decoder_header_gate.detach().float()
        metrics["decoder_header_gate_norm"] = float(gate.norm())
        metrics["decoder_header_gate_abs_mean"] = float(gate.abs().mean())
    else:
        metrics["decoder_header_gate_norm"] = 0.0
        metrics["decoder_header_gate_abs_mean"] = 0.0
    return loss, metrics, outputs


def _run_m_batch(
    model: SimpleWorldModel,
    batch: Mapping[str, Any],
    *,
    action_vocabulary: ActionVocabulary,
    deterministic: bool,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    with _cuda_bf16(batch["team_mu"].device):
        team_z = _sample_posterior(batch["team_mu"], batch["team_logvar"], deterministic=deterministic)
        state_z = _sample_posterior(batch["state_mu"], batch["state_logvar"], deterministic=deterministic)
        next_z = _sample_posterior(batch["next_mu"], batch["next_logvar"], deterministic=deterministic)
        outputs = model.forward_m(
            team_z=team_z,
            state_z=state_z,
            own_history_action_ids=batch["own_history_action_ids"],
            opponent_history_action_ids=batch["opponent_history_action_ids"],
            current_own_action_ids=batch["current_own_action_ids"],
            current_opponent_action_ids=batch["current_opponent_action_ids"],
            state_valid_mask=batch["state_mask"],
            action_logit_mask=_format_mask(action_vocabulary, batch["formats"], team_z.device),
        )
        loss, metrics = m_losses(
            outputs,
            next_z=next_z,
            opponent_action_ids=batch["current_opponent_action_ids"],
            done_targets=batch["done"], outcome_targets=batch["outcome"],
            lambda_opponent=args.lambda_opponent,
            lambda_mdn=args.lambda_mdn,
            lambda_done=args.lambda_done,
            lambda_value=args.lambda_value,
            done_pos_weight=args.done_pos_weight,
        )
    with torch.no_grad():
        for fmt in sorted(set(batch["formats"])):
            rows = torch.tensor([value == fmt for value in batch["formats"]], device=team_z.device, dtype=torch.bool)
            if not bool(rows.any()):
                continue
            opp_logits = outputs["opponent_logits"][rows]
            opponent = batch["current_opponent_action_ids"][rows]
            topk = opp_logits.topk(min(5, opp_logits.shape[-1]), dim=-1).indices
            metrics[f"m_{fmt}_opponent_top1"] = float(topk[:, :1].eq(opponent[:, None]).any(dim=-1).float().mean())
            metrics[f"m_{fmt}_opponent_top5"] = float(topk.eq(opponent[:, None]).any(dim=-1).float().mean())
            metrics[f"m_{fmt}_mdn_nll"] = float(mdn_nll(
                next_z[rows], outputs["mixture_logits"][rows], outputs["mixture_means"][rows],
                outputs["mixture_log_scales"][rows],
            ))
            value_probs = torch.softmax(outputs["value_logits"][rows].float(), dim=-1)
            one_hot = torch.nn.functional.one_hot(batch["outcome"][rows], NUM_OUTCOME_CLASSES).float()
            metrics[f"m_{fmt}_value_brier"] = float((value_probs - one_hot).square().sum(dim=-1).mean())
    return loss, metrics, outputs


def _run_c_batch(
    model: SimpleWorldModel,
    batch: Mapping[str, Any],
    *,
    eligible_ids: torch.Tensor,
    action_vocabulary: ActionVocabulary,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    with _cuda_bf16(batch["team_mu"].device):
        with torch.no_grad():
            history = model.encode_history(
                batch["team_mu"], batch["state_mu"], batch["own_history_action_ids"],
                batch["opponent_history_action_ids"], batch["state_mask"],
            )
            state_last = batch["state_mask"].long().sum(dim=-1).sub(1)
            z_t = batch["state_mu"][torch.arange(batch["state_mu"].shape[0], device=batch["state_mu"].device), state_last]
        outputs = model.forward_c(
            z_t=z_t, h_t=history["h"], legal_action_ids=batch["legal_action_ids"],
            legal_action_mask=batch["legal_action_mask"],
        )
        loss, metrics, selected = c_losses(
            outputs["controller_logits"], legal_action_ids=batch["legal_action_ids"],
            legal_action_mask=batch["legal_action_mask"], chosen_legal_action_idx=batch["chosen_legal_action_idx"],
            controller_eligible=eligible_ids,
        )
    with torch.no_grad():
        # Split C metrics by observed action kind and format.  This is only
        # evaluated on eligible rows, exactly matching the C objective.
        chosen_ids = batch["legal_action_ids"].gather(1, batch["chosen_legal_action_idx"][:, None]).squeeze(1)
        for fmt in sorted(set(batch["formats"])):
            rows = selected & torch.tensor([value == fmt for value in batch["formats"]], device=selected.device)
            if bool(rows.any()):
                pred = outputs["controller_logits"][rows].argmax(dim=-1)
                target = batch["chosen_legal_action_idx"][rows]
                metrics[f"c_{fmt}_top1"] = float(pred.eq(target).float().mean())
        for kind in ("move", "switch"):
            kind_rows = selected & torch.tensor(
                [action_vocabulary.decode(int(action_id)).startswith(f"{kind} ") for action_id in chosen_ids],
                device=selected.device,
                dtype=torch.bool,
            )
            if bool(kind_rows.any()):
                metrics[f"c_{kind}_top1"] = float(
                    outputs["controller_logits"][kind_rows].argmax(dim=-1)
                    .eq(batch["chosen_legal_action_idx"][kind_rows]).float().mean()
                )
            for fmt in sorted(set(batch["formats"])):
                fmt_rows = kind_rows & torch.tensor(
                    [value == fmt for value in batch["formats"]], device=selected.device, dtype=torch.bool
                )
                if bool(fmt_rows.any()):
                    metrics[f"c_{fmt}_{kind}_top1"] = float(
                        outputs["controller_logits"][fmt_rows].argmax(dim=-1)
                        .eq(batch["chosen_legal_action_idx"][fmt_rows]).float().mean()
                    )
        legal_counts = batch["legal_action_mask"].sum(dim=-1)
        for label, bucket in (("1_2", (1, 2)), ("3_4", (3, 4)), ("5_plus", (5, 10**9))):
            rows = selected & legal_counts.ge(bucket[0]) & legal_counts.le(bucket[1])
            if bool(rows.any()):
                metrics[f"c_legal_{label}_top1"] = float(
                    outputs["controller_logits"][rows].argmax(dim=-1)
                    .eq(batch["chosen_legal_action_idx"][rows]).float().mean()
                )
    return loss, metrics, outputs


def _validate_v(
    model: SimpleWorldModel, loader: DataLoader, device: torch.device,
    args: argparse.Namespace, tokenizer: PokemonTokenizer,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    mc_samples = int(args.val_mc_samples)
    reconstruction_ce_sum = 0.0
    reconstruction_correct = reconstruction_count = 0
    mc_recon_sums = [0.0] * mc_samples
    mc_correct = [0] * mc_samples
    mc_counts = [0] * mc_samples
    group_totals: dict[str, list[float]] = {}
    state_posterior_audit = _new_posterior_audit()
    team_posterior_audit = _new_posterior_audit()
    # Common random numbers make MC metrics comparable between checkpoints and
    # keep validation from advancing the RNG stream used by stochastic V
    # training. Generating epsilon on CPU also works uniformly on CUDA/MPS/CPU.
    mc_generator = torch.Generator(device="cpu")
    mc_generator.manual_seed(int(args.seed) + 17_031)
    vae = getattr(model.v, "_orig_mod", model.v)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model.encode_state(
                batch["header_tokens"], batch["state_tokens"], header_valid_mask=batch["header_mask"],
                state_valid_mask=batch["state_mask"],
                deterministic=True,
            )
            _, metrics = vae_losses(
                outputs, batch["state_tokens"], beta_kl=args.beta_kl, free_bits=args.free_bits,
                capacity=args.kl_capacity, capacity_weight=args.kl_capacity_weight,
            )
            ce_sum, correct, count = _reconstruction_totals(outputs, batch["state_tokens"])
            reconstruction_ce_sum += ce_sum
            reconstruction_correct += correct
            reconstruction_count += count
            metrics.update(_accumulate_group_totals(group_totals, _v_format_metrics(outputs, batch)))
            metrics.update(_accumulate_group_totals(
                group_totals, _v_token_category_metrics(outputs, batch, tokenizer),
            ))
            rows.append(metrics)
            # Audit the two deployed V posterior families independently.  State
            # latents condition on header + visible state, while the permanent
            # team latent is produced by the distinct header-only path used by
            # cache construction and online play.  This is validation-only: it
            # deliberately does not alter V's reconstruction objective.
            _accumulate_posterior_audit(
                state_posterior_audit, outputs["mu"], outputs["logvar"],
            )
            team_mu, team_logvar = vae.encode(
                batch["header_tokens"], header_valid_mask=batch["header_mask"],
            )
            _accumulate_posterior_audit(
                team_posterior_audit, team_mu, team_logvar,
            )
            for sample_index in range(mc_samples):
                epsilon = torch.randn(
                    outputs["mu"].shape, generator=mc_generator, dtype=torch.float32,
                ).to(device=outputs["mu"].device, dtype=outputs["mu"].dtype)
                sampled_z = outputs["mu"] + epsilon * torch.exp(0.5 * outputs["logvar"])
                sampled_outputs = {
                    "logits": vae.decode(
                        sampled_z,
                        outputs["state_valid_mask"],
                        **_decoder_header_kwargs(vae, batch),
                    ),
                    "mu": outputs["mu"],
                    "logvar": outputs["logvar"],
                    "state_valid_mask": outputs["state_valid_mask"],
                }
                ce_sum, correct, count = _reconstruction_totals(sampled_outputs, batch["state_tokens"])
                mc_recon_sums[sample_index] += ce_sum
                mc_correct[sample_index] += correct
                mc_counts[sample_index] += count
    model.train()
    result = _mean_metrics(rows)
    if reconstruction_count:
        result["recon_ce"] = reconstruction_ce_sum / reconstruction_count
        result["recon_token_acc"] = reconstruction_correct / reconstruction_count
    for base, (ce_sum, correct, count) in group_totals.items():
        if count:
            result[f"{base}_ce"] = ce_sum / count
            result[f"{base}_acc"] = correct / count
    mc_draw_means = [total / count for total, count in zip(mc_recon_sums, mc_counts) if count]
    if mc_draw_means:
        mc_mean = sum(mc_draw_means) / len(mc_draw_means)
        result["recon_ce_mc"] = mc_mean
        result["recon_ce_mc_std"] = math.sqrt(
            sum((value - mc_mean) ** 2 for value in mc_draw_means) / len(mc_draw_means)
        )
    mc_accuracy_means = [total / count for total, count in zip(mc_correct, mc_counts) if count]
    if mc_accuracy_means:
        mc_accuracy_mean = sum(mc_accuracy_means) / len(mc_accuracy_means)
        result["recon_token_acc_mc"] = mc_accuracy_mean
        result["recon_token_acc_mc_std"] = math.sqrt(
            sum((value - mc_accuracy_mean) ** 2 for value in mc_accuracy_means)
            / len(mc_accuracy_means)
        )
    if state_posterior_audit["count"]:
        # Use the same fixed directions for state and team so their held-out CF
        # discrepancies are directly comparable.  The seed is independent of
        # both training and the fixed common-random-number MC reconstruction.
        latent_dim = int(state_posterior_audit["mu_sum"].shape[-1])
        direction_generator = torch.Generator(device="cpu")
        direction_generator.manual_seed(int(args.seed) + 29_771)
        num_slices = int(getattr(args, "sigreg_num_slices", 128))
        fixed_directions = torch.randn(
            latent_dim, num_slices, generator=direction_generator, dtype=torch.float32,
        ).to(state_posterior_audit["mu_sum"].device)
        audit_kwargs = {
            "fixed_directions": fixed_directions,
            "num_points": int(getattr(args, "sigreg_num_points", 17)),
            "domain": float(getattr(args, "sigreg_domain", 3.0)),
        }
        state_metrics = _posterior_audit_metrics(
            state_posterior_audit, prefix="state", **audit_kwargs,
        )
        result.update(state_metrics)
        # Preserve all historical unprefixed names as exact aliases to the
        # state posterior; existing dashboards and checkpoint comparisons keep
        # their original semantics and values.
        result.update({
            name.removeprefix("state_"): value
            for name, value in state_metrics.items()
        })
        result.update(_posterior_audit_metrics(
            team_posterior_audit, prefix="team", **audit_kwargs,
        ))
    result["selection_score"] = result.get("recon_ce", float("inf"))
    return result


def _merge_v_train_eval_metrics(
    validation_metrics: Mapping[str, float],
    train_eval_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Attach matched train-split evaluation without changing V selection."""
    result = dict(validation_metrics)
    for name, value in train_eval_metrics.items():
        if name != "selection_score":
            result[f"train_eval_{name}"] = float(value)
    for name in ("loss", "recon_ce", "recon_token_acc", "recon_ce_mc", "recon_token_acc_mc"):
        train_name = f"train_eval_{name}"
        if name in result and train_name in result:
            # Positive CE/loss gaps mean validation is worse.  Accuracy uses
            # train - validation so positive consistently means overfitting.
            if "acc" in name:
                result[f"generalization_gap_{name}"] = result[train_name] - result[name]
            else:
                result[f"generalization_gap_{name}"] = result[name] - result[train_name]
    return result


def _validate_m(
    model: SimpleWorldModel, loader: DataLoader, device: torch.device,
    action_vocabulary: ActionVocabulary, args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    drift_values: dict[int, list[float]] = {1: [], 5: [], 10: []}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            _, metrics, _ = _run_m_batch(
                model, batch, action_vocabulary=action_vocabulary, deterministic=True, args=args
            )
            rows.append(metrics)
            for horizon, values in _teacher_forced_rollout_drift(model, batch).items():
                drift_values[horizon].extend(values)
    model.train()
    result = _mean_metrics(rows)
    # Fixed composite selection score: all four terms improve when lower.
    done_auc = result.get("done_auroc", float("nan"))
    auc_penalty = 1.0 - done_auc if done_auc == done_auc else 1.0
    result["selection_score"] = (
        result.get("mdn_nll", float("inf")) + result.get("opponent_ce", float("inf"))
        + auc_penalty + result.get("value_brier", float("inf"))
    )
    for horizon, values in drift_values.items():
        result[f"rollout_latent_drift_{horizon}"] = sum(values) / len(values) if values else float("nan")
    result.setdefault("done_auroc", float("nan"))
    result.setdefault("done_pr_auc", float("nan"))
    return result


@torch.no_grad()
def _teacher_forced_rollout_drift(model: SimpleWorldModel, batch: Mapping[str, Any]) -> dict[int, list[float]]:
    """Measure 1/5/10-step drift using true actions and recursively imagined z.

    This intentionally does not feed future V states back into M: after the
    first transition each context contains M's own mean prediction, which is
    the online rollout failure mode the diagnostic should expose.
    """
    team = batch["team_mu"]
    original_states = batch["state_mu"]
    original_mask = batch["state_mask"]
    capacity = model.m.max_context_transitions + 1
    batch_size, _, latent_dim = original_states.shape
    states = original_states.new_zeros(batch_size, capacity, latent_dim)
    state_mask = torch.zeros(batch_size, capacity, dtype=torch.bool, device=team.device)
    own = torch.zeros(batch_size, capacity - 1, dtype=torch.long, device=team.device)
    opponent = torch.zeros_like(own)
    for row in range(batch_size):
        count = int(original_mask[row].sum())
        keep = min(count, capacity)
        states[row, :keep] = original_states[row, count - keep : count]
        state_mask[row, :keep] = True
        action_keep = max(keep - 1, 0)
        if action_keep:
            original_actions = max(count - 1, 0)
            own[row, :action_keep] = batch["own_history_action_ids"][row, original_actions - action_keep : original_actions]
            opponent[row, :action_keep] = batch["opponent_history_action_ids"][row, original_actions - action_keep : original_actions]
    results: dict[int, list[float]] = {1: [], 5: [], 10: []}
    max_steps = int(batch["future_mask"].shape[1])
    for step in range(max_steps):
        active = batch["future_mask"][:, step]
        if not bool(active.any()):
            break
        history = model.encode_history(team, states, own, opponent, state_mask)
        own_ids = batch["future_own_action_ids"][:, step]
        opponent_ids = batch["future_opponent_action_ids"][:, step]
        params = model.transition_head(
            history["h"], model.action_embedding(own_ids), model.action_embedding(opponent_ids)
        )
        weights = torch.softmax(params["mixture_logits"].float(), dim=-1)
        predicted = (weights.unsqueeze(-1) * params["mixture_means"].float()).sum(dim=1)
        distance = (predicted - batch["future_mu"][:, step].float()).norm(dim=-1)
        if step + 1 in results:
            results[step + 1].extend(float(value) for value in distance[active].cpu())
        for row in range(batch_size):
            if not bool(active[row]):
                continue
            count = int(state_mask[row].sum())
            if count < capacity:
                states[row, count] = predicted[row]
                state_mask[row, count] = True
                if capacity > 1:
                    own[row, count - 1] = own_ids[row]
                    opponent[row, count - 1] = opponent_ids[row]
            else:
                if capacity > 1:
                    states[row, :-1] = states[row, 1:].clone()
                    states[row, -1] = predicted[row]
                    own[row, :-1] = own[row, 1:].clone()
                    opponent[row, :-1] = opponent[row, 1:].clone()
                    own[row, -1] = own_ids[row]
                    opponent[row, -1] = opponent_ids[row]
                else:
                    states[row, 0] = predicted[row]
    return results


def _validate_c(
    model: SimpleWorldModel, loader: DataLoader, device: torch.device,
    eligible_ids: torch.Tensor, action_vocabulary: ActionVocabulary,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            _, metrics, _ = _run_c_batch(
                model, batch, eligible_ids=eligible_ids, action_vocabulary=action_vocabulary
            )
            rows.append(metrics)
    model.train()
    result = _mean_metrics(rows)
    result["selection_score"] = -result.get("controller_acc", 0.0)
    return result


def _training_loop(
    *,
    stage: str,
    model: SimpleWorldModel,
    train_loader: DataLoader,
    train_sampler: Any,
    validation: Callable[[], dict[str, float]],
    batch_loss: Callable[[Mapping[str, Any], bool], tuple[torch.Tensor, dict[str, float]]],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    save_callback: Callable[[int, Mapping[str, float], bool], None],
    device: torch.device,
    wandb_logger: _WandbLogger | None = None,
    start_update: int = 0,
    initial_best: float = float("inf"),
    learning_rate_schedule: Callable[[int], float] | None = None,
    early_stop_patience: int = 0,
) -> None:
    updates_budget = _resolve_updates_budget(args, stage, start_update=start_update)
    best = float(initial_best)
    update = int(start_update)
    micro = 0
    pending_metric_rows: list[dict[str, float]] = []
    metric_history: deque[dict[str, float]] = deque(
        maxlen=max(int(getattr(args, "train_metric_window", 1)), 1)
    )
    stale_validations = 0
    clip_history: deque[float] = deque(maxlen=max(int(args.grad_clip_fraction_window), 1))
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    if update >= updates_budget:
        print(
            f"[{stage}] checkpoint is already at update {update:,}; "
            f"target is {updates_budget:,}, so no training is needed.",
            flush=True,
        )
        return
    print(
        f"[{stage}] {'resuming' if update else 'starting'} at update {update:,}; "
        f"target={updates_budget:,} optimizer updates "
        f"(batch_size={args.batch_size}, grad_accum={args.grad_accum_steps}, workers={args.num_workers}).",
        flush=True,
    )
    # A resumed compact sampler starts from a new deterministic seed instead
    # of replaying the original run's first ``start_update`` batches.
    for epoch in itertools.count(start_update):
        train_sampler.set_epoch(epoch)
        base_dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
        if hasattr(base_dataset, "set_epoch"):
            base_dataset.set_epoch(epoch)
        for batch in train_loader:
            if update == start_update and micro == 0:
                compile_note = "; first compiled step may take a little longer" if args.compile else ""
                print(f"[{stage}] first batch loaded; beginning forward/backward{compile_note}.", flush=True)
            model.train()
            batch = move_batch_to_device(batch, device)
            setattr(batch_loss, "optimizer_update", update)
            loss, metrics = batch_loss(batch, False)
            pending_metric_rows.append(dict(metrics))
            (loss / args.grad_accum_steps).backward()
            micro += 1
            if micro % args.grad_accum_steps:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], args.grad_clip
            )
            grad_norm_value = float(grad_norm)
            was_clipped = float(not math.isfinite(grad_norm_value) or grad_norm_value > args.grad_clip)
            clip_history.append(was_clipped)
            clip_fraction = sum(clip_history) / len(clip_history)
            if learning_rate_schedule is not None:
                scheduled_lr = float(learning_rate_schedule(update + 1))
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = scheduled_lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            update_metrics = _mean_metrics(pending_metric_rows)
            pending_metric_rows.clear()
            metric_history.append(update_metrics)
            smoothed_metrics = _mean_metrics(metric_history)
            elapsed = max(time.time() - started, 1e-6)
            updates_per_second = (update - start_update) / elapsed
            log_interval = args.wandb_log_interval or args.print_interval
            if wandb_logger is not None and log_interval > 0 and update % log_interval == 0:
                payload = {
                    "global_step": update,
                    **_numeric_metrics("train", update_metrics),
                    **_numeric_metrics("train_smooth", smoothed_metrics),
                    "train/grad_norm": grad_norm_value,
                    "train/grad_was_clipped": was_clipped,
                    "train/grad_clip_fraction": clip_fraction,
                    "train/updates_per_second": updates_per_second,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                }
                if device.type == "cuda":
                    payload["system/cuda_memory_allocated_gib"] = torch.cuda.memory_allocated(device) / 1024 ** 3
                    payload["system/cuda_memory_reserved_gib"] = torch.cuda.memory_reserved(device) / 1024 ** 3
                wandb_logger.log(payload, step=update)
            if args.print_interval and update % args.print_interval == 0:
                print(
                    f"[{stage}] update {update:6d}/{updates_budget} loss={update_metrics.get('loss', 0.0):.4f} "
                    f"avg{len(metric_history)}={smoothed_metrics.get('loss', 0.0):.4f} "
                    f"grad={grad_norm_value:.3f} clip={clip_fraction:.1%} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} updates/s={updates_per_second:.2f}",
                    flush=True,
                )
            should_validate = args.val_interval > 0 and update % args.val_interval == 0
            if should_validate or update >= updates_budget:
                val_metrics = validation()
                score = val_metrics.get("selection_score", float("inf"))
                improved = score < best
                if improved:
                    best = score
                    stale_validations = 0
                else:
                    stale_validations += 1
                print(f"[{stage}] validation @ {update}: {json.dumps(val_metrics, sort_keys=True)}", flush=True)
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "global_step": update,
                            **_numeric_metrics("val", val_metrics),
                            "checkpoint/is_best": float(improved),
                            "checkpoint/best_selection_score": float(best),
                            "checkpoint/early_stop_stale_validations": float(stale_validations),
                        },
                        step=update,
                    )
                save_callback(update, val_metrics, improved)
                if early_stop_patience > 0 and stale_validations >= early_stop_patience:
                    print(
                        f"[{stage}] early stopping after {stale_validations} validation checks "
                        f"without improvement; best selection score={best:.6f}.",
                        flush=True,
                    )
                    return
            if update >= updates_budget:
                return


def _check_v_resume_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    path: str | Path,
    tokenizer: PokemonTokenizer,
    source_hash: str,
    model_config: Mapping[str, Any],
) -> None:
    mismatches: list[str] = []
    if checkpoint.get("tokenizer_state") != tokenizer.to_state():
        mismatches.append("tokenizer_state")
    if checkpoint.get("dataset_manifest_hash") != source_hash:
        mismatches.append("dataset_manifest_hash")
    if checkpoint.get("model_config") != dict(model_config):
        mismatches.append("model_config")
    if int(checkpoint.get("vocab_size", -1)) != len(tokenizer):
        mismatches.append("vocab_size")
    if int(checkpoint.get("pad_id", -1)) != tokenizer.pad_token_id:
        mismatches.append("pad_id")
    if mismatches:
        raise ValueError(
            f"V resume checkpoint {path} is incompatible with this run: {', '.join(mismatches)}"
        )


def _resume_v(
    args: argparse.Namespace,
    *,
    model: SimpleWorldModel,
    optimizer: torch.optim.Optimizer,
    tokenizer: PokemonTokenizer,
    source_hash: str,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, float]:
    if not args.resume_checkpoint:
        return 0, float("inf")
    resume_path = Path(args.resume_checkpoint)
    if not resume_path.is_file():
        raise FileNotFoundError(f"V resume checkpoint not found: {resume_path}")
    checkpoint = load_stage_checkpoint(str(resume_path), device=device, expected_stage="v")
    _check_v_resume_compatibility(
        checkpoint, path=resume_path, tokenizer=tokenizer, source_hash=source_hash,
        model_config=model_config,
    )
    load_stage_weights(model, checkpoint)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        raise ValueError(f"V resume checkpoint {resume_path} has no optimizer state")
    optimizer.load_state_dict(optimizer_state)
    start_update = int(checkpoint.get("global_step", 0))
    best_score = float(checkpoint.get("best_val_loss") or float("inf"))

    # ``v_latest.pt`` stores the latest score.  Preserve the historical best
    # from the selection checkpoint when it is available.
    best_path = Path(args.checkpoint) if args.checkpoint else None
    if best_path is not None and best_path.is_file() and best_path.resolve() != resume_path.resolve():
        best_checkpoint = load_stage_checkpoint(str(best_path), device=device, expected_stage="v")
        _check_v_resume_compatibility(
            best_checkpoint, path=best_path, tokenizer=tokenizer, source_hash=source_hash,
            model_config=model_config,
        )
        best_score = min(best_score, float(best_checkpoint.get("best_val_loss") or float("inf")))
    print(
        f"[v] resumed model and optimizer from {resume_path} at update {start_update:,}; "
        f"best reconstruction CE={best_score:.6f}",
        flush=True,
    )
    return start_update, best_score


def _v_decoder_config(
    model_config: Mapping[str, Any],
) -> tuple[int, dict[str, Any], str, str, float | None]:
    """Return V bottleneck/base config and normalized posterior options."""
    latent_dim = int(model_config.get("latent_dim", -1))
    v_config = dict(model_config.get("v", {}))
    conditioning = str(v_config.pop("decoder_conditioning", "additive")).lower()
    header_conditioning = str(
        v_config.pop("decoder_header_conditioning", "none")
    ).lower()
    fixed_std_value = v_config.pop("fixed_posterior_std", None)
    fixed_std = None if fixed_std_value is None else float(fixed_std_value)
    return latent_dim, v_config, conditioning, header_conditioning, fixed_std


def _warm_start_v(
    args: argparse.Namespace,
    *,
    model: SimpleWorldModel,
    tokenizer: PokemonTokenizer,
    source_hash: str,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    """Safely warm start V across exact function-preserving transitions.

    This is intentionally separate from ``--resume_checkpoint``: a warm start
    loads model weights but starts a fresh optimizer and update schedule.  All
    data/tokenizer/base-V compatibility checks remain strict, and the only
    architecture differences accepted are the function-preserving AdaLN branch,
    raw-header decoder cross-attention behind a zero gate, and, with otherwise
    identical decoder/base settings, replacing a learned posterior scale with a
    fixed positive scale.  Each transition is isolated: simultaneous changes
    are rejected so the expected new state-dict surface stays auditable.
    """
    if not args.warm_start_checkpoint:
        return None
    warm_path = Path(args.warm_start_checkpoint)
    if not warm_path.is_file():
        raise FileNotFoundError(f"V warm-start checkpoint not found: {warm_path}")
    checkpoint = load_stage_checkpoint(str(warm_path), device=device, expected_stage="v")

    mismatches: list[str] = []
    if checkpoint.get("tokenizer_state") != tokenizer.to_state():
        mismatches.append("tokenizer_state")
    if checkpoint.get("dataset_manifest_hash") != source_hash:
        mismatches.append("dataset_manifest_hash")
    if int(checkpoint.get("vocab_size", -1)) != len(tokenizer):
        mismatches.append("vocab_size")
    if int(checkpoint.get("pad_id", -1)) != tokenizer.pad_token_id:
        mismatches.append("pad_id")

    (
        source_latent_dim,
        source_v_config,
        source_conditioning,
        source_header_conditioning,
        source_fixed_std,
    ) = _v_decoder_config(checkpoint.get("model_config", {}))
    (
        target_latent_dim,
        target_v_config,
        target_conditioning,
        target_header_conditioning,
        target_fixed_std,
    ) = _v_decoder_config(model_config)
    if source_latent_dim != target_latent_dim:
        mismatches.append("latent_dim")
    if source_v_config != target_v_config:
        mismatches.append("v_config")
    if source_conditioning not in {"additive", "adaln"}:
        mismatches.append(f"source_decoder_conditioning({source_conditioning})")
    if source_header_conditioning not in {"none", "cross_attention"}:
        mismatches.append(
            f"source_decoder_header_conditioning({source_header_conditioning})"
        )
    if target_header_conditioning not in {"none", "cross_attention"}:
        mismatches.append(
            f"target_decoder_header_conditioning({target_header_conditioning})"
        )
    if source_fixed_std is not None and (
        not math.isfinite(source_fixed_std) or source_fixed_std <= 0.0
    ):
        mismatches.append(f"source_fixed_posterior_std({source_fixed_std})")
    if target_fixed_std is not None and (
        not math.isfinite(target_fixed_std) or target_fixed_std <= 0.0
    ):
        mismatches.append(f"target_fixed_posterior_std({target_fixed_std})")
    fixed_std_changed = source_fixed_std != target_fixed_std
    header_conditioning_changed = (
        source_header_conditioning != target_header_conditioning
    )
    allowed_transition = (
        source_conditioning == target_conditioning
        or (
            source_conditioning == "additive"
            and target_conditioning == "adaln"
            and not fixed_std_changed
            and not header_conditioning_changed
        )
    )
    if not allowed_transition:
        mismatches.append(
            f"decoder_conditioning({source_conditioning}->{target_conditioning})"
        )
    allowed_fixed_std_transition = (
        not fixed_std_changed
        or (
            source_fixed_std is None
            and target_fixed_std is not None
            and math.isfinite(target_fixed_std)
            and target_fixed_std > 0.0
            and source_conditioning == target_conditioning
            and not header_conditioning_changed
        )
    )
    if not allowed_fixed_std_transition:
        mismatches.append(
            f"fixed_posterior_std({source_fixed_std}->{target_fixed_std})"
        )
    allowed_header_conditioning_transition = (
        not header_conditioning_changed
        or (
            source_header_conditioning == "none"
            and target_header_conditioning == "cross_attention"
            and source_conditioning == target_conditioning
            and not fixed_std_changed
        )
    )
    if not allowed_header_conditioning_transition:
        mismatches.append(
            "decoder_header_conditioning("
            f"{source_header_conditioning}->{target_header_conditioning})"
        )
    if mismatches:
        raise ValueError(
            f"V warm-start checkpoint {warm_path} is incompatible with this run: "
            f"{', '.join(mismatches)}"
        )

    allowed_missing: set[str] = set()
    if source_conditioning == "additive" and target_conditioning == "adaln":
        allowed_missing.update({
            key
            for key in model.state_dict()
            if key.startswith("v.decoder_blocks.") and ".adaln." in key
        })
        if not allowed_missing:
            raise ValueError("AdaLN warm start found no target conditioner tensors")
    if source_header_conditioning == "none" and target_header_conditioning == "cross_attention":
        header_missing = {
            key for key in model.state_dict() if key.startswith("v.decoder_header_")
        }
        if not header_missing:
            raise ValueError("Header-conditioned warm start found no target conditioner tensors")
        allowed_missing.update(header_missing)
    load_stage_weights(
        model,
        checkpoint,
        prefixes=("v.",),
        allowed_missing_keys=allowed_missing,
    )
    print(
        f"[v] warm started V weights from {warm_path}; "
        f"decoder conditioning {source_conditioning}->{target_conditioning}; "
        "decoder header conditioning "
        f"{source_header_conditioning}->{target_header_conditioning}; "
        f"fixed posterior std {source_fixed_std}->{target_fixed_std}; "
        "optimizer and update schedule start fresh.",
        flush=True,
    )
    return checkpoint


def _train_v(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.v_battle_sampling_alpha) <= 1.0:
        raise ValueError("--v_battle_sampling_alpha must be between 0 and 1")
    if not 0.0 <= float(args.encoder_token_mask_prob) <= 1.0:
        raise ValueError("--encoder_token_mask_prob must be between 0 and 1")
    if not 0.0 <= float(args.mean_recon_weight) <= 1.0:
        raise ValueError("--mean_recon_weight must be between 0 and 1")
    if float(args.lambda_mu_sigreg) < 0.0:
        raise ValueError("--lambda_mu_sigreg must be non-negative")
    if int(args.mu_sigreg_warmup_updates) < 0:
        raise ValueError("--mu_sigreg_warmup_updates must be non-negative")
    if float(args.lambda_sampled_sigreg) < 0.0:
        raise ValueError("--lambda_sampled_sigreg must be non-negative")
    if int(args.sampled_sigreg_warmup_updates) < 0:
        raise ValueError("--sampled_sigreg_warmup_updates must be non-negative")
    if float(args.lambda_aggregate_sigreg) < 0.0:
        raise ValueError("--lambda_aggregate_sigreg must be non-negative")
    if int(args.aggregate_sigreg_warmup_updates) < 0:
        raise ValueError("--aggregate_sigreg_warmup_updates must be non-negative")
    if float(args.lambda_team_aggregate_sigreg) < 0.0:
        raise ValueError("--lambda_team_aggregate_sigreg must be non-negative")
    if int(args.team_aggregate_sigreg_warmup_updates) < 0:
        raise ValueError("--team_aggregate_sigreg_warmup_updates must be non-negative")
    if int(args.team_sigreg_batch_size) < 1:
        raise ValueError("--team_sigreg_batch_size must be positive")
    if float(args.posterior_std_weight) < 0.0:
        raise ValueError("--posterior_std_weight must be non-negative")
    if int(args.posterior_std_warmup_updates) < 0:
        raise ValueError("--posterior_std_warmup_updates must be non-negative")
    if args.posterior_std_target is not None and float(args.posterior_std_target) <= 0.0:
        raise ValueError("--posterior_std_target must be positive")
    if float(args.posterior_std_weight) and args.posterior_std_target is None:
        raise ValueError("--posterior_std_weight requires --posterior_std_target")
    enabled_sigreg_targets = sum(bool(float(value)) for value in (
        args.lambda_mu_sigreg, args.lambda_sampled_sigreg, args.lambda_aggregate_sigreg,
    ))
    if enabled_sigreg_targets > 1:
        raise ValueError(
            "Use only one aggregate SIGReg target. --lambda_aggregate_sigreg analytically "
            "constrains q(z), --lambda_sampled_sigreg uses noisy posterior draws, and "
            "--lambda_mu_sigreg constrains only deterministic codes."
        )
    if int(args.sigreg_num_slices) < 1:
        raise ValueError("--sigreg_num_slices must be positive")
    if int(args.sigreg_num_points) < 2:
        raise ValueError("--sigreg_num_points must be at least two")
    if float(args.sigreg_domain) <= 0.0:
        raise ValueError("--sigreg_domain must be positive")
    if int(args.early_stop_patience) < 0:
        raise ValueError("--early_stop_patience must be non-negative")
    if int(args.grad_clip_fraction_window) < 1:
        raise ValueError("--grad_clip_fraction_window must be positive")
    if int(args.val_mc_samples) < 1:
        raise ValueError("--val_mc_samples must be positive")
    if int(args.train_eval_samples) < 0:
        raise ValueError("--train_eval_samples must be non-negative")
    if int(args.train_metric_window) < 1:
        raise ValueError("--train_metric_window must be positive")
    _warmup_cosine_lr(
        1,
        peak_lr=args.lr,
        min_lr=args.min_lr,
        warmup_updates=args.lr_warmup_updates,
        schedule_updates=args.lr_schedule_updates,
    )

    device = _device()
    tokenizer = _load_tokenizer(args.tokenizer_path)
    structural_token_lookup = torch.as_tensor(
        [
            word.startswith("<") and word.endswith(">")
            for word in tokenizer.detokenize(list(range(len(tokenizer) + 1)))
        ],
        dtype=torch.bool,
        device=device,
    )
    config = _load_config(args.config)
    _apply_loss_config(args, config)
    fixed_posterior_std = dict(config["model"].get("v", {})).get(
        "fixed_posterior_std"
    )
    if fixed_posterior_std is not None and float(args.posterior_std_weight):
        raise ValueError(
            "--posterior_std_weight cannot be used when model.v.fixed_posterior_std "
            "is configured; the scale is constant and has no gradient"
        )
    # V only sees headers/current states.  Scanning 70M action and legal
    # candidates here delayed the first optimizer step by many minutes while
    # serving no V objective.  The full canonical vocabulary is built once in
    # the cache stage, before M/C ever consume it.
    vocab = ActionVocabulary()
    print("[v] action vocabulary is deferred to --stage cache; starting state-posterior indexing.", flush=True)
    model = _model_from_config(config, tokenizer=tokenizer, action_vocabulary=vocab, max_context_transitions=args.max_context_transitions, device=device)
    _freeze(model.m); _freeze(model.action_embedding); _freeze(model.opponent_head); _freeze(model.transition_head); _freeze(model.done_head); _freeze(model.value_head); _freeze(model.c)
    train_ds = VStateDataset(
        discover_source_shards(args.data_root, "train", args.formats), data_root=args.data_root,
        max_state_tokens=model.v.max_state_tokens, formats=args.formats,
    )
    val_paths = discover_source_shards(args.data_root, "val", args.formats)
    val_ds = _fixed_subset(
        VStateDataset(
            val_paths, data_root=args.data_root, max_state_tokens=model.v.max_state_tokens, formats=args.formats,
        ), args.val_samples
    ) if val_paths else None
    train_loader, train_sampler = _loader(
        train_ds, batch_size=args.batch_size, balanced_formats=args.balanced_formats, shuffle=True, seed=args.seed,
        collate=functools.partial(collate_v, pad_id=tokenizer.pad_token_id), num_workers=args.num_workers,
        battle_sampling_alpha=args.v_battle_sampling_alpha,
    )
    team_train_loader = None
    team_train_sampler = None
    if float(args.lambda_team_aggregate_sigreg):
        # Team/cache deployment contains one permanent header latent per POV
        # and raw battle.  A state-weighted main V batch would over-represent
        # long battles, so draw this posterior family from its own uniform-
        # battle stream (alpha=0) instead of trying to correct it in the loss.
        team_train_loader, team_train_sampler = _loader(
            train_ds,
            batch_size=args.team_sigreg_batch_size,
            balanced_formats=True,
            shuffle=True,
            seed=args.seed + 41_009,
            collate=functools.partial(collate_v, pad_id=tokenizer.pad_token_id),
            num_workers=min(args.num_workers, 4),
            battle_sampling_alpha=0.0,
        )
    train_eval_ds = train_ds.fixed_subset(args.train_eval_samples) if args.train_eval_samples else None
    train_eval_loader = None
    if train_eval_ds is not None:
        train_eval_loader, _ = _loader(
            train_eval_ds, batch_size=args.batch_size, balanced_formats=False, shuffle=False,
            seed=args.seed, collate=functools.partial(collate_v, pad_id=tokenizer.pad_token_id),
            num_workers=args.num_workers,
        )
    print(
        f"[v] battle sampling alpha={args.v_battle_sampling_alpha:g} "
        "(0=uniform battles, 1=uniform states)",
        flush=True,
    )
    if train_eval_loader is not None:
        print(
            f"[v] matched train-eval uses {len(train_eval_ds):,} fixed samples; "
            f"validation uses {args.val_mc_samples} fixed posterior draws.",
            flush=True,
        )
    print(
        f"[v] LR warmup={args.lr_warmup_updates:,}, peak={args.lr:.2e}, "
        f"cosine minimum={args.min_lr:.2e} at update {args.lr_schedule_updates:,}; "
        f"early-stop patience={args.early_stop_patience} validations; "
        f"encoder content masking={args.encoder_token_mask_prob:.1%}; "
        f"mean reconstruction blend={args.mean_recon_weight:.1%}; "
        f"mu SIGReg={args.lambda_mu_sigreg:g} "
        f"(warmup {args.mu_sigreg_warmup_updates:,}); "
        f"sampled SIGReg={args.lambda_sampled_sigreg:g} "
        f"(warmup {args.sampled_sigreg_warmup_updates:,}); "
        f"analytic aggregate SIGReg={args.lambda_aggregate_sigreg:g} "
        f"(warmup {args.aggregate_sigreg_warmup_updates:,}); "
        f"team aggregate SIGReg={args.lambda_team_aggregate_sigreg:g} "
        f"on {args.team_sigreg_batch_size:,} uniform-battle headers "
        f"(warmup {args.team_aggregate_sigreg_warmup_updates:,}); "
        f"posterior std target={args.posterior_std_target} weight={args.posterior_std_weight:g} "
        f"(warmup {args.posterior_std_warmup_updates:,}).",
        flush=True,
    )
    if val_ds is None:
        raise ValueError("V stage requires a validation split for checkpoint selection")
    val_loader, _ = _loader(
        val_ds, batch_size=args.batch_size, balanced_formats=False, shuffle=False, seed=args.seed,
        collate=functools.partial(collate_v, pad_id=tokenizer.pad_token_id), num_workers=args.num_workers,
    )
    source_hash = dataset_manifest_hash(args.data_root)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    model_config = _full_model_config(config, args.max_context_transitions)
    if args.warm_start_checkpoint:
        warm_path = Path(args.warm_start_checkpoint).resolve()
        best_output = Path(args.checkpoint or save_dir / "v_best.pt").resolve()
        latest_output = (save_dir / "v_latest.pt").resolve()
        if warm_path in {best_output, latest_output}:
            raise ValueError(
                "--warm_start_checkpoint must not be the V best/latest output path; "
                "use a new --save_dir/--checkpoint so the source cannot be overwritten"
            )
    warm_start_checkpoint = _warm_start_v(
        args,
        model=model,
        tokenizer=tokenizer,
        source_hash=source_hash,
        model_config=model_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_update, initial_best = _resume_v(
        args, model=model, optimizer=optimizer, tokenizer=tokenizer, source_hash=source_hash,
        model_config=model_config, device=device,
    )
    if args.compile and device.type == "cuda":
        model.v = torch.compile(model.v, dynamic=True)
        print("torch.compile enabled on V")

    team_stream_epoch = int(start_update)
    if team_train_sampler is not None:
        team_train_sampler.set_epoch(team_stream_epoch)
    team_train_iterator = iter(team_train_loader) if team_train_loader is not None else None

    def batch_loss(batch: Mapping[str, Any], _: bool) -> tuple[torch.Tensor, dict[str, float]]:
        nonlocal team_train_iterator, team_stream_epoch
        warmup = min(1.0, (getattr(batch_loss, "optimizer_update", 0) + 1) / max(args.kl_warmup_updates, 1))
        phase_update = max(1, getattr(batch_loss, "optimizer_update", start_update) - start_update + 1)
        mu_sigreg_scale = min(
            1.0, phase_update / max(int(args.mu_sigreg_warmup_updates), 1),
        ) if args.mu_sigreg_warmup_updates else 1.0
        sampled_sigreg_scale = min(
            1.0, phase_update / max(int(args.sampled_sigreg_warmup_updates), 1),
        ) if args.sampled_sigreg_warmup_updates else 1.0
        aggregate_sigreg_scale = min(
            1.0, phase_update / max(int(args.aggregate_sigreg_warmup_updates), 1),
        ) if args.aggregate_sigreg_warmup_updates else 1.0
        team_aggregate_sigreg_scale = min(
            1.0,
            phase_update / max(int(args.team_aggregate_sigreg_warmup_updates), 1),
        ) if args.team_aggregate_sigreg_warmup_updates else 1.0
        posterior_std_scale = min(
            1.0, phase_update / max(int(args.posterior_std_warmup_updates), 1),
        ) if args.posterior_std_warmup_updates else 1.0
        team_batch = None
        if team_train_loader is not None:
            assert team_train_iterator is not None
            try:
                team_batch = next(team_train_iterator)
            except StopIteration:
                team_stream_epoch += 1
                assert team_train_sampler is not None
                team_train_sampler.set_epoch(team_stream_epoch)
                team_train_iterator = iter(team_train_loader)
                team_batch = next(team_train_iterator)
            team_batch = move_batch_to_device(team_batch, device)
        loss, metrics, _ = _run_v_batch(
            model,
            batch,
            beta=args.beta_kl * warmup,
            args=args,
            team_batch=team_batch,
            structural_token_lookup=structural_token_lookup,
            mask_token_id=tokenizer.unknown_token_id,
            mu_sigreg_scale=mu_sigreg_scale,
            sampled_sigreg_scale=sampled_sigreg_scale,
            aggregate_sigreg_scale=aggregate_sigreg_scale,
            team_aggregate_sigreg_scale=team_aggregate_sigreg_scale,
            posterior_std_scale=posterior_std_scale,
        )
        return loss, metrics

    def learning_rate_schedule(update: int) -> float:
        return _warmup_cosine_lr(
            update,
            peak_lr=args.lr,
            min_lr=args.min_lr,
            warmup_updates=args.lr_warmup_updates,
            schedule_updates=args.lr_schedule_updates,
        )

    def validate() -> dict[str, float]:
        validation_metrics = _validate_v(model, val_loader, device, args, tokenizer)
        if train_eval_loader is None:
            return validation_metrics
        train_eval_metrics = _validate_v(model, train_eval_loader, device, args, tokenizer)
        return _merge_v_train_eval_metrics(validation_metrics, train_eval_metrics)

    def save_callback(update: int, metrics: Mapping[str, float], improved: bool) -> None:
        _save(save_dir / "v_latest.pt", model=model, optimizer=optimizer, stage="v", update=update, config=config,
              tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=None,
              max_context_transitions=args.max_context_transitions, metrics=metrics,
              training_config=vars(args))
        if improved:
            _save(args.checkpoint or save_dir / "v_best.pt", model=model, optimizer=optimizer, stage="v", update=update, config=config,
                  tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=None,
                  max_context_transitions=args.max_context_transitions, metrics=metrics,
                  training_config=vars(args))

    if warm_start_checkpoint is not None:
        # AdaLN is an exact identity at initialization.  Persist and select an
        # update-zero checkpoint after evaluating that invariant on the real
        # held-out split, before any newly initialized parameter is updated.
        warm_start_metrics = validate()
        initial_best = float(warm_start_metrics["selection_score"])
        save_callback(0, warm_start_metrics, True)
        print(
            f"[v] warm-start baseline at update 0: "
            f"reconstruction CE={warm_start_metrics['recon_ce']:.6f}, "
            f"token accuracy={warm_start_metrics['recon_token_acc']:.4%}.",
            flush=True,
        )

    wandb_logger = _start_wandb(
        args, stage="v", model=model, model_config=_full_model_config(config, args.max_context_transitions),
        source_hash=source_hash, cache_hash=None, device=device, start_update=start_update,
    )
    try:
        _training_loop(
            stage="v", model=model, train_loader=train_loader, train_sampler=train_sampler, validation=validate,
            batch_loss=batch_loss, optimizer=optimizer, args=args, save_callback=save_callback, device=device,
            wandb_logger=wandb_logger, start_update=start_update, initial_best=initial_best,
            learning_rate_schedule=learning_rate_schedule, early_stop_patience=args.early_stop_patience,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


def _load_for_m_or_c(args: argparse.Namespace, stage: str) -> tuple[SimpleWorldModel, PokemonTokenizer, ActionVocabulary, dict[str, Any], str, str, torch.device]:
    device = _device()
    tokenizer = _load_tokenizer(args.tokenizer_path)
    config = _load_config(args.config)
    _apply_loss_config(args, config)
    if not args.v_checkpoint:
        raise ValueError(f"--stage {stage} requires --v_checkpoint")
    if not args.latent_cache_root:
        raise ValueError(f"--stage {stage} requires --latent_cache_root")
    v_checkpoint = load_stage_checkpoint(args.v_checkpoint, device=device, expected_stage="v")
    if v_checkpoint.get("tokenizer_state") != tokenizer.to_state():
        raise ValueError("--tokenizer_path does not match the V checkpoint")
    latent_dim = int(config["model"].get("latent_dim", 128))
    manifest = assert_matching_cache(
        args.latent_cache_root, data_root=args.data_root, tokenizer=tokenizer,
        v_checkpoint_path=args.v_checkpoint, latent_dim=latent_dim,
    )
    vocab = ActionVocabulary.from_state(manifest["action_vocabulary"])
    model = _model_from_config(config, tokenizer=tokenizer, action_vocabulary=vocab,
                               max_context_transitions=args.max_context_transitions, device=device)
    load_stage_weights(model, v_checkpoint, prefixes=("v.",))
    source_hash = dataset_manifest_hash(args.data_root)
    cache_hash = _checkpoint_manifest_hash(manifest)
    return model, tokenizer, vocab, config, source_hash, cache_hash, device


def _latent_loaders(args: argparse.Namespace, manifest: Mapping[str, Any]) -> tuple[Any, Any, DataLoader, BalancedFormatBatchSampler, DataLoader]:
    mapping = format_id_to_name(manifest.get("format_id_map", {}))
    train_ds = LatentTransitionDataset(
        args.latent_cache_root, split="train", max_context_transitions=args.max_context_transitions,
        format_id_map=mapping, formats=args.formats,
    )
    val_ds = _fixed_subset(
        LatentTransitionDataset(
            args.latent_cache_root, split="val", max_context_transitions=args.max_context_transitions,
            format_id_map=mapping, formats=args.formats,
        ),
        args.val_samples,
    )
    train_loader, train_sampler = _loader(train_ds, batch_size=args.batch_size, balanced_formats=args.balanced_formats,
                                          shuffle=True, seed=args.seed, collate=collate_latent, num_workers=args.num_workers)
    val_loader, _ = _loader(val_ds, batch_size=args.batch_size, balanced_formats=False,
                            shuffle=False, seed=args.seed, collate=collate_latent, num_workers=args.num_workers)
    return train_ds, val_ds, train_loader, train_sampler, val_loader


def _train_m(args: argparse.Namespace) -> None:
    model, tokenizer, vocab, config, source_hash, cache_hash, device = _load_for_m_or_c(args, "m")
    manifest = load_cache_manifest(args.latent_cache_root)
    _freeze(model.v)
    if args.compile and device.type == "cuda":
        model.m = torch.compile(model.m, dynamic=True)
        print("torch.compile enabled on M")
    _, _, train_loader, train_sampler, val_loader = _latent_loaders(args, manifest)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    def batch_loss(batch: Mapping[str, Any], _: bool) -> tuple[torch.Tensor, dict[str, float]]:
        loss, metrics, _ = _run_m_batch(model, batch, action_vocabulary=vocab, deterministic=False, args=args)
        return loss, metrics

    def validate() -> dict[str, float]:
        return _validate_m(model, val_loader, device, vocab, args)

    def save_callback(update: int, metrics: Mapping[str, float], improved: bool) -> None:
        _save(save_dir / "m_latest.pt", model=model, optimizer=optimizer, stage="m", update=update, config=config,
              tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=cache_hash,
              max_context_transitions=args.max_context_transitions, metrics=metrics,
              training_config=vars(args))
        if improved:
            _save(args.checkpoint or save_dir / "m_best.pt", model=model, optimizer=optimizer, stage="m", update=update, config=config,
                  tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=cache_hash,
                  max_context_transitions=args.max_context_transitions, metrics=metrics,
                  training_config=vars(args))

    wandb_logger = _start_wandb(
        args, stage="m", model=model, model_config=_full_model_config(config, args.max_context_transitions),
        source_hash=source_hash, cache_hash=cache_hash, device=device,
    )
    try:
        _training_loop(
            stage="m", model=model, train_loader=train_loader, train_sampler=train_sampler, validation=validate,
            batch_loss=batch_loss, optimizer=optimizer, args=args, save_callback=save_callback, device=device,
            wandb_logger=wandb_logger,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


def _train_c(args: argparse.Namespace) -> None:
    if not args.m_checkpoint:
        raise ValueError("--stage c requires --m_checkpoint")
    model, tokenizer, vocab, config, source_hash, cache_hash, device = _load_for_m_or_c(args, "c")
    m_checkpoint = load_stage_checkpoint(args.m_checkpoint, device=device, expected_stage="m")
    if m_checkpoint.get("latent_cache_manifest_hash") != cache_hash:
        raise ValueError("M checkpoint was trained against a different latent-cache manifest")
    load_stage_weights(
        model, m_checkpoint,
        prefixes=("v.", "action_embedding.", "m.", "opponent_head.", "transition_head.", "done_head.", "value_head."),
    )
    _freeze(model.v); _freeze(model.action_embedding); _freeze(model.m); _freeze(model.opponent_head)
    _freeze(model.transition_head); _freeze(model.done_head); _freeze(model.value_head)
    manifest = load_cache_manifest(args.latent_cache_root)
    _, _, train_loader, train_sampler, val_loader = _latent_loaders(args, manifest)
    optimizer = torch.optim.AdamW([parameter for parameter in model.c.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    eligible_ids = torch.tensor([vocab.is_controller_id(index) for index in range(len(vocab))], dtype=torch.bool, device=device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    def batch_loss(batch: Mapping[str, Any], _: bool) -> tuple[torch.Tensor, dict[str, float]]:
        loss, metrics, _ = _run_c_batch(
            model, batch, eligible_ids=eligible_ids, action_vocabulary=vocab
        )
        return loss, metrics

    def validate() -> dict[str, float]:
        return _validate_c(model, val_loader, device, eligible_ids, vocab)

    def save_callback(update: int, metrics: Mapping[str, float], improved: bool) -> None:
        _save(save_dir / "c_latest.pt", model=model, optimizer=optimizer, stage="c", update=update, config=config,
              tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=cache_hash,
              max_context_transitions=args.max_context_transitions, metrics=metrics,
              training_config=vars(args))
        if improved:
            _save(args.checkpoint or save_dir / "c_best.pt", model=model, optimizer=optimizer, stage="c", update=update, config=config,
                  tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=cache_hash,
                  max_context_transitions=args.max_context_transitions, metrics=metrics,
                  training_config=vars(args))

    wandb_logger = _start_wandb(
        args, stage="c", model=model, model_config=_full_model_config(config, args.max_context_transitions),
        source_hash=source_hash, cache_hash=cache_hash, device=device,
    )
    try:
        _training_loop(
            stage="c", model=model, train_loader=train_loader, train_sampler=train_sampler, validation=validate,
            batch_loss=batch_loss, optimizer=optimizer, args=args, save_callback=save_callback, device=device,
            wandb_logger=wandb_logger,
        )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


def train(args: argparse.Namespace) -> None:
    # The sampler already owns independent deterministic Python RNGs, but VAE
    # reparameterization and SIGReg projection directions use Torch's global
    # generators. Seed them explicitly so controlled branches with the same
    # CLI seed are comparable. Exact mid-run continuation still depends on the
    # checkpointed optimizer/data position and is not claimed here.
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if args.stage == "cache":
        if not args.v_checkpoint:
            raise ValueError("--stage cache requires --v_checkpoint")
        build_cache(args)
        return
    if args.stage == "v":
        _train_v(args)
    elif args.stage == "m":
        _train_m(args)
    elif args.stage == "c":
        _train_c(args)
    else:  # pragma: no cover - argparse owns the choices
        raise ValueError(args.stage)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train staged simple V/M/C world model.")
    parser.add_argument("--stage", choices=("v", "cache", "m", "c"), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--save_dir", default="simple-world-model-checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Best checkpoint output for the active training stage.")
    v_start_group = parser.add_mutually_exclusive_group()
    v_start_group.add_argument(
        "--resume_checkpoint", default=None,
        help="Resume V model, optimizer, and global update from this stage checkpoint.",
    )
    v_start_group.add_argument(
        "--warm_start_checkpoint", default=None,
        help=(
            "Warm start compatible V weights with a fresh optimizer/update schedule. "
            "Supports the function-preserving additive-to-AdaLN decoder upgrade."
        ),
    )
    parser.add_argument("--v_checkpoint", default=None)
    parser.add_argument("--m_checkpoint", default=None)
    parser.add_argument("--latent_cache_root", default=None)
    parser.add_argument("--max_context_transitions", type=int, default=32)
    parser.add_argument("--balanced_formats", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_updates", type=int, default=0, help="Optimizer updates; defaults to the stage pilot budget.")
    parser.add_argument("--max_steps", type=int, default=0, help="Deprecated alias for --max_updates.")
    parser.add_argument(
        "--additional_updates", type=int, default=0,
        help="Train this many optimizer updates beyond the resumed global step.",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=3e-6, help="V cosine-schedule minimum learning rate.")
    parser.add_argument("--lr_warmup_updates", type=int, default=2_000, help="V linear LR warmup length.")
    parser.add_argument("--lr_schedule_updates", type=int, default=200_000, help="V cosine LR schedule length.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--grad_clip_fraction_window", type=int, default=1_000,
        help="Recent optimizer-update window used for the logged clipping fraction.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=5_000)
    parser.add_argument("--val_samples", type=int, default=10_000)
    parser.add_argument(
        "--val_mc_samples", type=int, default=4,
        help="Posterior draws used for V validation recon_ce_mc and recon_ce_mc_std.",
    )
    parser.add_argument(
        "--train_eval_samples", type=int, default=2_000,
        help="Fixed train-split V samples evaluated identically to validation (0 disables).",
    )
    parser.add_argument(
        "--train_metric_window", type=int, default=100,
        help="Optimizer-update window for smoothed training metrics.",
    )
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--wandb", default=True, action=argparse.BooleanOptionalAction,
        help="Enable Weights & Biases logging for V/M/C stages (default: enabled). Use --no-wandb to disable.",
    )
    parser.add_argument("--wandb_project", default=None, help="W&B project (default: metamon-simple-world-model).")
    parser.add_argument("--wandb_name", default=None, help="Optional W&B run name.")
    parser.add_argument(
        "--wandb_log_interval", type=int, default=0,
        help="Log train metrics every N optimizer updates (0 = --print_interval).",
    )
    # V objective
    parser.add_argument("--beta_kl", type=float, default=None)
    parser.add_argument("--kl_warmup_updates", type=int, default=20_000)
    parser.add_argument("--free_bits", type=float, default=0.02)
    parser.add_argument("--kl_capacity", type=float, default=0.0)
    parser.add_argument("--kl_capacity_weight", type=float, default=0.0)
    parser.add_argument(
        "--encoder_token_mask_prob", type=float, default=0.0,
        help="Training-only probability of replacing valid non-structural V encoder tokens with UNK.",
    )
    parser.add_argument(
        "--mean_recon_weight", type=float, default=0.0,
        help=(
            "Fraction of the V reconstruction objective decoded from posterior mu; "
            "the remainder uses a posterior sample (default: 0)."
        ),
    )
    parser.add_argument(
        "--lambda_mu_sigreg", type=float, default=0.0,
        help=(
            "Weight for aggregate Gaussian SIGReg on posterior means. This directly "
            "regularizes the deterministic latent used by caches and online inference."
        ),
    )
    parser.add_argument(
        "--mu_sigreg_warmup_updates", type=int, default=2_000,
        help="Phase-local linear warmup for --lambda_mu_sigreg after a fresh start or resume.",
    )
    parser.add_argument(
        "--lambda_sampled_sigreg", type=float, default=0.0,
        help=(
            "Weight for aggregate Gaussian SIGReg on reparameterized posterior samples. "
            "This constrains q(z) without directly penalizing mutual information."
        ),
    )
    parser.add_argument(
        "--sampled_sigreg_warmup_updates", type=int, default=2_000,
        help="Phase-local linear warmup for --lambda_sampled_sigreg.",
    )
    parser.add_argument(
        "--lambda_aggregate_sigreg", type=float, default=0.0,
        help=(
            "Weight for the analytic characteristic-function Gaussian loss on the full "
            "diagonal-Gaussian aggregate posterior."
        ),
    )
    parser.add_argument(
        "--aggregate_sigreg_warmup_updates", type=int, default=2_000,
        help="Phase-local linear warmup for --lambda_aggregate_sigreg.",
    )
    parser.add_argument(
        "--lambda_team_aggregate_sigreg", type=float, default=0.0,
        help=(
            "Independent analytic characteristic-function Gaussian loss for the "
            "deployed header-only team posterior."
        ),
    )
    parser.add_argument(
        "--team_aggregate_sigreg_warmup_updates", type=int, default=2_000,
        help="Phase-local linear warmup for --lambda_team_aggregate_sigreg.",
    )
    parser.add_argument(
        "--team_sigreg_batch_size", type=int, default=32,
        help=(
            "Maximum approximately format-balanced raw headers used by the team "
            "aggregate posterior loss per V batch."
        ),
    )
    parser.add_argument(
        "--posterior_std_target", type=float, default=None,
        help=(
            "Optional target for per-example posterior standard deviation. With aggregate "
            "q(z) matching, a small target moves prior variance into informative means."
        ),
    )
    parser.add_argument(
        "--posterior_std_weight", type=float, default=0.0,
        help="Weight for squared error to --posterior_std_target.",
    )
    parser.add_argument(
        "--posterior_std_warmup_updates", type=int, default=2_000,
        help="Phase-local linear warmup for the posterior-std target loss.",
    )
    parser.add_argument("--sigreg_num_slices", type=int, default=128)
    parser.add_argument("--sigreg_num_points", type=int, default=17)
    parser.add_argument("--sigreg_domain", type=float, default=3.0)
    parser.add_argument(
        "--early_stop_patience", type=int, default=10,
        help="Stop V after this many validation checks without improvement (0 disables).",
    )
    parser.add_argument(
        "--v_battle_sampling_alpha", type=float, default=0.5,
        help=(
            "Temper V raw-battle sampling by num_states**alpha: "
            "0 is uniform battles, 1 is uniform states (default: 0.5)."
        ),
    )
    # M objective
    parser.add_argument("--lambda_opponent", type=float, default=None)
    parser.add_argument("--lambda_mdn", type=float, default=None)
    parser.add_argument("--lambda_done", type=float, default=None)
    parser.add_argument("--lambda_value", type=float, default=None)
    parser.add_argument("--done_pos_weight", type=float, default=4.0)
    # Cache arguments are shared with cache_latents.py.
    parser.add_argument("--batch_token_budget", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-low-disk", dest="allow_low_disk", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
