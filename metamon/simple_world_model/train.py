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
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:  # Keep local training usable if an optional environment lacks wandb.
    import wandb as _wandb
except ImportError:  # pragma: no cover - wandb is a declared dependency.
    _wandb = None

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
    c_losses,
    mdn_nll,
    m_losses,
    vae_losses,
)
from metamon.tokenizer import PokemonTokenizer


DEFAULT_UPDATES = {"v": 100_000, "m": 100_000, "c": 50_000}


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
        "wandb_log_interval": int(args.wandb_log_interval),
        "balanced_formats": bool(args.balanced_formats),
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
) -> tuple[DataLoader, Any]:
    if hasattr(dataset, "draw_ref") and hasattr(dataset, "total_by_format"):
        sampler: Any = CompactFormatBatchSampler(
            dataset, batch_size=batch_size, balanced=balanced_formats, shuffle=shuffle, seed=seed,
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
        result[f"v_{fmt}_token_ce"] = float(ce[valid.reshape(-1)].mean())
        result[f"v_{fmt}_token_acc"] = float(logits[rows].argmax(dim=-1)[valid].eq(targets[rows][valid]).float().mean())
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
            result[f"v_{name}_token_ce"] = float(ce[mask].mean())
            result[f"v_{name}_token_acc"] = float(pred[mask].eq(targets[mask]).float().mean())
        else:
            result[f"v_{name}_token_ce"] = float("nan")
            result[f"v_{name}_token_acc"] = float("nan")
    return result


def _run_v_batch(model: SimpleWorldModel, batch: Mapping[str, Any], *, beta: float, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    with _cuda_bf16(batch["header_tokens"].device):
        outputs = model.encode_state(
            batch["header_tokens"], batch["state_tokens"],
            header_valid_mask=batch["header_mask"], state_valid_mask=batch["state_mask"], deterministic=False,
        )
        loss, metrics = vae_losses(
            outputs, batch["state_tokens"], beta_kl=beta, free_bits=args.free_bits,
            capacity=args.kl_capacity, capacity_weight=args.kl_capacity_weight,
        )
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
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model.encode_state(
                batch["header_tokens"], batch["state_tokens"], header_valid_mask=batch["header_mask"],
                state_valid_mask=batch["state_mask"], deterministic=True,
            )
            _, metrics = vae_losses(
                outputs, batch["state_tokens"], beta_kl=args.beta_kl, free_bits=args.free_bits,
                capacity=args.kl_capacity, capacity_weight=args.kl_capacity_weight,
            )
            metrics.update(_v_format_metrics(outputs, batch))
            metrics.update(_v_token_category_metrics(outputs, batch, tokenizer))
            rows.append(metrics)
    model.train()
    result = _mean_metrics(rows)
    result["selection_score"] = result.get("recon_ce", float("inf"))
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
) -> None:
    updates_budget = _resolve_updates_budget(args, stage, start_update=start_update)
    best = float(initial_best)
    update = int(start_update)
    micro = 0
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
            (loss / args.grad_accum_steps).backward()
            micro += 1
            if micro % args.grad_accum_steps:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], args.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            elapsed = max(time.time() - started, 1e-6)
            updates_per_second = (update - start_update) / elapsed
            log_interval = args.wandb_log_interval or args.print_interval
            if wandb_logger is not None and log_interval > 0 and update % log_interval == 0:
                payload = {
                    "global_step": update,
                    **_numeric_metrics("train", metrics),
                    "train/grad_norm": float(grad_norm),
                    "train/updates_per_second": updates_per_second,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                }
                if device.type == "cuda":
                    payload["system/cuda_memory_allocated_gib"] = torch.cuda.memory_allocated(device) / 1024 ** 3
                    payload["system/cuda_memory_reserved_gib"] = torch.cuda.memory_reserved(device) / 1024 ** 3
                wandb_logger.log(payload, step=update)
            if args.print_interval and update % args.print_interval == 0:
                print(
                    f"[{stage}] update {update:6d}/{updates_budget} loss={metrics.get('loss', 0.0):.4f} "
                    f"grad={float(grad_norm):.3f} updates/s={updates_per_second:.2f}", flush=True
                )
            should_validate = args.val_interval > 0 and update % args.val_interval == 0
            if should_validate or update >= updates_budget:
                val_metrics = validation()
                score = val_metrics.get("selection_score", float("inf"))
                improved = score < best
                if improved:
                    best = score
                print(f"[{stage}] validation @ {update}: {json.dumps(val_metrics, sort_keys=True)}", flush=True)
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "global_step": update,
                            **_numeric_metrics("val", val_metrics),
                            "checkpoint/is_best": float(improved),
                            "checkpoint/best_selection_score": float(best),
                        },
                        step=update,
                    )
                save_callback(update, val_metrics, improved)
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


def _train_v(args: argparse.Namespace) -> None:
    device = _device()
    tokenizer = _load_tokenizer(args.tokenizer_path)
    config = _load_config(args.config)
    _apply_loss_config(args, config)
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
    )
    if val_ds is None:
        raise ValueError("V stage requires a validation split for checkpoint selection")
    val_loader, _ = _loader(
        val_ds, batch_size=args.batch_size, balanced_formats=False, shuffle=False, seed=args.seed,
        collate=functools.partial(collate_v, pad_id=tokenizer.pad_token_id), num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    source_hash = dataset_manifest_hash(args.data_root)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    model_config = _full_model_config(config, args.max_context_transitions)
    start_update, initial_best = _resume_v(
        args, model=model, optimizer=optimizer, tokenizer=tokenizer, source_hash=source_hash,
        model_config=model_config, device=device,
    )
    if args.compile and device.type == "cuda":
        model.v = torch.compile(model.v, dynamic=True)
        print("torch.compile enabled on V")

    def batch_loss(batch: Mapping[str, Any], _: bool) -> tuple[torch.Tensor, dict[str, float]]:
        warmup = min(1.0, (getattr(batch_loss, "optimizer_update", 0) + 1) / max(args.kl_warmup_updates, 1))
        loss, metrics, _ = _run_v_batch(model, batch, beta=args.beta_kl * warmup, args=args)
        return loss, metrics

    def validate() -> dict[str, float]:
        return _validate_v(model, val_loader, device, args, tokenizer)

    def save_callback(update: int, metrics: Mapping[str, float], improved: bool) -> None:
        _save(save_dir / "v_latest.pt", model=model, optimizer=optimizer, stage="v", update=update, config=config,
              tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=None,
              max_context_transitions=args.max_context_transitions, metrics=metrics)
        if improved:
            _save(args.checkpoint or save_dir / "v_best.pt", model=model, optimizer=optimizer, stage="v", update=update, config=config,
                  tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=None,
                  max_context_transitions=args.max_context_transitions, metrics=metrics)

    wandb_logger = _start_wandb(
        args, stage="v", model=model, model_config=_full_model_config(config, args.max_context_transitions),
        source_hash=source_hash, cache_hash=None, device=device, start_update=start_update,
    )
    try:
        _training_loop(
            stage="v", model=model, train_loader=train_loader, train_sampler=train_sampler, validation=validate,
            batch_loss=batch_loss, optimizer=optimizer, args=args, save_callback=save_callback, device=device,
            wandb_logger=wandb_logger, start_update=start_update, initial_best=initial_best,
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
              max_context_transitions=args.max_context_transitions, metrics=metrics)
        if improved:
            _save(args.checkpoint or save_dir / "m_best.pt", model=model, optimizer=optimizer, stage="m", update=update, config=config,
                  tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=cache_hash,
                  max_context_transitions=args.max_context_transitions, metrics=metrics)

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
              max_context_transitions=args.max_context_transitions, metrics=metrics)
        if improved:
            _save(args.checkpoint or save_dir / "c_best.pt", model=model, optimizer=optimizer, stage="c", update=update, config=config,
                  tokenizer=tokenizer, action_vocabulary=vocab, source_hash=source_hash, cache_hash=cache_hash,
                  max_context_transitions=args.max_context_transitions, metrics=metrics)

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
    parser.add_argument(
        "--resume_checkpoint", default=None,
        help="Resume V model, optimizer, and global update from this stage checkpoint.",
    )
    parser.add_argument("--v_checkpoint", default=None)
    parser.add_argument("--m_checkpoint", default=None)
    parser.add_argument("--latent_cache_root", default=None)
    parser.add_argument("--max_context_transitions", type=int, default=32)
    parser.add_argument("--balanced_formats", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_updates", type=int, default=0, help="Optimizer updates; defaults to the stage pilot budget.")
    parser.add_argument("--max_steps", type=int, default=0, help="Deprecated alias for --max_updates.")
    parser.add_argument(
        "--additional_updates", type=int, default=0,
        help="Train this many optimizer updates beyond the resumed global step.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=5_000)
    parser.add_argument("--val_samples", type=int, default=10_000)
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
    parser.add_argument("--kl_warmup_updates", type=int, default=10_000)
    parser.add_argument("--free_bits", type=float, default=0.02)
    parser.add_argument("--kl_capacity", type=float, default=0.0)
    parser.add_argument("--kl_capacity_weight", type=float, default=0.0)
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
