"""Compete a JEPA model against a registered baseline.

Usage:
    uv run python -m metamon.jepa.compete_baseline \\
        --checkpoint /path/to/paired_best.pt \\
        --format gen1ou \\
        --baseline Gen1BossAI \\
        --n_battles 10
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string

import torch
import yaml
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from metamon.baselines import get_baseline
from metamon.env import get_metamon_teams
from metamon.jepa.baseline import register_jepa_baseline
from metamon.jepa.checkpointing import require_tokenizer_from_checkpoint
from metamon.jepa.model import PairedJEPAModel


def _patch_null_max_seq_len(model_cfg: dict) -> None:
    """Replace null max_seq_len with safe defaults for YAML-fallback configs."""
    for section, fallback in [("encoder", 256), ("temporal_encoder", 6144)]:
        sec = model_cfg.setdefault(section, {})
        if sec.get("max_seq_len") is None:
            sec["max_seq_len"] = fallback


def main():
    parser = argparse.ArgumentParser(description="JEPA vs Baseline competition")
    parser.add_argument("--checkpoint", required=True, help="Path to JEPA checkpoint .pt file")
    parser.add_argument("--format", default="gen1ou", help="Battle format (default: gen1ou)")
    parser.add_argument("--baseline", default=None, help="Name of the baseline to compete against")
    parser.add_argument("--all-baselines", action="store_true",
                        help="Compete against every registered baseline for this format")
    parser.add_argument("--n_battles", type=int, default=10, help="Number of battles (default: 10)")
    parser.add_argument("--team_set", default="competitive", help="Team set to use (default: competitive)")
    parser.add_argument("--server", default="localhost",
                        choices=["localhost", "showdown"],
                        help="Server to use (default: localhost)")
    parser.add_argument("--password", default=None, help="Showdown password (required for --server showdown)")
    parser.add_argument("--config", default="metamon/jepa/configs/default.yaml",
                        help="Model config YAML")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn JEPA output")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent battles (default: 5)")
    args = parser.parse_args()
    if not args.baseline and not args.all_baselines:
        parser.error("either --baseline or --all-baselines is required")
    if args.baseline and args.all_baselines:
        parser.error("--baseline and --all-baselines are mutually exclusive")

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Load tokenizer from checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config")
    if not model_cfg:
        with open(args.config, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]
    # Replace null max_seq_len (YAML sentinel) with safe defaults.
    _patch_null_max_seq_len(model_cfg)

    tokenizer = require_tokenizer_from_checkpoint(ckpt, args.checkpoint)
    print(f"Loaded tokenizer from checkpoint (vocab={len(tokenizer)}, name={tokenizer.name})")

    vocab_size = ckpt.get("vocab_size", len(tokenizer))
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]

    # Extract training-time windowing parameter (0 = unlimited).
    max_history_blocks = ckpt.get("max_history_blocks", 100)

    model = PairedJEPAModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        latent_dim=model_cfg.get("latent_dim", 192),
        action_latent_dim=model_cfg.get("action_latent_dim", 32),
        encoder_cfg=model_cfg.get("encoder", {}),
        temporal_encoder_cfg=model_cfg.get("temporal_encoder", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        opponent_belief_predictor_cfg=model_cfg.get(
            "opponent_belief_predictor",
            model_cfg.get("opponent_state_predictor", {})
        ),
        next_state_predictor_cfg=model_cfg.get("next_state_predictor", {}),
        rank_head_cfg=model_cfg.get("rank_head", {}),
        decision_state_encoder_cfg=model_cfg.get("decision_state_encoder", {}),
        value_head_cfg=model_cfg.get("value_head", {}),
        action_projector_cfg=model_cfg.get("action_projector", {}),
        action_value_head_cfg=model_cfg.get("action_value_head", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded JEPA checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Server config ──
    server_config = ShowdownServerConfiguration if args.server == "showdown" else LocalhostServerConfiguration

    # ── Teams ──
    team_set = get_metamon_teams(args.format, args.team_set)

    # ── Usernames ──
    if args.server == "showdown":
        if not args.password:
            suffix = "-" + "".join(random.choices(string.ascii_lowercase, k=6))
            jepa_user = "jepabot" + suffix
            baseline_user = "baseline" + suffix
        else:
            jepa_user = args.baseline + "-jepa"
            baseline_user = args.baseline + "-base"
    else:
        jepa_user = "jepabot-test"
        baseline_user = f"baseline-{args.baseline}"

    # ── Register JEPA as a baseline (uses shared model) ──
    register_jepa_baseline(model, tokenizer, fmt=args.format,
                          max_history_blocks=max_history_blocks)

    # ── Create the JEPA player via the baseline registry ──
    JEPACls = get_baseline("JEPABaseline")
    jepa = JEPACls(
        battle_format=args.format,
        team=team_set,
        max_concurrent_battles=args.max_concurrent,
        account_configuration=AccountConfiguration(jepa_user, args.password),
        server_configuration=server_config,
        start_timer_on_battle_start=False,
    )

    # ── Determine which baselines to test ──
    if args.all_baselines:
        # Parse gen and format from the battle format string.
        import re
        m = re.match(r"gen(\d+)(\w+)", args.format)
        gen = int(m.group(1)) if m else 1
        tier = m.group(2) if m else args.format
        from metamon.baselines import BASELINES_BY_GEN
        baselines_for_format = BASELINES_BY_GEN.get(gen, {}).get(tier, [])
        if not baselines_for_format:
            print(f"No baselines registered for gen {gen}, format {tier}")
            return
        baseline_names = sorted([cls.__name__ for cls in baselines_for_format])
        print(f"Testing against {len(baseline_names)} baselines: {', '.join(baseline_names)}")
    else:
        baseline_names = [args.baseline]

    # ── Run battles against each baseline ──
    all_results: list[dict] = []
    for baseline_name in baseline_names:
        BaselineCls = get_baseline(baseline_name)
        # Showdown usernames are limited to 18 characters.
        base_username = f"base-{baseline_name}"
        if len(base_username) > 18:
            # Truncate the baseline name part, keeping "base-" prefix.
            max_name_len = 18 - len("base-")
            base_username = f"base-{baseline_name[:max_name_len]}"
        baseline = BaselineCls(
            battle_format=args.format,
            team=team_set,
            max_concurrent_battles=args.max_concurrent,
            account_configuration=AccountConfiguration(base_username, args.password),
            server_configuration=server_config,
        )

        print(f"\n── JEPA vs {baseline_name} ({args.n_battles} battles) ──")

        async def _compete():
            jepa.reset_battles()
            baseline.reset_battles()
            await jepa.battle_against(baseline, n_battles=args.n_battles)

        asyncio.run(_compete())

        total = jepa.n_finished_battles
        wins, losses, ties = jepa.n_won_battles, jepa.n_lost_battles, jepa.n_tied_battles
        print(f"  {wins}W  {losses}L  {ties}T")
        if total:
            wr = wins / total
            print(f"  win rate: {wr:.1%}")
            import math
            adj_w = wins + 0.5 * ties
            adj_l = losses + 0.5 * ties
            if adj_w > 0 and adj_l > 0:
                bt = 400.0 * math.log10(adj_w / adj_l)
                print(f"  BT ELO: {bt:+.1f}")
        all_results.append({
            "baseline": baseline_name,
            "wins": wins, "losses": losses, "ties": ties,
            "total": total,
        })

    # ── Summary table (if multiple baselines) ──
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Baseline':<30} {'W':>4} {'L':>4} {'T':>4} {'Win%':>7} {'BT ELO':>8}")
        print(f"{'-'*60}")
        for r in all_results:
            w, l, t, tot = r["wins"], r["losses"], r["ties"], r["total"]
            if tot:
                wr = w / tot
                adj_w = w + 0.5 * t
                adj_l = l + 0.5 * t
                if adj_w > 0 and adj_l > 0:
                    bt = 400.0 * math.log10(adj_w / adj_l)
                    bt_str = f"{bt:+.1f}"
                elif adj_w > 0:
                    bt_str = "+∞"
                elif adj_l > 0:
                    bt_str = "−∞"
                else:
                    bt_str = "0"
                print(f"{r['baseline']:<30} {w:>4} {l:>4} {t:>4} {wr:>6.1%} {bt_str:>8}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
