"""Play Pokémon Showdown battles with JEPA world-model diagnostics.

Usage:
    uv run python -m metamon.jepa.play \\
        --checkpoint /workspace/poke-datasets/jepa-checkpoints/paired_best.pt \\
        --format gen1ou \\
        --username JEPABot

Interactive REPL (press keys during battle):
    R = raw protocol logs    P = state/action blocks
    V = toggle verbose       O = battle overview    Q = quit REPL
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import string

import torch
import yaml
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from metamon.env import get_metamon_teams
from metamon.jepa.checkpointing import require_tokenizer_from_checkpoint
from metamon.jepa.model import PairedJEPAModel
from metamon.jepa.player import JEPAWorldModelPlayer
from metamon.tui import TuiMixin


def random_battle_format_for(fmt: str) -> str:
    """Return the random-battle equivalent for a Showdown format id."""
    import re

    match = re.match(r"^(gen\d+)", fmt)
    if match is None:
        return "gen1randombattle"
    return f"{match.group(1)}randombattle"


def _patch_null_max_seq_len(model_cfg: dict) -> None:
    """Replace null max_seq_len with safe defaults for YAML-fallback configs.

    Checkpoints saved by train_paired.py already contain concrete integers;
    this only matters when loading the standalone default.yaml directly.
    """
    for section, fallback in [("encoder", 256), ("temporal_encoder", 6144)]:
        sec = model_cfg.setdefault(section, {})
        if sec.get("max_seq_len") is None:
            sec["max_seq_len"] = fallback


async def main() -> None:
    parser = argparse.ArgumentParser(description="Play Showdown with JEPA diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", type=str, default="gen1ou",
                        help="Battle format (default: gen1ou).")
    parser.add_argument("--username", default="JEPABot")
    parser.add_argument("--num_battles", type=int, default=30,
                        help="Number of battles to play (default: 30).")
    parser.add_argument("--max_concurrent_battles", type=int, default=30,
                        help="Maximum simultaneous battles per bot instance "
                             "(default: 30).")
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--config",
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose_blocks", action="store_true",
                        help="Print full JEPA input state/action blocks for each decision.")
    parser.add_argument("--heuristic", default="max_delta",
                        choices=["max_delta"],
                        help="Action-scoring heuristic (default: max_delta).")
    parser.add_argument("--ladder", action="store_true",
                        help="Search for random ladder battles instead of waiting for challenges.")
    parser.add_argument("--server", default="localhost",
                        choices=["localhost", "showdown"],
                        help="Server to connect to (default: localhost).")
    parser.add_argument("--password", default=None,
                        help="Showdown password (required for real server).")
    parser.add_argument("--save-raw-replay", action="store_true", default=None,
                        help="Save raw replays (default: on for showdown, off for localhost).")
    parser.add_argument("--no-save-raw-replay", dest="save_raw_replay",
                        action="store_false",
                        help="Disable saving raw replays.")
    parser.add_argument("--raw-replay-dir", default=None,
                        help="Directory for saved raw replays (default: "
                             "$METAMON_CACHE_DIR/online-raw-replays).")
    parser.add_argument("--save", action="store_true",
                        help="Save finished online battles in parsed replay text format "
                             "under $METAMON_CACHE_DIR/online-play/<format>.")
    parser.add_argument("--save-dir", default=None,
                        help="Root directory for --save output (default: "
                             "$METAMON_CACHE_DIR/online-play).")
    args = parser.parse_args()
    if args.max_concurrent_battles < 1:
        parser.error("--max_concurrent_battles must be >= 1")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config")
    if not model_cfg:
        with open(args.config, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]
    # Ensure max_seq_len is never None (YAML null) — the model constructor
    # needs explicit ints or its own defaults.
    _patch_null_max_seq_len(model_cfg)

    # ── Load tokenizer from checkpoint ──
    tokenizer = require_tokenizer_from_checkpoint(ckpt, args.checkpoint)
    print(f"Loaded tokenizer from checkpoint (vocab={len(tokenizer)}, name={tokenizer.name})")

    # All token IDs derived from the loaded tokenizer.
    vocab_size = ckpt.get("vocab_size", len(tokenizer))
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]

    # Extract training-time windowing parameter (0 = unlimited).
    max_history_blocks = ckpt.get("max_history_blocks", 100)
    if max_history_blocks:
        print(f"Using max_history_blocks={max_history_blocks} from checkpoint")
    else:
        print("Using unlimited history (max_history_blocks=0 from checkpoint)")

    model = PairedJEPAModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        latent_dim=model_cfg.get("latent_dim", 100),
        encoder_cfg=model_cfg.get("encoder", {}),
        self_belief_encoder_cfg=model_cfg.get("self_belief_encoder", {}),
        opponent_belief_predictor_cfg=model_cfg.get("opponent_belief_predictor", {}),
        opponent_policy_belief_cfg=model_cfg.get("opponent_policy_belief", {}),
        next_state_predictor_cfg=model_cfg.get("next_state_predictor", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded paired JEPA checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create lightweight bot instances sharing one model.
    server_config = ShowdownServerConfiguration if args.server == "showdown" else LocalhostServerConfiguration
    main_format = args.format
    random_format = random_battle_format_for(main_format)
    team_set = get_metamon_teams(main_format, args.team_set)

    # Generate unique usernames for the real server.
    if args.server == "showdown":
        if args.password:
            username = args.username
            username_rb = args.username + "-rb"
        else:
            suffix = "-" + "".join(random.choices(string.ascii_lowercase, k=6))
            username = args.username.lower() + suffix
            username_rb = args.username.lower() + "-rb" + suffix
    else:
        username = args.username
        username_rb = args.username + "-rb"

    # Configure raw‑replay saving.
    save_replay = args.save_raw_replay
    if save_replay is None:
        save_replay = (args.server == "showdown")  # on by default for online play
    if save_replay:
        base = args.raw_replay_dir or os.path.join(
            os.environ.get("METAMON_CACHE_DIR", "."), "online-raw-replays"
        )
        TuiMixin._repl_save_raw_dir = os.path.join(base, "jepa")
        TuiMixin._repl_save_raw_by_format = True
        os.makedirs(TuiMixin._repl_save_raw_dir, exist_ok=True)
        print(f"Saving raw replays to {TuiMixin._repl_save_raw_dir}")
    else:
        TuiMixin._repl_save_raw_dir = None
        TuiMixin._repl_save_raw_by_format = False

    parsed_save_root = None
    if args.save:
        parsed_save_root = args.save_dir or os.path.join(
            os.environ.get("METAMON_CACHE_DIR", "."), "online-play"
        )
        os.makedirs(parsed_save_root, exist_ok=True)
        print(f"Saving parsed online replays to {parsed_save_root}/<format>")

    player_ou = JEPAWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt=main_format,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        max_history_blocks=max_history_blocks,
        save_online_play_root=parsed_save_root,
        account_configuration=AccountConfiguration(username, args.password),
        server_configuration=server_config,
        battle_format=main_format,
        team=team_set,
        start_timer_on_battle_start=False,
        max_concurrent_battles=args.max_concurrent_battles,
    )
    player_rb = JEPAWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt=random_format,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        max_history_blocks=max_history_blocks,
        save_online_play_root=parsed_save_root,
        account_configuration=AccountConfiguration(username_rb, args.password),
        server_configuration=server_config,
        battle_format=random_format,
        team=None,
        start_timer_on_battle_start=False,
        max_concurrent_battles=args.max_concurrent_battles,
    )

    await asyncio.sleep(2)

    # Start interactive REPL (shared class-level key listener).
    JEPAWorldModelPlayer._start_repl()

    tasks = []
    if args.ladder:
        print(f"Searching for {args.num_battles} {main_format} ladder battles...")
        tasks.append(player_ou.ladder(args.num_battles))
    else:
        print(f"Bot online: {username} ({main_format})")
        print(f"Challenge with: /challenge {username}, {main_format}")
        tasks.append(player_ou.accept_challenges(None, args.num_battles))

    # Random battle bot always ladders.
    print(f"Searching for {args.num_battles} {random_format} ladder battles as {username_rb}...")
    tasks.append(player_rb.ladder(args.num_battles))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        JEPAWorldModelPlayer._stop_repl()
        print(f"\nResults for {username} ({main_format}):")
        print(f"  Wins: {player_ou.n_won_battles}  Losses: {player_ou.n_lost_battles}  Ties: {player_ou.n_tied_battles}")
        print(f"Results for {username_rb} ({random_format}):")
        print(f"  Wins: {player_rb.n_won_battles}  Losses: {player_rb.n_lost_battles}  Ties: {player_rb.n_tied_battles}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.")
