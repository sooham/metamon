"""Play Pokémon Showdown battles with JEPA world-model diagnostics.

Usage:
    uv run python -m metamon.jepa.play \\
        --checkpoint /workspace/poke-datasets/jepa-checkpoints/paired_best.pt \\
        --tokenizer_path /workspace/poke-datasets/tokenizers/WorldModelObservationSpace-v1.json \\
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

from metamon.data.download import METAMON_CACHE_DIR
from metamon.env import get_metamon_teams
from metamon.jepa.model import PairedJEPAModel
from metamon.jepa.player import JEPAWorldModelPlayer
from metamon.tokenizer import PokemonTokenizer
from metamon.tui import TuiMixin


async def main() -> None:
    parser = argparse.ArgumentParser(description="Play Showdown with JEPA diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tokenizer_path",
        default=os.path.join(METAMON_CACHE_DIR, "tokenizers", "WorldModelObservationSpace-v1.json"),
    )
    parser.add_argument("--format", type=str, default="gen1ou",
                        help="Battle format (default: gen1ou).")
    parser.add_argument("--username", default="JEPABot")
    parser.add_argument("--num_battles", type=int, default=30,
                        help="Number of battles to play (default: 30).")
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--config",
                        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose_blocks", action="store_true",
                        help="Print full JEPA input state/action blocks for each decision.")
    parser.add_argument("--ladder", action="store_true",
                        help="Search for random ladder battles instead of waiting for challenges.")
    parser.add_argument("--server", default="localhost",
                        choices=["localhost", "showdown"],
                        help="Server to connect to (default: localhost).")
    parser.add_argument("--password", default=None,
                        help="Showdown password (required for real server).")
    parser.add_argument("--heuristic", default="max-rank",
                        choices=["max-rank", "max-self-state-delta", "max-opponent-state-delta"],
                        help="Action selection heuristic (default: max-rank).")
    parser.add_argument("--save-raw-replay", action="store_true", default=None,
                        help="Save raw replays (default: on for showdown, off for localhost).")
    parser.add_argument("--no-save-raw-replay", dest="save_raw_replay",
                        action="store_false",
                        help="Disable saving raw replays.")
    parser.add_argument("--raw-replay-dir", default=None,
                        help="Directory for saved raw replays (default: "
                             "$METAMON_CACHE_DIR/online-raw-replays).")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(args.tokenizer_path)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config")
    if not model_cfg:
        with open(args.config, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)["model"]

    model = PairedJEPAModel(
        vocab_size=ckpt.get("vocab_size", len(tokenizer)),
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer["<bos>"],
        eos_id=tokenizer["<eos>"],
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded paired JEPA checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create lightweight bot instances sharing one model.
    server_config = ShowdownServerConfiguration if args.server == "showdown" else LocalhostServerConfiguration
    team_set = get_metamon_teams("gen1ou", args.team_set)

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
        import re
        m = re.match(r"gen(\d+)(\w+)", args.format)
        gen = f"gen{m.group(1)}" if m else args.format
        tier = m.group(2) if m else ""
        base = args.raw_replay_dir or os.path.join(
            os.environ.get("METAMON_CACHE_DIR", "."), "online-raw-replays"
        )
        TuiMixin._repl_save_raw_dir = os.path.join(base, "jepa", gen, tier)
        os.makedirs(TuiMixin._repl_save_raw_dir, exist_ok=True)
        print(f"Saving raw replays to {TuiMixin._repl_save_raw_dir}")
    else:
        TuiMixin._repl_save_raw_dir = None

    player_ou = JEPAWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt="gen1ou",
        heuristic=args.heuristic,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        account_configuration=AccountConfiguration(username, args.password),
        server_configuration=server_config,
        battle_format="gen1ou",
        team=team_set,
        start_timer_on_battle_start=False,
        max_concurrent_battles=30,
    )
    player_rb = JEPAWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt="gen1randombattle",
        heuristic=args.heuristic,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        account_configuration=AccountConfiguration(username_rb, args.password),
        server_configuration=server_config,
        battle_format="gen1randombattle",
        team=None,
        start_timer_on_battle_start=False,
        max_concurrent_battles=30,
    )

    await asyncio.sleep(2)

    # Start interactive REPL (shared class-level key listener).
    JEPAWorldModelPlayer._start_repl()

    tasks = []
    if args.ladder:
        print(f"Searching for {args.num_battles} gen1ou ladder battles...")
        tasks.append(player_ou.ladder(args.num_battles))
    else:
        print(f"Bot online: {username} (gen1ou)")
        print(f"Challenge with: /challenge {username}, gen1ou")
        tasks.append(player_ou.accept_challenges(None, args.num_battles))

    # Random battle bot always ladders.
    print(f"Searching for {args.num_battles} gen1randombattle ladder battles as {username_rb}...")
    tasks.append(player_rb.ladder(args.num_battles))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        JEPAWorldModelPlayer._stop_repl()
        print(f"\nResults for {username} (gen1ou):")
        print(f"  Wins: {player_ou.n_won_battles}  Losses: {player_ou.n_lost_battles}  Ties: {player_ou.n_tied_battles}")
        print(f"Results for {username_rb} (gen1randombattle):")
        print(f"  Wins: {player_rb.n_won_battles}  Losses: {player_rb.n_lost_battles}  Ties: {player_rb.n_tied_battles}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.")
