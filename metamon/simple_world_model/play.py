"""Play Pokémon Showdown battles with the simple-world-model controller.

Usage:
    uv run python -m metamon.simple_world_model.play \\
        --checkpoint /workspace/poke-datasets/simple-world-model-checkpoints/simple_world_model_best.pt \\
        --format gen1ou \\
        --username SimpleWMBot

Interactive REPL:
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
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from metamon.env import get_metamon_teams
from metamon.jepa.play import (
    _maintain_ladder_battles,
    random_battle_format_for,
    uses_random_battle_team,
)
from metamon.simple_world_model.action_vocab import ActionVocabulary
from metamon.simple_world_model.checkpointing import _strip_compile_prefixes, load_stage_checkpoint
from metamon.simple_world_model.model import SimpleWorldModel
from metamon.simple_world_model.player import SimpleWorldModelPlayer
from metamon.tokenizer import PokemonTokenizer
from metamon.tui import TuiMixin


def _active_battle_count(player: SimpleWorldModelPlayer) -> int:
    return sum(1 for battle in player._battles.values() if not battle.finished)


def _load_tokenizer_from_checkpoint(ckpt: dict, checkpoint_path: str) -> PokemonTokenizer:
    tokenizer_state = ckpt.get("tokenizer_state")
    if tokenizer_state is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain tokenizer_state. "
            "simple-world-model online play requires checkpoints saved by train.py."
        )
    return PokemonTokenizer.from_state(tokenizer_state)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Play Showdown with simple-world-model diagnostics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", type=str, default="gen1ou")
    parser.add_argument("--username", default="SimpleWMBot")
    parser.add_argument("--num_battles", type=int, default=30)
    parser.add_argument("--max_concurrent_battles", type=int, default=30)
    parser.add_argument("--team_set", default="competitive")
    parser.add_argument("--rollout_horizon", type=int, default=4)
    parser.add_argument("--rollouts_per_action", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose_blocks", action="store_true")
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--keep_ladder_battles", type=int, default=0)
    parser.add_argument("--no_random_battle_bot", action="store_true")
    parser.add_argument("--timer", dest="timer", action="store_true", default=True)
    parser.add_argument("--no-timer", dest="timer", action="store_false")
    parser.add_argument("--server", default="localhost", choices=["localhost", "showdown"])
    parser.add_argument("--password", default=None)
    parser.add_argument("--save-raw-replay", action="store_true", default=None)
    parser.add_argument("--no-save-raw-replay", dest="save_raw_replay", action="store_false")
    parser.add_argument("--raw-replay-dir", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    if args.max_concurrent_battles < 1:
        parser.error("--max_concurrent_battles must be >= 1")
    if args.keep_ladder_battles < 0:
        parser.error("--keep_ladder_battles must be >= 0")
    if args.keep_ladder_battles > args.max_concurrent_battles:
        args.max_concurrent_battles = args.keep_ladder_battles
        print(
            "Increasing max_concurrent_battles to "
            f"{args.max_concurrent_battles} for --keep_ladder_battles"
        )
    if args.rollout_horizon < 1 or args.rollouts_per_action < 1:
        parser.error("--rollout_horizon and --rollouts_per_action must be >= 1")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = load_stage_checkpoint(args.checkpoint, device=device, expected_stage="c")
    model_cfg = ckpt["model_config"]

    tokenizer = _load_tokenizer_from_checkpoint(ckpt, args.checkpoint)
    print(f"Loaded tokenizer from checkpoint (vocab={len(tokenizer)}, name={tokenizer.name})")

    vocab_size = int(ckpt.get("vocab_size", len(tokenizer)))
    pad_id = int(ckpt.get("pad_id", tokenizer.pad_token_id))
    max_context_transitions = int(ckpt.get("max_context_transitions", 32))
    action_vocab_state = ckpt.get("action_vocabulary")
    if action_vocab_state is None:
        raise ValueError("C checkpoint is missing canonical action_vocabulary")
    action_vocabulary = ActionVocabulary.from_state(action_vocab_state)
    print(f"Using max_context_transitions={max_context_transitions} from checkpoint")

    model = SimpleWorldModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        action_vocab_size=len(action_vocabulary),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        v_cfg=model_cfg.get("v", {}),
        action_encoder_cfg=model_cfg.get("action_encoder", {}),
        m_cfg=model_cfg.get("m", {}),
        controller_cfg=model_cfg.get("controller", {}),
    ).to(device)
    model.load_state_dict(_strip_compile_prefixes(ckpt["model_state_dict"]), strict=True)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    print(f"Loaded simple-world-model checkpoint: {args.checkpoint}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    server_config = ShowdownServerConfiguration if args.server == "showdown" else LocalhostServerConfiguration
    main_format = args.format
    random_format = random_battle_format_for(main_format)
    team_set = None if uses_random_battle_team(main_format) else get_metamon_teams(main_format, args.team_set)

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

    save_replay = args.save_raw_replay
    if save_replay is None:
        save_replay = (args.server == "showdown")
    if save_replay:
        base = args.raw_replay_dir or os.path.join(
            os.environ.get("METAMON_CACHE_DIR", "."), "online-raw-replays"
        )
        TuiMixin._repl_save_raw_dir = os.path.join(base, "simple-world-model")
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

    player_ou = SimpleWorldModelPlayer(
        model=model,
        tokenizer=tokenizer,
        fmt=main_format,
        verbose=not args.quiet,
        verbose_blocks=args.verbose_blocks,
        action_vocabulary=action_vocabulary,
        max_context_transitions=max_context_transitions,
        rollout_horizon=args.rollout_horizon,
        rollouts_per_action=args.rollouts_per_action,
        save_online_play_root=parsed_save_root,
        account_configuration=AccountConfiguration(username, args.password),
        server_configuration=server_config,
        battle_format=main_format,
        team=team_set,
        start_timer_on_battle_start=args.timer,
        max_concurrent_battles=args.max_concurrent_battles,
    )
    player_rb = None
    if not args.no_random_battle_bot:
        player_rb = SimpleWorldModelPlayer(
            model=model,
            tokenizer=tokenizer,
            fmt=random_format,
            verbose=not args.quiet,
            verbose_blocks=args.verbose_blocks,
            action_vocabulary=action_vocabulary,
            max_context_transitions=max_context_transitions,
            rollout_horizon=args.rollout_horizon,
            rollouts_per_action=args.rollouts_per_action,
            save_online_play_root=parsed_save_root,
            account_configuration=AccountConfiguration(username_rb, args.password),
            server_configuration=server_config,
            battle_format=random_format,
            team=None,
            start_timer_on_battle_start=args.timer,
            max_concurrent_battles=args.max_concurrent_battles,
        )

    await asyncio.sleep(2)
    SimpleWorldModelPlayer._start_repl()

    tasks = []
    if args.keep_ladder_battles:
        tasks.append(_maintain_ladder_battles(
            player_ou,
            target_active=args.keep_ladder_battles,
            label=main_format,
        ))
    elif args.ladder:
        print(f"Searching for {args.num_battles} {main_format} ladder battles...")
        tasks.append(player_ou.ladder(args.num_battles))
    else:
        print(f"Bot online: {username} ({main_format})")
        print(f"Challenge with: /challenge {username}, {main_format}")
        tasks.append(player_ou.accept_challenges(None, args.num_battles))

    if player_rb is not None:
        if args.keep_ladder_battles:
            tasks.append(_maintain_ladder_battles(
                player_rb,
                target_active=args.keep_ladder_battles,
                label=f"{random_format} as {username_rb}",
            ))
        else:
            print(f"Searching for {args.num_battles} {random_format} ladder battles as {username_rb}...")
            tasks.append(player_rb.ladder(args.num_battles))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        SimpleWorldModelPlayer._stop_repl()
        print(f"\nResults for {username} ({main_format}):")
        print(f"  Wins: {player_ou.n_won_battles}  Losses: {player_ou.n_lost_battles}  Ties: {player_ou.n_tied_battles}")
        if player_rb is not None:
            print(f"Results for {username_rb} ({random_format}):")
            print(f"  Wins: {player_rb.n_won_battles}  Losses: {player_rb.n_lost_battles}  Ties: {player_rb.n_tied_battles}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.")
