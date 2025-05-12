import os
import multiprocessing as mp
from functools import partial

import wandb

import amago
from amago.cli_utils import *
from amago.agent import binary_filter, exp_filter

import metamon
from metamon.env import MetaShowdown
from metamon.task_distributions import FixedGenOpponentDistribution
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    TokenizedObservationSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)
from metamon.task_distributions import get_task_distribution
from metamon.data.tokenizer import get_tokenizer
from metamon.data.replay_dataset.parsed_replays.loading import ParsedReplayDataset
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonAMAGOWrapper,
    MetamonTstepEncoder,
)
from metamon import baselines


def add_cli(parser):
    # fmt: off
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    parser.add_argument("--reward_function", type=str, default="DefaultShapedReward")
    parser.add_argument("--parsed_replay_dir", required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--batch_size_per_gpu", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--il", action="store_true")
    parser.add_argument("--arch_size", required=True, choices=["small", "medium", "large"])
    parser.add_argument("--token_aug", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="allreplays-v3")
    parser.add_argument("--log", action="store_true")
    # fmt: on
    return parser


live_opponents = [
    baselines.heuristic.basic.PokeEnvHeuristic,
    baselines.heuristic.basic.Gen1BossAI,
    baselines.heuristic.basic.Grunt,
    baselines.heuristic.basic.GymLeader,
    baselines.heuristic.kaizo.EmeraldKaizo,
]


def make_baseline_env(
    gen,
    format,
    observation_space: ObservationSpace,
    reward_function: RewardFunction,
    player_split: str,
    opponent_split: str,
    opponent,
):
    """
    Battle against a built-in baseline opponent
    """
    env = MetaShowdown(
        task_distribution=FixedGenOpponentDistribution(
            format=f"gen{gen}{format}",
            opponent=opponent,
            player_split=player_split,
            opponent_split=opponent_split,
            reward_function=reward_function,
        ),
        new_task_every=1,
        observation_space=observation_space,
    )
    return MetamonAMAGOWrapper(env)


def configure(args):
    config = {
        "amago.agent.Agent.reward_multiplier": 10.0,
        "amago.agent.Agent.offline_coeff": 1.0,
        "amago.agent.Agent.online_coeff": 0.0,
        "amago.agent.Agent.fake_filter": args.il,
        "amago.agent.Agent.use_multigamma": not args.il,
        "amago.agent.Agent.fbc_filter_func": binary_filter,
        "amago.agent.Agent.tau": 0.004,
        "MetamonTstepEncoder.token_mask_aug": args.token_aug,
        "MetamonTstepEncoder.tokenizer": get_tokenizer(args.tokenizer),
        "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.VanillaAttention,
    }
    config_file = os.path.join(
        os.path.dirname(__file__), "configs", f"{args.arch_size}.gin"
    )
    use_config(config, [config_file])


if __name__ == "__main__":
    from argparse import ArgumentParser

    mp.set_start_method("spawn")
    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()
    configure(args)

    obs_space = TokenizedObservationSpace(
        DefaultObservationSpace(), get_tokenizer(args.tokenizer)
    )
    reward_function = DefaultShapedReward()
    parsed_replay_dataset = ParsedReplayDataset(
        dset_root=args.parsed_replay_dir,
        observation_space=obs_space,
        reward_function=reward_function,
    )

    make_envs = [
        partial(
            make_baseline_env,
            gen=i,
            format="ou",
            observation_space=obs_space,
            reward_function=reward_function,
            player_split="train",
            opponent_split="train",
            opponent=opponent,
        )
        for i in range(1, 5)
        for opponent in live_opponents
    ]

    experiment = MetamonAMAGOExperiment(
        run_name=args.run_name,
        ckpt_dir=args.ckpt_dir,
        make_train_env=make_envs,
        make_val_env=make_envs,
        parsed_replay_dataset=parsed_replay_dataset,
        log_to_wandb=args.log,
        train_batches_per_epoch=25_000 * args.grad_accum,
        batches_per_update=args.grad_accum,
        batch_size=args.batch_size_per_gpu,
    )

    experiment.start()
    if args.ckpt is not None:
        experiment.load_checkpoint(args.ckpt)
    experiment.learn()
    wandb.finish()
