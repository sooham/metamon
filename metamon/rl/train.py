import os
import warnings
import multiprocessing as mp
from functools import partial

import wandb

import amago
from amago import cli_utils
from amago.agent import binary_filter, exp_filter
from amago.utils import AmagoWarning

from metamon.env import BattleAgainstBaseline, TeamSet, get_metamon_teams
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    TokenizedObservationSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)
from metamon.tokenizer import get_tokenizer
from metamon.datasets import ParsedReplayDataset
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonAMAGOWrapper,
    MetamonAMAGODataset,
    MetamonTstepEncoder,
)
from metamon import baselines


def add_cli(parser):
    # fmt: off
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    parser.add_argument("--reward_function", type=str, default="DefaultShapedReward")
    parser.add_argument("--parsed_replay_dir", type=str, default=None, help="Path to the parsed replay directory. Defaults to the official huggingface version.")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--batch_size_per_gpu", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--il", action="store_true")
    parser.add_argument("--arch_size", required=True, choices=["small", "medium", "large", "synthetic"])
    parser.add_argument("--token_aug", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="allreplays-v3")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--agent_type", type=str, default="agent", choices=["agent", "multitask"])
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
    battle_format: str,
    observation_space: ObservationSpace,
    reward_function: RewardFunction,
    team_set: TeamSet,
    opponent,
):
    """
    Battle against a built-in baseline opponent
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=AmagoWarning)
    env = BattleAgainstBaseline(
        battle_format=battle_format,
        observation_space=observation_space,
        reward_function=reward_function,
        team_set=team_set,
        opponent_type=opponent,
    )
    return MetamonAMAGOWrapper(env)


def configure(args):
    """
    This is all customizable. When we've trained a model we like, we can recover
    the config from wandb or the config.txt in the checkpoint directory to set up
    an inference checkpoint.
    """
    config = {
        "MetamonTstepEncoder.token_mask_aug": args.token_aug,
        "MetamonTstepEncoder.tokenizer": get_tokenizer(args.tokenizer),
        # change to FlashAttention if possible (or any of the other options)
        "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.VanillaAttention,
    }
    agent_type = cli_utils.switch_agent(
        config=config,
        agent=args.agent_type,
        reward_multiplier=10.0,
        offline_coeff=1.0,
        online_coeff=0.0,
        fake_filter=args.il,
        use_multigamma=not args.il,
        fbc_filter_func=binary_filter,
        tau=0.004,
        num_actions_for_value_in_critic_loss=5,
    )
    config_file = os.path.join(
        os.path.dirname(__file__), "configs", f"{args.arch_size}.gin"
    )
    cli_utils.use_config(config, [config_file])
    return agent_type


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()
    agent_type = configure(args)

    obs_space = TokenizedObservationSpace(
        DefaultObservationSpace(), get_tokenizer(args.tokenizer)
    )
    reward_function = DefaultShapedReward()
    parsed_replay_dataset = ParsedReplayDataset(
        dset_root=args.parsed_replay_dir,
        observation_space=obs_space,
        reward_function=reward_function,
    )

    amago_dataset = MetamonAMAGODataset(
        dset_name="Metamon Parsed Replays",
        parsed_replay_dset=parsed_replay_dataset,
    )

    make_envs = [
        partial(
            make_baseline_env,
            battle_format=f"gen{i}ou",
            observation_space=obs_space,
            reward_function=reward_function,
            team_set=get_metamon_teams(f"gen{i}ou", "paper_variety"),
            opponent=opponent,
        )
        for i in range(1, 5)
        for opponent in live_opponents
    ]
    experiment = MetamonAMAGOExperiment(
        run_name=args.run_name,
        agent_type=agent_type,
        ckpt_base_dir=args.ckpt_dir,
        make_train_env=make_envs,
        make_val_env=make_envs,
        dataset=amago_dataset,
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
