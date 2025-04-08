import multiprocessing as mp
from functools import partial

import wandb
import amago
from amago.cli_utils import *

from metamon.env import MetaShowdown, TokenizedEnv
from metamon.task_distributions import get_task_distribution
from metamon.rl.metamon_to_amago import MetamonAMAGOWrapper, MetamonTstepEncoder


def add_cli(parser):
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--task_dist", default="Tutorial")
    return parser


def make_metamon_env(task_dist):
    env = MetaShowdown(get_task_distribution(task_dist)(), new_task_every=10)
    env = TokenizedEnv(env)
    env = MetamonAMAGOWrapper(env)
    return env


def make_metamon_val_env(task_dist):
    env = MetaShowdown(get_task_distribution(task_dist)(), new_task_every=3)
    env = TokenizedEnv(env)
    env = MetamonAMAGOWrapper(env)
    return env


if __name__ == "__main__":
    from argparse import ArgumentParser

    mp.set_start_method("spawn")

    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    config = {
        "amago.agent.Agent.reward_multiplier": 10.0,
    }
    traj_encoder_type = switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    use_config(config, args.configs)

    run_name = f"{args.run_name}_{args.task_dist}_online_l_{args.max_seq_len}"
    group_name = run_name

    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        experiment = create_experiment_from_cli(
            args,
            tstep_encoder_type=MetamonTstepEncoder,
            agent_type=amago.agent.Agent,
            traj_encoder_type=traj_encoder_type,
            make_train_env=partial(make_metamon_env, task_dist=args.task_dist),
            make_val_env=partial(make_metamon_val_env, task_dist=args.task_dist),
            max_seq_len=args.max_seq_len,
            traj_save_len=1000,
            stagger_traj_file_lengths=False,
            run_name=run_name,
            group_name=group_name,
            val_timesteps_per_epoch=1500,
            wandb_entity="jakegrigsby",
            wandb_project="metamon",
        )
        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        wandb.finish()
