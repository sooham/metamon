import os
import warnings
import multiprocessing as mp
from functools import partial

import wandb

import amago
from amago import cli_utils
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
)
from metamon import baselines


WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")


def add_cli(parser):
    # fmt: off
    parser.add_argument("--run_name", required=True, help="Give the run a name to identify logs and checkpoints.")
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    parser.add_argument("--reward_function", type=str, default="DefaultShapedReward")
    parser.add_argument("--parsed_replay_dir", type=str, default=None, help="Path to the parsed replay directory. Defaults to the official huggingface version.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to save checkpoints. Find checkpoints under {ckpt_dir}/{run_name}/ckpts/")
    parser.add_argument("--ckpt", type=int, default=None, help="Resume training from an existing run with this run_name. Provide the epoch checkpoint to load.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=12, help="Batch size per GPU. Total batch size is batch_size_per_gpu * num_gpus.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulations per update.")
    parser.add_argument("--il", action="store_true", help="Overrides amago settings to use imitation learning.")
    parser.add_argument("--model_gin_config", type=str, required=True, help="Path to a gin config file (that might edit the model architecture). See provided rl/configs/models/)")
    parser.add_argument("--train_gin_config", type=str, required=True, help="Path to a gin config file (that might edit the training or hparams).")
    parser.add_argument("--tokenizer", type=str, default="DefaultObservationSpace-v0", help="The tokenizer to use for the text observation space. See metamon.tokenizer for options.")
    parser.add_argument("--log", action="store_true", help="Log to wandb.")
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
        "MetamonTstepEncoder.tokenizer": get_tokenizer(args.tokenizer),
        "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.VanillaAttention,
    }
    if args.il:
        # NOTE: would break for a custom agent, but ultimately just creates some wasted params that aren't trained
        config.update(
            {
                "amago.agent.Agent.use_multigamma": False,
                "amago.agent.MultiTaskAgent.use_multigamma": False,
                "amago.agent.Agent.fake_filter": True,
                "amago.agent.MultiTaskAgent.fake_filter": True,
            }
        )
    cli_utils.use_config(config, [args.model_gin_config, args.train_gin_config])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()
    configure(args)

    # metamon dataset
    obs_space = TokenizedObservationSpace(
        DefaultObservationSpace(), get_tokenizer(args.tokenizer)
    )
    reward_function = DefaultShapedReward()
    parsed_replay_dataset = ParsedReplayDataset(
        dset_root=args.parsed_replay_dir,
        observation_space=obs_space,
        reward_function=reward_function,
        verbose=True,
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
        for i in range(4, 5)
        for opponent in live_opponents
    ]
    experiment = MetamonAMAGOExperiment(
        ## required ##
        run_name=args.run_name,
        ckpt_base_dir=args.ckpt_dir,
        # max_seq_len = should be set in the gin file
        dataset=amago_dataset,
        # tstep_encoder_type = should be set in the gin file
        # traj_encoder_type = should be set in the gin file
        # agent_type = should be set in the gin file
        val_timesteps_per_epoch=200,
        ## environment ##
        make_train_env=make_envs,
        make_val_env=make_envs,
        env_mode="async",
        async_env_mp_context="spawn",
        parallel_actors=len(make_envs),
        exploration_wrapper_type=None,
        sample_actions=True,
        force_reset_train_envs_every=None,
        ## logging ##
        log_to_wandb=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        verbose=True,
        log_interval=300,
        ## replay ##
        padded_sampling="none",
        dloader_workers=8,
        ## learning schedule ##
        epochs=100,
        start_learning_at_epoch=0,
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        train_batches_per_epoch=25_000 * args.grad_accum,
        val_interval=1,
        ckpt_interval=2,
        ## optimization ##
        batch_size=args.batch_size_per_gpu,
        batches_per_update=args.grad_accum,
        learning_rate=1.5e-4,
        critic_loss_weight=10.0,
        lr_warmup_steps=1000,
        grad_clip=1.5,
        l2_coeff=1e-4,
        mixed_precision="no",
    )

    experiment.start()
    if args.ckpt is not None:
        experiment.load_checkpoint(args.ckpt)
    experiment.learn()
    wandb.finish()
