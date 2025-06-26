import os
from pathlib import Path
import json
from functools import partial
import multiprocessing as mp
import warnings
from typing import Type, Optional

warnings.filterwarnings("ignore")


def red_warning(msg: str):
    print(f"\033[91m{msg}\033[0m")


import gymnasium as gym
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import amago
from amago import cli_utils

from metamon.env import get_metamon_teams
from metamon.rl.metamon_to_amago import (
    make_placeholder_experiment,
    make_baseline_env,
    make_ladder_env,
)
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    ALL_OBSERVATION_SPACES,
    ALL_REWARD_FUNCTIONS,
    DefaultObservationSpace,
    DefaultShapedReward,
    TokenizedObservationSpace,
    ActionSpace,
    DefaultActionSpace,
    MinimalActionSpace,
)
from metamon.baselines.heuristic.basic import *
from metamon.baselines.heuristic.kaizo import EmeraldKaizo
from metamon.baselines.model_based.bcrnn_baselines import BaseRNN, WinsOnlyRNN, MiniRNN
from metamon.tokenizer import PokemonTokenizer, get_tokenizer
from metamon.download import METAMON_CACHE_DIR

HEURISTIC_COMPOSITE_BASELINES = [
    RandomBaseline,
    PokeEnvHeuristic,
    Gen1BossAI,
    Grunt,
    GymLeader,
    EmeraldKaizo,
]

IL = [BaseRNN]

if METAMON_CACHE_DIR is None:
    raise ValueError("Set METAMON_CACHE_DIR environment variable")
# downloads checkpoints to the metamon cache dir where we're putting all the other data
MODEL_DOWNLOAD_DIR = os.path.join(METAMON_CACHE_DIR, "pretrained_models")


class PretrainedModel:
    """
    Create an AMAGO agent and load a pretrained checkpoint from the HuggingFace Hub
    """

    HF_REPO_ID = "jakegrigsby/metamon"

    # fmt: off
    def __init__(
        self,
        # gin files modify the model architecture (layers, size, etc.)
        gin_config : str,
        # model name is used to identify the model in the HuggingFace Hub
        model_name: str,
        # whether the model is an IL model (vs RL) (IL expects fewer params)
        is_il_model: bool,
        # tokenize the text component of the observation space
        tokenizer: PokemonTokenizer = get_tokenizer("allreplays-v3"),
        # use original paper observation space and reward function
        observation_space: ObservationSpace = DefaultObservationSpace(),
        action_space: ActionSpace = DefaultActionSpace(),
        reward_function: RewardFunction = DefaultShapedReward(),
        # cache directory for the HuggingFace Hub (note that these files are large)
        hf_cache_dir: Optional[str] = None,
        default_checkpoint: int = 40,  # a.k.a. 1M grad steps w/ original paper training settings
    ):
    # fmt: on

        self.model_name = model_name
        self.gin_config = os.path.join(os.path.dirname(__file__), "configs", gin_config)
        self.is_il_model = is_il_model
        self.hf_cache_dir = hf_cache_dir or MODEL_DOWNLOAD_DIR
        self.tokenizer = tokenizer
        self.observation_space = TokenizedObservationSpace(
            base_obs_space=observation_space,
            tokenizer=tokenizer,
        )
        self.action_space = action_space
        self.reward_function = reward_function
        self.default_checkpoint = default_checkpoint
        os.makedirs(self.hf_cache_dir, exist_ok=True)

    @property
    def base_config(self) -> dict:
        has_gpu = torch.cuda.is_available()
        try:
            import flash_attn
            has_flash_attn = True
        except ImportError:
            has_flash_attn = False
        if has_flash_attn and has_gpu:
            attn_type = amago.nets.transformer.FlashAttention
        else:
            attn_type = amago.nets.transformer.VanillaAttention
            red_warning("Warning: Using unofficial VanillaAttention implementation")
        return {
            # NOTE: assumes the pretrained models in this file are using
            # built-in agents we know about and can change the settings for...
            "amago.agent.Agent.fake_filter": self.is_il_model,
            "amago.agent.MultiTaskAgent.fake_filter": self.is_il_model,
            "amago.agent.Agent.use_multigamma": not self.is_il_model,
            "amago.agent.MultiTaskAgent.use_multigamma": not self.is_il_model,
            # attention and tokenizer
            "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": attn_type,
            "MetamonTstepEncoder.tokenizer": self.tokenizer,
            # skip cpu-intensive init, because we're going to be replacing the weights
            # with a checkpoint anyway.... If you get an error about this, pull `amago`.
            "amago.nets.transformer.SigmaReparam.fast_init": True,
        }
    
    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        # Download checkpoint from HF Hub
        checkpoint_path = hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=f"{self.model_name}/ckpts/policy_weights/policy_epoch_{checkpoint}.pt",
            cache_dir=self.hf_cache_dir,
        )
        return checkpoint_path

    def initialize_agent(self, checkpoint: Optional[int] = None, log: bool = False) -> amago.Experiment:
        # use the base config and the gin file to configure the model
        cli_utils.use_config(self.base_config, [self.gin_config], finalize=False)
        checkpoint = checkpoint or self.default_checkpoint
        ckpt_path = self.get_path_to_checkpoint(checkpoint or self.default_checkpoint)
        ckpt_base_dir = str(Path(ckpt_path).parents[2])
        # build an experiment
        experiment = make_placeholder_experiment(
            ckpt_base_dir=ckpt_base_dir,
            run_name=self.model_name,
            log=log,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        # starting the experiment will build the initial model
        experiment.start()
        if checkpoint > 0:
            # replace the weights with the pretrained checkpoint
            experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)
        return experiment


class LocalPretrainedModel(PretrainedModel):
    """
    Evaluate a model from a custom training run.

    Args:
        amago_run_path: Path to the AMAGO run directory containing a config.txt,
            ckpts/, and wandb logs.
    """

    def __init__(self, amago_run_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amago_run_path = amago_run_path

    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        return os.path.join(
            self.amago_run_path,
            "ckpts",
            "policy_weights",
            f"policy_epoch_{checkpoint}.pt",
        )


class SmallIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il",
            gin_config="models/small_agent.gin",
            is_il_model=True,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SmallILFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il-filled-actions",
            gin_config="models/small_agent.gin",
            is_il_model=True,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SmallRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl",
            gin_config="models/small_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SmallRL_ExtremeFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-exp-extreme",
            gin_config="models/small_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SmallRL_BinaryFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-binary",
            gin_config="models/small_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SmallRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-aug",
            gin_config="models/small_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SmallRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-maxq",
            gin_config="models/small_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class MediumIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-il",
            gin_config="models/medium_agent.gin",
            is_il_model=True,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class MediumRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl",
            gin_config="models/medium_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class MediumRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-aug",
            gin_config="models/medium_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class MediumRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-maxq",
            gin_config="models/medium_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class LargeRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-rl",
            gin_config="models/large_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class LargeIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-il",
            gin_config="models/large_agent.gin",
            is_il_model=True,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SyntheticRLV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v0",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SyntheticRLV1(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


class SyntheticRLV1_SelfPlay(PretrainedModel):

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1+sp",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
            default_checkpoint=48,
            action_space=MinimalActionSpace(),
        )


class SyntheticRLV1_PlusPlus(PretrainedModel):

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1++",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
            default_checkpoint=38,
            action_space=MinimalActionSpace(),
        )


class SyntheticRLV2(PretrainedModel):

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v2",
            gin_config="models/synthetic_multitaskagent.gin",
            is_il_model=False,
            default_checkpoint=48,
            action_space=MinimalActionSpace(),
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--agent",
        required=True,
        choices=[
            "SmallIL",
            "SmallILFA",
            "SmallRL",
            "SmallRL_ExtremeFilter",
            "SmallRL_BinaryFilter",
            "SmallRL_Aug",
            "SmallRL_MaxQ",
            "SmallRLPostPaper",
            "MediumIL",
            "MediumRL",
            "MediumRL_Aug",
            "MediumRL_MaxQ",
            "LargeIL",
            "LargeRL",
            "SyntheticRLV0",
            "SyntheticRLV1",
            "SyntheticRLV1_SelfPlay",
            "SyntheticRLV1_PlusPlus",
            "SyntheticRLV2",
        ],
        help="Choose a pretrained model to evaluate.",
    )
    parser.add_argument(
        "--gens",
        type=int,
        nargs="+",
        default=1,
        help="Specify the Pokémon generations to evaluate.",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default="ou",
        choices=["ubers", "ou", "uu", "nu"],
        help="Specify the battle format/tier.",
    )
    parser.add_argument(
        "--username",
        default="Metamon",
        help="Username for the Showdown server.",
    )
    parser.add_argument(
        "--n_challenges",
        type=int,
        default=10,
        help=(
            "Number of battles to run before returning eval stats. "
            "Note this is the total sample size across all parallel actors."
        ),
    )
    parser.add_argument(
        "--avatar",
        default="red-gen1main",
        help="Avatar to use for the battles.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[None],
        help="Checkpoints to evaluate.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["heuristic", "il", "ladder"],
        help=(
            "Type of evaluation to perform. 'heuristic' will run against 6 "
            "heuristic baselines, 'il' will run against a BCRNN baseline, "
            "'ladder' will queue the agent for battles on your self-hosted Showdown ladder."
        ),
    )
    parser.add_argument(
        "--team_set",
        default="competitive",
        choices=["competitive", "paper_variety", "paper_replays", "modern_replays"],
        help="Team Set.",
    )
    parser.add_argument(
        "--save_trajectories_to",
        default=None,
        help="Save replays (in the parsed replay format) to a directory.",
    )
    parser.add_argument(
        "--wait_for_input",
        action="store_true",
        help="Wait for user input before starting.",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help=(
            "Method for interpreting Showdown's requests and simulator messages. "
            "poke-env is the default. metamon is an experimental option that aims to "
            "remove sim2sim gap by reusing the code that generates our huggingface "
            "replay dataset."
        ),
    )
    args = parser.parse_args()

    agent_maker = eval(args.agent)()

    for gen in args.gens:
        for format in args.formats:
            battle_format = f"gen{gen}{format.lower()}"
            player_team_set = get_metamon_teams(battle_format, args.team_set)
            for checkpoint in args.checkpoints:
                agent = agent_maker.initialize_agent(
                    checkpoint=checkpoint, log=args.log_to_wandb
                )
                # create envs
                env_kwargs = dict(
                    battle_format=battle_format,
                    player_team_set=player_team_set,
                    observation_space=agent_maker.observation_space,
                    action_space=agent_maker.action_space,
                    reward_function=agent_maker.reward_function,
                    save_trajectories_to=args.save_trajectories_to,
                    battle_backend=args.battle_backend,
                )
                if args.eval_type == "heuristic":
                    make_envs = [
                        partial(make_baseline_env, **env_kwargs, opponent_type=o)
                        for o in HEURISTIC_COMPOSITE_BASELINES
                    ]
                    make_envs *= 5
                elif args.eval_type == "il":
                    make_envs = [
                        partial(make_baseline_env, **env_kwargs, opponent_type=o)
                        for o in IL
                    ]
                    make_envs *= 1
                elif args.eval_type == "ladder":
                    agent.env_mode = "sync"
                    make_envs = [
                        partial(
                            make_ladder_env,
                            **env_kwargs,
                            num_battles=args.n_challenges + 1,
                            username=args.username,
                            avatar=args.avatar,
                        )
                    ]
                    # disables AMAGO tqdm because we'll be rendering the poke-env battle bar
                    agent.verbose = False
                else:
                    raise ValueError(f"Invalid eval_type: {args.eval_type}")

                agent.parallel_actors = len(make_envs)

                # evaluate
                results = agent.evaluate_test(
                    make_envs,
                    # sets upper bound on total timesteps
                    timesteps=args.n_challenges * 250,
                    # terminates after n_challenges
                    episodes=args.n_challenges,
                )
                print(json.dumps(results, indent=4, sort_keys=True))
