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

from metamon.env import (
    BattleAgainstBaseline,
    QueueOnLocalLadder,
    get_metamon_teams,
    TeamSet,
)
from metamon.rl.metamon_to_amago import (
    PSLadderAMAGOWrapper,
    MetamonAMAGOWrapper,
    MetamonAMAGOExperiment,
)
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    ALL_OBSERVATION_SPACES,
    ALL_REWARD_FUNCTIONS,
    DefaultObservationSpace,
    DefaultShapedReward,
    TokenizedObservationSpace,
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

WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")

if METAMON_CACHE_DIR is None:
    raise ValueError("Set METAMON_CACHE_DIR environment variable")
# downloads checkpoints to the metamon cache dir where we're putting all the other data
MODEL_DOWNLOAD_DIR = os.path.join(METAMON_CACHE_DIR, "pretrained_models")


def make_placeholder_env(observation_space: ObservationSpace):
    """
    Create an environment that does nothing, but will be used to initialize the network
    """

    class _PlaceholderShowdown(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = observation_space.gym_space
            self.action_space = gym.spaces.Discrete(9)
            self.metamon_battle_format = "PlaceholderShowdown"
            self.metamon_opponent_name = "PlaceholderOpponent"

        def reset(self, *args, **kwargs):
            obs = {
                key: np.zeros(value.shape, dtype=value.dtype)
                for key, value in self.observation_space.items()
            }
            return obs, {}

        def take_long_break(self):
            pass

        def resume_from_break(self):
            pass

    env = _PlaceholderShowdown()
    return MetamonAMAGOWrapper(env)


def make_ladder_env(
    battle_format: str,
    player_team_set: TeamSet,
    observation_space: ObservationSpace,
    reward_function: RewardFunction,
    num_battles: int,
    username: str,
    avatar: str,
    save_trajectories_to: Optional[str] = None,
):
    """
    Battle on the local Showdown ladder
    """
    env = QueueOnLocalLadder(
        battle_format=battle_format,
        num_battles=num_battles,
        observation_space=observation_space,
        reward_function=reward_function,
        player_team_set=player_team_set,
        player_username=username,
        player_avatar=avatar,
        save_trajectories_to=save_trajectories_to,
    )
    return PSLadderAMAGOWrapper(env)


def make_baseline_env(
    battle_format: str,
    player_team_set: TeamSet,
    observation_space: ObservationSpace,
    reward_function: RewardFunction,
    opponent_type: Type[Player],
    save_trajectories_to: Optional[str] = None,
):
    """
    Battle against a built-in baseline opponent
    """
    env = BattleAgainstBaseline(
        battle_format=battle_format,
        observation_space=observation_space,
        reward_function=reward_function,
        team_set=player_team_set,
        opponent_type=opponent_type,
        turn_limit=200,
        save_trajectories_to=save_trajectories_to,
    )
    return MetamonAMAGOWrapper(env)


def create_placeholder_experiment(
    ckpt_base_dir: str,
    run_name: str,
    log: bool,
    observation_space: ObservationSpace,
):
    """
    Initialize an AMAGO experiment that will be used to load a pretrained checkpoint
    and manage agent/env interaction.
    """
    # the environment is only used to initialize the network
    # before loading the correct checkpoint
    env = make_placeholder_env(
        observation_space=observation_space,
    )
    dummy_dset = amago.loading.DoNothingDataset()
    dummy_env = lambda: env
    experiment = MetamonAMAGOExperiment(
        # assumes that positional args
        # agent_type, tstep_encoder_type,
        # traj_encoder_type, and max_seq_len
        # are set in the gin file
        ckpt_base_dir=ckpt_base_dir,
        run_name=run_name,
        dataset=dummy_dset,
        make_train_env=dummy_env,
        make_val_env=dummy_env,
        env_mode="async",
        async_env_mp_context="spawn",
        parallel_actors=1,
        exploration_wrapper_type=None,
        epochs=0,
        start_learning_at_epoch=float("inf"),
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        stagger_traj_file_lengths=False,
        train_batches_per_epoch=0,
        val_interval=None,
        val_timesteps_per_epoch=0,
        ckpt_interval=None,
        always_save_latest=False,
        always_load_latest=False,
        log_interval=1,
        batch_size=1,
        dloader_workers=0,
        log_to_wandb=log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        verbose=True,
    )
    return experiment


class PretrainedModel:
    """
    Create an AMAGO agent and load a pretrained checkpoint from the HuggingFace Hub
    """

    HF_REPO_ID = "jakegrigsby/metamon"
    DEFAULT_CKPT = 40  # a.k.a. 1M grad steps

    # fmt: off
    def __init__(
        self,
        # gin files modify the model architecture (layers, size, etc.)
        gin_config : str,
        # model name is used to identify the model in the HuggingFace Hub
        model_name: str,
        # whether the model is an IL model (vs RL) (IL expects slightly less params)
        is_il_model: bool,
        # tokenize the text component of the observation space
        tokenizer: PokemonTokenizer = get_tokenizer("allreplays-v3"),
        # use original paper observation space and reward function
        observation_space: ObservationSpace = DefaultObservationSpace(),
        reward_function: RewardFunction = DefaultShapedReward(),
        # cache directory for the HuggingFace Hub (note that these files are large)
        hf_cache_dir: Optional[str] = None,
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
        self.reward_function = reward_function
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

    def initialize_agent(self, checkpoint: Optional[int] = None, log: bool = False) -> amago.Experiment:
        # use the base config and the gin file to configure the model
        cli_utils.use_config(self.base_config, [self.gin_config], finalize=False)
        checkpoint = checkpoint or self.DEFAULT_CKPT
        # Download checkpoint from HF Hub
        checkpoint_path = hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=f"{self.model_name}/ckpts/policy_weights/policy_epoch_{checkpoint}.pt",
            cache_dir=self.hf_cache_dir,
        )
        model_dir = Path(os.path.dirname(os.path.dirname(checkpoint_path)))
        ckpt_base_dir = str(model_dir.parents[1])
        # build an experiment
        experiment = create_placeholder_experiment(
            ckpt_base_dir=ckpt_base_dir,
            run_name=self.model_name,
            log=log,
            observation_space=self.observation_space,
        )
        # starting the experiment will build the initial model
        experiment.start()
        if checkpoint > 0:
            # replace the weights with the pretrained checkpoint
            experiment.load_checkpoint(checkpoint, resume_training_state=False)
        return experiment


class SmallIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il",
            gin_config="models/small_agent.gin",
            is_il_model=True,
        )


class SmallILFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il-filled-actions",
            gin_config="models/small_agent.gin",
            is_il_model=True,
        )


class SmallRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl",
            gin_config="models/small_agent.gin",
            is_il_model=False,
        )


class SmallRL_ExtremeFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-exp-extreme",
            gin_config="models/small_agent.gin",
            is_il_model=False,
        )


class SmallRL_BinaryFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-binary",
            gin_config="models/small_agent.gin",
            is_il_model=False,
        )


class SmallRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-aug",
            gin_config="models/small_agent.gin",
            is_il_model=False,
        )


class SmallRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-maxq",
            gin_config="models/small_agent.gin",
            is_il_model=False,
        )


class MediumIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-il",
            gin_config="models/medium_agent.gin",
            is_il_model=True,
        )


class MediumRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl",
            gin_config="models/medium_agent.gin",
            is_il_model=False,
        )


class MediumRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-aug",
            gin_config="models/medium_agent.gin",
            is_il_model=False,
        )


class MediumRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-maxq",
            gin_config="models/medium_agent.gin",
            is_il_model=False,
        )


class LargeRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-rl",
            gin_config="models/large_agent.gin",
            is_il_model=False,
        )


class LargeIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-il",
            gin_config="models/large_agent.gin",
            is_il_model=True,
        )


class SyntheticRLV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v0",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
        )


class SyntheticRLV1(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
        )


class SyntheticRLV1_SelfPlay(PretrainedModel):
    DEFAULT_CKPT = 48

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1+sp",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
        )


class SyntheticRLV1_PlusPlus(PretrainedModel):
    DEFAULT_CKPT = 38

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1++",
            gin_config="models/synthetic_agent.gin",
            is_il_model=False,
        )


class SyntheticRLV2(PretrainedModel):
    DEFAULT_CKPT = 48

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v2",
            gin_config="models/synthetic_multitaskagent.gin",
            is_il_model=False,
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
        help="Specify the generations to evaluate.",
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Log results to Weights & Biases."
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default="ou",
        choices=["ubers", "ou", "uu", "nu"],
        help="Specify the battle formats.",
    )
    parser.add_argument(
        "--username", default="Metamon", help="Username for the Showdown server."
    )
    parser.add_argument(
        "--n_challenges", type=int, default=10, help="Number of battles to run."
    )
    parser.add_argument(
        "--avatar", default="red-gen1main", help="Avatar to use for the battles."
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
        choices=[
            "heuristic",
            "il",
            "ladder",
        ],
        help="Type of evaluation to perform. 'heuristic' will run the agent against the heuristic baselines, 'il' will run the agent against the IL baselines, 'local-ladder' will run the agent on your self-hosted Showdown ladder. If you set two agents to play on the local-ladder, they will be battling each other!",
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
                    reward_function=agent_maker.reward_function,
                    save_trajectories_to=args.save_trajectories_to,
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
