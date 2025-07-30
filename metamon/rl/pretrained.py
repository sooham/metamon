import os
from pathlib import Path
import json
from functools import partial
import warnings
from typing import Optional

warnings.filterwarnings("ignore")


def red_warning(msg: str):
    print(f"\033[91m{msg}\033[0m")


import huggingface_hub
import torch
import amago

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
    ExpandedObservationSpace,
    DefaultShapedReward,
    TokenizedObservationSpace,
    ActionSpace,
    DefaultActionSpace,
    MinimalActionSpace,
)
from metamon.baselines import ALL_BASELINES

# Import baseline modules to ensure registration happens
from metamon.baselines import heuristic, model_based
from metamon.tokenizer import PokemonTokenizer, get_tokenizer
from metamon import METAMON_CACHE_DIR


def get_heuristic_baselines():
    """Get all heuristic baseline classes from the registration system."""
    # Filter for heuristic baselines (exclude model-based ones like BCRNN)
    heuristic_names = [
        "RandomBaseline",
        "PokeEnvHeuristic",
        "Gen1BossAI",
        "Grunt",
        "GymLeader",
        "EmeraldKaizo",
    ]
    return [ALL_BASELINES[name] for name in heuristic_names if name in ALL_BASELINES]


def get_il_baselines():
    """Get all imitation learning baseline classes from the registration system."""
    # Filter for IL baselines
    il_names = ["BaseRNN", "WinsOnlyRNN", "MiniRNN"]
    return [ALL_BASELINES[name] for name in il_names if name in ALL_BASELINES]


# Get baselines from registration system
HEURISTIC_COMPOSITE_BASELINES = get_heuristic_baselines()
IL = get_il_baselines()

if METAMON_CACHE_DIR is None:
    raise ValueError("Set METAMON_CACHE_DIR environment variable")
# downloads checkpoints to the metamon cache dir where we're putting all the other data
MODEL_DOWNLOAD_DIR = os.path.join(METAMON_CACHE_DIR, "pretrained_models")

# Registry for pretrained models
ALL_PRETRAINED_MODELS = {}


def pretrained_model(name: Optional[str] = None):
    """
    Decorator to register pretrained model classes.

    Args:
        name: Optional custom name for the model. If not provided, uses the class name.

    Usage:
        @pretrained_model()
        class MyModel(PretrainedModel):
            pass

        @pretrained_model("CustomName")
        class AnotherModel(PretrainedModel):
            pass
    """

    def _register(cls):
        model_name = name if name is not None else cls.__name__
        if model_name in ALL_PRETRAINED_MODELS:
            raise ValueError(f"Pretrained model '{model_name}' is already registered!")
        ALL_PRETRAINED_MODELS[model_name] = cls
        return cls

    return _register


def get_pretrained_model_names():
    return sorted(ALL_PRETRAINED_MODELS.keys())


def get_pretrained_model(name: str):
    """Get a pretrained model class by name."""
    if name not in ALL_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model '{name}' (available models: {get_pretrained_model_names()})"
        )
    return ALL_PRETRAINED_MODELS[name]


class PretrainedModel:
    """
    Create an AMAGO agent and load a pretrained checkpoint from the HuggingFace Hub
    """

    HF_REPO_ID = "jakegrigsby/metamon"

    # fmt: off
    def __init__(
        self,
        # gin files modify the model architecture (layers, size, etc.)
        model_gin_config : str,
        # training gin file does not have to be 1:1 with training, but should match any architecture changes that were used
        train_gin_config : str,
        # model name is used to identify the model in the HuggingFace Hub
        model_name: str,
        # tokenize the text component of the observation space
        tokenizer: PokemonTokenizer = get_tokenizer("allreplays-v3"),
        # use original paper observation space and reward function
        # (paper action space is now called MinimalActionSpace)
        observation_space: ObservationSpace = DefaultObservationSpace(),
        action_space: ActionSpace = DefaultActionSpace(),
        reward_function: RewardFunction = DefaultShapedReward(),
        # cache directory for the HuggingFace Hub (note that these files are large)
        hf_cache_dir: Optional[str] = None,
        default_checkpoint: int = 40,  # a.k.a. 1M grad steps w/ original paper training settings
    ):
    # fmt: on

        self.model_name = model_name
        self.model_gin_config = os.path.join(os.path.dirname(__file__), "configs", model_gin_config)
        self.train_gin_config = os.path.join(os.path.dirname(__file__), "configs", train_gin_config)
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
            "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": attn_type,
            "MetamonTstepEncoder.tokenizer": self.tokenizer,
            # skip cpu-intensive init, because we're going to be replacing the weights
            # with a checkpoint anyway....
            "amago.nets.transformer.SigmaReparam.fast_init": True,
        }
    
    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        # Download checkpoint from HF Hub
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=f"{self.model_name}/ckpts/policy_weights/policy_epoch_{checkpoint}.pt",
            cache_dir=self.hf_cache_dir,
        )
        return checkpoint_path

    def initialize_agent(self, checkpoint: Optional[int] = None, log: bool = False) -> amago.Experiment:
        # use the base config and the gin file to configure the model
        amago.cli_utils.use_config(self.base_config, [self.model_gin_config, self.train_gin_config], finalize=False)
        checkpoint = checkpoint if checkpoint is not None else self.default_checkpoint
        ckpt_path = self.get_path_to_checkpoint(checkpoint)
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


@pretrained_model()
class SmallIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_il.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SmallILFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il-filled-actions",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_il.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SmallRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SmallRL_ExtremeFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-exp-extreme",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SmallRL_BinaryFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-binary",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SmallRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-aug",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SmallRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-maxq",
            model_gin_config="models/small_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class MediumIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-il",
            model_gin_config="models/medium_agent.gin",
            train_gin_config="training/base_il.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class MediumRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl",
            model_gin_config="models/medium_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class MediumRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-aug",
            model_gin_config="models/medium_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class MediumRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-maxq",
            model_gin_config="models/medium_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class LargeRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-rl",
            model_gin_config="models/large_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class LargeIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-il",
            model_gin_config="models/large_agent.gin",
            train_gin_config="training/base_il.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SyntheticRLV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v0",
            model_gin_config="models/synthetic_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SyntheticRLV1(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1",
            model_gin_config="models/synthetic_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=40,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SyntheticRLV1_SelfPlay(PretrainedModel):

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1+sp",
            model_gin_config="models/synthetic_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=48,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SyntheticRLV1_PlusPlus(PretrainedModel):

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1++",
            model_gin_config="models/synthetic_agent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=38,
            action_space=MinimalActionSpace(),
        )


@pretrained_model()
class SyntheticRLV2(PretrainedModel):

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v2",
            model_gin_config="models/synthetic_multitaskagent.gin",
            train_gin_config="training/base_offline_rl.gin",
            default_checkpoint=48,
            action_space=MinimalActionSpace(),
        )
