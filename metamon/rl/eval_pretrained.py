import os
from pathlib import Path
import json
from functools import partial
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")


def red_warning(msg: str):
    print(f"\033[91m{msg}\033[0m")


from huggingface_hub import hf_hub_download
import torch
import amago
from amago.cli_utils import *

from metamon.env import MetaShowdown, TokenizedEnv, LocalLadder
from metamon.rl.metamon_to_amago import (
    PSLadderAMAGOWrapper,
    MetamonAMAGOWrapper,
    MetamonTstepEncoder,
)
from metamon.task_distributions import (
    get_task_distribution,
    FixedGenOpponentDistribution,
)
from metamon.baselines.heuristic.basic import *
from metamon.baselines.heuristic.kaizo import EmeraldKaizo
from metamon.baselines.model_based.bcrnn_baselines import BaseRNN, WinsOnlyRNN, MiniRNN


def make_placeholder_env():
    env = MetaShowdown(
        task_distribution=get_task_distribution("AllGenOU")(
            specify_opponents=[RandomBaseline]
        ),
        new_task_every=100,
    )
    env = TokenizedEnv(env)
    return MetamonAMAGOWrapper(env)


def make_ladder_env(
    gen: int,
    format: str,
    username: str,
    avatar: str,
    n_challenges: int,
    team_split: str,
    wait_for_input: bool = False,
):
    env = LocalLadder(
        gen=gen,
        format=format,
        username=username[-18:],
        avatar=avatar,
        team_split=team_split,
    )
    if wait_for_input:
        input("Hit any key to start challenging")
    env.start_laddering(n_challenges=n_challenges)
    env = TokenizedEnv(env)
    return PSLadderAMAGOWrapper(env)


def make_baseline_env(gen, format, player_split: str, opponent_split: str, opponent):
    env = MetaShowdown(
        task_distribution=FixedGenOpponentDistribution(
            format=f"gen{gen}{format}",
            opponent=opponent,
            player_split=player_split,
            opponent_split=opponent_split,
        ),
        new_task_every=1,
    )
    env = TokenizedEnv(env)
    return MetamonAMAGOWrapper(env)


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
METAMON_CACHE_DIR = os.environ.get(
    "METAMON_CACHE_DIR", os.path.expanduser("~/.cache/metamon")
)


def _create_placeholder_experiment(
    dset_root, dset_name, run_name, max_seq_len, log, agent_type
):
    # the environment is only used to initialize the network
    # before loading the correct checkpoint
    env = make_placeholder_env()
    dummy_env = lambda: env
    experiment = amago.Experiment(
        max_seq_len=max_seq_len,
        run_name=run_name,
        traj_save_len=1000,
        make_train_env=dummy_env,
        make_val_env=dummy_env,
        parallel_actors=1,
        traj_encoder_type=amago.nets.traj_encoders.TformerTrajEncoder,
        tstep_encoder_type=MetamonTstepEncoder,
        agent_type=agent_type,
        exploration_wrapper_type=None,
        dset_root=dset_root,
        dset_name=dset_name,
        dset_max_size=float("inf"),
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
    HF_REPO_ID = "jakegrigsby/metamon"
    DEFAULT_CKPT = 40

    def __init__(
        self,
        gin_config,
        model_name,
        is_il_model,
        max_seq_len=200,
        agent_type=amago.agent.Agent,
        hf_cache_dir=None,
    ):
        self.model_name = model_name
        self.gin_config = os.path.join(os.path.dirname(__file__), "configs", gin_config)
        self.is_il_model = is_il_model
        self.max_seq_len = max_seq_len
        self.agent_type = agent_type
        self.hf_cache_dir = hf_cache_dir or METAMON_CACHE_DIR
        os.makedirs(self.hf_cache_dir, exist_ok=True)

    @property
    def base_config(self):
        has_gpu = torch.cuda.is_available()
        try:
            import flash_attn
            has_flash_attn = True
        except ImportError:
            has_flash_attn = False
        if has_flash_attn and has_gpu:
            attn_type = amago.nets.transformer.FlashAttention
            red_warning("Using FlashAttention")
        else:
            attn_type = amago.nets.transformer.VanillaAttention
            red_warning("Warning: Using unofficial VanillaAttention implementation")
        return {
            "amago.agent.Agent.reward_multiplier": 10.0,
            "amago.agent.Agent.fake_filter": self.is_il_model,
            "amago.agent.Agent.use_multigamma": not self.is_il_model,
            "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": attn_type,
        }

    def initialize_agent(self, checkpoint: Optional[int] = None, log: bool = False):
        use_config(self.base_config, [self.gin_config], finalize=False)
        checkpoint = checkpoint or self.DEFAULT_CKPT
        # Download checkpoint from HF Hub
        checkpoint_path = hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=f"{self.model_name}/ckpts/policy_weights/policy_epoch_{checkpoint}.pt",
            cache_dir=self.hf_cache_dir,
        )
        base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        full_path = Path(base_dir)
        dset_root = str(full_path.parents[2])
        dset_name = full_path.parents[1].name
        experiment = _create_placeholder_experiment(
            dset_root=dset_root,
            dset_name=dset_name,
            run_name=self.model_name,
            max_seq_len=self.max_seq_len,
            log=log,
            agent_type=self.agent_type,
        )
        experiment.start()
        if checkpoint > 0:
            experiment.load_checkpoint(checkpoint, resume_training_state=False)
        return experiment


class SmallIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il",
            gin_config="small.gin",
            is_il_model=True,
        )


class SmallILFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il-fa",
            gin_config="small.gin",
            is_il_model=True,
        )


class SmallRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl",
            gin_config="small.gin",
            is_il_model=False,
        )


class SmallRL_ExtremeFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-exp-extreme",
            gin_config="small.gin",
            is_il_model=False,
        )


class SmallRL_BinaryFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-binary",
            gin_config="small.gin",
            is_il_model=False,
        )


class SmallRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-aug",
            gin_config="small.gin",
            is_il_model=False,
        )


class SmallRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-maxq",
            gin_config="small.gin",
            is_il_model=False,
        )


class MediumIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-il",
            gin_config="medium.gin",
            is_il_model=True,
        )


class MediumRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl",
            gin_config="medium.gin",
            is_il_model=False,
        )


class MediumRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-aug",
            gin_config="medium.gin",
            is_il_model=False,
        )


class MediumRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-maxq",
            gin_config="medium.gin",
            is_il_model=False,
        )


class LargeRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-rl-aug",
            gin_config="large.gin",
            is_il_model=False,
            max_seq_len=128,
        )


class LargeIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-il",
            gin_config="large.gin",
            is_il_model=True,
            max_seq_len=128,
        )


class SyntheticRLV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v0",
            gin_config="synthetic.gin",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV1(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1",
            gin_config="synthetic.gin",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV1_SelfPlay(PretrainedModel):
    DEFAULT_CKPT = 48

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1+sp",
            gin_config="synthetic.gin",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV1_PlusPlus(PretrainedModel):
    DEFAULT_CKPT = 38

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1++",
            gin_config="synthetic.gin",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV2(PretrainedModel):
    DEFAULT_CKPT = 48

    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v2",
            gin_config="synthetic.gin",
            is_il_model=False,
            max_seq_len=128,
            agent_type=amago.agent.MultiTaskAgent,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    mp.set_start_method("spawn")

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
            "local-ladder",
        ],
        help="Type of evaluation to perform. 'heuristic' will run the agent against the heuristic baselines, 'il' will run the agent against the IL baselines, 'local-ladder' will run the agent on your self-hosted Showdown ladder. If you set two agents to play on the local-ladder, they will be battling each other!",
    )
    parser.add_argument(
        "--save_trajs_to",
        default=None,
        help="Path to save (amago-format) trajectories of completed battles.",
    )
    parser.add_argument(
        "--team_split",
        default="competitive",
        choices=["competitive", "train", "replays", "random_lead"],
        help="Team split strategy.",
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
            for checkpoint in args.checkpoints:
                agent = agent_maker.initialize_agent(
                    checkpoint=checkpoint, log=args.log_to_wandb
                )
                if args.eval_type == "heuristic":
                    make_envs = [
                        partial(
                            make_baseline_env,
                            gen=gen,
                            format=format,
                            player_split=args.team_split,
                            opponent_split=args.team_split,
                            opponent=o,
                        )
                        for o in HEURISTIC_COMPOSITE_BASELINES
                    ]
                    make_envs *= 5
                    agent.parallel_actors = len(make_envs)
                elif args.eval_type == "il":
                    make_envs = [
                        partial(
                            make_baseline_env,
                            gen=gen,
                            format=format,
                            player_split=args.team_split,
                            opponent_split=args.team_split,
                            opponent=o,
                        )
                        for o in IL
                    ]
                    make_envs *= 1
                    agent.parallel_actors = len(make_envs)
                elif "ladder" in args.eval_type:
                    make_envs = partial(
                        make_ladder_env,
                        gen=gen,
                        format=format,
                        username=args.username,
                        wait_for_input=args.wait_for_input,
                        avatar=args.avatar,
                        n_challenges=args.n_challenges + 1,
                        team_split=args.team_split,
                    )
                    agent.env_mode = "sync"
                else:
                    raise ValueError(f"Invalid eval_type: {args.eval_type}")

                agent.verbose = False
                results = agent.evaluate_test(
                    make_envs,
                    # sets upper bound on total timesteps
                    timesteps=args.n_challenges * 200,
                    save_trajs_to=args.save_trajs_to,
                    # terminates after n_challenges
                    episodes=args.n_challenges,
                )
                print(json.dumps(results, indent=4, sort_keys=True))
