import os
import json
from functools import partial
import multiprocessing as mp

import amago
from amago.cli_utils import *

from metamon.env import MetaShowdown, TokenizedEnv, PSLadder, LocalLadder
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
    local: bool = False,
    password: str = None,
    wait_for_input: bool = False,
):
    Ladder = PSLadder if not local else LocalLadder
    kwargs = {
        "gen": gen,
        "format": format,
        "username": username[-18:],
        "avatar": avatar,
    }
    if not local:
        assert password is not None, "provide --ps_password to use the ladder"
        kwargs["password"] = password
    env = Ladder(**kwargs)
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


def _create_placeholder_experiment(
    dset_root, dset_name, run_name, max_seq_len, log, agent_type
):
    # the environment is only used to initialize the network
    # before loading the correct checkpoint
    env = make_placeholder_env()
    print("made dummy env")
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
    BASE_DIR = "/mnt/nfs_client/jake/metamon_official_ckpts"
    CKPT_DIR = "bulk_runs"

    def __init__(
        self,
        username,
        avatar,
        gin_config,
        amago_run_name,
        is_il_model,
        max_seq_len=200,
        agent_type=amago.agent.Agent,
    ):
        self.username = username
        self.avatar = avatar
        self.amago_run_name = amago_run_name
        self.gin_config = os.path.join(os.path.dirname(__file__), "configs", gin_config)
        self.gin_config = gin_config
        self.is_il_model = is_il_model
        self.max_seq_len = max_seq_len
        self.agent_type = agent_type

    @property
    def base_config(self):
        return {
            "amago.agent.Agent.reward_multiplier": 10.0,
            "amago.agent.Agent.offline_coeff": 1.0,
            "amago.agent.Agent.online_coeff": 0.0,
            "amago.agent.Agent.fake_filter": self.is_il_model,
            "amago.agent.Agent.use_multigamma": not self.is_il_model,
        }

    def initialize_agent(self, checkpoint: int, log: bool):
        use_config(self.base_config, [self.gin_config], finalize=False)
        experiment = _create_placeholder_experiment(
            dset_root=self.BASE_DIR,
            dset_name=self.CKPT_DIR,
            run_name=self.amago_run_name,
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
            username="SmallSparks",
            avatar="preschooler",
            gin_config="small.gin",
            amago_run_name="small_il_512x3",
            is_il_model=True,
        )


class SmallILFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallILFA",
            avatar="preschooler",
            gin_config="small.gin",
            amago_run_name="small_il_512x3_filled_actions",
            is_il_model=True,
        )


class SmallRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallRL",
            avatar="youngster-gen2",
            gin_config="small.gin",
            amago_run_name="small_rl_512x3",
            is_il_model=False,
        )


class SmallRL_ExtremeFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallRLExtremeFilter",
            avatar="youngster-gen3",
            gin_config="small.gin",
            amago_run_name="small_rl_exp_extreme_512x3",
            is_il_model=False,
        )


class SmallRL_BinaryFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallRLBinaryFA",
            avatar="youngster-gen3",
            gin_config="small.gin",
            amago_run_name="small_rl_binary_filter_512x3_filled_actions",
            is_il_model=False,
        )


class SmallRL_BinaryFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallRLBinaryFilter",
            avatar="youngster-gen4",
            gin_config="small.gin",
            amago_run_name="small_rl_binary_512x3",
            is_il_model=False,
        )


class SmallRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallRLAug",
            avatar="youngster-gen6",
            gin_config="small.gin",
            amago_run_name="small_rl_aug_3x512",
            is_il_model=False,
        )


class SmallRL_DPG(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="SmallRLAug",
            avatar="youngster-gen6",
            gin_config="small.gin",
            amago_run_name="small_rl_dpg_binary_512x3",
            is_il_model=False,
        )


class MediumIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="MediumIL",
            avatar="hiker-gen6",
            gin_config="medium.gin",
            amago_run_name="medium_il_768x6",
            is_il_model=True,
        )


class MediumRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="MediumRL",
            avatar="hiker-gen3rs",
            gin_config="medium.gin",
            amago_run_name="medium_rl_768x6",
            is_il_model=False,
        )


class MediumRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="MediumRLAug",
            avatar="hiker-gen3rs",
            gin_config="medium.gin",
            amago_run_name="medium_rl_aug_768x6",
            is_il_model=False,
        )


class MediumRL_DPG(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="MediumRLDPG",
            avatar="hiker-gen3rs",
            gin_config="medium.gin",
            amago_run_name="medium_rl_dpg_binary_768x6",
            is_il_model=False,
        )


class LargeRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="Montezuma2600",
            avatar="bugcatcher-gen3rs",
            gin_config="large.gin",
            amago_run_name="large_rl_1280x9",
            is_il_model=False,
            max_seq_len=128,
        )


class LargeIL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="DittoIsAllYouNeed",
            avatar="clown",
            gin_config="large.gin",
            amago_run_name="large_il_1280x9",
            is_il_model=True,
            max_seq_len=128,
        )


class SyntheticRLV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="Metamon1",
            avatar="rancher",
            gin_config="synthetic.gin",
            # the "768x6" model size claimed in these checkpoint names is a harmless mistake.
            # it's the 1280x9 model
            amago_run_name="large_rl_dpg_binary_synthetic_data_768x6",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV1(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="TheDeadlyTriad",
            avatar="sailor-gen1rb",
            gin_config="synthetic.gin",
            amago_run_name="large_rl_dpg_binary_synthetic_v2",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV1_SelfPlay(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="ABitterLesson",
            avatar="red-gen1main",
            gin_config="synthetic.gin",
            amago_run_name="large_rl_dpg_binary_sp_v1",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV1_PlusPlus(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="QPrime",
            avatar="fisherman-gen1rb",
            gin_config="synthetic.gin",
            amago_run_name="synthetic_v3",
            is_il_model=False,
            max_seq_len=128,
        )


class SyntheticRLV2(PretrainedModel):
    def __init__(self):
        super().__init__(
            username="MetamonII",
            avatar="sailor-gen1rb",
            gin_config="synthetic.gin",
            amago_run_name="twohot_v0",
            is_il_model=False,
            max_seq_len=128,
            agent_type=amago.agent.MultiTaskAgent,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    mp.set_start_method("spawn")

    parser = ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--gens", type=int, nargs="+", default=1)
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument(
        "--formats", nargs="+", default="ou", choices=["ubers", "ou", "uu", "nu"]
    )
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--ps_password", default=None)
    parser.add_argument("--n_challenges", type=int, default=150)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument(
        "--eval_type",
        choices=[
            "heuristic",
            "il",
            "ladder",
            "local-ladder",
        ],
    )
    parser.add_argument("--save_trajs_to", default=None)
    parser.add_argument(
        "--team_split",
        default="competitive",
        choices=["competitive", "train", "replays", "random_lead"],
    )
    parser.add_argument("--wait_for_input", action="store_true")
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
                    username = agent_maker.username
                    if "local" in args.eval_type:
                        username += str(checkpoint)
                        episodes = args.n_challenges
                    make_envs = partial(
                        make_ladder_env,
                        gen=gen,
                        format=format,
                        username=username,
                        password=args.ps_password,
                        local="local" in args.eval_type,
                        wait_for_input=args.wait_for_input,
                        avatar=agent_maker.avatar,
                        n_challenges=args.n_challenges + 1,
                    )
                    agent.env_mode = "sync"
                else:
                    raise ValueError(f"Invalid eval_type: {args.eval_type}")

                agent.verbose = False
                results = agent.evaluate_test(
                    make_envs,
                    timesteps=args.n_challenges * 200,
                    save_trajs_to=args.save_trajs_to,
                    episodes=episodes,
                )
                print(json.dumps(results, indent=4, sort_keys=True))
