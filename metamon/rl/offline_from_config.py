import os
import multiprocessing as mp
from functools import partial

import wandb

import amago
from amago.cli_utils import *
from amago.agent import exp_filter, binary_filter

import metamon
from metamon.env import MetaShowdown, TokenizedEnv
from metamon.task_distributions import get_task_distribution
from metamon.data.tokenizer import get_tokenizer
from metamon.baselines.model_based.nn_baseline import load_pretrained_model_to_cpu
from metamon.rl.online_meta_rl import MetamonAMAGOWrapper, MetamonTstepEncoder
from metamon import baselines


def add_cli(parser):
    # fmt: off
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--task_dist", default="AllGenOU")
    parser.add_argument("--amago_replay_dir", required=True)
    parser.add_argument("--amago_dset_name", default="replays_v6")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--actors_per_gpu", type=int, default=10)
    parser.add_argument("--online_epoch", type=int, default=None)
    parser.add_argument("--batch_size_per_gpu", type=int, default=12)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--il", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--arch_size", required=True, choices=["small", "medium", "large"])
    parser.add_argument("--token_aug", action="store_true")
    # fmt: on
    return parser


live_opponents = [
    baselines.heuristic.basic.RandomBaseline,
    baselines.heuristic.basic.GymLeader,
    baselines.heuristic.basic.Grunt,
    baselines.heuristic.kaizo.EmeraldKaizo,
    baselines.heuristic.basic.PokeEnvHeuristic,
]


def make_train_env(task_dist):
    env = MetaShowdown(
        get_task_distribution(task_dist)(
            k_shot_range=[0, 0],
            player_split="train-competitive",
            opponent_split="train-competitive",
            specify_opponents=live_opponents,
        ),
        new_task_every=4,
    )
    env = TokenizedEnv(env)
    env = MetamonAMAGOWrapper(env)
    return env


def make_val_env(task_dist):
    env = MetaShowdown(
        get_task_distribution(task_dist)(
            k_shot_range=[0, 0],
            player_split="competitive",
            opponent_split="competitive",
            specify_opponents=live_opponents,
        ),
        new_task_every=4,
    )
    env = TokenizedEnv(env)
    env = MetamonAMAGOWrapper(env)
    return env


def configure(args):
    config = {
        "amago.agent.Agent.reward_multiplier": 10.0,
        "amago.agent.Agent.offline_coeff": 1.0,
        "amago.agent.Agent.online_coeff": 0.0,
        "amago.agent.Agent.fake_filter": args.il,
        "amago.agent.Agent.use_multigamma": not args.il,
        "amago.agent.Agent.fbc_filter_func": binary_filter,
        "amago.agent.Agent.tau": 0.004,
        "exp_filter.clip_weights_high": 50,
        "exp_filter.clip_weights_low": 1e-8,
        "exp_filter.beta": 0.5,
        "MetamonTstepEncoder.token_mask_aug": args.token_aug,
    }
    config_file = os.path.join(
        os.path.dirname(__file__), "configs", f"{args.arch_size}.gin"
    )
    use_config(config, [config_file])


class RebuildValEnvExperiment(amago.Experiment):
    def evaluate_val(self):
        print("Rebuilding Envs...")
        self.init_envs()
        self.train_envs.close()
        del self.train_envs
        out = super().evaluate_val()
        self.val_envs.close()
        del self.val_envs
        return out

    def _get_grad_norms(self):
        base = super()._get_grad_norms()
        base[
            "metamon_text_emb_grad_norm"
        ] = (
            self.policy.tstep_encoder.turn_embedding.token_embedding.text_emb.weight.grad.norm()
        )
        return base

    def x_axis_metrics(self):
        return {
            "Epoch": self.epoch,
            "gradient_steps": self.grad_update_counter,
        }


if __name__ == "__main__":
    from argparse import ArgumentParser

    mp.set_start_method("spawn")

    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()
    configure(args)
    for trial in range(args.trials):
        experiment = RebuildValEnvExperiment(
            run_name=args.run_name,
            max_seq_len=args.max_seq_len,
            traj_save_len=10000,
            agent_type=amago.agent.Agent,
            tstep_encoder_type=MetamonTstepEncoder,
            traj_encoder_type=amago.nets.traj_encoders.TformerTrajEncoder,
            # Environment
            make_train_env=partial(make_train_env, args.task_dist),
            make_val_env=partial(make_val_env, args.task_dist),
            parallel_actors=args.actors_per_gpu,
            env_mode="async",
            exploration_wrapper_type=amago.envs.exploration.EpsilonGreedy,
            # Use Existing Replay Buffer
            dset_root=args.amago_replay_dir,
            dset_name=args.amago_dset_name,
            # only for the online buffer
            dset_max_size=float("inf"),
            save_trajs_as="npz",
            # Learning Schedule
            epochs=5000,
            start_learning_at_epoch=0,
            start_collecting_at_epoch=float("inf"),
            train_timesteps_per_epoch=25_000,  # hack to put optimer reset in the next epoch if enabled
            train_batches_per_epoch=25_000 * args.grad_accum,
            val_interval=1,
            val_timesteps_per_epoch=200,
            # do not override any checkpoints
            ckpt_interval=2,
            always_save_latest=True,
            always_load_latest=False,
            log_interval=20,
            # optimization
            batch_size=args.batch_size_per_gpu,  # per gpu
            batches_per_update=args.grad_accum,  # grad accumulation
            dloader_workers=10,
            learning_rate=0.00015,
            critic_loss_weight=10.0,
            lr_warmup_steps=1000,
            grad_clip=1.5,
            l2_coeff=1e-4,
            local_time_optimizer=False,
            mixed_precision="no",
            # Main Logging Process
            log_to_wandb=args.log,
            wandb_project="metamon",
            wandb_entity="ut-austin-rpl-metamon",
            verbose=True,
        )

        experiment.start()
        if args.ckpt is not None:
            experiment.load_checkpoint(args.ckpt)
        experiment.learn()
        wandb.finish()
