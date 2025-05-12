import os
import time
import random
import glob
import warnings
import gymnasium as gym
from typing import Optional, Dict, List, Type, Callable, Iterable

import gin
import numpy as np
import amago
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

import metamon
from metamon.il.model import TransformerTurnEmbedding
from metamon.data.tokenizer import PokemonTokenizer
from metamon.interface import ObservationSpace, RewardFunction
from metamon.data.tokenizer.tokenizer import UNKNOWN_TOKEN
from metamon.data.replay_dataset.parsed_replays.loading import ParsedReplayDataset

import amago
from amago.envs import AMAGOEnv
from amago.nets.utils import symlog
from amago.loading import TrajDset, RLData, Batch, MAGIC_PAD_VAL
from amago.agent import Agent
from amago.nets.traj_encoders import TformerTrajEncoder, TrajEncoder
from amago.nets.tstep_encoders import TstepEncoder
from amago.envs.amago_env import AMAGO_ENV_LOG_PREFIX


class MetamonAMAGOWrapper(AMAGOEnv):
    def __init__(self, metamon_env):
        self.reset_counter = 0
        super().__init__(
            env=metamon_env,
            env_name="metamon",
            batched_envs=1,
        )

    def step(self, action):
        try:
            *out, info = super().step(action)
            if "win_rate" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Success"] = info["win_rate"]
            if "valid_action_count" in info and "invalid_action_count" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Valid Actions"] = info[
                    "valid_action_count"
                ] / (info["valid_action_count"] + info["invalid_action_count"])
            return *out, info
        except:
            print("Force resetting due to long-tail error")
            self.reset()
            next_state, reward, terminated, truncated, info = self.step(action)
            reward *= 0.0
            terminated[:] = False
            truncated[:] = True
            return next_state, reward, terminated, truncated, info

    @property
    def env_name(self):
        return f"{self.current_task.battle_format}_vs_{self.current_task.opponent_type.__name__}"


class PSLadderAMAGOWrapper(MetamonAMAGOWrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def env_name(self):
        return f"psladder_{self.env.env.env.username}"


def unknown_token_mask(tokens, skip_prob: float = 0.2, batch_max_prob: float = 0.33):
    B, L, tok = tokens.shape
    dev = tokens.device
    batch_mask = torch.rand(B) < (1.0 - skip_prob)  # mask tokens from this batch index
    batch_thresh = (
        torch.rand(B) * batch_max_prob
    )  # mask this % of tokens from the sequence
    thresh = (
        batch_mask * batch_thresh
    )  # 0 if batch index isn't masked, % to mask otherwise
    mask = torch.rand(tokens.shape) < thresh.view(-1, 1, 1)
    tokens[mask.to(dev)] = UNKNOWN_TOKEN
    return tokens.to(dev)


@gin.configurable
class MetamonTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 100,
        n_layers: int = 3,
        n_heads: int = 5,
        scratch_tokens: int = 4,
        token_mask_aug: bool = False,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        self.turn_embedding = TransformerTurnEmbedding(
            tokenizer=tokenizer,
            token_embedding_dim=d_model,
            numerical_features=48 + extra_emb_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            scratch_tokens=scratch_tokens,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    @torch.compile
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        return turn_emb


class MetamonAMAGODataset(TrajDset):
    def __init__(
        self,
        items_per_epoch: int,
        parsed_replay_dset: ParsedReplayDataset,
        selfplay_replay_dset: Optional[ParsedReplayDataset] = None,
        max_seq_len: Optional[int] = None,
        parsed_replay_sampling_rate: float = 1.0,
        padded_sampling: str = "none",
    ):
        self.items_per_epoch = items_per_epoch
        self.parsed_replay_dset = parsed_replay_dset
        self.selfplay_replay_dset = selfplay_replay_dset
        self.max_seq_len = max_seq_len
        if not 0 <= parsed_replay_sampling_rate <= 1:
            raise ValueError(
                f"parsed_replay_sampling_rate must be between 0 and 1, got {parsed_replay_sampling_rate}"
            )
        self.parsed_replay_sampling_rate = parsed_replay_sampling_rate
        self.padded_sampling = padded_sampling

    def __len__(self):
        return self.items_per_epoch

    @property
    def disk_usage(self):
        # avoid slowdowns for extremely large datasets
        return -1

    def clear(self, delete_protected: bool = False):
        warnings.warn(
            "Metamon protects against deleting replay dataset files via the AMAGO Dataset. Nothing will happen."
        )
        return

    def refresh_files(self):
        self.parsed_replay_dset.refresh_files()
        if self.selfplay_replay_dset is not None:
            self.selfplay_replay_dset.refresh_files()

    def count_deletable_trajectories(self):
        if self.selfplay_replay_dset is None:
            return 0
        return len(self.selfplay_replay_dset)

    def count_protected_trajectories(self):
        return len(self.parsed_replay_dset)

    def count_trajectories(self):
        return self.count_deletable_trajectories() + self.count_protected_trajectories()

    def filter(self, new_size: int):
        if self.count_deletable_trajectories() < new_size:
            return
        warnings.warn("Dataset FIFO filtering is not yet implemented.")
        return

    def __getitem__(self, i):
        if (
            self.selfplay_replay_dset is None
            or random.random() < self.parsed_replay_sampling_rate
        ):
            data = self.parsed_replay_dset.random_sample()
        else:
            data = self.selfplay_replay_dset.random_sample()

        obs, actions, rewards, dones, missing_actions = data
        rl_data = MetamonRLData(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            missing_actions=missing_actions,
        )
        if self.max_seq_len is not None:
            rl_data = rl_data.random_slice(
                length=self.max_seq_len, padded_sampling=self.padded_sampling
            )
        return rl_data


class MetamonRLData(RLData):
    def __init__(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        missing_actions: np.ndarray,
    ):
        self.obs = {k: torch.from_numpy(np.array(v)) for k, v in obs.items()}
        # a bit of a hack: put the action mask here to avoid rewriting more code
        self.obs["missing_action_mask"] = (
            torch.from_numpy(missing_actions).bool().unsqueeze(-1)
        )

        # 1. imitate AMAGOEnv one-hot action representation
        # (AMAGOEnv.make_action_rep)
        actions_one_hot = torch.zeros((len(actions), 9), dtype=torch.float32)
        actions_one_hot[torch.arange(len(actions)), actions] = 1.0
        self.actions = actions_one_hot[:-1, :]

        # 2. imitate AMAGOEnv time index
        self.time_idxs = torch.arange(len(actions)).long().unsqueeze(-1)

        # 3. imitate AMAGO reward
        self.rews = torch.from_numpy(rewards).float().unsqueeze(-1)

        # 4. imitate AMAGO done
        self.dones = torch.from_numpy(dones).bool().unsqueeze(-1)

        blank_action = torch.zeros((1, 9), dtype=torch.float32)
        blank_rew = torch.zeros((1, 1), dtype=torch.float32)
        # "rl2s" are what AMAGO calls the array of (prev_action, prev_reward) values
        # `amago.hindsight.Timestep`
        self.rl2s = torch.cat(
            (
                torch.cat((blank_rew, self.rews), dim=0),
                torch.cat((blank_action, self.actions), dim=0),
            ),
            dim=-1,
        )

        # used by random sampling
        self.safe_randrange = lambda l, h: random.randrange(l, max(h, l + 1))


@gin.configurable
class MetamonAMAGOExperiment(amago.Experiment):
    def __init__(
        self,
        # required
        run_name: str,
        ckpt_dir: str,
        make_train_env: Iterable[Callable],
        make_val_env: Iterable[Callable],
        parsed_replay_dataset: ParsedReplayDataset,
        # agent
        max_seq_len: int = 200,
        agent_type: Type[Agent] = Agent,
        tstep_encoder_type: Type[TstepEncoder] = MetamonTstepEncoder,
        traj_encoder_type: Type[TrajEncoder] = TformerTrajEncoder,
        # logging
        log_to_wandb: bool = False,
        wandb_project: str = os.environ.get("METAMON_WANDB_PROJECT"),
        wandb_entity: str = os.environ.get("METAMON_WANDB_ENTITY"),
        verbose: bool = True,
        log_interval: int = 250,
        # dataset
        padded_sampling: str = "none",
        dloader_workers: int = 8,
        # learning schedule (update : data / online vs. offline)
        epochs: int = 100,
        train_batches_per_epoch: int = 25_000,
        val_interval: int = 1,
        val_timesteps_per_epoch: int = 0,  # FIXME
        ckpt_interval: int = 2,
        # optimization
        learning_rate: float = 1.5e-4,
        batches_per_update: int = 1,
        batch_size: int = 32,
        critic_loss_weight: float = 10.0,
        lr_warmup_steps: int = 1000,
        grad_clip=1.5,
        l2_coeff=1e-4,
        mixed_precision: str = "no",
    ):

        assert len(make_train_env) == len(make_val_env)
        parallel_actors = len(make_train_env)
        self.parsed_replay_dataset = parsed_replay_dataset

        super().__init__(
            run_name=run_name,
            max_seq_len=max_seq_len,
            traj_save_len=1000,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            parallel_actors=parallel_actors,
            env_mode="async",
            exploration_wrapper_type=None,
            sample_actions=True,
            force_reset_train_envs_every=None,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            verbose=verbose,
            log_interval=log_interval,
            # we will not be using AMAGO's dataset saving,
            # but this establishes the checkpoint directory
            dset_root=os.path.dirname(ckpt_dir),
            dset_name=os.path.basename(ckpt_dir),
            # only for the online buffer
            dset_max_size=float("inf"),
            # disable saving trajectories
            save_trajs_as=None,
            padded_sampling=padded_sampling,
            dloader_workers=dloader_workers,
            epochs=epochs,
            start_learning_at_epoch=0,
            start_collecting_at_epoch=float("inf"),
            train_timesteps_per_epoch=0,
            train_batches_per_epoch=train_batches_per_epoch,
            val_interval=val_interval,
            val_timesteps_per_epoch=val_timesteps_per_epoch,
            ckpt_interval=ckpt_interval,
            always_save_latest=True,
            always_load_latest=False,
            batch_size=batch_size,
            batches_per_update=batches_per_update,
            learning_rate=learning_rate,
            critic_loss_weight=critic_loss_weight,
            lr_warmup_steps=lr_warmup_steps,
            grad_clip=grad_clip,
            l2_coeff=l2_coeff,
            mixed_precision=mixed_precision,
        )

    def init_dsets(self):
        """
        Builds pytorch training dataset

        Modified to point to a custom Metamon dataset instead of AMAGO's on-disk replay buffer
        """
        self.train_dset = MetamonAMAGODataset(
            items_per_epoch=self.train_batches_per_epoch
            * self.batch_size
            * self.accelerator.num_processes,
            parsed_replay_dset=self.parsed_replay_dataset,
            max_seq_len=self.max_seq_len,
            padded_sampling=self.padded_sampling,
            # currently hardcoded to ignore new online selfplay mode
            selfplay_replay_dset=None,
            parsed_replay_sampling_rate=1.0,
        )
        return self.train_dset

    def evaluate_val(self):
        """
        Handles evaluation loop

        Modified to get around an issue with resuming interaction of poke-env envs
        """
        print("Rebuilding Envs...")
        self.init_envs()
        self.train_envs.close()
        del self.train_envs
        out = super().evaluate_val()
        self.val_envs.close()
        del self.val_envs
        return out

    def x_axis_metrics(self):
        """
        Modified to remove frame/timestep counting because we are tearing down the train envs
        """
        return {
            "Epoch": self.epoch,
            "gradient_steps": self.grad_update_counter,
        }

    def compute_loss(self, batch: Batch, log_step: bool):
        """
        Core computation of the actor and critic RL loss terms from a `Batch` of data.

        Modified to also mask metamon's missing actions
        """
        critic_loss, actor_loss = self.policy_aclr(batch, log_step=log_step)
        update_info = self.policy.update_info
        B, L_1, G, _ = actor_loss.shape
        C = len(self.policy.critics)
        amago_pad_mask = (
            ~((batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True))
        ).float()[:, 1:, ...]
        # amago_mask : 1.0 if valid, 0.0 if padded by the dataloader
        metamon_missing_action_mask = (
            ~batch.obs["missing_action_mask"][:, :-1, :]
        ).float()
        # metamon_missing_actions : 1.0 if action was provided, 0.0 if missing
        combined_mask = amago_pad_mask * metamon_missing_action_mask
        critic_state_mask = repeat(combined_mask, f"B L 1 -> B L {C} {G} 1")
        actor_state_mask = repeat(combined_mask, f"B L 1 -> B L {G} 1")

        masked_actor_loss = amago.utils.masked_avg(actor_loss, actor_state_mask)
        if isinstance(critic_loss, torch.Tensor):
            masked_critic_loss = amago.utils.masked_avg(critic_loss, critic_state_mask)
        else:
            assert critic_loss is None
            masked_critic_loss = 0.0

        return {
            "critic_loss": masked_critic_loss,
            "actor_loss": masked_actor_loss,
            "seq_len": L_1 + 1,
            "mask": combined_mask,
        } | update_info
