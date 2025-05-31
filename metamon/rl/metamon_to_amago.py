import os
from typing import Optional, Dict, List, Type, Callable, Iterable, Any

import gin
import numpy as np
import amago
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

import metamon
from metamon.il.model import TransformerTurnEmbedding
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.datasets import ParsedReplayDataset
from metamon.env import PokeEnvWrapper

import amago

assert amago.__version__ >= "3.1.0", "Update to the latest AMAGO version!"
from amago.envs import AMAGOEnv
from amago.envs.exploration import ExplorationWrapper
from amago.nets.utils import symlog
from amago.loading import RLData, RLDataset, Batch
from amago.agent import Agent
from amago.nets.traj_encoders import TformerTrajEncoder, TrajEncoder
from amago.nets.tstep_encoders import TstepEncoder
from amago.envs.amago_env import AMAGO_ENV_LOG_PREFIX


class MetamonAMAGOWrapper(AMAGOEnv):
    """AMAGOEnv wrapper with success rate and valid action rate logging."""

    def __init__(self, metamon_env: PokeEnvWrapper):
        self.reset_counter = 0
        super().__init__(
            env=metamon_env,
            env_name="metamon",
            batched_envs=1,
        )

    def step(self, action):
        try:
            *out, info = super().step(action)
            # amago will average these stats over episodes, devices, and parallel actors.
            if "won" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Win Rate"] = info["won"]
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
        return f"{self.env.metamon_battle_format}_vs_{self.env.metamon_opponent_name}"


class PSLadderAMAGOWrapper(MetamonAMAGOWrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def env_name(self):
        return f"psladder_{self.env.env.username}"


def unknown_token_mask(tokens, skip_prob: float = 0.2, batch_max_prob: float = 0.33):
    """Randomly set entries in the text component of the observation space to UNKNOWN_TOKEN.

    Args:
        skip_prob: Probability of entirely skipping the mask for any given sequence
        batch_max_prob: For each sequence, randomly mask tokens with [0, batch_max_prob) prob
            (if not skipped).
    """
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
    """Token + numerical embedding for Metamon.

    Fuses multi-modal input with attention and summary tokens.
    Visualized on the README and in the paper architecture figure.
    """

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
        numerical_tokens: int = 6,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        self.turn_embedding = TransformerTurnEmbedding(
            tokenizer=tokenizer,
            token_embedding_dim=d_model,
            numerical_features=48 + extra_emb_dim,
            numerical_tokens=numerical_tokens,
            scratch_tokens=scratch_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
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


class MetamonAMAGODataset(RLDataset):
    """A wrapper around the ParsedReplayDataset that converts to an AMAGO RLDataset.

    Args:
        dset_name: Give the dataset an arbitrary name for logging. Defaults to class name.
        parsed_replay_dset: The ParsedReplayDataset to wrap.
    """

    def __init__(
        self,
        parsed_replay_dset: ParsedReplayDataset,
        dset_name: Optional[str] = None,
    ):
        super().__init__(dset_name=dset_name)
        self.parsed_replay_dset = parsed_replay_dset

    @property
    def save_new_trajs_to(self):
        # disables AMAGO's trajetory saving; metamon
        # will handle this in its own replay format.
        return None

    def on_end_of_collection(self, experiment) -> dict[str, Any]:
        # TODO: implement FIFO replay buffer
        return {}

    def get_description(self) -> str:
        return f"Metamon Replay Dataset ({self.dset_name})"

    def sample_random_trajectory(self) -> RLData:
        data = self.parsed_replay_dset.random_sample()
        obs, actions, rewards, dones, missing_actions = data
        obs_torch = {k: torch.from_numpy(np.stack(v, axis=0)) for k, v in obs.items()}
        actions_torch = F.one_hot(
            torch.from_numpy(actions).long().clamp(min=0), num_classes=9
        )[:-1]
        rewards_torch = torch.from_numpy(rewards).unsqueeze(-1)
        dones_torch = torch.from_numpy(dones).unsqueeze(-1)
        obs_torch["missing_action_mask"] = torch.from_numpy(missing_actions).unsqueeze(
            -1
        )
        time_idxs = torch.arange(len(missing_actions)).long().unsqueeze(-1)
        rl_data = RLData(
            obs=obs_torch,
            actions=actions_torch,
            rews=rewards_torch,
            dones=dones_torch,
            time_idxs=time_idxs,
        )
        return rl_data


@gin.configurable
class MetamonAMAGOExperiment(amago.Experiment):
    """
    Verbose wrapper around the AMAGO Experiment that sets some default kwargs,
    adds a missing action mask, and leaves room for further tweaks.
    """

    def __init__(
        self,
        # main
        run_name: str,
        ckpt_base_dir: str,
        make_train_env: Iterable[Callable],
        make_val_env: Iterable[Callable],
        dataset: RLDataset,
        max_seq_len: int = 200,
        agent_type: Type[Agent] = Agent,
        tstep_encoder_type: Type[TstepEncoder] = MetamonTstepEncoder,
        traj_encoder_type: Type[TrajEncoder] = TformerTrajEncoder,
        # environment
        val_timesteps_per_epoch: int = 200,
        env_mode: str = "async",
        async_env_mp_context: str = "spawn",
        exploration_wrapper_type: Optional[Type[ExplorationWrapper]] = None,
        sample_actions: bool = True,
        force_reset_train_envs_every: Optional[int] = None,
        # logging
        log_to_wandb: bool = False,
        wandb_project: str = os.environ.get("METAMON_WANDB_PROJECT"),
        wandb_entity: str = os.environ.get("METAMON_WANDB_ENTITY"),
        verbose: bool = True,
        log_interval: int = 300,
        # replay
        dloader_workers: int = 8,
        padded_sampling: str = "none",
        # schedule
        epochs: int = 100,
        start_learning_at_epoch: int = 0,
        start_collecting_at_epoch: int = float("inf"),
        train_timesteps_per_epoch: int = 0,
        train_batches_per_epoch: int = 25_000,
        val_interval: int = 1,
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

        super().__init__(
            run_name=run_name,
            max_seq_len=max_seq_len,
            ckpt_base_dir=ckpt_base_dir,
            dataset=dataset,
            tstep_encoder_type=tstep_encoder_type,
            traj_encoder_type=traj_encoder_type,
            agent_type=agent_type,
            make_train_env=make_train_env,
            make_val_env=make_val_env,
            parallel_actors=parallel_actors,
            env_mode=env_mode,
            async_env_mp_context=async_env_mp_context,
            exploration_wrapper_type=exploration_wrapper_type,
            sample_actions=sample_actions,
            force_reset_train_envs_every=force_reset_train_envs_every,
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            verbose=verbose,
            log_interval=log_interval,
            padded_sampling=padded_sampling,
            dloader_workers=dloader_workers,
            epochs=epochs,
            start_learning_at_epoch=start_learning_at_epoch,
            start_collecting_at_epoch=start_collecting_at_epoch,
            train_timesteps_per_epoch=train_timesteps_per_epoch,
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

    def edit_actor_mask(
        self, batch: Batch, actor_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, L, G, _ = actor_loss.shape
        missing_action_mask = repeat(
            ~batch.obs["missing_action_mask"][:, :-1, :], "b l 1 -> b l g 1", g=G
        )
        return pad_mask & missing_action_mask

    def edit_critic_mask(
        self, batch: Batch, critic_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, L, C, G, _ = pad_mask.shape
        missing_action_mask = repeat(
            ~batch.obs["missing_action_mask"][:, :-1, :], "b l 1 -> b l c g 1", g=G, c=C
        )
        return pad_mask & missing_action_mask
