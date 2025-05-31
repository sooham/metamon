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
from amago.nets.utils import symlog
from amago.loading import RLData, RLDataset, Batch
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
        obs, actions, rewards, dones, missing_acts = data
        # amago expects discrete actions to be one-hot encoded
        actions_torch = F.one_hot(
            torch.from_numpy(actions).long().clamp(min=0), num_classes=9
        )
        # a bit of a hack: make the action mask (which is the same size as actions)
        # one timestep longer to match the size of observations, then put it in the amago
        # observation dict, let the network ignore it, and make it accessible to
        # mask the actor/critic loss later on.
        missing_acts = np.concatenate([missing_acts, np.ones(1, dtype=bool)], axis=0)
        obs_torch = {k: torch.from_numpy(np.stack(v, axis=0)) for k, v in obs.items()}
        obs_torch["missing_action_mask"] = torch.from_numpy(missing_acts).unsqueeze(-1)
        rewards_torch = torch.from_numpy(rewards).unsqueeze(-1)
        dones_torch = torch.from_numpy(dones).unsqueeze(-1)
        time_idxs = torch.arange(len(actions) + 1).long().unsqueeze(-1)
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
    Adds actions masking to the main AMAGO experiment, and leaves room for further tweaks.
    """

    def edit_actor_mask(
        self, batch: Batch, actor_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, L, G, _ = actor_loss.shape
        # missing_action_mask is one timestep too long to match the size of observations
        # True where the action is missing, False where it's provided.
        # pad_mask is True where the timestep should count towards loss, False where it shouldn't.
        missing_action_mask = repeat(
            ~batch.obs["missing_action_mask"][:, :-1], "b l 1 -> b l g 1", g=G
        )
        return pad_mask & missing_action_mask

    def edit_critic_mask(
        self, batch: Batch, critic_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, L, C, G, _ = pad_mask.shape
        missing_action_mask = repeat(
            ~batch.obs["missing_action_mask"][:, :-1], "b l 1 -> b l c g 1", g=G, c=C
        )
        return pad_mask & missing_action_mask
