import os
import glob
import gymnasium as gym

import gin
import numpy as np
import amago
import torch
import torch.nn as nn
import torch.nn.functional as F

import metamon
from metamon.il.model import TransformerTurnEmbedding
from metamon.data.tokenizer import PokemonTokenizer, get_tokenizer

from amago.envs import AMAGOEnv, SequenceWrapper
from amago.nets.utils import symlog


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
                info["AMAGO_LOG_METRIC Success"] = info["win_rate"]
            if "valid_action_count" in info and "invalid_action_count" in info:
                info["AMAGO_LOG_METRIC Valid Actions"] = info["valid_action_count"] / (
                    info["valid_action_count"] + info["invalid_action_count"]
                )
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


class ReplayConversionEnv(gym.Env):
    def __init__(self, input_dir, tokenizer=get_tokenizer("allreplays-v3")):
        self.tokenizer = tokenizer
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Dict(
            {
                "tokens": gym.spaces.Box(
                    low=-1, high=len(tokenizer), shape=(87,), dtype=np.int32
                ),
                "numbers": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(48,),
                    dtype=np.float32,
                ),
            }
        )
        self.paths = glob.glob(f"{input_dir}/**/*.npz", recursive=True)
        self.file_idx = 0

    def get_obs(self, t):
        text = self.traj_text[t]
        num = self.traj_nums[t]
        tokens = self.tokenizer.tokenize(text.tolist())
        return {"tokens": tokens, "numbers": num}

    def reset(self, *args, **kwargs):
        path = self.paths[self.file_idx]
        with np.load(path) as replay:
            self.traj_text = replay["obs_text"]
            self.traj_nums = replay["obs_num"]
            self.traj_actions = replay["actions"]
            self.traj_rewards = replay["rewards"]

        self.time_idx = 1
        self.file_idx += 1
        return self.get_obs(0), {"take_action": self.traj_actions[0]}

    def step(self, action):
        t = self.time_idx
        next_obs = self.get_obs(t)
        rew = self.traj_rewards[t].item()
        done = t == len(self.traj_text) - 1
        next_action = self.traj_actions[t]
        self.time_idx += 1
        return next_obs, rew, done, False, {"take_action": next_action}


from metamon.data.tokenizer.tokenizer import UNKNOWN_TOKEN


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
        if token_mask_aug:
            print("Using token mask aug")
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
