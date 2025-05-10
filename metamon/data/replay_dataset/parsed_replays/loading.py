import os
import json
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np

from metamon.interface import ObservationSpace, RewardFunction, UniversalState


class ParsedReplayDataset(Dataset):
    def __init__(
        self,
        dset_root: str,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        formats: list[str],
        wins_losses_both: str = "both",
    ):
        assert dset_root is not None and os.path.exists(dset_root)
        assert wins_losses_both in ["wins", "losses", "both"]
        self.observation_space = observation_space
        self.reward_function = reward_function
        self.filenames = []
        for format in formats:
            path = os.path.join(dset_root, format)
            for filename in os.listdir(path):
                if wins_losses_both == "wins" and not "WIN" in filename:
                    continue
                elif wins_losses_both == "losses" and not "LOSS" in filename:
                    continue
                self.filenames.append(os.path.join(path, filename))
        print(f"Dataset contains {len(self.filenames)} battles...")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with open(self.filenames[i], "r") as f:
            data = json.load(f)
        states = [UniversalState.from_dict(s) for s in data["states"]]
        obs = [self.observation_space.state_to_obs(s) for s in states]
        actions = torch.LongTensor(data["actions"])
        rewards = torch.Tensor(
            [
                self.reward_function(s_t, s_t1)
                for s_t, s_t1 in zip(states[:-1], states[1:])
            ]
        )
        dones = torch.zeros_like(rewards, dtype=bool)
        dones[-1] = True
        return obs, actions, rewards, dones


if __name__ == "__main__":
    from metamon.interface import DefaultObservationSpace, DefaultShapedReward

    dset = ParsedReplayDataset(
        dset_root="/mnt/nfs_client/jake/metamon_parsed_hf_replays",
        observation_space=DefaultObservationSpace(),
        reward_function=DefaultShapedReward(),
        formats=["gen1ou"],
    )
    print(len(dset))
    obs, actions, rewards, dones = dset[0]
    breakpoint()
