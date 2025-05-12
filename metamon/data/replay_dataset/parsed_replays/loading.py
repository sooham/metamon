import os
import json
import random
from typing import Optional, Dict, Tuple, List
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm

from metamon.interface import ObservationSpace, RewardFunction, UniversalState


class ParsedReplayDataset(Dataset):
    def __init__(
        self,
        dset_root: str,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        formats: Optional[List[str]] = None,
        wins_losses_both: str = "both",
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        verbose: bool = False,
    ):
        assert dset_root is not None and os.path.exists(dset_root)
        formats = formats or [
            f"gen{g}{t}" for g in range(1, 5) for t in ["ou", "uu", "nu", "ubers"]
        ]
        self.observation_space = observation_space
        self.reward_function = reward_function
        self.dset_root = dset_root
        self.formats = formats
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.min_date = min_date
        self.max_date = max_date
        self.wins_losses_both = wins_losses_both
        self.verbose = verbose
        self.refresh_files()

    @property
    def disk_usage(self):
        return sum(os.path.getsize(f) for f in self.filenames)

    def refresh_files(self):
        self.filenames = []

        def _rating_to_int(rating: str) -> int:
            try:
                return int(rating)
            except ValueError:
                return 1000

        bar = lambda it, desc: (
            it if not self.verbose else tqdm.tqdm(it, desc=desc, colour="red")
        )

        for format in self.formats:
            path = os.path.join(self.dset_root, format)
            for filename in bar(os.listdir(path), desc=f"Finding {format} battles"):
                if not filename.endswith(".json"):
                    continue
                try:
                    battle_id, rating, p1_name, _, p2_name, mm_dd_yyyy, result = (
                        filename[:-5].split("_")
                    )
                except ValueError:
                    continue
                rating = _rating_to_int(rating)
                date = datetime.strptime(mm_dd_yyyy, "%m-%d-%Y")
                battle_id = (
                    battle_id.replace("[", "").replace("]", "").replace(" ", "").lower()
                )
                if (
                    format not in battle_id
                    or (self.min_rating is not None and rating < self.min_rating)
                    or (self.max_rating is not None and rating > self.max_rating)
                    or (self.min_date is not None and date < self.min_date)
                    or (self.max_date is not None and date > self.max_date)
                    or (self.wins_losses_both == "wins" and result != "WIN")
                    or (self.wins_losses_both == "losses" and result != "LOSS")
                ):
                    continue
                self.filenames.append(os.path.join(path, filename))

        if self.verbose:
            print(f"Dataset contains {len(self.filenames)} battles")

    def __len__(self):
        return len(self.filenames)

    def load_filename(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
        states = [UniversalState.from_dict(s) for s in data["states"]]
        obs = [self.observation_space.state_to_obs(s) for s in states]
        nested_obs = defaultdict(list)
        for o in obs:
            for k, v in o.items():
                nested_obs[k].append(v)
        actions = np.array(data["actions"], dtype=np.int32)
        missing_actions = actions == -1
        rewards = np.array(
            [
                self.reward_function(s_t, s_t1)
                for s_t, s_t1 in zip(states[:-1], states[1:])
            ],
            dtype=np.float32,
        )
        dones = np.zeros_like(rewards, dtype=bool)
        dones[-1] = True
        return dict(nested_obs), actions, rewards, dones, missing_actions

    def random_sample(self):
        filename = random.choice(self.filenames)
        return self.load_filename(filename)

    def __getitem__(self, i) -> Tuple[
        Dict[str, np.ndarray] | np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        return self.load_filename(self.filenames[i])


if __name__ == "__main__":
    from metamon.interface import (
        DefaultObservationSpace,
        DefaultShapedReward,
        TokenizedObservationSpace,
    )
    from metamon.data.tokenizer import get_tokenizer

    dset = ParsedReplayDataset(
        dset_root="/mnt/nfs_client/jake/metamon_parsed_hf_replays",
        observation_space=TokenizedObservationSpace(
            DefaultObservationSpace(),
            tokenizer=get_tokenizer("allreplays-v3"),
        ),
        reward_function=DefaultShapedReward(),
        formats=["gen1ou"],
        verbose=True,
    )
    print(len(dset))
    obs, actions, rewards, dones, missing_actions = dset[0]
    dset.refresh_files()
    obs, actions, rewards, dones, missing_actions = dset[0]
    print(len(dset))
    breakpoint()
