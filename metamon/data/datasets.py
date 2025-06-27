import os
import json
import random
from typing import Optional, Dict, Tuple, List
from datetime import datetime
from collections import defaultdict


from torch.utils.data import Dataset
import lz4.frame
import numpy as np
import tqdm

from metamon.interface import ObservationSpace, RewardFunction, UniversalState
from metamon.data.download import download_parsed_replays


class ParsedReplayDataset(Dataset):
    """An iterable dataset of "parsed replays"

    Parsed replays are records of Pokémon Showdown battles that have been converted to the partially observed
    point-of-view of a single player, matching the problem our agents face in the RL environment. They are created by the
    `metamon.data.replay_dataset.replay_parser` module from "raw" Showdown replay logs
    downloaded from publicly available battles.

    This is a pytorch `Dataset` that returns (nested_obs, actions, rewards, dones, missing_actions) trajectory tuples,
    where:
    - nested_obs: lists of numpy arrays with length seq_len (arrays may have different shapes). If the observation space
    is a dict, this becomes a dict of lists of arrays for each key....
    - actions: A numpy array of shape (seq_len, dim)
    - rewards: A numpy array of shape (seq_len, dim)
    - dones: A numpy array of shape (seq_len, dim)
    - missing_actions: A numpy array of shape (seq_len, dim)

    Note that depending on the observation space, you may need a custom pad_collate_fn in the pytorch dataloader
    to handle the variable-shaped arrays in nested_obs.

    Missing actions are a bool mask where idx i = True if action i is missing (actions[i] == -1, or was originally
    missing but has since been filled by some prediction scheme). Missing actions are caused by player choices that
    are not revealed to spectators and do not show up in the replay logs (e.g., paralysis, sleep, flinch).

    Data is stored as interface.UniversalStates and observations and rewards are created on the fly. This
    means we no longer have to create new versions of the parsed replay dataset to experiment with different
    observation spaces or reward functions.

    Example:
        ```python
        dset = ParsedReplayDataset(
            observation_space=TokenizedObservationSpace(
                DefaultObservationSpace(),
                tokenizer=get_tokenizer("allreplays-v3"),
            ),
            reward_function=DefaultShapedReward(),
            formats=["gen1nu"],
            verbose=True,
        )

        obs, actions, rewards, dones, missing_actions = dset[0]
        ```

    Args:
        observation_space: The observation space to use. Must be an instance of `interface.ObservationSpace`.
        reward_function: The reward function to use. Must be an instance of `interface.RewardFunction`.
        dset_root: The root directory of the parsed replays. If not specified, the parsed replays will be
            downloaded and extracted from the latest version of the huggingface dataset, but this may take minutes.
        formats: A list of formats to load (e.g. ["gen1ou", "gen2ubers"]). Defaults to all supported formats
            (Gen 1-4 ou, uu, nu, and ubers), but this will take a long time to download and extract the first time.
        wins_losses_both: Whether to only load the perspective of players who won their battle, lost their
            battle, or both. {"wins", "losses", "both"}
        min_rating: The minimum rating of battles to load (in ELO). Note that most replays are Unrated, which
            is mapped to 1000 ELO (the minimum rating on Showdown). In reality many of these battles were played
            as part of tournaments and should probably not be ignored.
        max_rating: The maximum rating of battles to load (in ELO). In Generations 1-4, ELO ratings above 1500
            are very good.
        min_date: The minimum date of battles to load (as a datetime). Our dataset begins in 2014. Many replays
            from 2021-2024 are missing due to a Showdown database issue. See the raw-replay dataset README on
            HF for a visual timeline of the dataset.
        max_date: The maximum date of battles to load (as a datetime). The latest date available will depend on
            the current version of the parsed replays dataset.
        max_seq_len: The maximum sequence length to load. Trajectories are randomly sliced to this length.
        verbose: Whether to print progress bars while loading large datasets.
    """

    def __init__(
        self,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        dset_root: Optional[str] = None,
        formats: Optional[List[str]] = None,
        wins_losses_both: str = "both",
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        max_seq_len: Optional[int] = None,
        verbose: bool = False,
    ):
        formats = formats or [
            f"gen{g}{t}" for g in range(1, 5) for t in ["ou", "uu", "nu", "ubers"]
        ]

        if dset_root is None:
            for format in formats:
                path_to_format_data = download_parsed_replays(format)
            dset_root = os.path.dirname(path_to_format_data)

        assert dset_root is not None and os.path.exists(dset_root)
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
        self.max_seq_len = max_seq_len
        self.refresh_files()

    def parse_battle_date(self, filename: str) -> datetime:
        # parsed replays saved by our own gym env will have hour/minute/sec
        # while Showdown replays will not.
        date_str = filename.split("_")[-2]
        formats = ["%m-%d-%Y-%H:%M:%S", "%m-%d-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse date string: {date_str}")

    def refresh_files(self):
        self.filenames = []

        def _rating_to_int(rating: str) -> int:
            # cast "Unrated" to 1000 (the minimum rating)
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
                if not (filename.endswith(".json") or filename.endswith(".json.lz4")):
                    continue
                try:
                    (
                        battle_id,
                        rating,
                        p1_name,
                        _,
                        p2_name,
                        mm_dd_yyyy,
                        result,
                    ) = filename[:-5].split("_")
                except ValueError:
                    continue
                rating = _rating_to_int(rating)
                # abstracted to let RL replay buffers delete the oldest battles
                date = self.parse_battle_date(filename)
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
        if filename.endswith(".json.lz4"):
            # compressed (v2 format)
            with lz4.frame.open(filename, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
        elif filename.endswith(".json"):
            # uncompressed (v1 format)
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unknown file extension: {filename}")

        states = [UniversalState.from_dict(s) for s in data["states"]]
        # reset the observation space, then call once on each state, which lets
        # any history-dependent features behave as they would in an online battle
        self.observation_space.reset()
        obs = [self.observation_space.state_to_obs(s) for s in states]
        # TODO: handle case where observation space is not a dict. don't have one to test yet.
        nested_obs = defaultdict(list)
        for o in obs:
            for k, v in o.items():
                nested_obs[k].append(v)
        # NOTE: the replay parser currently leaves a blank (-1) final action
        actions = np.array(data["actions"][:-1], dtype=np.int32)
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

        if self.max_seq_len is not None:
            # s s s s s s s s
            # a a a a a a a
            # r r r r r r r
            # d d d d d d d
            safe_start = random.randint(0, max(len(actions) - self.max_seq_len, 0))
            actions = actions[safe_start : safe_start + self.max_seq_len]
            rewards = rewards[safe_start : safe_start + self.max_seq_len]
            dones = dones[safe_start : safe_start + self.max_seq_len]
            missing_actions = missing_actions[
                safe_start : safe_start + self.max_seq_len
            ]
            nested_obs = {
                k: v[safe_start : safe_start + 1 + self.max_seq_len]
                for k, v in nested_obs.items()
            }

        return dict(nested_obs), actions, rewards, dones, missing_actions

    def random_sample(self):
        filename = random.choice(self.filenames)
        return self.load_filename(filename)

    def __getitem__(self, i) -> Tuple[
        Dict[str, list[np.ndarray]] | list[np.ndarray],
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
    from metamon.tokenizer import get_tokenizer

    dset = ParsedReplayDataset(
        observation_space=TokenizedObservationSpace(
            DefaultObservationSpace(),
            tokenizer=get_tokenizer("allreplays-v3"),
        ),
        reward_function=DefaultShapedReward(),
        formats=["gen1ou"],
        verbose=True,
    )
    for i in tqdm.tqdm(range(min(2000, len(dset)))):
        obs, actions, rewards, dones, missing_actions = dset[i]
