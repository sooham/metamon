import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np

from metamon.data.tokenizer import PokemonTokenizer, get_tokenizer


class ShowdownReplayDataset(Dataset):
    def __init__(
        self,
        dset_root: str,
        max_seq_len: int,
        formats: list[str],
        tokenizer: PokemonTokenizer,
        max_size: Optional[int] = None,
        wins_losses_both: str = "both",
        dset_split: str = "train",
        verbose: bool = True,
        as_numpy: bool = False,
    ):
        assert dset_root is not None and os.path.exists(dset_root)
        assert dset_split in ["train", "val"]
        assert wins_losses_both in ["wins", "losses", "both"]
        self.max_seq_len = max_seq_len
        self.dset_split = dset_split
        self.tokenizer = tokenizer
        self.filenames = []
        for format in formats:
            path = os.path.join(dset_root, format, dset_split)
            for filename in os.listdir(path):
                if wins_losses_both == "wins" and not "WIN" in filename:
                    continue
                elif wins_losses_both == "losses" and not "LOSS" in filename:
                    continue
                if formats and not any(f in filename for f in formats):
                    continue
                self.filenames.append(os.path.join(path, filename))

        if max_size is not None:
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:max_size]

        self.as_numpy = as_numpy
        self.verbose = verbose
        if self.verbose:
            print(
                f"Dataset {dset_split} split contains {len(self.filenames)} replays..."
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        try:
            # temporary ugliness until we stop interrupting replay parser runs mid pickle
            with np.load(self.filenames[i]) as replay:
                obs_text = replay["obs_text"]
                obs_num = replay["obs_num"]
                actions = replay["actions"]
                rewards = replay["rewards"]
        except:
            if self.verbose:
                print(f"Skipping replay {self.filenames[i]} due to loading error.")
            return self[i + 1]
        obs_tokens = np.stack([self.tokenizer.tokenize(o) for o in obs_text], axis=0)
        length = obs_tokens.shape[0]
        dones = np.zeros_like(rewards, dtype=bool)
        dones[-1] = True

        idx = random.randrange(0, max(obs_tokens.shape[0] - self.max_seq_len, 1))
        end = min(idx + self.max_seq_len, length - 1)
        t = slice(idx, end)
        t1 = slice(idx + 1, end + 1)
        tensor = lambda seq: torch.from_numpy(seq) if not self.as_numpy else seq

        obs_t = tensor(obs_tokens[t]), tensor(obs_num[t])
        obs_t1 = tensor(obs_tokens[t1]), tensor(obs_num[t1])
        rewards = tensor(rewards[t1])
        dones = tensor(dones[t1])
        actions = tensor(actions[t])

        return obs_t, actions, rewards, obs_t1, dones
