import os
import random
import re
import pathlib
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from metamon.data.team_prediction.team import TeamSet
from metamon.data.team_prediction.vocabulary import Vocabulary
from poke_env.data import to_id_str


class TeamPredictionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mask_pokemon_prob_range: Tuple[float, float] = (0.1, 0.1),
        mask_attrs_prob_range: Tuple[float, float] = (0.1, 0.1),
        seed: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing .team files (will be searched recursively)
            tokenizer: Tokenizer for the team (from vocabulary.py)
            mask_pokemon_prob: Probability of masking an entire Pokemon
            mask_attrs_prob: Probability of masking individual attributes
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.mask_pokemon_prob_low, self.mask_pokemon_prob_high = (
            mask_pokemon_prob_range
        )
        self.mask_attrs_prob_low, self.mask_attrs_prob_high = mask_attrs_prob_range
        assert self.mask_pokemon_prob_low <= self.mask_pokemon_prob_high
        assert self.mask_attrs_prob_low <= self.mask_attrs_prob_high
        self.vocab = Vocabulary()
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        self.team_files = []
        # Find all files with extensions like gen1_team, gen2_team, etc.
        data_dir = pathlib.Path(data_dir)
        self.team_files = [
            str(f)
            for f in data_dir.rglob("*")
            if f.is_file() and f.suffix.endswith("team")
        ]
        print(f"Found {len(self.team_files)} team files")

    def __len__(self) -> int:
        return len(self.team_files)

    def __getitem__(self, idx: int) -> Tuple[TeamSet, TeamSet]:
        """
        Returns:
            x: Masked team
            y: Complete team (ground truth)
        """
        path = self.team_files[idx]
        # Extract format from file extension (e.g. .gen4ou_team -> gen4ou)
        format = to_id_str(os.path.splitext(path)[1].split("_")[0])
        assert format.startswith("gen"), f"Invalid format: {format}"
        team = TeamSet.from_showdown_file(path, format=format)
        mask_pokemon_prob = random.uniform(
            self.mask_pokemon_prob_low, self.mask_pokemon_prob_high
        )
        mask_attrs_prob = random.uniform(
            self.mask_attrs_prob_low, self.mask_attrs_prob_high
        )
        x, y = team.to_prediction_pair(
            mask_pokemon_prob=mask_pokemon_prob,
            mask_attrs_prob=mask_attrs_prob,
        )
        x_seq = x.to_seq(include_stats=False)
        y_seq = y.to_seq(include_stats=False)
        x_tokens = self.vocab.pokeset_seq_to_ints(x_seq)
        y_tokens = self.vocab.pokeset_seq_to_ints(y_seq)
        return x_tokens, y_tokens


if __name__ == "__main__":
    # Test dataset loading
    dataset = TeamPredictionDataset(
        data_dir="/mnt/data1/shared_pokemon_project/metamon_team_files",
        seed=42,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a single item
    x, y = dataset[0]
    print("\nMasked team (x):")
    print(x)
    print("\nComplete team (y):")
    print(y)
