import os
import random
import re
import pathlib
from typing import List, Tuple, Optional, Union, Iterable, Literal

import torch
from torch.utils.data import Dataset

from metamon.data.team_prediction.team import TeamSet
from metamon.data.team_prediction.vocabulary import Vocabulary
from metamon.data import DATA_PATH
from poke_env.data import to_id_str


class TeamPredictionDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Iterable[str]],
        split: Literal["train", "val"] = "train",
        validation_ratio: float = 0.1,
        mask_pokemon_prob_range: Tuple[float, float] = (0.1, 0.1),
        mask_attrs_prob_range: Tuple[float, float] = (0.1, 0.1),
        seed: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory or iterable of directories containing .team files (will be searched recursively)
            split: Whether this is the training or validation split
            validation_ratio: Fraction of data to use for validation
            mask_pokemon_prob: Probability of masking an entire Pokemon
            mask_attrs_prob: Probability of masking individual attributes
            seed: Random seed for reproducibility
        """
        self.mask_pokemon_prob_low, self.mask_pokemon_prob_high = (
            mask_pokemon_prob_range
        )
        self.mask_attrs_prob_low, self.mask_attrs_prob_high = mask_attrs_prob_range
        assert self.mask_pokemon_prob_low <= self.mask_pokemon_prob_high
        assert self.mask_attrs_prob_low <= self.mask_attrs_prob_high
        assert 0 <= validation_ratio < 1, "validation_ratio must be in [0, 1)"

        self.vocab = Vocabulary()
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Accept a string or an iterable of strings for data_dir
        if isinstance(data_dir, str):
            data_dirs = [data_dir]
        else:
            data_dirs = list(data_dir)

        # Collect all team files
        team_files_set = set()
        for d in data_dirs:
            d_path = pathlib.Path(d)
            for f in d_path.rglob("*"):
                if f.is_file() and f.suffix.endswith("team"):
                    team_files_set.add(str(f))
        all_team_files = sorted(team_files_set)

        # Create deterministic train/val split
        n_total = len(all_team_files)
        n_val = int(n_total * validation_ratio)

        # Use a separate random state for splitting to ensure same split regardless of other randomness
        split_rng = random.Random(seed)
        indices = list(range(n_total))
        split_rng.shuffle(indices)

        val_indices = set(indices[:n_val])

        # Assign files based on split
        if split == "train":
            self.team_files = [
                f for i, f in enumerate(all_team_files) if i not in val_indices
            ]
        else:  # val
            self.team_files = [
                f for i, f in enumerate(all_team_files) if i in val_indices
            ]

        print(f"Created {split} split with {len(self.team_files)} team files")

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


class CompetitiveTeamPredictionDataset(TeamPredictionDataset):
    def __init__(
        self,
        mask_pokemon_prob_range: Tuple[float, float] = (0.1, 0.1),
        mask_attrs_prob_range: Tuple[float, float] = (0.1, 0.1),
    ):
        team_dirs = []
        for gen in range(1, 5):
            for tier in ["ou", "uu", "ubers", "nu"]:
                team_dirs.append(
                    os.path.join(DATA_PATH, "teams", f"gen{gen}", tier, "competitive")
                )
        super().__init__(
            data_dir=team_dirs,
            split="train",
            validation_ratio=0.0,
            mask_pokemon_prob_range=mask_pokemon_prob_range,
            mask_attrs_prob_range=mask_attrs_prob_range,
        )


if __name__ == "__main__":
    # Test dataset loading
    dataset = CompetitiveTeamPredictionDataset()

    print(f"Dataset size: {len(dataset)}")

    # Test loading a single item
    x, y = dataset[0]
    print("\nMasked team (x):")
    print(x)
    print("\nComplete team (y):")
    print(y)
