import os
from typing import Optional
from collections import defaultdict

import numpy as np
import torch

from metamon.data.tokenizer.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.data.team_prediction.team import PokemonSet
from metamon.data.team_builder.stat_reader import PreloadedSmogonStat


def create_vocabularies():
    # Initialize tokenizers for each vocabulary type
    team_tokenizer = PokemonTokenizer()
    team_tokenizer.unfreeze()
    for gen in range(1, 5):
        for tier in ["ou", "uu", "ubers", "nu"]:
            format = f"Format: gen{gen}{tier}"
            team_tokenizer.add_token_for(format)
    # Add special tokens
    team_tokenizer.add_token_for(f"Mon: {PokemonSet.MISSING_NAME}")
    team_tokenizer.add_token_for(f"Ability: {PokemonSet.MISSING_ABILITY}")
    team_tokenizer.add_token_for(f"Ability: {PokemonSet.NO_ABILITY}")
    team_tokenizer.add_token_for(f"Item: {PokemonSet.MISSING_ITEM}")
    team_tokenizer.add_token_for(f"Item: {PokemonSet.NO_ITEM}")
    team_tokenizer.add_token_for(f"Nature: {PokemonSet.MISSING_NATURE}")
    team_tokenizer.add_token_for(f"Nature: {PokemonSet.NO_NATURE}")
    team_tokenizer.add_token_for(f"Move: {PokemonSet.MISSING_MOVE}")
    team_tokenizer.add_token_for(f"Move: {PokemonSet.NO_MOVE}")
    team_tokenizer.add_token_for(f"EV: {PokemonSet.MISSING_EV}")
    team_tokenizer.add_token_for(f"IV: {PokemonSet.MISSING_IV}")

    # Populate vocabularies from Smogon stats
    for gen in range(1, 5):
        for tier in ["ou", "uu", "ubers", "nu"]:
            format = f"gen{gen}{tier}"
            stat = PreloadedSmogonStat(format, inclusive=True)

            for pokemon_name, data in stat._inclusive.items():
                team_tokenizer.add_token_for(f"Mon: {pokemon_name}")

                for ability in data["abilities"]:
                    ability = ability.strip()
                    if ability != "No Ability":
                        team_tokenizer.add_token_for(f"Ability: {ability}")

                for move in data["moves"]:
                    move = move.strip()
                    # have to fold Hidden Power types into one move since
                    # that's how they'd appear in replays and team files.
                    if move.startswith("Hidden Power"):
                        move = "Hidden Power"
                    team_tokenizer.add_token_for(f"Move: {move}")

                for item in data["items"]:
                    item = item.strip()
                    if item != "Nothing":
                        team_tokenizer.add_token_for(f"Item: {item}")

                for spread in data["spreads"]:
                    nature = spread.split(":")[0].strip()
                    team_tokenizer.add_token_for(f"Nature: {nature}")

    # Sort all tokenizers
    team_tokenizer.sort_tokens()
    team_tokenizer.freeze()
    return team_tokenizer


class _TeamTokenizer(PokemonTokenizer):
    def __init__(self):
        super().__init__()
        self._inv_data = None

    @property
    def all_words(self) -> list[str]:
        return sorted(self._data.keys(), key=lambda k: self._data[k])

    def load_tokens_from_disk(self, path):
        super().load_tokens_from_disk(path)
        self._inv_data = {v: k for k, v in self._data.items()}
        return self

    def tokenize(self, seq: list[str]) -> np.ndarray:
        for i, s in enumerate(seq):
            if s.startswith("Move:") and "Hidden Power" in s:
                print(f"Found Hidden Power: {s}")
                breakpoint()
                seq[i] = "Move: Hidden Power"

        out = np.array([self[s] for s in seq], dtype=np.int32)
        for i, token in enumerate(out):
            if token == UNKNOWN_TOKEN:
                print(f"Unknown token: {seq[i]}")
                breakpoint()
        return out

    def invert(self, tokens: np.ndarray) -> list[str]:
        out = []
        for token in tokens:
            if token in self._inv_data:
                out.append(self._inv_data[token])
            else:
                out.append(f"<unknown>")
        return out


class Vocabulary:
    def __init__(self):
        vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
        self.tokenizer = _TeamTokenizer().load_tokens_from_disk(vocab_path)
        prefixes = ["Mon:", "Ability:", "Item:", "Nature:", "Move:", "EV:", "IV:"]
        for prefix in prefixes:
            attr_name = f"{prefix.lower().rstrip(':')}_mask"
            setattr(
                self,
                attr_name,
                [
                    i
                    for i, token in enumerate(self.tokenizer.all_words)
                    if token.startswith(prefix)
                ],
            )
        self.masks = {
            "mon": self.mon_mask,
            "ability": self.ability_mask,
            "item": self.item_mask,
            "nature": self.nature_mask,
            "move": self.move_mask,
            "ev": self.ev_mask,
            "iv": self.iv_mask,
        }
        self.type_ids = defaultdict(
            lambda: UNKNOWN_TOKEN,
            {
                "Format": 0,
                "Mon": 1,
                "Ability": 2,
                "Item": 3,
                "Nature": 4,
                "Move": 5,
                "EV": 6,
                "IV": 6,
            },
        )

    def pokeset_seq_to_ints(self, seq: list[str]) -> np.ndarray:
        tokens = self.tokenizer.tokenize(seq)
        type_ids = np.array(
            [self.type_ids[s.split(":")[0].strip()] for s in seq], dtype=np.int32
        )
        return tokens, type_ids

    def ints_to_pokeset_seq(self, ints: np.ndarray) -> list[str]:
        return self.tokenizer.invert(ints)

    def filter_probs(self, probs: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        # TODO
        breakpoint()
        # return filtered_probs


if __name__ == "__main__":
    import os

    vocabularies = create_vocabularies()
    vocabularies.save_tokens_to_disk("vocab.json")
