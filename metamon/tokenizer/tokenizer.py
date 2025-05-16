import json
import os

import numpy as np

UNKNOWN_TOKEN: int = -1


class PokemonTokenizer:
    def __init__(self):
        self._data = {}
        self._frozen = True
        self.name = "custom"

    def unfreeze(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def __len__(self):
        return len(self._data.keys())

    @property
    def all_words(self) -> list[str]:
        return list(self._data.keys())

    @property
    def new_token(self):
        return len(self)

    def __getitem__(self, string: str) -> int:
        if string in self._data:
            return self._data[string]
        return UNKNOWN_TOKEN

    def save_tokens_to_disk(self, path):
        with open(path, "w") as f:
            json.dump(self._data, f)

    def load_tokens_from_disk(self, path):
        with open(path, "r") as f:
            self._data = json.load(f)
        return self

    def add_token_for(self, string: str):
        if string in self._data:
            return
        print(f"Adding: `{string}`")
        self._data[string] = self.new_token

    def sort_tokens(self):
        self._data = {k: i for i, k in enumerate(sorted(self._data.keys()))}

    def tokenize(self, text: str) -> np.ndarray:
        words = text.split(" ")
        if not self._frozen:
            for word in words:
                self.add_token_for(word)
        return np.array([self[word] for word in words], dtype=np.int32)


PREMADE_TOKEN_LISTS = {
    # pre-history token lists for backwards compatibility
    # with old models before release
    "allreplays-v1": "allreplaysv1.json",
    "allreplays-v2": "allreplaysv2.json",
    "allreplays-v3": "allreplaysv3.json",
    # post v1.0 official token lists -- now named by the observation space
    # they are confirmed to be compatible with
    "DefaultObservationSpace-v0": "DefaultObservationSpace-v0.json",
}


def get_tokenizer(choice: str) -> PokemonTokenizer:
    tokenizer = PokemonTokenizer()
    if choice not in PREMADE_TOKEN_LISTS:
        raise KeyError(
            f"`get_tokenizer` `choice = {choice}` is invalid. Options are: {list(PREMADE_TOKEN_LISTS.keys())}"
        )
    path = os.path.join(os.path.dirname(__file__), PREMADE_TOKEN_LISTS[choice])
    tokenizer.load_tokens_from_disk(path)
    tokenizer.name = choice
    return tokenizer


if __name__ == "__main__":
    from argparse import ArgumentParser
    import tqdm

    from metamon.interface import *
    from metamon.datasets import ParsedReplayDataset
    from metamon.data.legacy_team_builder.stat_reader import PreloadedSmogonStat

    parser = ArgumentParser()
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--start_tokens", type=str, default=None)
    parser.add_argument("--save_tokens", type=str, default=None)
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    args = parser.parse_args()

    tokenizer = PokemonTokenizer()
    tokenizer.unfreeze()
    if args.start_tokens:
        tokenizer.load_tokens_from_disk(args.start_tokens)

    dset = ParsedReplayDataset(
        dset_root=args.parsed_replay_root,
        observation_space=eval(args.obs_space)(),
        reward_function=DefaultShapedReward(),
        verbose=True,
    )

    for (obs_seq, *_) in tqdm.tqdm(dset):
        for text_obs in obs_seq["text"]:
            tokenizer.tokenize(text_obs.tolist())

    # catch stray names from Smogon stats
    for gen in range(1, 5):
        for tier in ["ou", "uu", "ubers", "nu"]:
            format = f"gen{gen}{tier}"
            stat = PreloadedSmogonStat(format, inclusive=True)
            for pokemon_name_str, data in tqdm.tqdm(stat._inclusive.items()):
                tokenizer.add_token_for(pokemon_name(pokemon_name_str))

                for ability in data["abilities"]:
                    ability = ability.strip()
                    if ability != "No Ability":
                        tokenizer.add_token_for(clean_no_numbers(ability))

                for move in data["moves"]:
                    move = move.strip()
                    if move.startswith("Hidden Power"):
                        move = "Hidden Power"
                    tokenizer.tokenize(clean_no_numbers(move))

                for item in data["items"]:
                    item = item.strip()
                    if item != "Nothing":
                        tokenizer.tokenize(clean_no_numbers(item))

                for spread in data["spreads"]:
                    nature = spread.split(":")[0].strip()
                    tokenizer.tokenize(clean_no_numbers(nature))

    tokenizer.sort_tokens()

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)
