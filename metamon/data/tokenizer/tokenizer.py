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
    "gen1fromreplays": "gen1fromreplays.json",
    "allreplays-v1": "allreplaysv1.json",
    "allreplays-v2": "allreplaysv2.json",
    "allreplays-v3": "allreplaysv3.json",
}


def get_tokenizer(choice: str) -> PokemonTokenizer:
    # temporary version control sanity check
    assert choice == "allreplays-v3"
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
    import glob
    import tqdm

    parser = ArgumentParser()
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--online_formats", required=False, nargs="*", default=[])
    parser.add_argument("--online_episodes", type=int, default=20)
    parser.add_argument("--start_tokens", type=str, default=None)
    parser.add_argument("--save_tokens", type=str, default=None)
    args = parser.parse_args()

    tokenizer = PokemonTokenizer()
    tokenizer.unfreeze()
    if args.start_tokens:
        tokenizer.load_tokens_from_disk(args.start_tokens)

    filenames = glob.glob(
        os.path.join(args.parsed_replay_root, "**/*.npz"), recursive=True
    )

    print(len(tokenizer))
    for filename in tqdm.tqdm(filenames):
        try:
            with np.load(filename) as replay:
                text = replay["obs_text"]
                lengths.append(len(text))
                for string in text:
                    tokenizer.tokenize(string)
        except:
            # os.remove(filename)
            print(f"Failed to load: {filename}")

    from metamon.task_distributions import TASK_DISTRIBUTIONS
    from metamon.env import MetaShowdown

    for online_format in args.online_formats:
        env = MetaShowdown(TASK_DISTRIBUTIONS[online_format]())
        for ep in tqdm.tqdm(range(args.online_episodes)):
            obs, info = env.reset()
            done = False
            steps = 0
            while not done:
                tokenizer.tokenize(obs["text"].tolist())
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                steps += 1
                done = terminated or truncated or steps > 10

    tokenizer.sort_tokens()

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)
