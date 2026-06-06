import orjson
import os
from datetime import date

import numpy as np

from metamon.config import SUPPORTED_BATTLE_FORMATS
from metamon.backend.replay_parser.str_parsing import (
    clean_no_numbers,
    pokemon_name,
    move_name,
    clean_name,
)

UNKNOWN_TOKEN: int = 0  # non-negative so it works cleanly with nn.Embedding(padding_idx=0)


class PokemonTokenizer:
    def __init__(self):
        self._initial_ids: dict[str, int] = {}
        self._new_ids: dict[str, int] = {}
        self._frozen: bool = True
        self.name: str = "custom"

    def unfreeze(self):
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def __len__(self):
        return len(self._initial_ids.keys()) + len(self._new_ids.keys())

    @property
    def all_words(self) -> list[str]:
        return list(self._initial_ids.keys()) + list(self._new_ids.keys())

    @property
    def new_token(self):
        # Token IDs start at 1 so that 0 is reserved for UNKNOWN_TOKEN.
        return len(self) + 1

    def __getitem__(self, string: str) -> int:
        if string in self._initial_ids:
            return self._initial_ids[string]
        if string in self._new_ids:
            return self._new_ids[string]
        return UNKNOWN_TOKEN

    def save_tokens_to_disk(self, path):
        with open(path, "wb") as f:
            f.write(orjson.dumps({**self._initial_ids, **self._new_ids}))

    def load_tokens_from_disk(self, path):
        with open(path, "rb") as f:
            ids = orjson.loads(f.read())
        # Auto-migrate old 0-based tokenizer files to 1-based
        # (UNKNOWN_TOKEN changed from -1 to 0).
        if ids and min(ids.values()) == 0:
            ids = {k: v + 1 for k, v in ids.items()}
        self._initial_ids = ids
        self._reverse_ids = None  # invalidate detokenizer cache
        return self

    def load_tokens(self, tokens: dict[str, int]):
        if tokens and min(tokens.values()) == 0:
            tokens = {k: v + 1 for k, v in tokens.items()}
        self._initial_ids = tokens
        self._reverse_ids = None  # invalidate detokenizer cache
        return self

    def add_token_for(self, string: str) -> bool:
        """Add a token, returning True if it was newly added (vs already known)."""
        if string in self._initial_ids:
            return False
        if string in self._new_ids:
            return False
        print(f"Adding: `{string}`")
        self._new_ids[string] = self.new_token
        return True

    def sort_tokens(self) -> None:
        self._new_ids = {
            k: i + len(self._initial_ids)
            for i, k in enumerate(sorted(self._new_ids.keys()))
        }

    def tokenize(self, text: str) -> np.ndarray:
        words = text.split(" ")
        if not self._frozen:
            for word in words:
                self.add_token_for(word)
        return np.array([self[word] for word in words], dtype=np.int32)

    def tokenize_text_only(self, text: str) -> bool:
        """Lightweight tokenize: only register new tokens, returning True if any were new.

        Skips the numpy-array return and the second dict lookup (``self[word]``)
        that ``tokenize`` performs after adding tokens — for vocabulary-building
        we only care about side effects.
        """
        added_any = False
        for word in text.split(" "):
            if self.add_token_for(word):
                added_any = True
        return added_any

    def detokenize(self, ids: list[int] | np.ndarray) -> list[str]:
        """Convert token IDs back to their string tokens.

        Unknown IDs (including UNKNOWN_TOKEN=0 used as padding) map to ``<unk>``.

        Args:
            ids: A list or 1-D numpy array of integer token IDs.

        Returns:
            A list of string tokens, one per input ID.
        """
        # Build the reverse mapping lazily (cached after first call).
        if not hasattr(self, "_reverse_ids") or self._reverse_ids is None:
            self._reverse_ids: dict[int, str] = {}
            for token, tid in self._initial_ids.items():
                self._reverse_ids[tid] = token
            for token, tid in self._new_ids.items():
                self._reverse_ids[tid] = token
        return [self._reverse_ids.get(int(tid), "<unk>") for tid in ids]


PREMADE_TOKEN_LISTS = {
    # pre-history token lists for backwards compatibility
    # with old models before release
    "allreplays-v1": "allreplaysv1.json",
    "allreplays-v2": "allreplaysv2.json",
    "allreplays-v3": "allreplaysv3.json",
    # post v1.0 official token lists -- now named by the observation space
    # they are confirmed to be compatible with
    "DefaultObservationSpace-v0": "DefaultObservationSpace-v0.json",
    # adds ~1k new words for gen 9
    "DefaultObservationSpace-v1": "DefaultObservationSpace-v1.json",
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


def _load_text_observations(filename: str, obs_space_template) -> list[str]:
    """Load a single replay file and extract all text observations.

    This function is designed to be called from worker threads.  Each thread
    deep-copies *obs_space_template* so that the mutable per-episode state
    (e.g. ``reset()`` / ``state_to_obs()``) is never shared across threads.

    Returns a list of space-joined token strings, one per battle state.
    """
    import copy
    import orjson
    import lz4.frame

    from metamon.interface import UniversalState

    if filename.endswith(".json.lz4"):
        with lz4.frame.open(filename, "rb") as f:
            data = orjson.loads(f.read())
    else:
        with open(filename, "r") as f:
            data = orjson.loads(f.read())

    states = [UniversalState.from_dict(s) for s in data["states"]]
    obs_space = copy.deepcopy(obs_space_template)
    obs_space.reset()
    obs_list = [obs_space.state_to_obs(s) for s in states]
    return [o["text"].tolist() for o in obs_list]


if __name__ == "__main__":
    import copy
    import concurrent.futures
    from argparse import ArgumentParser
    import tqdm

    from metamon.interface import (
        get_observation_space,
        DefaultShapedReward,
        DefaultActionSpace,
    )
    from metamon.data import ParsedReplayDataset
    from metamon.backend.team_prediction.usage_stats import get_usage_stats

    parser = ArgumentParser()
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--start_tokens", type=str, default=None)
    parser.add_argument("--save_tokens", type=str, default=None)
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    parser.add_argument("--num_workers", type=int, default=1,
        help="Number of worker threads for parallel replay loading (default: 1 = single-threaded).")
    parser.add_argument("--early_stop", type=int, default=10000,
        help="Stop after this many consecutive battles produce no new tokens (default: 10000).")
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Formats to tokenize (e.g., gen1ou gen9ou). "
            "If not provided, auto-detects from the parsed_replay_root directory name "
            "or falls back to all supported formats."
        ),
    )
    args = parser.parse_args()

    # Determine which formats to process
    if args.formats:
        formats = args.formats
    else:
        basename = os.path.basename(args.parsed_replay_root)
        if basename in SUPPORTED_BATTLE_FORMATS:
            # parsed_replay_root points to a single format directory (legacy mode)
            formats = [basename]
            args.parsed_replay_root = os.path.dirname(args.parsed_replay_root)
        else:
            formats = SUPPORTED_BATTLE_FORMATS

    tokenizer = PokemonTokenizer()
    tokenizer.unfreeze()
    if args.start_tokens:
        tokenizer.load_tokens_from_disk(args.start_tokens)

    # catch stray names from Smogon stats (only for requested formats)
    for format in formats:
        stat = get_usage_stats(format)
        for pokemon_name_str, data in tqdm.tqdm(stat._inclusive.items()):
            tokenizer.add_token_for(pokemon_name(pokemon_name_str))

            for ability in data["abilities"]:
                ability = ability.strip()
                if ability != "No Ability":
                    tokenizer.add_token_for(clean_no_numbers(ability))

            for move in data["moves"]:
                move = move.strip()
                tokenizer.tokenize(move_name(move))

            for item in data["items"]:
                item = item.strip()
                if item != "Nothing":
                    tokenizer.tokenize(clean_no_numbers(item))

            for spread in data["spreads"]:
                nature = spread.split(":")[0].strip()
                tokenizer.tokenize(clean_no_numbers(nature))

    # Pre-register structural tokens used by every world-model-compatible
    # observation space.  These are guaranteed to appear in every battle, so
    # giving them deterministic low IDs avoids drift across tokenizer builds.
    WORLD_MODEL_STRUCTURAL_TOKENS = [
        "<opponent_moveset>",
        "unknownmove",
        "<ongoing>",
        "<won>",
        "<lost>",
        "<opponent_switch>",
        "<fainted>",
        "<opponent_fainted>",
        "<bos>",
        "<eos>",
        "<boa>",
        "<eoa>",
    ]
    for token in WORLD_MODEL_STRUCTURAL_TOKENS:
        tokenizer.add_token_for(token)

    obs_space = get_observation_space(args.obs_space)
    dset = ParsedReplayDataset(
        dset_root=args.parsed_replay_root,
        formats=formats,
        observation_space=obs_space,
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        verbose=True,
        shuffle=True,
    )

    total_dataset_size = 0
    battles_processed = 0
    # Staleness counter: increments for clean battles, decays for battles with new tokens.
    # A single new token costs STALENESS_DECAY battles of progress instead of resetting
    # to zero, so rare long-tail tokens don't prevent early exit.
    staleness = 0
    STALENESS_DECAY = 200
    early_stop_battles = args.early_stop

    def _process_text_str(text_str: str, tokenizer) -> bool:
        """Return True if any new token was registered."""
        return tokenizer.tokenize_text_only(text_str)

    if args.num_workers > 1:
        # ── parallel path: threads load files, main thread tokenizes ──
        obs_space_template = copy.deepcopy(obs_space)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            future_to_fn = {
                executor.submit(_load_text_observations, fn, obs_space_template): fn
                for fn in dset.filenames
            }
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_fn),
                total=len(dset),
                desc="Tokenizing replays",
            ):
                added_in_battle = False
                for text_str in future.result():
                    total_dataset_size += 1
                    if _process_text_str(text_str, tokenizer):
                        added_in_battle = True
                battles_processed += 1
                if added_in_battle:
                    staleness = max(0, staleness - STALENESS_DECAY)
                else:
                    staleness += 1
                    if staleness >= early_stop_battles:
                        print(
                            f"\nEarly stopping at battle {battles_processed} "
                            f"(staleness={staleness}, vocabulary size={len(tokenizer)})"
                        )
                        # Cancel pending futures so shutdown doesn't block
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        # ── sequential path (original behaviour) ──
        for obs_seq, *_ in tqdm.tqdm(dset):
            added_in_battle = False
            for text_obs in obs_seq["text"]:
                total_dataset_size += 1
                if _process_text_str(text_obs.tolist(), tokenizer):
                    added_in_battle = True
            battles_processed += 1
            if added_in_battle:
                staleness = max(0, staleness - STALENESS_DECAY)
            else:
                staleness += 1
                if staleness >= early_stop_battles:
                    print(
                        f"\nEarly stopping at battle {battles_processed} "
                        f"(staleness={staleness}, vocabulary size={len(tokenizer)})"
                    )
                    break

    print(f"Total dataset size: {total_dataset_size}")

    tokenizer.sort_tokens()

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)

    if args.start_tokens and args.save_tokens:
        original_tokenizer = PokemonTokenizer()
        original_tokenizer.load_tokens_from_disk(args.start_tokens)
        new_tokenizer = PokemonTokenizer()
        new_tokenizer.load_tokens_from_disk(args.save_tokens)

        for token, id in original_tokenizer._initial_ids.items():
            if token not in new_tokenizer._initial_ids:
                print(f"Token `{token}` is missing from the new tokenizer")
            elif new_tokenizer._initial_ids[token] != id:
                print(f"Token `{token}` has the wrong id in the new tokenizer")

        for word in original_tokenizer.all_words:
            if word not in new_tokenizer.all_words:
                print(f"Word `{word}` is missing from the new tokenizer")
