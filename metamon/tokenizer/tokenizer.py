"""Pokémon battle tokenizer — builds vocabulary from new-format parsed replay .txt files.

Token IDs start at 1 so that index 0 is permanently unused inside
``nn.Embedding``.  ``<pad>`` and ``<unk>`` are assigned non-zero IDs.
"""

import os
import re
from typing import Optional
from collections import Counter

import orjson
import numpy as np

from metamon.backend.replay_parser.str_parsing import (
    clean_no_numbers,
    pokemon_name,
    move_name,
    clean_name,
)

# ── Special token defaults ──────────────────────────────────────────────
_DEFAULT_UNKNOWN_TOKEN: int = 0
_DEFAULT_PADDING_TOKEN: int = 0

UNKNOWN_TOKEN: int = 0
PADDING_TOKEN: int = 0


class PokemonTokenizer:
    """1-based string→int vocabulary with reserved ``<unk>`` and ``<pad>`` tokens."""

    def __init__(self):
        self._initial_ids: dict[str, int] = {}
        self._new_ids: dict[str, int] = {}
        self._frozen: bool = True
        self.name: str = "custom"
        self.unknown_token_id: int = _DEFAULT_UNKNOWN_TOKEN
        self.pad_token_id: int = _DEFAULT_PADDING_TOKEN

    def __len__(self):
        return len(self._initial_ids) + len(self._new_ids)

    @property
    def all_words(self) -> list[str]:
        return list(self._initial_ids) + list(self._new_ids)

    @property
    def new_token(self) -> int:
        return len(self) + 1

    def __getitem__(self, string: str) -> int:
        if string in self._initial_ids:
            return self._initial_ids[string]
        if string in self._new_ids:
            return self._new_ids[string]
        return self.unknown_token_id

    def __contains__(self, string: str) -> bool:
        return string in self._initial_ids or string in self._new_ids

    def save_tokens_to_disk(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(orjson.dumps({**self._initial_ids, **self._new_ids}))

    def load_tokens_from_disk(self, path: str) -> "PokemonTokenizer":
        with open(path, "rb") as f:
            ids = orjson.loads(f.read())
        if ids and min(ids.values()) == 0:
            ids = {k: v + 1 for k, v in ids.items()}
        self._initial_ids = ids
        self._new_ids = {}
        self._reverse_ids = None
        self._ensure_special_tokens()
        self._frozen = True
        return self

    def load_tokens(self, tokens: dict[str, int]) -> "PokemonTokenizer":
        if tokens and min(tokens.values()) == 0:
            tokens = {k: v + 1 for k, v in tokens.items()}
        self._initial_ids = tokens
        self._new_ids = {}
        self._reverse_ids = None
        self._ensure_special_tokens()
        self._frozen = True
        return self

    def add_token_for(self, string: str, verbose: bool = False) -> bool:
        if self._frozen:
            raise RuntimeError(
                f"Cannot add token '{string}' to a frozen tokenizer. "
                "Set _frozen = False first if building a vocabulary."
            )
        if string in self._initial_ids or string in self._new_ids:
            return False
        if verbose:
            print(f"Adding: `{string}`")
        self._new_ids[string] = self.new_token
        return True

    def sort_tokens(self) -> None:
        base = len(self._initial_ids) + 1
        self._new_ids = {
            k: i + base
            for i, k in enumerate(sorted(self._new_ids.keys()))
        }

    def _ensure_special_tokens(self) -> None:
        added = False
        if "<unk>" not in self:
            self._new_ids["<unk>"] = self.new_token
            added = True
        if "<pad>" not in self:
            self._new_ids["<pad>"] = self.new_token
            added = True
        for token in WORLD_MODEL_STRUCTURAL_TOKENS:
            if token not in self:
                self._new_ids[token] = self.new_token
                added = True
        for token in WORLD_MODEL_ACTION_TOKENS:
            if token not in self:
                self._new_ids[token] = self.new_token
                added = True

        if added:
            self.sort_tokens()

        self.unknown_token_id = self["<unk>"]
        self.pad_token_id = self["<pad>"]

    def get_action_token_id(self, action_idx: int) -> int:
        return self[f"<action_{action_idx}>"]

    def tokenize(self, text: str) -> np.ndarray:
        words = text.split()
        return np.array([self[word] for word in words], dtype=np.int32)

    def tokenize_text_only(self, text: str) -> bool:
        added_any = False
        for word in text.split():
            if self.add_token_for(word):
                added_any = True
        return added_any

    def detokenize(self, ids: list[int] | np.ndarray) -> list[str]:
        if not hasattr(self, "_reverse_ids") or self._reverse_ids is None:
            self._reverse_ids: dict[int, str] = {}
            for token, tid in self._initial_ids.items():
                self._reverse_ids[tid] = token
            for token, tid in self._new_ids.items():
                self._reverse_ids[tid] = token
        return [self._reverse_ids.get(int(tid), "<unk>") for tid in ids]


# ── World-model structural tokens (new text format) ──────────────────────
WORLD_MODEL_STRUCTURAL_TOKENS = [
    "<begin_team>", "<end_team>",
    "<begin_opponent_team>", "<end_opponent_team>",
    "<bos>", "<eos>", "<boa>", "<eoa>",
    "<format>", "<end_format>", "<turn>", "<end_turn>",
    "<arena>", "<end_arena>",
    "<active>", "<end_active>",
    "<opponent>", "<end_opponent>",
    "<active1>", "<end_active1>", "<active2>", "<end_active2>",
    "<opponent1>", "<end_opponent1>", "<opponent2>", "<end_opponent2>",
    "<begin_moves>", "<end_moves>", "<move>", "<end_move>",
    "<begin_moves:1>", "<begin_moves:2>",
    "<bench>", "<end_bench>",
    "<conditions>", "<end_conditions>",
    "<conditions_empty>",
    "<you>", "<end_you>", "<you_empty>",
    "<opponent_empty>",
    "<boosts>", "<end_boosts>",
    "<chosen_move>", "<end_chosen_move>",
    "<chosen_move:1>", "<chosen_move:2>",
    "<opponent_chosen_move>", "<end_opponent_chosen_move>",
    "<opponent_chosen_move:1>", "<opponent_chosen_move:2>",
    "<terminal>", "<end_terminal>",
]

# ── World-model action tokens ────────────────────────────────────────────
WORLD_MODEL_ACTION_TOKENS = [
    "<action_-1>",
    *[f"<action_{i}>" for i in range(13)],
]

# ── Premade token lists ──────────────────────────────────────────────────

PREMADE_TOKEN_LISTS = {
    "allreplays-v1": "allreplaysv1.json",
    "allreplays-v2": "allreplaysv2.json",
    "allreplays-v3": "allreplaysv3.json",
    "DefaultObservationSpace-v0": "DefaultObservationSpace-v0.json",
    "DefaultObservationSpace-v1": "DefaultObservationSpace-v1.json",
}


def get_tokenizer(choice: str) -> PokemonTokenizer:
    tokenizer = PokemonTokenizer()
    if choice not in PREMADE_TOKEN_LISTS:
        raise KeyError(
            f"`get_tokenizer` `choice = {choice}` is invalid. "
            f"Options are: {list(PREMADE_TOKEN_LISTS.keys())}"
        )
    path = os.path.join(os.path.dirname(__file__), PREMADE_TOKEN_LISTS[choice])
    tokenizer.load_tokens_from_disk(path)
    tokenizer.name = choice
    return tokenizer


# ── New-format file reader ────────────────────────────────────────────────

def _load_text_from_file(filename: str) -> str:
    """Load a single new-format .txt replay file and return its full text content."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def _worker_process_batch_new(args_tuple):
    """Multiprocessing worker: load new-format .txt files and collect tokens + n-grams.

    Args:
        args_tuple: ``(filenames, tokenizer_base_path)``

    Returns:
        ``(new_tokens, ngram_counts, files_processed, files_with_new)``
        where *ngram_counts* is a dict mapping n→Counter of n-gram tuples.
    """
    filenames, tokenizer_base_path = args_tuple

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(tokenizer_base_path)
    tokenizer._frozen = False

    new_tokens = set()
    # n-gram counters: 2-gram, 3-gram, 5-gram
    ngram_counters: dict[int, Counter] = {2: Counter(), 3: Counter(), 5: Counter()}
    files_processed = 0
    files_with_new = 0

    for fn in filenames:
        try:
            text = _load_text_from_file(fn)
        except Exception:
            continue

        words = text.split()
        files_processed += 1
        file_had_new = False

        # Collect n-grams
        for n in ngram_counters:
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i : i + n])
                ngram_counters[n][ngram] += 1

        # Register new tokens
        for word in words:
            if tokenizer.add_token_for(word):
                new_tokens.add(word)
                file_had_new = True

        if file_had_new:
            files_with_new += 1

    return new_tokens, dict(ngram_counters), files_processed, files_with_new


# ── CLI vocabulary builder ──────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp
    import tempfile
    from argparse import ArgumentParser
    import tqdm
    import glob

    from metamon.config import SUPPORTED_BATTLE_FORMATS
    from metamon.backend.team_prediction.usage_stats import get_usage_stats

    parser = ArgumentParser()
    parser.add_argument("--parsed_replay_root", required=True)
    parser.add_argument("--save_tokens", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=10000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--formats", type=str, nargs="+", default=None,
        help="Formats to tokenize (e.g., gen1ou gen9ou).",
    )
    args = parser.parse_args()

    # Determine formats
    if args.formats:
        formats = args.formats
    else:
        basename = os.path.basename(args.parsed_replay_root)
        if basename in SUPPORTED_BATTLE_FORMATS:
            formats = [basename]
            args.parsed_replay_root = os.path.dirname(args.parsed_replay_root)
        else:
            formats = SUPPORTED_BATTLE_FORMATS

    # Build base tokenizer from usage stats
    tokenizer = PokemonTokenizer()
    tokenizer._frozen = False

    for fmt in formats:
        stat = get_usage_stats(fmt)
        for pname, data in tqdm.tqdm(stat._inclusive.items(), desc=f"Loading usage stats for {fmt}"):
            tokenizer.add_token_for(pokemon_name(pname), verbose=args.verbose)
            for ability in data["abilities"]:
                ability = ability.strip()
                if ability != "No Ability":
                    tokenizer.add_token_for(clean_no_numbers(ability), verbose=args.verbose)
            for move in data["moves"]:
                move = move.strip()
                tokenizer.add_token_for(move_name(move), verbose=args.verbose)
            for item in data["items"]:
                item = item.strip()
                if item != "Nothing":
                    tokenizer.add_token_for(clean_no_numbers(item), verbose=args.verbose)
            for spread in data["spreads"]:
                nature = spread.split(":")[0].strip()
                tokenizer.add_token_for(clean_no_numbers(nature), verbose=args.verbose)

    # Pre-register structural tokens
    for token in WORLD_MODEL_STRUCTURAL_TOKENS:
        tokenizer.add_token_for(token, verbose=args.verbose)
    for token in WORLD_MODEL_ACTION_TOKENS:
        tokenizer.add_token_for(token, verbose=args.verbose)

    # Collect all .txt filenames
    all_filenames = []
    for fmt in formats:
        fmt_dir = os.path.join(args.parsed_replay_root, fmt)
        if os.path.isdir(fmt_dir):
            all_filenames.extend(glob.glob(os.path.join(fmt_dir, "*.txt")))
    print(f"Found {len(all_filenames)} parsed replay .txt files across {len(formats)} formats")

    total_battles = 0
    staleness = 0
    STALENESS_DECAY = 200
    early_stop_battles = args.early_stop

    # Master n-gram counters
    master_ngrams: dict[int, Counter] = {2: Counter(), 3: Counter(), 5: Counter()}

    if args.num_workers > 1:
        tmpdir = tempfile.mkdtemp(prefix="tokenizer_")
        base_tok_path = os.path.join(tmpdir, "base_tokenizer.json")
        tokenizer.save_tokens_to_disk(base_tok_path)

        BATCH_SIZE = 500
        batches = [all_filenames[i:i + BATCH_SIZE] for i in range(0, len(all_filenames), BATCH_SIZE)]
        WAVE_SIZE = max(args.num_workers * 4, 16)

        pbar = tqdm.tqdm(total=len(all_filenames), desc="Tokenizing replays")
        pool = mp.Pool(processes=args.num_workers)

        try:
            batch_idx = 0
            stopped_early = False

            while batch_idx < len(batches) and not stopped_early:
                wave = batches[batch_idx:batch_idx + WAVE_SIZE]
                batch_idx += len(wave)
                wave_args = [(b, base_tok_path) for b in wave]

                for new_tokens, ngram_counts, n_battles, n_with_new in pool.imap_unordered(
                    _worker_process_batch_new, wave_args
                ):
                    total_battles += n_battles
                    pbar.update(n_battles)

                    # Merge n-grams
                    for n, counter in ngram_counts.items():
                        master_ngrams[n].update(counter)

                    if n_with_new > 0:
                        for word in new_tokens:
                            tokenizer.add_token_for(word, verbose=args.verbose)
                        staleness = max(0, staleness - STALENESS_DECAY * n_with_new)
                    else:
                        staleness += n_battles
                        if early_stop_battles > 0 and staleness >= early_stop_battles:
                            print(f"\nEarly stopping at battle {total_battles}")
                            stopped_early = True
                            break

                if not stopped_early:
                    tokenizer.save_tokens_to_disk(base_tok_path)

        finally:
            if stopped_early:
                pool.terminate()
            pool.close()
            pool.join()

        pbar.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # Sequential path
        for fn in tqdm.tqdm(all_filenames, desc="Tokenizing replays"):
            try:
                text = _load_text_from_file(fn)
            except Exception:
                continue

            words = text.split()
            total_battles += 1

            # Collect n-grams
            for n in master_ngrams:
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i : i + n])
                    master_ngrams[n][ngram] += 1

            added_in_battle = False
            for word in words:
                if tokenizer.add_token_for(word):
                    added_in_battle = True

            if added_in_battle:
                staleness = max(0, staleness - STALENESS_DECAY)
            else:
                staleness += 1
                if early_stop_battles > 0 and staleness >= early_stop_battles:
                    print(f"\nEarly stopping at battle {total_battles}")
                    break

    print(f"Total battles processed: {total_battles}")

    # Compact vocabulary
    tokenizer.sort_tokens()
    tokenizer._ensure_special_tokens()
    tokenizer._frozen = True

    print(f"Vocabulary: {len(tokenizer)} tokens "
          f"(<unk>={tokenizer.unknown_token_id}, <pad>={tokenizer.pad_token_id})")

    # ── n-gram output ──
    TOP_N = 30
    for n in [2, 3, 5]:
        print(f"\n{'=' * 60}")
        print(f"Top {TOP_N} {n}-grams by count:")
        print(f"{'=' * 60}")
        for ngram, count in master_ngrams[n].most_common(TOP_N):
            print(f"  {count:>8d}  {' '.join(ngram)}")

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)
