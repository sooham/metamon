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

    # ── checkpoint serialization (full state, not just IDs) ──────────

    def to_state(self) -> dict:
        """Return a JSON-serialisable dict of the full tokenizer state.

        Includes the complete token→ID mapping, special-token IDs, and name.
        Safe to store inside a PyTorch checkpoint.
        """
        return {
            "initial_ids": dict(self._initial_ids),
            "new_ids": dict(self._new_ids),
            "unknown_token_id": self.unknown_token_id,
            "pad_token_id": self.pad_token_id,
            "name": self.name,
        }

    @classmethod
    def from_state(cls, state: dict) -> "PokemonTokenizer":
        """Restore a full tokenizer from a ``to_state()`` dict.

        Special tokens that may have been added after the original build
        (e.g. structural tokens, action tokens) are re-ensured.
        """
        tokenizer = cls()
        tokenizer._initial_ids = state["initial_ids"]
        tokenizer._new_ids = state.get("new_ids", {})
        tokenizer.unknown_token_id = state.get("unknown_token_id", 0)
        tokenizer.pad_token_id = state.get("pad_token_id", 0)
        tokenizer.name = state.get("name", "custom")
        tokenizer._reverse_ids = None
        tokenizer._frozen = False
        tokenizer._ensure_special_tokens()
        tokenizer._frozen = True
        return tokenizer

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

    @staticmethod
    def _natural_sort_key(token: str):
        """Sort key so numeric tokens appear in natural (not lexicographic) order.

        Group ordering:
          0. Decimal percentages (``0.00`` – ``1.00``) — sorted by float value
          1. Non-negative integers (``0`` – ``2000``) — sorted by int value
          2. Everything else — lexicographic
        """
        # Decimal percentages: "0.00" through "1.00"
        if re.match(r'^\d+\.\d{2}$', token):
            return (0, float(token), token)
        # Non-negative integers
        if token.isdigit():
            return (1, int(token), token)
        # Everything else (words, structural tokens, etc.)
        return (2, 0, token)

    def sort_tokens(self) -> None:
        base = len(self._initial_ids) + 1
        self._new_ids = {
            k: i + base
            for i, k in enumerate(
                sorted(self._new_ids.keys(), key=self._natural_sort_key)
            )
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
        if added:
            self.sort_tokens()

        self.unknown_token_id = self["<unk>"]
        self.pad_token_id = self["<pad>"]

    def tokenize(self, text: str) -> np.ndarray:
        words = text.split()
        return np.array([self[word] for word in words], dtype=np.int32)

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
    # ── team header (§2) ──
    "<begin_team>", "<end_team>",
    "<begin_opponent_team>", "<end_opponent_team>",
    *[f"<poke{i}>" for i in range(1, 7)],
    *[f"<end_poke{i}>" for i in range(1, 7)],
    # ── state / action frames (§1, §4, §5) ──
    "<bos>", "<eos>", "<boa>", "<eoa>",
    "<format>", "<end_format>", "<turn>", "<end_turn>",
    # ── last_turn_results (§4.3b) ──
    "<last_turn_results>", "<end_last_turn_results>",
    # ── arena (§4.4) ──
    "<arena>", "<end_arena>",
    "<active>", "<end_active>",
    "<opponent>", "<end_opponent>",
    "<active1>", "<end_active1>", "<active2>", "<end_active2>",
    "<opponent1>", "<end_opponent1>", "<opponent2>", "<end_opponent2>",
    # ── available moves (§4.5) ──
    "<begin_moves>", "<end_moves>", "<move>", "<end_move>",
    "<begin_moves:1>", "<begin_moves:2>",
    # ── bench (§4.6) ──
    "<bench>", "<end_bench>",
    # ── conditions (§4.7) ──
    "<conditions>", "<end_conditions>",
    "<empty_conditions>",
    "<you>", "<end_you>", "<you_empty>",
    "<opponent_empty>",
    # ── boosts (§4.4 arena) ──
    "<boosts>", "<end_boosts>",
    # ── action block content (§5) ──
    "<chosen_move>", "<end_chosen_move>",
    "<chosen_move:1>", "<chosen_move:2>",
    "<opponent_chosen_move>", "<end_opponent_chosen_move>",
    "<opponent_chosen_move:1>", "<opponent_chosen_move:2>",
    # ── terminal (§4.8) ──
    "<terminal>", "<end_terminal>",
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
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling files across formats (default: 42).",
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

    # ── Guarantee all Dex Pokémon (including forms, variants, megas) ──
    # The tokenizer must know every Pokémon that could appear in battle,
    # even species that never occur in the training dataset.
    import re as _re
    _dex_gens = sorted(set(
        int(_re.search(r"gen(\d+)", fmt).group(1))
        for fmt in formats
        if _re.search(r"gen(\d+)", fmt)
    ))
    _dex_dir = os.path.join(os.path.dirname(__file__), "..", "backend",
                            "showdown_dex", "static", "pokemon")
    _dex_added = 0
    for _gen in _dex_gens:
        _dex_path = os.path.join(_dex_dir, f"gen{_gen}pokedex.json")
        if not os.path.isfile(_dex_path):
            continue
        with open(_dex_path, "r", encoding="utf-8") as _f:
            _pokedex = orjson.loads(_f.read())
        for _pname in _pokedex:
            if tokenizer.add_token_for(pokemon_name(_pname), verbose=args.verbose):
                _dex_added += 1
    if _dex_added:
        print(f"Added {_dex_added} Dex Pokémon not found in usage stats")

    # Pre-register structural tokens
    for token in WORLD_MODEL_STRUCTURAL_TOKENS:
        tokenizer.add_token_for(token, verbose=args.verbose)
    # Collect all .txt filenames (recursive, matching wm-dataset generator)
    all_filenames = []
    per_format_counts: dict[str, int] = {}
    for fmt in formats:
        fmt_dir = os.path.join(args.parsed_replay_root, fmt)
        if not os.path.isdir(fmt_dir):
            print(f"Skipping {fmt}: directory not found at {fmt_dir}")
            continue
        fmt_files: list[str] = []
        for root, _, files in os.walk(fmt_dir):
            for f in files:
                if f.endswith(".txt"):
                    fmt_files.append(os.path.join(root, f))
        if fmt_files:
            fmt_files.sort()
            all_filenames.extend(fmt_files)
            per_format_counts[fmt] = len(fmt_files)
        else:
            print(f"No .txt files found in {fmt_dir}")

    # Shuffle files across formats so early stopping doesn't favour the
    # first format and n-gram statistics are format-interleaved.
    if len(formats) > 1 and all_filenames:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_filenames)

    fmt_detail = ", ".join(f"{fmt}: {n}" for fmt, n in sorted(per_format_counts.items()))
    print(f"Found {len(all_filenames)} parsed replay .txt files across {len(formats)} formats ({fmt_detail})")

    total_battles = 0
    staleness = 0
    STALENESS_DECAY = 200
    early_stop_battles = args.early_stop

    # Master n-gram counters
    master_ngrams: dict[int, Counter] = {2: Counter(), 3: Counter(), 5: Counter(), 4: Counter()}

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

    # ── Guarantee HP token coverage ──────────────────────────────────
    # The online serializer emits HP as [percentage, current_hp, max_hp]
    # individual tokens.  We must ensure every possible HP percentage
    # (0.00–1.00) and every integer from 0 up to a safe ceiling are in
    # the vocabulary so that online play never produces <unk> for valid
    # HP values, even when the training data did not happen to include
    # every one.
    hp_added = 0
    for i in range(101):
        if tokenizer.add_token_for(f"{i / 100:.2f}"):
            hp_added += 1
    # The ceiling is chosen to cover max-HP values across all standard
    # generations, including doubles (Blissey 714 × 2 with Dynamax ≈ 1428)
    # and edge cases.  2000 is the safe upper bound observed in existing
    # tokenizers built from large multi-generation datasets.
    MAX_HP_INT = 2000
    for i in range(MAX_HP_INT + 1):
        if tokenizer.add_token_for(str(i)):
            hp_added += 1
    if hp_added:
        print(f"Added {hp_added} HP tokens (0.00–1.00, 0–{MAX_HP_INT}) "
              f"not found in training data")

    # Compact vocabulary
    tokenizer.sort_tokens()
    tokenizer._ensure_special_tokens()
    tokenizer._frozen = True

    print(f"Vocabulary: {len(tokenizer)} tokens "
          f"(<unk>={tokenizer.unknown_token_id}, <pad>={tokenizer.pad_token_id})")

    # ── n-gram output ──
    TOP_N = 30
    for n in [2, 3, 4, 5]:
        print(f"\n{'=' * 60}")
        print(f"Top {TOP_N} {n}-grams by count:")
        print(f"{'=' * 60}")
        for ngram, count in master_ngrams[n].most_common(TOP_N):
            print(f"  {count:>8d}  {' '.join(ngram)}")

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)
