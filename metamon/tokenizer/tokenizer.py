import orjson
import os
from datetime import date
from typing import Optional

import numpy as np

from metamon.config import SUPPORTED_BATTLE_FORMATS
from metamon.backend.replay_parser.str_parsing import (
    clean_no_numbers,
    pokemon_name,
    move_name,
    clean_name,
)

# ── Special token sentinel IDs (set per-tokenizer after vocabulary is built) ──
# These are instance-level now — no longer a module constant — so that each
# tokenizer can own its IDs.  For backward compatibility the default is 0, but
# after `_ensure_special_tokens()` they point to real, non-zero IDs.
_DEFAULT_UNKNOWN_TOKEN: int = 0
_DEFAULT_PADDING_TOKEN: int = 0

# Backward-compatible module-level exports (deprecated; use tokenizer.unknown_token_id).
UNKNOWN_TOKEN: int = 0
PADDING_TOKEN: int = 0


class PokemonTokenizer:
    """1-based string→int vocabulary with reserved ``<unk>`` and ``<pad>`` tokens.

    Token IDs start at **1** so that index 0 can be left permanently unused
    inside ``nn.Embedding`` (the embedding table still has a slot at 0, but
    no real token ever maps to it).  ``<pad>`` is assigned a non-zero ID and
    passed as ``padding_idx`` to the embedding, which zeroes its vector.

    ``<unk>`` is also a non-zero ID.  Unlike ``<pad>``, loss IS computed on
    ``<unk>`` targets — the model should learn to represent genuinely
    unknown words from context.

    Typical workflow
    ----------------
    1. ``t = PokemonTokenizer()``
    2. ``t._frozen = False``
    3. Scan replays, calling ``t.add_token_for(w, verbose=True)`` for every word.
    4. ``t.sort_tokens()``        # compact, sorted IDs starting at 1
    5. ``t._ensure_special_tokens()``  # appends <unk> and <pad>
    6. ``t._frozen = True``
    7. ``t.save_tokens_to_disk(path)``
    """

    def __init__(self):
        self._initial_ids: dict[str, int] = {}
        self._new_ids: dict[str, int] = {}
        self._frozen: bool = True
        self.name: str = "custom"

        # Set by _ensure_special_tokens() — until then, both default to 0.
        self.unknown_token_id: int = _DEFAULT_UNKNOWN_TOKEN
        self.pad_token_id: int = _DEFAULT_PADDING_TOKEN

    # ── sizing ───────────────────────────────────────────────────────

    def __len__(self):
        return len(self._initial_ids.keys()) + len(self._new_ids.keys())

    @property
    def all_words(self) -> list[str]:
        return list(self._initial_ids.keys()) + list(self._new_ids.keys())

    @property
    def new_token(self) -> int:
        """Next available token ID (called only while unfrozen, before sort)."""
        # Token IDs start at 1 so that 0 is never a real token.
        return len(self) + 1

    # ── lookup ───────────────────────────────────────────────────────

    def __getitem__(self, string: str) -> int:
        """Return the token ID for *string*, or ``unknown_token_id`` if unseen."""
        if string in self._initial_ids:
            return self._initial_ids[string]
        if string in self._new_ids:
            return self._new_ids[string]
        return self.unknown_token_id

    def __contains__(self, string: str) -> bool:
        return string in self._initial_ids or string in self._new_ids

    # ── persistence ──────────────────────────────────────────────────

    def save_tokens_to_disk(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(orjson.dumps({**self._initial_ids, **self._new_ids}))

    def load_tokens_from_disk(self, path: str) -> "PokemonTokenizer":
        with open(path, "rb") as f:
            ids = orjson.loads(f.read())
        # Auto-migrate old 0-based tokenizer files to 1-based
        # (UNKNOWN_TOKEN changed from -1 to 0 in an earlier version).
        if ids and min(ids.values()) == 0:
            ids = {k: v + 1 for k, v in ids.items()}
        self._initial_ids = ids
        self._new_ids = {}          # ← clear stale dynamically-added tokens
        self._reverse_ids = None    # invalidate detokenizer cache
        self._ensure_special_tokens()
        self._frozen = True         # loaded tokenizers are read-only
        return self

    def load_tokens(self, tokens: dict[str, int]) -> "PokemonTokenizer":
        if tokens and min(tokens.values()) == 0:
            tokens = {k: v + 1 for k, v in tokens.items()}
        self._initial_ids = tokens
        self._new_ids = {}          # ← clear stale dynamically-added tokens
        self._reverse_ids = None
        self._ensure_special_tokens()
        self._frozen = True         # loaded tokenizers are read-only
        return self

    # ── vocabulary building ──────────────────────────────────────────

    def add_token_for(self, string: str, verbose: bool = False) -> bool:
        """Register *string*, returning ``True`` iff it was newly added.

        Parameters
        ----------
        string: The token string to register.
        verbose: If True, print a message for every newly-added token.

        Raises
        ------
        RuntimeError: If the tokenizer is frozen.
        """
        if self._frozen:
            raise RuntimeError(
                f"Cannot add token '{string}' to a frozen tokenizer. "
                "Set _frozen = False first if building a vocabulary."
            )
        if string in self._initial_ids:
            return False
        if string in self._new_ids:
            return False
        if verbose:
            print(f"Adding: `{string}`")
        self._new_ids[string] = self.new_token
        return True

    def sort_tokens(self) -> None:
        """Re-number ``_new_ids`` in sorted order with IDs continuing after initial IDs.

        Initial token IDs are 1-based (1 … N).  After sorting, new tokens
        receive IDs ``N+1, N+2, …`` — no collision with any existing token.
        """
        base = len(self._initial_ids) + 1  # first available ID after initial block
        self._new_ids = {
            k: i + base
            for i, k in enumerate(sorted(self._new_ids.keys()))
        }

    def _ensure_special_tokens(self) -> None:
        """Make sure ``<unk>``, ``<pad>``, and action tokens exist with non-zero IDs.

        Called automatically by ``load_tokens_from_disk`` / ``load_tokens``,
        and should be called manually after ``sort_tokens()`` when building
        a tokenizer from scratch.

        If the loaded tokenizer already contains these tokens, their IDs are
        simply adopted.  Otherwise they are appended at the end of the ID
        space.
        """
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

        # Re-sort only if we actually appended new tokens so that their
        # IDs come after the existing vocabulary.
        if added:
            self.sort_tokens()

        self.unknown_token_id = self["<unk>"]
        self.pad_token_id = self["<pad>"]

    def get_action_token_id(self, action_idx: int) -> int:
        """Return the token ID for a world-model action index (-1 … 12).

        Action indices follow the ``UniversalAction`` convention:
        -1 = missing/unknown, 0 = no-op, 1–3 = moves, 4–8 = switches,
        9–12 = tera-boosted moves.
        """
        return self[f"<action_{action_idx}>"]

    # ── tokenize / detokenize ────────────────────────────────────────

    def tokenize(self, text: str) -> np.ndarray:
        """Convert space-delimited *text* to an array of token IDs.

        Uses ``str.split()`` (any whitespace, discards empties).  Words not
        in the vocabulary become ``unknown_token_id``.

        This is a **pure lookup** — it never modifies the vocabulary.
        Use ``add_token_for()`` / ``tokenize_text_only()`` during vocabulary
        building to register new tokens.
        """
        words = text.split()
        return np.array([self[word] for word in words], dtype=np.int32)

    def tokenize_text_only(self, text: str) -> bool:
        """Lightweight tokenize: only register new tokens, returns True if any were new.

        Skips the numpy-array return and the second dict lookup that
        ``tokenize`` performs — for vocabulary-building we only care about
        side effects.
        """
        added_any = False
        for word in text.split():
            if self.add_token_for(word):
                added_any = True
        return added_any

    def detokenize(self, ids: list[int] | np.ndarray) -> list[str]:
        """Convert token IDs back to their string tokens.

        Unknown IDs (including padding) map to ``"<unk>"``.  The special
        ``unknown_token_id`` itself maps to ``"<unk>"``.

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


# ── World-model structural tokens ───────────────────────────────────────
# These tokens are guaranteed to appear in every world-model battle.
# They are added by _ensure_special_tokens() so that both freshly-built and
# legacy tokenizers include them.
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

# ── World-model action tokens (one per action index, -1 … 12) ──────────
# These map raw action indices from parsed replays to text tokens so that
# the world-model prompt can include the action as a regular token instead
# of injecting a separate embedding table.
WORLD_MODEL_ACTION_TOKENS = [
    "<action_-1>",
    *[f"<action_{i}>" for i in range(13)],  # 0 … 12
]

# ── Premade token lists (backward-compatible aliases) ──────────────────

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
            f"`get_tokenizer` `choice = {choice}` is invalid. "
            f"Options are: {list(PREMADE_TOKEN_LISTS.keys())}"
        )
    path = os.path.join(os.path.dirname(__file__), PREMADE_TOKEN_LISTS[choice])
    tokenizer.load_tokens_from_disk(path)
    tokenizer.name = choice
    return tokenizer


# ── helper for parallel tokenizer building ──────────────────────────────

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


# ── CLI vocabulary builder ──────────────────────────────────────────────

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
    parser.add_argument("--save_tokens", type=str, default=None)
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    parser.add_argument("--num_workers", type=int, default=1,
        help="Number of worker threads for parallel replay loading (default: 1 = single-threaded).")
    parser.add_argument("--early_stop", type=int, default=10000,
        help="Stop after this many consecutive battles produce no new tokens (default: 10000).")
    parser.add_argument("--verbose", action="store_true",
        help="Print every newly-added token during vocabulary building.")
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
            formats = [basename]
            args.parsed_replay_root = os.path.dirname(args.parsed_replay_root)
        else:
            formats = SUPPORTED_BATTLE_FORMATS

    tokenizer = PokemonTokenizer()
    tokenizer._frozen = False

    # catch stray names from Smogon stats (only for requested formats)
    for format in formats:
        stat = get_usage_stats(format)
        for pokemon_name_str, data in tqdm.tqdm(stat._inclusive.items()):
            tokenizer.add_token_for(pokemon_name(pokemon_name_str), verbose=args.verbose)

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

    # Pre-register structural tokens (module-level constant).
    # These are guaranteed to appear in every battle, so giving them
    # deterministic low IDs avoids drift across tokenizer builds.
    for token in WORLD_MODEL_STRUCTURAL_TOKENS:
        tokenizer.add_token_for(token, verbose=args.verbose)
    # Action tokens map raw replay action indices to vocabulary entries.
    for token in WORLD_MODEL_ACTION_TOKENS:
        tokenizer.add_token_for(token, verbose=args.verbose)

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
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        # ── sequential path ──
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

    # Compact vocabulary and append <unk>/<pad> with non-zero IDs.
    tokenizer.sort_tokens()
    tokenizer._ensure_special_tokens()
    tokenizer._frozen = True

    print(f"Vocabulary: {len(tokenizer)} tokens "
          f"(<unk>={tokenizer.unknown_token_id}, <pad>={tokenizer.pad_token_id})")

    if args.save_tokens:
        tokenizer.save_tokens_to_disk(args.save_tokens)
