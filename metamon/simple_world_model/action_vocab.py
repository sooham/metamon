"""Canonical action vocabulary used by the staged simple world model.

The replay tokenizer is intentionally not used as the dynamics action space:
an action such as ``move surf`` is one categorical event for M/C, regardless
of how many text tokens spell it.  Keeping this vocabulary separately also
makes format-specific action masks cheap at training and play time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


ACTION_PAD = "<pad>"
ACTION_NONE = "none"
ACTION_UNKNOWN = "unknown unknown"


def canonicalize_action_text(value: str | Sequence[str] | None) -> str:
    """Return the stable action spelling used in cached sidecars.

    Showdown-facing code commonly writes ``move: surf`` while replay shards
    contain ``move surf``.  Role delimiters and empty actions are normalized
    here so the two paths round-trip through the same vocabulary.
    """
    if value is None:
        return ACTION_NONE
    if isinstance(value, str):
        words = value.strip().lower().replace(":", " ").split()
    else:
        words = [str(word).strip().lower() for word in value if str(word).strip()]
    words = [word for word in words if not (word.startswith("<") and word.endswith(">"))]
    if not words:
        return ACTION_NONE
    try:
        start = next(i for i, word in enumerate(words) if word in {"move", "switch", "none", "unknown"})
        words = words[start:]
    except StopIteration:
        return ACTION_UNKNOWN
    if words[0] == "none":
        return ACTION_NONE
    if words[0] == "unknown":
        return ACTION_UNKNOWN
    if words[0] not in {"move", "switch"} or len(words) < 2:
        return ACTION_UNKNOWN
    return " ".join(words)


def canonicalize_action_ids(
    token_ids: np.ndarray,
    *,
    tokenizer: object,
    pad_id: int,
) -> str:
    """Canonicalize a delimiter-free action token block from a paired shard."""
    ids = [int(token) for token in np.asarray(token_ids).reshape(-1) if int(token) != pad_id]
    if not ids:
        return ACTION_NONE
    words = tokenizer.detokenize(ids)  # type: ignore[attr-defined]
    return canonicalize_action_text(words)


def is_controller_action(action: str) -> bool:
    """Whether behavior cloning is defined for *action*.

    ``none`` and unknown actions must remain in M's timeline because they are
    observations from the replay protocol, but are deliberately not policy
    targets for C.
    """
    return action.startswith("move ") or action.startswith("switch ")


@dataclass
class ActionVocabulary:
    """Stable categorical action vocabulary plus per-format admissibility."""

    id_to_action: list[str] = field(default_factory=lambda: [ACTION_PAD, ACTION_NONE, ACTION_UNKNOWN])
    format_action_ids: dict[str, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id_to_action or self.id_to_action[0] != ACTION_PAD:
            raise ValueError("ActionVocabulary id 0 must be <pad>")
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.action_to_id = {action: idx for idx, action in enumerate(self.id_to_action)}

    def __len__(self) -> int:
        return len(self.id_to_action)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def none_id(self) -> int:
        return self.action_to_id[ACTION_NONE]

    @property
    def unknown_id(self) -> int:
        return self.action_to_id[ACTION_UNKNOWN]

    def add(self, action: str, *, fmt: str | None = None) -> int:
        action = canonicalize_action_text(action)
        action_id = self.action_to_id.get(action)
        if action_id is None:
            action_id = len(self.id_to_action)
            self.id_to_action.append(action)
            self.action_to_id[action] = action_id
        if fmt is not None:
            self.format_action_ids.setdefault(str(fmt), set()).add(action_id)
        return action_id

    def encode(self, action: str | Sequence[str] | None, *, fmt: str | None = None) -> int:
        canonical = canonicalize_action_text(action)
        action_id = self.action_to_id.get(canonical, self.unknown_id)
        if fmt is not None:
            self.format_action_ids.setdefault(str(fmt), set()).add(action_id)
        return action_id

    def decode(self, action_id: int) -> str:
        if 0 <= int(action_id) < len(self.id_to_action):
            return self.id_to_action[int(action_id)]
        return ACTION_UNKNOWN

    def is_controller_id(self, action_id: int) -> bool:
        return is_controller_action(self.decode(action_id))

    def format_mask(self, fmt: str | None = None) -> np.ndarray:
        """Return a bool action mask; special timeline actions stay legal.

        A missing format is deliberately permissive.  It is useful for random
        battles or a new online format and avoids converting a valid observed
        action into an impossible target merely because no cache was built for
        that format yet.
        """
        mask = np.zeros(len(self), dtype=np.bool_)
        mask[self.none_id] = True
        mask[self.unknown_id] = True
        if fmt is None or fmt not in self.format_action_ids:
            mask[1:] = True
            return mask
        for action_id in self.format_action_ids[fmt]:
            if 0 <= action_id < len(mask):
                mask[action_id] = True
        return mask

    def to_state(self) -> dict:
        return {
            "schema_version": 1,
            "id_to_action": list(self.id_to_action),
            "format_action_ids": {
                fmt: sorted(int(action_id) for action_id in ids)
                for fmt, ids in sorted(self.format_action_ids.items())
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "ActionVocabulary":
        vocab = cls(id_to_action=[str(value) for value in state["id_to_action"]])
        raw_masks = state.get("format_action_ids", {})
        vocab.format_action_ids = {
            str(fmt): {int(action_id) for action_id in ids}
            for fmt, ids in dict(raw_masks).items()
        }
        return vocab

    @classmethod
    def build(cls, actions: Iterable[tuple[str, str | None] | str]) -> "ActionVocabulary":
        vocab = cls()
        # Sort canonical strings for deterministic checkpoint/cache hashes.
        normalized: list[tuple[str, str | None]] = []
        for item in actions:
            if isinstance(item, str):
                action, fmt = item, None
            else:
                action, fmt = item
            normalized.append((canonicalize_action_text(action), fmt))
        for action, fmt in sorted(normalized, key=lambda item: (item[0], item[1] or "")):
            vocab.add(action, fmt=fmt)
        return vocab
