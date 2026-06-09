# World Model Dataset Generator — Design Plan

## Overview

A modular dataset generator that converts parsed Pokémon replay files into
`(state_history, action, next_state)` training pairs suitable for world model
training (next-state prediction given past states and the chosen action).

## Data Format

### Input: Parsed Replay JSON

```json
{
  "states": [state_0, state_1, ..., state_N],
  "actions": [a_0, a_1, ..., a_N]
}
```

- `states[t]` → full battle state at turn `t` (serialized `UniversalState`)
- `actions[t]` → action chosen by the player at state `t`:
  - `0–3`: use move (alphabetically sorted)
  - `4–8`: switch to bench Pokémon (alphabetically sorted)
  - `9–12`: Tera + move (gen 9 only)
  - `-1`: not observed / no-op (terminal, recharge, etc.)

**Relationship**: `(state[t], action[t]) → state[t+1]` for `t = 0 … N-1`.
The final `actions[N]` is typically `-1` and unused.

### Output: Training Sample

Each sample is a tuple `(state_history, action, next_state)`:

```
state_history: list of observations from t=0 to t=T (or fixed-length window)
action:        integer action index chosen at state[T]
next_state:    observation at state[T+1]
```

Where an **observation** is the output of `WorldModelObservationSpace.state_to_obs()`:

```python
{
    "text":   np.str_    # whitespace-separated string, 310 tokens
    "numbers": np.float32 # shape (63,)
}
```

After tokenization (see §Tokenization below), the text becomes an integer array
of shape `(310,)` with values in `[0, vocab_size)`.

### HP Encoding

HP is treated as **fixed-point** (not floating point). Values are rounded to 2 decimal
places (`round(hp_pct, 2)`) and stored as space-separated characters in the text:

```
1 . 0 0    (100% HP)
0 . 7 3    (73% HP)
0 . 0 0    (0% HP / fainted)
```

This means the tokenizer only needs 11 tokens for all HP values: digits `0`–`9`
and `.`. Without this, each unique HP string like `0.73` would be a separate
token, bloating the vocabulary by ~100 entries.

Each HP value consumes 4 text tokens (was 1 before character-level split).
There are 22 HP values per state (active, 5 bench, opponent active, 5 opponent bench,
5 fainted, 5 opponent fainted).

### Text Token Count (310 total)

| Section | Tokens |
|---------|--------|
| Format + forced switch | 2 |
| Player active (name + 4-char HP + item + ability + 2 types + effect + status) | 12 |
| 4 moves × (name + type + category) | 16 |
| 5 bench slots × (name + 4-char HP + item + ability + moveset + 4 moves) | 65 |
| Opponent active (same as player) | 12 |
| 5 opponent bench slots | 65 |
| 5 fainted slots | 65 |
| 5 opponent fainted slots | 65 |
| Conditions (weather + player + opponent) | 4 |
| Previous moves (player + opponent) | 4 |
| **Total** | **310** |

## Module Structure

```
metamon/data/world_model_dataset.py   # PyTorch Dataset + generator
metamon/tokenizer/tokenizer.py         # Already exists — may need minor updates
```

### Class: `WorldModelDataset`

A PyTorch `Dataset` that lazily loads parsed replays and yields training samples.

```python
class WorldModelDataset(torch.utils.data.Dataset):
    """
    Args:
        replay_dir:      Path to directory of parsed replay JSON files.
        obs_space:       WorldModelObservationSpace instance.
        tokenizer:        PokemonTokenizer instance (frozen).
        context_mode:     "full" (all past states) or "window" (fixed K).
        context_len:      Number of past states when mode="window" (default 8).
        min_context:      Minimum number of past states to include (default 1).
        skip_missing:     Skip samples where action == -1 (default True).
        shuffle_files:    Shuffle file order (default True).
    """
```

### Class: `WorldModelDataGenerator`

A standalone generator that processes a single replay file and yields samples.
Used internally by `WorldModelDataset` but also callable standalone.

```python
class WorldModelDataGenerator:
    """
    Processes one parsed replay file into training samples.

    Args:
        replay_path:     Path to a parsed replay JSON file.
        obs_space:       WorldModelObservationSpace instance.
        tokenizer:        PokemonTokenizer instance.
        context_mode:     "full" or "window".
        context_len:      Window size for "window" mode.
        min_context:      Minimum history length.
        skip_missing:     Skip samples with missing actions.
    """

    def __iter__(self) -> Iterator[Sample]:
        ...
```

### Sample dataclass

```python
@dataclass
class WorldModelSample:
    state_texts:     np.ndarray    # (seq_len, 310) int32  — tokenized text
    state_numbers:   np.ndarray    # (seq_len, 63) float32 — numerical features
    action:          int           # chosen action index
    next_text:       np.ndarray    # (310,) int32  — tokenized next text
    next_numbers:    np.ndarray    # (63,) float32 — next numerical features
    seq_len:         int           # number of past states in this sample
```

## Tokenization Pipeline

### Step 1: Build Vocabulary

Train the `PokemonTokenizer` on the `text` field of `WorldModelObservationSpace`
across the full parsed replay dataset (or a representative subset).

```bash
python -m metamon.tokenizer.tokenizer \
    --parsed_replay_root /path/to/parsed-replays \
    --obs_space WorldModelObservationSpace \
    --save_tokens world_model_tokens.json
```

This iterates over all replays, tokenizes every state's text, and accumulates
unique words. The tokenizer's `unfreeze()` mode adds new words on-the-fly.
After scanning the dataset, `sort_tokens()` assigns sorted IDs and the result
is saved as JSON.

**Estimated vocabulary size**: The `DefaultObservationSpace` (87 tokens)
has ~2,000 unique words (`DefaultObservationSpace-v1.json`). The
`WorldModelObservationSpace` (310 tokens) adds Pokémon names,
fainted/bench structural tokens, and HP digits. Expect **~3,000–5,000**
unique tokens (HP only contributes 11: `0`–`9` + `.`).

### Step 2: Tokenize During Dataset Generation

The `WorldModelDataGenerator` uses a frozen tokenizer to convert each state's
text string into an integer array. The `WorldModelObservationSpace.tokenizable`
property indicates that the `"text"` key should be tokenized to length 310.

### Step 3: Unknown Handling

Any word not in the vocabulary maps to `UNKNOWN_TOKEN = -1`. This should be
rare if the tokenizer was trained on the full dataset.

**HP tokenization**: HP is fixed-point with character-level splitting.
The tokenizer sees individual digits `0`–`9` and `.`, so only 11 tokens
cover all possible HP values. No binning needed.

## Context Modes

### Mode 1: Full History (variable-length)

```
Sample 0: ([s0],        a0, s1)
Sample 1: ([s0, s1],    a1, s2)
Sample 2: ([s0, s1, s2], a2, s3)
...
```

- **Pros**: Model sees all past information; no information loss.
- **Cons**: Variable-length sequences are harder to batch; requires padding
  or packing. Long battles (50+ turns) produce very long sequences.

**Batching strategy**: Pad to `max_seq_len` within a batch, use attention
mask. Or use PyTorch's `pack_padded_sequence`.

### Mode 2: Sliding Window (fixed-length)

With `context_len = 4`, `min_context = 1`:

```
Sample 0: ([s0],          a0, s1)     # pad with 3 empty states
Sample 1: ([s0, s1],      a1, s2)     # pad with 2 empty states
Sample 2: ([s0, s1, s2],  a2, s3)     # pad with 1 empty state
Sample 3: ([s0, s1, s2, s3], a3, s4)  # full window
Sample 4: ([s1, s2, s3, s4], a4, s5)  # sliding forward
...
```

- **Pros**: Fixed-length → easy batching. Compatible with standard transformer
  architectures.
- **Cons**: Older context is lost when the window slides past it.

**Padding**: Empty/padding states use a special "empty" observation (all zeros
for numbers, all `<blank>` tokens for text, or a dedicated padding token).

### Recommendation

Start with **Mode 1 (full history)** for maximum flexibility. The world model
can learn its own compression of history. If memory becomes an issue, switch
to Mode 2.

## Output Storage Format

| Format | Pros | Cons |
|--------|------|------|
| **JSONL** | Human-readable, easy to inspect | Large file size, slow to load |
| **.npz** | Compressed, fast NumPy I/O | Must load entire file into memory |
| **.pt** (PyTorch) | Directly loadable by DataLoader | PyTorch-specific |
| **Memory-mapped .npy** | O(1) random access, low memory | Complex setup |
| **Parquet** | Columnar, compressed, fast | Requires pyarrow |

**Recommendation**: Use **JSONL** for development/debugging and **memory-mapped
.npy** or **Parquet** for production training. The generator should support
multiple output formats via a configurable writer.

## Processing Flow

```
┌─────────────────────┐
│ Parsed Replay JSON  │
│ {states, actions}   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────┐
│ WorldModelDataGenerator     │
│                             │
│ For t in 0..N-1:           │
│   1. Tokenize states[0..t] │
│   2. Get action[t]         │
│   3. Tokenize states[t+1]  │
│   4. Yield sample          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Output Writer       │
│ (JSONL / .npz / .pt)│
└─────────────────────┘
```

## Integration with Existing Code

### Dependencies

- `metamon.interface.WorldModelObservationSpace` — produces text + numbers
- `metamon.tokenizer.PokemonTokenizer` — word-level tokenizer
- `metamon.data.parsed_replay_dset.ParsedReplayDataset` — existing replay loader
  (can be reused for file discovery)

### File Changes

| File | Change |
|------|--------|
| `metamon/data/world_model_dataset.py` | **NEW** — Dataset + generator |
| `metamon/tokenizer/tokenizer.py` | Minor: support `WorldModelObservationSpace` tokenizable length (310) |
| `metamon/tokenizer/WorldModelObservationSpace-v0.json` | **NEW** — vocabulary file |
| `scripts/generate_world_model_data.py` | **NEW** — CLI script for batch generation |

## Action Encoding

Actions are already integers in the parsed JSON. The mapping is:

| Action Range | Meaning |
|-------------|---------|
| `-1` | Missing / no-op (Recharge, sleep, flinch) |
| `0–3` | Use move (moves sorted alphabetically) |
| `4–8` | Switch to bench Pokémon (sorted alphabetically) |
| `9–12` | Tera + move (gen 9 only) |

**World model action encoding**: Keep as a single integer. The world model
predicts the next state given this action; it doesn't need to know the action
space structure. If needed, the action can be embedded via a lookup table.

**Missing actions** (`-1`): Should be skipped during training (the player
had no choice). The `skip_missing` flag controls this.

## Next Steps

1. Implement `WorldModelDataGenerator` — single-file processor
2. Implement `WorldModelDataset` — PyTorch Dataset with file-level shuffling
3. Update tokenizer to support `WorldModelObservationSpace`
4. Build vocabulary from the dataset (`--obs_space WorldModelObservationSpace`)
5. Create CLI script for batch generation
6. Add unit tests for round-trip correctness
