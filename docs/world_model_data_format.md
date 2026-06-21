# World Model Data Format

`scripts/generate_world_model_data.py` converts paired parsed-replay `.txt`
files into tokenized `paired_shard_*.npz` files for `metamon.jepa.train_paired`.
The current schema is `paired_pov_rollout_v2`.

## Generation

```bash
uv run python scripts/generate_world_model_data.py \
    --parsed_replay_root $METAMON_CACHE_DIR/parsed-replays \
    --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \
    --output_dir $METAMON_CACHE_DIR/world-model-samples \
    --formats gen1ou \
    --battles_per_shard 1000 \
    --rollout_len 1 \
    --processes 8
```

`--rollout_len K` controls how many contiguous aligned transitions are stored in
one experience-replay sample. `K=1` is the legacy single-transition case. Larger
values produce rollout windows from the same battle:

```text
sample row r, step j:
  state_idx[r, j]       -> state at t+j
  action_idx[r, j]      -> action chosen at t+j
  next_state_idx[r, j]  -> state at t+j+1
```

Rows are only emitted when all `K` transitions are contiguous for both POVs.
Windows that cross skipped subturn gaps are dropped. If a battle has fewer than
`K` aligned action steps, both POVs for that battle are skipped and the generator
prints a warning summary such as `too_short_for_rollout_len_K` or
`no_contiguous_rollout_windows_K`.

## Split And Shuffle

The train/validation split is by raw battle key, so the WIN and LOSS parsed POV
files for a raw battle always land in the same split. After splitting, battle
keys inside each split are shuffled before sharding. Each shard also shuffles
its rollout rows after deciding which battle IDs belong in that shard, so
adjacent rows in a training batch are less likely to be neighboring windows from
the same battle. `PairedJEPADataset` additionally shuffles shard order and row
order when `shuffle_shards=True`.

## Shard Layout

Each shard stores variable-length token arrays plus index matrices:

| Array | Shape | Meaning |
|---|---:|---|
| `p1_states`, `p2_states` | `(total_tokens,)` | Flattened state/header token blocks for each POV |
| `p1_state_offsets`, `p2_state_offsets` | `(num_states,)` | Start offset for each state block |
| `p1_state_lengths`, `p2_state_lengths` | `(num_states,)` | Length for each state block |
| `p1_actions`, `p2_actions` | `(total_action_tokens,)` | Flattened canonical own-action content |
| `p1_opponent_actions`, `p2_opponent_actions` | `(total_action_tokens,)` | Flattened canonical opponent-action content from that POV |
| `*_action_offsets`, `*_action_lengths` | `(num_actions,)` | Offsets and lengths for each action row |
| `p1_state_idx`, `p2_state_idx` | `(num_samples, K)` | Current state indices per rollout step |
| `p1_next_state_idx`, `p2_next_state_idx` | `(num_samples, K)` | Next state indices per rollout step |
| `p1_action_idx`, `p2_action_idx` | `(num_samples, K)` | Action indices per rollout step |
| `battle_id` | `(num_samples,)` | Local battle index for each rollout row |
| `turn_idx`, `turn_number`, `subturn_idx`, `format_id` | `(num_samples, K)` | Metadata per rollout step |
| `p1_won`, `p2_won`, `rank_valid` | `(num_battles,)` | Per-battle labels |
| `p1_battle_start`, `p2_battle_start` | `(num_battles+1,)` | Per-POV cumulative state starts |
| `p1_battle_action_start`, `p2_battle_action_start` | `(num_battles+1,)` | Per-POV cumulative action starts |
| `rollout_len` | scalar | `K` for this shard |

The legacy aliases `state_idx`, `next_state_idx`, `action_idx`, `battle_start`,
and `battle_action_start` are still written for compatibility, but new code
should use the explicit `p1_*` and `p2_*` arrays.

## Action Encoding

Action arrays store only canonical content, without `<chosen_move>` or
`<opponent_chosen_move>` role delimiters:

```text
move bodyslam
switch starmie
unknown unknown
```

Because role delimiters are removed, the same clicked action has the same token
sequence regardless of whether it is viewed as a player's own action or the
opponent's action. In a correctly aligned paired battle, `p2_action` and
`actual_p2_action_from_p1_perspective` encode the same action text, and
`p1_action` and `actual_p1_action_from_p2_perspective` encode the same action
text. The separate names are retained so histories remain perspective-local and
the JEPA losses can name the target they supervise.

## Dataset Sample

`PairedJEPADataset` yields one rollout sample at a time. Before collation, each
value is a list of length `K`:

```python
{
    "p1_state_T": [p1_history_to_T0, ..., p1_history_to_TK],
    "p1_state_T1": [p1_history_to_T1, ..., p1_history_to_TK_plus_1],
    "p1_player_hist_T": [...],
    "p1_opponent_hist_T": [...],
    "p1_player_hist_T1": [...],
    "p1_opponent_hist_T1": [...],

    "p2_state_T": [...],
    "p2_state_T1": [...],
    "p2_player_hist_T": [...],
    "p2_opponent_hist_T": [...],
    "p2_player_hist_T1": [...],
    "p2_opponent_hist_T1": [...],

    "p1_action": [p1_action_0, ..., p1_action_K_minus_1],
    "p2_action": [p2_action_0, ..., p2_action_K_minus_1],
    "actual_p2_action_from_p1_perspective": [...],
    "actual_p1_action_from_p2_perspective": [...],
    "p1_won": [bool, ..., bool],
    "p2_won": [bool, ..., bool],
    "rank_valid": [bool, ..., bool],
}
```

`collate_paired_fn` pads these to tensors with an explicit rollout axis:

| Batch field | Shape |
|---|---:|
| State/history block tensors | `[B, K, max_blocks, max_tokens]` |
| State/history valid masks | `[B, K, max_blocks]` |
| Action tensors | `[B, K, max_action_tokens]` |
| `p1_won`, `p2_won`, `rank_valid` | `[B, K]` |

`max_history_blocks=0` keeps full history. A positive value keeps the team
header plus the last `N` state/action blocks for each rollout step.

## Model Contract

`PairedJEPAModel.forward()` accepts the rollout axis directly. The encoders
preserve the leading `[B, K]` dimensions and emit latent tensors shaped
`[B, K, latent_dim]` or `[B, K, action_latent_dim]`. Losses reduce over both
batch and rollout steps; there is no special flattening path for `K=1`.

All samples in a collated batch must have the same `K`. Mixed rollout lengths in
one batch are rejected by `collate_paired_fn`.
