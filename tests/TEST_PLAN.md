# Test Plan: Granular Replay Parser Tests (Fixed Battles)

This document outlines a test plan for writing granular, targeted tests against
the replay parsing pipeline using **fixed, known battle IDs**.  Each test loads a
specific raw replay JSON, runs the component under test, and compares the result
against a **golden output file** stored in `./tests/golden_outputs/`.

## Golden output convention

```
tests/golden_outputs/
├── gen1ou-316031019/
│   ├── raw.json                         # copy of the raw replay (for reference)
│   ├── forward.json                     # serialized ParsedReplay after forward_fill
│   ├── pov_win.json                     # POVReplay → states+actions for WIN side
│   ├── pov_loss.json                    # POVReplay → states+actions for LOSS side
│   ├── e2e_win.json                     # final serialized UniversalState/Action (WIN)
│   └── e2e_loss.json                    # final serialized UniversalState/Action (LOSS)
├── gen1ou-XXXXXXX/                      # next fixed battle...
└── ...
```

**Golden generation command** (you run once when you pick a new fixed battle):
```
python -m tests.generate_golden --battle-id gen1ou-316031019
```
This script (`tests/generate_golden.py`, you write it) parses the replay once,
serializes the intermediate and final structures, and writes them to disk.
Subsequent test runs compare against these snapshots.  If the parser behavior
changes *intentionally*, you regenerate the goldens.

---

## 1. Fixed battle selection criteria

Choose 1–2 battles per generation that exercise specific mechanics.  The
current `FIXED_TEST_BATTLES` has `gen1ou-316031019` (a faint+switch edge case).
You should add at least one battle per gen that is:

- **Vanilla** — no Transform, no Zoroark, no Mimic, no foreign-called moves.
  6v6, both sides play a normal battle.  This is your "everything works" baseline.
- **Mechanically interesting** — exercises one or more of the tricky cases below.

For Gen 1, `gen1ou-316031019` already covers a faint+forced-switch mid-turn
scenario.  You might also want a vanilla gen1 battle and optionally one with
Transform (Ditto).

---

## 2. Forward-fill tests (`forward.py`)

### 2.1 Basic structural invariants (per-battle, golden comparison)

After running `forward_fill()` on the raw replay, serialize the `ParsedReplay`
and compare against `forward.json`.  This single golden comparison covers:

| Property | Where it lives |
|---|---|
| `gen` is correct | `ParsedReplay.gen` |
| `format` string | `ParsedReplay.format` |
| `winner` is correct | `ParsedReplay.winner` |
| `players[0]`, `players[1]` are correct | `ParsedReplay.players` |
| `ratings` are correct | `ParsedReplay.ratings` |
| Turn count matches expected | `len(replay.turnlist)` |
| Turn numbers are sequential | each `turn.turn_number` |
| Team sizes ≤ 6 | each `turn.pokemon_1`, `turn.pokemon_2` |
| Active slots are length 2 | `turn.active_pokemon_1`, `turn.active_pokemon_2` |

If the golden comparison passes, most structural invariants are automatically
verified.

### 2.2 Per-turn semantic checks (golden comparison — what to serialize)

When generating `forward.json`, serialize enough detail per turn to catch
regressions in the message handlers.  Suggested structure:

```json
{
  "gameid": "...",
  "gen": 1,
  "format": "gen1ou",
  "winner": "PLAYER_1",
  "players": ["Alice", "Bob"],
  "ratings": [1200, "Unrated"],
  "rules": ["Species Clause: ..."],
  "check_warnings": [],
  "showteam_data": null,
  "turns": [
    {
      "turn_number": 0,
      "pokemon_1": [{"name": "Charizard", "had_name": "Charizard", "unique_id": "uuid1", "lvl": 100, "current_hp": 100, "max_hp": 100, "status": "NO_STATUS", "moves": {}, "had_moves": {}, "had_item": null, "had_ability": null, "active_item": null, "active_ability": null, "type": ["Fire", "Flying"], "base_stats": {"hp": 78, "atk": 84, ...}, "boosts": {...}, "tera_type": "NO_TERA_TYPE", "transformed_into": null}, ...],
      "pokemon_2": [...],
      "active_pokemon_1": [...],
      "active_pokemon_2": [...],
      "moves_1": [{"name": "Switch", "is_switch": true, "is_noop": false, "user": "Charizard", "target": "Blastoise"}, null],
      "moves_2": [{"name": "Earthquake", "is_switch": false, ...}, null],
      "choices_1": [...],
      "choices_2": [...],
      "subturns": [...],
      "weather": "NO_WEATHER",
      "battle_field": {},
      "conditions_1": {},
      "conditions_2": {},
      "can_tera_1": false,
      "can_tera_2": false,
      "is_force_switch": false,
      "replacements_1": [],
      "replacements_2": [],
      "teampreview_1": [...],
      "teampreview_2": [...]
    }
  ]
}
```

This covers **every field** the parser touches.  One golden comparison catches
regressions in any message handler.

### 2.3 Targeted error-condition tests

These test specific exception paths.  They use synthetic logs (not real replays)
because you need exact control over the log content.

| Test | What to assert |
|---|---|
| `UnfinishedReplayException` | < 5 turns in the log → raise |
| `NoSpeciesClause` | Species Clause rule missing from `|rule|` lines → raise |
| `SoftLockedGen` | `|gen|5` (unsupported) → raise |
| `Scalemons` | Rule contains "Scalemons Mod" → raise |
| `ZoroarkException` | `|replace|` message where the replacement isn't Zoroark/Zorua → raise |
| `UnimplementedMessage` | Unknown pipe message type → raise |
| `UnfinishedMessageException` | Truncated `|move|`, `|-damage|`, or `|switch|` messages → raise |

The existing `test_forward_edge_cases.py` already has the first three.  You
could expand it or add a new `test_forward_errors.py` that tests the full set.

### 2.4 Specific mechanic tests (per-battle, golden comparison)

For battles that exercise specific mechanics, add targeted assertions on top
of the golden comparison.  These are the "does this specific turn look right?"
tests.

#### 2.4.1 Faint + forced switch (gen1ou-316031019, already in helpers)

- Turn 23: active Zapdos faints → forced switch state created
- Replacement is Snorlax (not Tauros)
- Action target points to Snorlax
- `is_force_switch` is True on the correct state

Already covered by `test_faint_switch_regression.py`.  When you golden-ify it,
the golden comparison should catch most of this automatically.

#### 2.4.2 Transform (Ditto / Mew)

Find or add a fixed battle with `|-transform|`.  Test:

- `transformed_into` pointer is set on the transforming Pokémon
- `transformed_this_turn` is True on the transform turn
- The opposing Pokémon's moves appear on the transformer's `moves` dict
- After transformation ends (switch-out), `transformed_into` is cleared
- `check_warnings` contains `WarningFlags.TRANSFORM`
- Moves discovered during the transformation window are propagated backward
  by `_resolve_transforms`

#### 2.4.3 Zoroark / Zorua (Illusion) — Gen 5+ only, but if you add those later

- `|replace|` message → `_parse_replace` fires
- `check_warnings` contains `WarningFlags.ZOROARK`
- The disguise Pokémon's state is rewound (items/abilities/moves that were
  unique to the illusion window transfer to Zoroark)
- `replacements_1` or `replacements_2` contains a `Replacement` tuple
- `_resolve_zoroark` fixes action targets that pointed to the disguise

#### 2.4.4 Foreign-summoned moves (Metronome, Sleep Talk, Mirror Move)

Find a battle where a Pokémon uses Metronome or Sleep Talk.  Test:

- The called move is NOT added to `had_moves` (Metronome case)
- The called move IS added to `had_moves` (Sleep Talk case — if it draws from own moveset)
- `pending_foreign_move` is set during the foreign-call sequence
- Follow-up turns of the called multi-turn move (e.g. Metronome → Outrage) are
  suppressed
- `_pending_foreign_charge_remaining` is handled for charge moves (e.g.
  Metronome → Solar Beam)

#### 2.4.5 Consecutive / multi-turn moves (Outrage, Thrash, Petal Dance)

Find a battle with Outrage or Petal Dance.  Test:

- First turn: move revealed and PP used
- Second/third turn: move auto-repeated, no additional PP used, `[still]` tag handled
- Confusion after the move ends is handled
- If the move is called by Metronome, the entire sequence is suppressed

#### 2.4.6 Gen 1 PP rollover (Wrap, Bind, Fire Spin, Clamp)

Find a battle with Wrap or Bind.  Test:

- First use of Bind: PP decremented normally
- When PP hits 0, the rollover to 63 is applied (`pp_used = -63`)
- Subsequent auto-continuation turns use 0 PP (`pp_used = 0`)

#### 2.4.7 Forced switches (U-turn, Volt Switch, Roar, Dragon Tail, etc.)

Find battles with U-turn / Volt Switch.  Test:

- `mark_forced_switch` creates a subturn
- The subturn is filled when the replacement switches in
- If the move fails (blocked by Protect, absorbed by Lightning Rod), the subturn
  is cancelled via `remove_empty_subturn`
- Eject Button / Eject Pack create forced switches via `_parse_item_enditem`
- Red Card forces the attacker out

#### 2.4.8 Revival Blessing

Find a Gen 9 battle with Revival Blessing.  Test:

- Fainted Pokémon is revived (status changes from FNT to NO_STATUS)
- The action is marked with `is_revival=True`
- The revived Pokémon becomes a legal switch target

#### 2.4.9 Tera (Gen 9 only)

Find a Gen 9 battle with Terastallization.  Test:

- `|-terastallize|` sets `tera_type` on the Pokémon
- The move action has `is_tera=True`
- `can_tera_1` / `can_tera_2` is toggled (only one Tera per battle per side)
- `type` changes to the Tera type (single element)

#### 2.4.10 Item manipulation (Trick, Thief, Knock Off)

Find battles with Trick, Knock Off, or Thief.  Test:

- Trick: items swap between two Pokémon (`tricking` pointers set, then resolved)
- Knock Off: target's `active_item` becomes `NO_ITEM`, `had_item` revealed
- Thief/Covet: item stolen from named target
- `BackwardMarkers.FORCE_UNKNOWN` set when the item can never be known

#### 2.4.11 Ability tracking (Trace, Intimidate, Skill Swap)

Find battles with Trace.  Test:

- `|-ability|` with `[from] ability: Trace [of] <target>` — the tracer reveals its own
  ability and copies the target's
- `had_ability` is the Pokémon's original ability, `active_ability` is the copied one
- Skill Swap exchanges abilities between two Pokémon
- `ABILITY_OVERWRITES_ABILITY` (Mummy, Lingering Aroma, Wandering Spirit) reveals
  both abilities

#### 2.4.12 Weather, field conditions, side conditions

- Weather set (`|-weather|RainDance`) and cleared (`|-weather|none`)
- Field conditions (`|-fieldstart|Electric Terrain`)
- Side conditions (`|-sidestart|p1|move: Stealth Rock`)
- Stackable conditions (Spikes — counter increments)
- Conditions swap (`|-swapsideconditions`)

#### 2.4.13 Status and volatile effects

- Status applied (`|-status|p1a: Gengar|slp`), cured (`|-curestatus|`)
- Volatile effects started/ended (`|-start|p1a: Slowbro|Leech Seed`, `|-end|`)
- `|cant|` messages (paralysis, sleep, flinch — action is left as None)
- `|-mustrecharge|` → "Recharge" no-op action
- `|-immune|` → cancels opponent's switch-out move via `_cancel_opponent_switch_based_on_user_immunity`
- `|-fail|` → cancels switch-out move via `_cancel_user_switch_based_on_failure`

#### 2.4.14 Boosts and boost manipulation

- `|-boost|` / `|-unboost|` — individual stat changes
- `|-swapboost|` (Heart Swap, Guard Swap)
- `|-clearboost|`, `|-clearallboost|`, `|-clearpositiveboost|`, `|-clearnegativeboost|`
- `|-copyboost|`, `|-invertboost|`, `|-restoreboost|`, `|-setboost|`

#### 2.4.15 HP manipulation

- `|-damage|` / `|-heal|` — standard damage/heal
- `|-sethp|` — direct HP set (Pain Split, Endeavor, Super Fang)
- `fnt` in HP string → `current_hp = 0`, `status = FNT`
- `[from] item: Life Orb` → reveals item
- `[from] ability: ...` → reveals ability

#### 2.4.16 Choice message parsing

Find battles that have `|choice|` messages (not all do).  Test:

- Named move choices (`move Ice Beam`) are parsed and added to `choices_1`/`choices_2`
- Move is revealed on the active Pokémon's `had_moves`
- Numeric choices (`move 1`) are skipped (current limitation)
- Empty choices (`|choice||`) are skipped
- Choices for Pokémon that already have 4 known moves are skipped (assumed to be
  pre-emptive choices for the replacement)
- Choices during `pending_foreign_move` are suppressed

#### 2.4.17 Team preview

- `|poke|` messages populate `teampreview_1` / `teampreview_2` (frozen copies)
- Team preview Pokémon don't change with the battle state
- `|teamsize|` adjusts team list lengths
- Team size ≠ 6 produces a warning but doesn't crash

#### 2.4.18 Forme changes and detailschange

- `|detailschange|` / `|-formechange|` — `had_name` persists, `name` changes
- Pokedex info is updated via `update_pokedex_info`

#### 2.4.19 Showteam data

- `|showteam|` messages are stored in `ParsedReplay.showteam_data` (not applied during forward fill)
- The packed format is rejoined correctly after `clean_log` splits on `|`

---

## 3. Backward-fill tests (`backward.py`)

### 3.1 Basic structural invariants (per-battle, golden comparison)

After running `backward_fill()` with `NoPredictor`, serialize both POVReplays and
compare against `pov_win.json` and `pov_loss.json`.

| Property | Where it lives |
|---|---|
| `povturnlist` is non-empty | each POV |
| `actionlist` length matches `povturnlist` | each POV |
| `winner` is consistent (p1 wins ⇔ p2 loses) | cross-POV |
| `format`, `gen`, `gameid` match | cross-POV |
| `rating` is set | each POV |
| `revealed_team` is not None | each POV |
| `actionlist` entries are length-2 lists | each POV |

### 3.2 Backfill propagation checks

These are things the golden comparison won't easily catch because they're
cross-turn relationships.

#### 3.2.1 Information flows backward

- A Pokémon that reveals its item on turn 5 has that item set on turn 4, 3, 2, 1, 0
- A move revealed on turn 10 appears in `had_moves` on all prior turns where that
  Pokémon existed
- HP values propagate backward (a Pokémon seen at 200/300 HP on turn 3 has
  `current_hp=200, max_hp=300` on turns 0–2 as well, unless it took damage)
- Ability and item propagate backward

#### 3.2.2 No opponent backfill leakage (with NoPredictor)

- On turn 0, the opponent's bench Pokémon have 0 moves in `had_moves`
- Opponent items and abilities are only set from what the forward pass observed
- Only the active opponent Pokémon may have partial info on turn 0

#### 3.2.3 Prediction fills player side only (with NaiveUsagePredictor)

- Player's Pokémon on turn 0 have > 0 moves (predicted)
- Opponent's bench Pokémon on turn 0 still have 0 moves
- Player's Pokémon have 4 moves filled on the final turn
- Cross-POV: P1's opponent species ⊆ P2's player species (prediction adds to
  player, not opponent)

#### 3.2.4 Showteam data as ground truth

If the battle has `|showteam|` messages:

- `showteam_data` is unpacked and applied (items, abilities, all 4 moves)
- No usage stats are consulted (no tar.gz access)
- Pokémon that never appeared in the battle are still filled from showteam data
- showteam's display names (e.g. "Altaria-Mega") are matched to `had_name` (base species)

#### 3.2.5 Missing info remains missing (with NoPredictor)

- With `NoPredictor`, unrevealed items stay `None`
- Unrevealed abilities stay `None`
- Unrevealed moves stay `{}` / empty
- The backward pass doesn't fabricate data when the predictor says "I don't know"

### 3.3 Backward-specific mechanic tests

#### 3.3.1 Transform resolution (`_resolve_transforms`)

- Moves the transformed opponent had at the time of transformation are propagated
  to the transformer through the entire transformation window
- After transformation ends, the transformer's moves revert to its own
- Moves copied via Transform start with 5 PP (`Move.from_transform`)

#### 3.3.2 Zoroark resolution (`_resolve_zoroark`)

- Action targets that pointed to the disguise are changed to point to Zoroark
- The disguise Pokémon's moves are replaced with Zoroark's real moves for the
  illusion window (so action validation passes)
- Items/abilities transferred to Zoroark during `_parse_replace` are also copied
  to the disguise for the illusion window

#### 3.3.3 State-action alignment (`_align_states_actions`)

- Subturns generate state-action pairs with the forced switch action
- The main action list combines `moves_*` (preferred) and falls back to `choices_*`
- The last action in the list is always `None` (terminal state)
- `actionlist` entries are length 2 (singles format — slot 'b' is always None)

---

## 4. End-to-end tests (`parse_replays.py`)

### 4.1 Full pipeline golden comparison

Run `ReplayParser.parse_replay()` on the fixed battle and compare the output
JSON files against `e2e_win.json` and `e2e_loss.json`.

This covers:
- `clean_log()` → correct splitting of the raw log string
- `forward_fill()` → correct ParsedReplay
- `backward_fill()` → correct POVReplay
- `povreplay_to_state_action()` → correct state/action extraction
- `state_action_to_obs_action_reward()` → correct UniversalState/Action conversion
- `save_to_disk()` → correct serialization

### 4.2 Serialized output structure

| Check | Detail |
|---|---|
| `states` is a list of dicts | every state |
| `actions` is a list of ints | every action |
| `len(states) == len(actions)` | |
| Required keys present in every state | `format`, `player_active_pokemon`, `opponent_active_pokemon`, `available_switches`, `opponent_bench`, `fainted_pokemon`, `opponent_fainted`, `player_prev_move`, `opponent_prev_move`, `opponents_remaining`, `player_conditions`, `opponent_conditions`, `weather`, `battle_field`, `forced_switch`, `battle_won`, `battle_lost`, `can_tera`, `opponent_teampreview` |
| Action indices in `[-1, 13]` | |
| Last action is `-1` (no action from terminal state) | |
| Terminal state has `battle_won=True` or `battle_lost=True` | |
| Non-terminal states have both `False` | |
| Format string matches the replay's format | |
| Weather is `None`, `str`, or `int` | |
| `battle_field` is `dict` or `"nofield"` | |
| `opponent_teampreview` is a list | |

### 4.3 File naming convention

- Output filename encodes gameid, rating, players, date, and WIN/LOSS
- Both POV files are produced (WIN and LOSS)
- With `compress=False`, files have `.json` extension
- With `compress=True`, files have `.json.lz4` extension

### 4.4 Error handling

- `ForwardException` during parse → logged to `error_history["Forward"]`
- `BackwardException` during parse → logged to `error_history["Backward"]`
- Corrupted JSON → warning, skipped
- `summarize_errors()` returns a dict of exception types → file paths

---

## 5. Patterns for writing the tests

### 5.1 Test file organization (suggested)

```
tests/
├── test_fixed_forward.py       # Fixed-battle forward tests + golden comparison
├── test_fixed_backward.py      # Fixed-battle backward tests + golden comparison
├── test_fixed_e2e.py           # Fixed-battle e2e tests + golden comparison
├── test_forward_mechanics.py   # Per-mechanic targeted tests (Transform, Zoroark, etc.)
├── test_backward_mechanics.py  # Per-mechanic backward tests
├── test_forward_errors.py      # Synthetic-log error-condition tests
├── generate_golden.py          # Script to (re)generate golden outputs
└── golden_outputs/             # Golden files directory
```

### 5.2 Fixture pattern for a fixed battle

```python
# In conftest.py or a dedicated file
@pytest.fixture(scope="module")
def fixed_gen1ou_forward():
    """Load gen1ou-316031019 and run forward_fill."""
    return run_forward_fill_on_fixed("gen1ou")

@pytest.fixture(scope="module")
def fixed_gen1ou_pov():
    """Load gen1ou-316031019 and run full parse with NoPredictor."""
    return run_full_parse_on_fixed("gen1ou", NoPredictor())

@pytest.fixture(scope="module")
def fixed_gen1ou_e2e(tmp_path_factory):
    """Run full pipeline end-to-end on gen1ou-316031019."""
    ...
```

### 5.3 Golden comparison pattern

```python
def test_forward_golden(fixed_gen1ou_forward):
    """Forward fill output matches golden."""
    golden = load_golden("gen1ou-316031019", "forward.json")
    actual = serialize_parsed_replay(fixed_gen1ou_forward)
    assert actual == golden  # or use deepdiff / custom compare for tolerance
```

### 5.4 What to serialize for the golden

The key question is: **what should `serialize_parsed_replay()` output?**  You want
enough detail to catch regressions but not so much that uuid changes break every
test.  Recommendations:

- **Use `unique_id`** in the golden but make comparisons tolerant of UUID changes
  (e.g., compare by position/turn rather than exact UUID value, or use a
  deterministic fake UUID seeded on the battle ID).
- **Serialize only the fields that matter for correctness.**  For `Pokemon`, that's:
  name, had_name, unique_id, lvl, current_hp, max_hp, status, type, had_type,
  active_item, had_item, active_ability, had_ability, moves (name+pp), had_moves
  (name+pp), boosts, tera_type, transformed_into (unique_id), effects.
- **For `Action`:** name, is_switch, is_noop, is_tera, is_revival, user
  (unique_id), target (unique_id).
- **For `Turn`:** all the lists above, plus weather, battle_field, conditions,
  replacements, teampreviews, subturns.
- **For `ParsedReplay`:** all metadata + list of turns.

Use `orjson` for consistent serialization (it's already used throughout the
codebase and sorts dict keys deterministically).

### 5.5 Tolerance for intentional changes

When you fix a bug or improve the parser, the golden will differ.  That's
expected.  The `generate_golden.py` script regenerates the goldens from the
current parser output.  Always review the diff before checking in new goldens.

---

## 6. Priority ordering — what to test first

If you're building incrementally, here's the recommended order:

1. **Vanilla gen1 battle golden comparison (forward)** — catches 80% of regressions
2. **Vanilla gen1 battle golden comparison (backward, NoPredictor)** — catches
   backfill bugs
3. **Vanilla gen1 battle golden comparison (e2e)** — catches pipeline integration
   bugs
4. **Synthetic error-condition tests** — `NoSpeciesClause`, `UnfinishedReplay`,
   `SoftLockedGen`, `Scalemons`, `UnimplementedMessage`
5. **Faint + forced switch (gen1ou-316031019)** — the existing regression test,
   now golden-ified
6. **Transform test** — find a battle with Ditto, golden-compare forward+backward
7. **prediction tests (NaiveUsagePredictor)** — golden-compare backward with
   prediction, verify player-only fills
8. **Mechanic-specific tests** — one per tricky mechanic from section 2.4
9. **Additional gens** — repeat the vanilla + mechanic tests for gen2, gen3, gen4, gen9
