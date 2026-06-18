# New Parsed-Replay Format Specification (v2)

## Overview

Each raw replay produces **two** parsed-replay text files — one per player POV
(`WIN` / `LOSS`).  A file is a sequence of state blocks (`<bos>`…`<eos>`) and
action blocks (`<boa>`…`<eoa>`) interleaved, preceded by a team header.

The format is **stateful**: each state shows only the battlefield information
visible to the POV player at that moment.  The model derives cumulative
knowledge by reading the state sequence from beginning to end.  No history is
aggregated into individual states.

**Tag rules:**

- Every structural tag has a matching close tag, always in `<end_*>` form
  (never `</foo>` XML-style).
- Value tokens — species names, status codes, weather, boost tokens, etc. —
  are bare words without angle brackets.

**Generation awareness** is built in: fields that don't exist in a generation
(items in Gen 1, abilities in Gen 1–2, Tera outside Gen 9, field effects before
Gen 4, team preview before Gen 5) are simply omitted from the output.

---

## 1. File Structure (top-down)

```
<begin_team>
  <poke1> … <end_poke1>
  …
<end_team>

<begin_opponent_team>       ← only when team preview exists (Gen 5+); species only
  <poke1> … <end_poke1>
  …
<end_opponent_team>

<bos>                       ← state 0 (team preview / lead selection)
  …
<eos>
<boa>                       ← action 0 (lead choice)
  …
<eoa>

<bos>                       ← state 1
  …
<eos>
<boa>                       ← action 1
  …
<eoa>

…                           ← repeating state/action pairs

<bos>                       ← terminal state (final battlefield)
  …
  <terminal>won<end_terminal>
<eos>
```

### Singles vs. Doubles
The format defaults to singles.  For doubles the arena block uses `<active1>` /
`<active2>` / `<opponent1>` / `<opponent2>` and action blocks carry `slot="1"` /
`slot="2"` attributes (see §4, §6).

---

## 2. Team Header (`<begin_team>` / `<end_team>`)

Shows the POV player's full team with backward-filled knowledge: all species,
movesets, items, and abilities are known.  Pokémon are ordered alphabetically by
canonical (cleaned) species name, matching `consistent_pokemon_order()`.

### 2.1 Per-Pokémon entry

```
<pokeN>
<species> <type1> <type2> [<item>] [<ability>]
<begin_moves>
<move>move1<end_move>
<move>move2<end_move>
<move>move3<end_move>
<move>move4<end_move>
<end_moves>
<end_pokeN>
```

| Field | Gen 1 | Gen 2 | Gen 3+ | Notes |
|-------|-------|-------|--------|-------|
| `species` | ✓ | ✓ | ✓ | canonical dex name |
| `type1` | ✓ | ✓ | ✓ | e.g. `water` |
| `type2` | ✓ | ✓ | ✓ | omitted if single-typed |
| `item` | ✗ | ✓ | ✓ | e.g. `leftovers`; omitted in Gen 1 |
| `ability` | ✗ | ✗ | ✓ | e.g. `intimidate`; omitted in Gen 1–2 |
| `<begin_moves>`… | ✓ | ✓ | ✓ | 4 moves, alphabetically ordered |

Moves are shown as **names only** in the team header (no type/category — those
appear in the per-state `<begin_moves>` blocks).

**Example (Gen 1):**
```
<begin_team>
<poke1>
alakazam psychic
<begin_moves>
<move>psychic<end_move>
<move>recover<end_move>
<move>seismictoss<end_move>
<move>thunderwave<end_move>
<end_moves>
<end_poke1>
<poke2>
chansey normal
<begin_moves>
<move>counter<end_move>
<move>icebeam<end_move>
<move>seismictoss<end_move>
<move>thunderwave<end_move>
<end_moves>
<end_poke2>
<poke3>
jinx ice psychic
<begin_moves>
<move>blizzard<end_move>
<move>bodyslam<end_move>
<move>lovelykiss<end_move>
<move>rest<end_move>
<end_moves>
<end_poke3>
<poke4>
snorlax normal
<begin_moves>
<move>bodyslam<end_move>
<move>hyperbeam<end_move>
<move>reflect<end_move>
<move>rest<end_move>
<end_moves>
<end_poke4>
<poke5>
starmie psychic water
<begin_moves>
<move>psychic<end_move>
<move>recover<end_move>
<move>surf<end_move>
<move>thunderwave<end_move>
<end_moves>
<end_poke5>
<poke6>
tauros normal
<begin_moves>
<move>blizzard<end_move>
<move>bodyslam<end_move>
<move>earthquake<end_move>
<move>hyperbeam<end_move>
<end_moves>
<end_poke6>
<end_team>
```

**Example (Gen 9):**
```
<begin_team>
<poke1>
gholdengo steel ghost leftovers goodasgold
<begin_moves>
<move>focusblast<end_move>
<move>makeitrain<end_move>
<move>nastyplot<end_move>
<move>shadowball<end_move>
<end_moves>
<end_poke1>
<poke2>
greattusk ground fighting heavy-dutyboots protosynthesis
<begin_moves>
<move>earthquake<end_move>
<move>headlongrush<end_move>
<move>icepinner<end_move>
<move>rapidspin<end_move>
<end_moves>
<end_poke2>
…
<end_team>
```

---

## 3. Opponent Team Preview (`<begin_opponent_team>`)

Only emitted when the raw replay contains `|teampreview|` data (Gen 5+).
Species only (no moves, items, or abilities — those aren't revealed at preview).

```
<begin_opponent_team>
<poke1>
<species>
<end_poke1>
<poke2>
<species>
<end_poke2>
…
<end_opponent_team>
```

Order: alphabetical by canonical species name.  If team preview is absent
(Gen 1–4), this entire block is omitted.

---

## 4. State Block (`<bos>` … `<eos>`)

Every state is a snapshot of what the POV player can observe at one timestep.

### 4.1 Structure

```
<bos>
<format>gen1ou<end_format>
<turn>1<end_turn>
<arena>
<active>
<species> <hp> <type1> <type2> [<item>] [<ability>] <effect> <status> [tera:<type>] <boosts_section>
<end_active>
<opponent>
<species> <hp> <type1> <type2> [<item>] [<ability>] <effect> <status> [tera:<type>] <boosts_section>
<end_opponent>
<end_arena>
<begin_moves>
<move>
<name> <type> <category>
<end_move>
…                           ×0–4 (active Pokémon's currently available moves)
<end_moves>
<bench>
<pokeN>
<species> <hp> <type1> <type2> [<item>] [<ability>] [<status>]
<end_pokeN>
…                           ×0–5 (benched + fainted, only when present)
<end_bench>
<conditions>
<weather> [<battle_field>]
<you> [forceswitch|forcedrevival|cantera] [<side_cond>…] <end_you>
<opponent> [<side_cond>…] <end_opponent>
<end_conditions>
<terminal>won<end_terminal>   ← only in final state
<eos>
```

### 4.2 `<format>`
The battle format string, e.g. `gen1ou`, `gen9ou`, `gen9ubers`.  On one line:
```
<format>gen1ou<end_format>
```

### 4.3 `<turn>`
The turn number from the raw replay's `|turn|N` message.  Turn 0 is the
pre-battle state (team preview / lead selection).  On one line:
```
<turn>1<end_turn>
```

### 4.4 `<arena>` — Active Pokémon

#### Singles
```
<arena>
<active>
<species> <hp> <type1> <type2> [<item>] [<ability>] <effect> <status> [tera:<type>] <boosts_section>
<end_active>
<opponent>
<species> <hp> <type1> <type2> [<item>] [<ability>] <effect> <status> [tera:<type>] <boosts_section>
<end_opponent>
<end_arena>
```

#### Doubles
```
<arena>
<active1>
…<end_active1>
<active2>
…<end_active2>
<opponent1>
…<end_opponent1>
<opponent2>
…<end_opponent2>
<end_arena>
```

#### Field details

| Position | Content | Examples | Notes |
|----------|---------|----------|-------|
| `species` | Canonical dex name | `snorlax`, `rotom-wash` | For Transform: `ditto snorlax` (actual-species transformed-species); for Zoroark from own POV: real species shown |
| `hp` | Fixed-point `X.XX` | `1.00`, `0.63`, `0.00` | Two decimals always. `unknown` only before first sighting (edge case). |
| `type1` / `type2` | Type names | `fire`, `water` | second type omitted for single-typed Pokémon |
| `item` | Item name or `unknownitem` | `leftovers`, `lifeorb`, `unknownitem` | For POV player: always known. For opponent: `unknownitem` until revealed. Omitted in Gen 1. |
| `ability` | Ability name or `unknownability` | `intimidate`, `levitate`, `unknownability` | Omitted Gen 1–2. For opponent: `unknownability` until revealed. |
| `condition` | Combined effect+status+boosts, or `clean` | `clean`, `par`, `slp`, `noeffect par noboosts`, `<boosts> atk+1 <end_boosts>` | `clean` when all three are default (no effect, healthy, zero boosts). Otherwise space-separated individual tokens. |
| `tera:<type>` | Tera type | `tera:rock` | Gen 9 only. |
| `boosts_section` | `noboosts` or `<boosts> … <end_boosts>` | `noboosts`, `<boosts> atk+1 spa-2 <end_boosts>` | Only on active Pokémon (reset on switch-out). Part of the `condition` field. |

**Gen-conditional omissions:**

| Field | Gen 1 | Gen 2 | Gen 3–8 | Gen 9 |
|-------|-------|-------|---------|-------|
| `item` | ✗ | ✓ | ✓ | ✓ |
| `ability` | ✗ | ✗ | ✓ | ✓ |
| `tera:<type>` | ✗ | ✗ | ✗ | ✓ |

**Concrete example (Gen 1):**
```
<arena>
<active>
jinx 1.00 ice psychic par noboosts
<end_active>
<opponent>
starmie 1.00 psychic water slp
<end_opponent>
<end_arena>
```

**Concrete example (Gen 9 with boosts and Tera):**
```
<arena>
<active>
garganacl 0.72 rock leftovers purifying salt clean tera:rock <boosts> def+2 spd+1 <end_boosts>
<end_active>
<opponent>
greattusk 0.45 ground fighting unknownitem unknownability clean
<end_opponent>
<end_arena>
```

### 4.5 `<begin_moves>` — Available Moves

Shows the **active Pokémon's currently available moves** (after accounting for
Transform, Mimic, PP depletion).  Sorted alphabetically by move name.

```
<begin_moves>
<move>
<name> <type> <category>
<end_move>
<move>
<name> <type> <category>
<end_move>
…
<end_moves>
```

- `<name>`: canonical move name, e.g. `blizzard`, `recover`
- `<type>`: move type, e.g. `ice`, `normal`
- `<category>`: `physical`, `special`, or `status`

**Edge cases:**
- **Struggle / Recharge / Fight (Gen 1):** shown as a normal move entry, e.g.
  `<move>recharge normal status<end_move>`.
- **Transform / Mimic:** shows the transformed/copied moves, not the original.
- **Empty moveset:** `0` or more entries (no padding — the model counts them).

**Example:**
```
<begin_moves>
<move>
blizzard ice special
<end_move>
<move>
bodyslam normal physical
<end_move>
<move>
lovelykiss normal status
<end_move>
<move>
rest psychic status
<end_move>
<end_moves>
```

### 4.6 `<bench>` — Benched & Fainted Pokémon

Shows **POV player's** non-active Pokémon.  Includes both healthy benched and
fainted Pokémon (marked with `fnt` status and `0.00` HP).  Ordered by
`consistent_pokemon_order()`.

```
<bench>
<pokeN>
<species> <hp> <type1> <type2> [<item>] [<ability>] [<status>]
<end_pokeN>
…
<end_bench>
```

| Position | Notes |
|----------|-------|
| `species` | Canonical name |
| `hp` | Fixed-point `X.XX`. Preserved across switch-outs. `0.00` for fainted. |
| `type1`/`type2` | Always known (from dex). |
| `item` | Omitted Gen 1. Always known for own Pokémon. |
| `ability` | Omitted Gen 1–2. Always known for own Pokémon. |
| `status` | Omitted when `nostatus` (healthy). Shown when `par`, `slp`, `psn`, `tox`, `brn`, `frz`, or `fnt`. |

**Omissions:**
- Opponent bench is **not** shown in POVReplay output (the model infers it from
  switch-in events across the sequence).
- Stat boosts are **not** shown on bench Pokémon (they reset on switch-out).
- Fainted Pokémon stay in `<bench>` with `0.00` HP and `fnt` status. No
  separate `<fainted>` section.

**Example:**
```
<bench>
<poke1>
alakazam 0.50 psychic par
<end_poke1>
<poke2>
chansey 1.00 normal
<end_poke2>
<poke3>
jinx 0.00 ice psychic fnt
<end_poke3>
<poke4>
snorlax 1.00 normal
<end_poke4>
<poke5>
starmie 0.00 psychic water fnt
<end_poke5>
<end_bench>
```

### 4.7 `<conditions>` — Weather, Field, Side Conditions

```
<conditions>
<weather>
[<battle_field>]
<you> [forceswitch|forcedrevival|cantera] [<side_cond>…] <end_you>
<opponent> [<side_cond>…] <end_opponent>
<end_conditions>
```

#### Weather (one token per line within `<conditions>`)
| Token | Meaning |
|-------|---------|
| `noweather` | No weather active |
| `sandstorm` | Sandstorm |
| `raindance` | Rain Dance / Drizzle |
| `sunnyday` | Sunny Day / Drought |
| `hail` | Hail (Gen 2–8) |
| `snow` | Snow (Gen 9+) |
| `deltastream` | Delta Stream (Primal) |
| `primordialsea` | Primordial Sea |
| `desolateland` | Desolate Land |

#### Battle field (Gen 4+ only, one token)
| Token | Meaning |
|-------|---------|
| `nofield` | No field effect (omitted when no field) |
| `electricterrain` | Electric Terrain |
| `grassyterrain` | Grassy Terrain |
| `psychicterrain` | Psychic Terrain |
| `mistyterrain` | Misty Terrain |
| `trickroom` | Trick Room |
| *(others from PEField)* | |

#### Side conditions

| Token | Meaning | Stackable? |
|-------|---------|------------|
| `reflect` | Reflect | No |
| `lightscreen` | Light Screen | No |
| `auroraveil` | Aurora Veil | No |
| `tailwind` | Tailwind | No |
| `safeguard` | Safeguard | No |
| `mist` | Mist | No |
| `spikes_1` | Spikes (1 layer) | Yes |
| `spikes_2` | Spikes (2 layers) | Yes |
| `spikes_3` | Spikes (3 layers) | Yes |
| `toxicspikes_1` | Toxic Spikes (1 layer) | Yes |
| `toxicspikes_2` | Toxic Spikes (2 layers) | Yes |
| `stealthrock` | Stealth Rock | No |
| `stickyweb` | Sticky Web | No |
| *(others from PESideCondition)* | |

#### Special markers in `<you>`

| Token | Meaning |
|-------|---------|
| `forceswitch` | Player MUST switch (U-turn hit, Eject Button, Roar, etc.) — only switch actions are legal |
| `forcedrevival` | Revival Blessing triggered — may switch to a previously-fainted Pokémon |
| `cantera` | Tera is still available for the player (Gen 9 only) |

**Examples:**

No weather, no side conditions, no special state (fully empty — collapsed):
```
<conditions_empty>
```

Sandstorm with your Reflect + Light Screen up:
```
<conditions>
 sandstorm
 <you> reflect lightscreen <end_you>
 <opponent_empty>
<end_conditions>
```

Forced switch (U-turn) with Stealth Rock on opponent's side:
```
<conditions>
 noweather
 <you> forceswitch <end_you>
 <opponent> stealthrock <end_opponent>
<end_conditions>
```

Tera available, Spikes (2 layers) on your side, Toxic Spikes (1) on opponent's:
```
<conditions>
 noweather
 <you> cantera spikes_2 <end_you>
 <opponent> toxicspikes_1 stealthrock <end_opponent>
<end_conditions>
```

### 4.8 `<terminal>` — Battle Outcome

Only present in the **final** state block. Options:

| Content | Meaning |
|---------|---------|
| `<terminal>won<end_terminal>` | POV player won |
| `<terminal>lost<end_terminal>` | POV player lost |
| `<terminal>tie<end_terminal>` | Battle ended in a tie |
| `<terminal>forfeit<end_terminal>` | Opponent forfeited / timed out (treated as win) |

The final state has **no** following `<boa>` block.

---

## 5. Action Block (`<boa>` … `<eoa>`)

Shows the actions chosen by the POV player and the opponent for a given turn.
One action block between every pair of states (except after the terminal state).

### 5.1 Structure

#### Singles
```
<boa>
<turn>1<end_turn>
<chosen_move>[switch] <name> [cant=<reason>]<end_chosen_move>
<opponent_chosen_move>[switch] <name> [cant=<reason>]|unknown<end_opponent_chosen_move>
<eoa>
```

#### Doubles
```
<boa>
<turn>1<end_turn>
<chosen_move slot="1">[switch] <name><end_chosen_move>
<chosen_move slot="2">[switch] <name><end_chosen_move>
<opponent_chosen_move slot="1">…<end_opponent_chosen_move>
<opponent_chosen_move slot="2">…<end_opponent_chosen_move>
<eoa>
```

### 5.2 Action format

**Moves:**
```
<boa>
<turn>3<end_turn>
<chosen_move>blizzard<end_chosen_move>
<opponent_chosen_move>thunderbolt<end_opponent_chosen_move>
<eoa>
```

**Moves with `cant` (player chose the move but couldn't execute):**
```
<boa>
<turn>2<end_turn>
<chosen_move>lovelykiss cant=par<end_chosen_move>
<opponent_chosen_move>unknown<end_opponent_chosen_move>
<eoa>
```

**Opponent moves with `cant` (opponent chose the move but couldn't execute):**
```
<boa>
<turn>3<end_turn>
<chosen_move>blizzard<end_chosen_move>
<opponent_chosen_move>thunderbolt cant=par<end_opponent_chosen_move>
<eoa>
```

**Switches:**
```
<boa>
<turn>3<end_turn>
<chosen_move>switch alakazam<end_chosen_move>
<opponent_chosen_move>switch chansey<end_opponent_chosen_move>
<eoa>
```

**Recharge / forced no-op:**
```
<boa>
<turn>35<end_turn>
<chosen_move>recharge<end_chosen_move>
<opponent_chosen_move>blizzard<end_opponent_chosen_move>
<eoa>
```

**Opponent action unknown (unrevealed):**
```
<boa>
<turn>2<end_turn>
<chosen_move>lovelykiss<end_chosen_move>
<opponent_chosen_move>unknown<end_opponent_chosen_move>
<eoa>
```

### 5.3 `cant` attribute

| Reason | Meaning |
|--------|---------|
| `par` | Fully paralysed |
| `slp` | Asleep |
| `frz` | Frozen |
| `flinch` | Flinched |
| `partiallytrapped` | Trapped by Wrap/Bind/etc. |
| `move: Taunt` | Taunted |
| `move: Heal Block` | Heal Blocked |
| `ability: Truant` | Truant "loafing around" turn |
| *(other reasons from `|cant|`)* | |

When the `|choice|` message exists, we use the exact chosen move + `cant`
reason.  When `|choice|` is absent, we randomly pick a valid move from the
active Pokémon's available moveset and attach the `cant` reason.

### 5.4 Move inference for opponent

When `|choice|` messages are absent (common), we infer the opponent's action:

1. **`|move|p2a: …|MoveName|…`** → infer from the executed move.
2. **`|-damage|…|[from] move: MoveName`** → infer from the `[from]` tag.
3. **`|switch|p2a: …|Species|…`** → infer switch.
4. **`pokemon.last_used_move`** → tracked per Pokémon; used when damage/heal
   messages don't carry `[from]` tags.
5. Otherwise → `unknown`.

---

## 6. Forced Switches (Subturns)

When a move triggers a forced switch (U-turn, Volt Switch, Eject Button, Red
Card, Roar, Dragon Tail, etc.), it creates an **extra state-action pair**.

The sequence is: *original state* → *move action* → *intermediate state with
`forceswitch` marker* → *switch action* → *new state with replacement active
Pokémon*.

```
<bos>
<format>gen9ou<end_format>
<turn>5<end_turn>
<arena>
<active>
landorus-therian 0.82 ground flying leftovers intimidate clean
<end_active>
<opponent>
ferrothorn 0.45 grass steel leftovers ironbarbs clean
<end_opponent>
<end_arena>
<begin_moves>…<end_moves>
<bench>…<end_bench>
<conditions>
 noweather
 <you_empty>
 <opponent_empty>
<end_conditions>
<eos>

<boa>
<turn>5<end_turn>
<chosen_move>uturn<end_chosen_move>
<opponent_chosen_move>unknown<end_opponent_chosen_move>
<eoa>

<bos>
<format>gen9ou<end_format>
<turn>5<end_turn>
<arena>
<active>
landorus-therian 0.82 ground flying leftovers intimidate clean
<end_active>
<opponent>
ferrothorn 0.45 grass steel leftovers ironbarbs clean
<end_opponent>
<end_arena>
<begin_moves>…<end_moves>
<bench>…<end_bench>
<conditions>
 noweather
 <you> forceswitch <end_you>
 <opponent_empty>
<end_conditions>
<eos>

<boa>
<turn>5<end_turn>
<chosen_move>switch rotom-wash<end_chosen_move>
<opponent_chosen_move>unknown<end_opponent_chosen_move>
<eoa>

<bos>
<format>gen9ou<end_format>
<turn>5<end_turn>
<arena>
<active>
rotom-wash 1.00 electric water leftovers levitate clean
<end_active>
<opponent>
ferrothorn 0.45 grass steel leftovers ironbarbs clean
<end_opponent>
<end_arena>
…
<eos>
```

The key marker is `forceswitch` as the first token inside `<you> … <end_you>`.
Revival Blessing uses `forcedrevival` instead.

---

## 7. Edge Cases & Mechanics

### 7.1 Transform (Ditto, Mew)

When the active Pokémon has transformed, `<active>` shows both identities:

```
<active>
ditto snorlax 1.00 normal noitem noability clean
<end_active>
```

Format: `<actual_species> <transformed_species> <hp> …`

The `<begin_moves>` block shows the transformed (copied) moves.  On switch-out,
Ditto reverts — subsequent states show `<active>` with `ditto` alone and its
original Transform move again.

### 7.2 Zoroark / Zorua (Illusion)

From the POV player's perspective, their own Zoroark is known.  The `<active>`
and `<bench>` entries always use the **real** species name.  The illusion
disguise is **not** shown in the output — the POV player knows it's Zoroark.

The opponent's illusion (if visible) is handled by the opponent's POV file,
which is generated separately.

Internally, the parser must still track Zoroark's `|replace|` events to
correctly attribute moves/items/abilities learned during the illusion window.
Use the `unique_id` + nickname for disambiguation.

### 7.3 Mimic

When Mimic replaces a move slot, `<begin_moves>` shows the copied move:

```
<begin_moves>
<move>
blizzard ice special
<end_move>
<move>
bodyslam normal physical
<end_move>
<move>
earthquake ground physical
<end_move>
<move>
rest psychic status
<end_move>
<end_moves>
```

The `<begin_team>` header shows the original moveset (including Mimic).

On switch-out, Mimic is restored. Track via `move_change_to_from`.

### 7.4 Consecutive / Multi-turn Moves (Outrage, Thrash, Hyper Beam)

- **Outrage / Thrash / Petal Dance (2–3 turn lock-in):** `<chosen_move>` shows
  the move name on each turn. The `<begin_moves>` shows only that one move
  available (the player is locked in).
- **Hyper Beam recharge:** `<chosen_move>recharge<end_chosen_move>`. The
  `<begin_moves>` shows only `recharge`.
- **Charge moves (Fly, Dig, Solar Beam):** Turn 1 shows
  `<chosen_move>fly<end_chosen_move>` (charge turn). Turn 2 also shows
  `<chosen_move>fly<end_chosen_move>` (attack turn).  Track internally via
  `charge_move` flag + `[still]` in the protocol messages.

### 7.5 `|cant|` — Player Unable to Move

When the player's Pokémon is paralysed, asleep, frozen, flinched, etc.:

- **If `|choice|` message exists:** use the exact chosen move with `cant` attribute.
- **If no `|choice|`:** randomly pick a valid move from the active Pokémon's
  available moveset. Attach the `cant` reason attribute.
- **Opponent `|cant|`:** same as player — use the opponent's known or randomly-chosen
  move with the `cant` reason attribute, e.g.
  `<opponent_chosen_move> blizzard cant=par <end_opponent_chosen_move>`.
  If the opponent's move is completely unknown (no reveals, no moveset info),
  output `<opponent_chosen_move>unknown<end_opponent_chosen_move>`.

### 7.6 Self-KO Moves (Self-Destruct, Explosion, Destiny Bond)

Both active Pokémon can faint in the same turn. The resulting state shows both
with `0.00` HP and `fnt` status.  The following `<boa>` blocks show both sides
switching (or the battle ending).

### 7.7 Same-Species Pokémon on Same Team

Standard formats enforce Species Clause.  If a custom format permits duplicates,
disambiguate internally using `unique_id` (UUID) + nickname.  **Do not include
nicknames in output.**  If disambiguation is impossible, skip the replay.

### 7.8 HP Preservation Across Switch-Outs

A Pokémon's HP when it switches out is remembered and shown in subsequent
`<bench>` entries.  The backward fill propagates HP from the last-known-active
state back through the turn sequence.  Entry hazards (Stealth Rock) reduce HP on
switch-**in**, which is reflected in the state where the Pokémon appears as
active.

### 7.9 Forme Changes (Palafin, Zygarde, etc.)

When a Pokémon changes form mid-battle (`|detailschange|`), the new species name
appears in subsequent states.  No special tag is needed — the model observes the
name change in the state sequence.

Example: Palafin switches out → switches back in as Palafin-Hero:
```
<bos> … <active>palafin 1.00 water …<end_active> … <eos>
<boa> <chosen_move>switch palafin<end_chosen_move> … <eoa>
<bos> … <active>palafin-hero 1.00 water …<end_active> … <eos>
```

### 7.10 Terastallization (Gen 9)

Before Terastallization, the Tera type may be unknown (omitted) or known (from
team preview data).  After Terastallization, the `<active>` line includes the
Tera type:

```
<active>
garganacl 1.00 rock leftovers purifying salt clean tera:rock noboosts
<end_active>
```

The `<you>` section in conditions carries `cantera` until it's consumed:
```
<conditions>
noweather
<you> cantera <end_you>
<opponent> <end_opponent>
<end_conditions>
```

---

## 8. Generation Quick Reference

| Feature | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 | Gen 6 | Gen 7 | Gen 8 | Gen 9 |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Held Items | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Abilities | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Natures | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Field effects | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Team Preview | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fairy type | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Tera | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Teleport switches | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Z-Moves | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Mega Evolution | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Dynamax | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |

(Replays with Mega/Z-moves/Dynamax in a non-native gen are skipped — they're
ROM hacks or custom formats.)

---

## 9. Token / Vocabulary Reference

### Structural tags (always paired `<foo>`…`<end_foo>`)
`<begin_team>` `<end_team>` `<pokeN>` `<end_pokeN>` `<begin_moves>` `<end_moves>`
`<move>` `<end_move>` `<begin_opponent_team>` `<end_opponent_team>` `<bos>` `<eos>`
`<boa>` `<eoa>` `<arena>` `<end_arena>` `<active>` `<end_active>` `<active1>`
`<end_active1>` `<active2>` `<end_active2>` `<opponent>` `<end_opponent>`
`<opponent1>` `<end_opponent1>` `<opponent2>` `<end_opponent2>` `<bench>`
`<end_bench>` `<conditions>` `<end_conditions>` `<you>` `<end_you>` `<boosts>`
`<end_boosts>` `<chosen_move>` `<end_chosen_move>` `<opponent_chosen_move>`
`<end_opponent_chosen_move>` `<format>` `<end_format>` `<turn>` `<end_turn>`
`<terminal>` `<end_terminal>`

### Status tokens (bare words)
`par` `slp` `psn` `tox` `brn` `frz` `fnt`

### Effect tokens (bare words)
`confusion` `leechseed` `protect` `substitute` `curse` `reflect` `lightscreen`
*(…full set from `PEEffect` enum)*

### Weather tokens (bare words)
`noweather` `sandstorm` `raindance` `sunnyday` `hail` `snow`

### Special markers (bare words)
`clean` `forceswitch` `forcedrevival` `cantera` `unknown` `unknownitem`
`unknownability` `nofield` `switch` `recharge` `tera:`

### Boost tokens (bare words)
`atk+N` `def+N` `spa+N` `spd+N` `spe+N` `accuracy+N` `evasion+N`
(where N is a signed integer, e.g. `atk+1`, `spa-2`)

### Turn number
Integer, e.g. `0`, `1`, `47`

### HP format
Fixed-point with two decimals: `1.00`, `0.63`, `0.00`.

---

## 10. Comparison: POVReplay vs. UniversalState

The new parsed-replay format described here is the **POVReplay** output — one
file per player (WIN / LOSS) with limited opponent visibility.

When the world-model data generator needs **full-knowledge** states, it can
produce a **UniversalState** variant that additionally includes:

- `<opponent_bench>` block showing all opponent bench Pokémon (with
  backward-filled items, abilities, movesets)
- Full opponent movesets in `<opponent>` arena entries (including unrevealed
  moves filled by backward prediction)

This is analogous to the current `WorldModelObservationSpace` text format but
with the new tag syntax.  The UniversalState variant is a **separate output
stage** built on top of the POVReplay, not part of the raw→parsed pipeline
itself.
