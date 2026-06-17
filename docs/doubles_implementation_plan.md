# Doubles Support — Implementation Plan

## Design Principle

**Separate classes, not `if is_doubles` branches.** The singles code path stays
untouched.  A parallel doubles path inherits from the same bases and overrides
only the places where slot-count matters.  A dispatch function chooses the right
path at the serialization boundary.

This keeps the code easy to read: open the singles serializer → see singles
logic.  Open the doubles serializer → see doubles logic.  No interleaved
conditionals.

---

## Architecture

```
                   ┌──────────────────────┐
                   │   forward_fill()     │  ← unchanged (already handles both slots)
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │  backward_fill()     │  ← unchanged
                   └──────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │                               │
    ┌─────────▼──────────┐      ┌─────────────▼──────────────┐
    │  POVReplay         │      │  POVReplayDoubles(POVReplay)│
    │  (singles)         │      │  overrides:                 │
    │  _align_states…    │      │  _align_states_actions()    │
    └─────────┬──────────┘      └─────────────┬──────────────┘
              │                               │
    ┌─────────▼──────────┐      ┌─────────────▼──────────────┐
    │  serialize_pov_    │      │  serialize_pov_replay_      │
    │  replay()          │      │  doubles()                  │
    │  → text (singles)  │      │  → text (doubles tags)      │
    └────────────────────┘      └────────────────────────────┘
```

### Files

| File | What changes |
|------|-------------|
| `backward.py` | Add `POVReplayDoubles` subclass (~40 lines) |
| `text_serializer.py` | Add doubles serialization functions (~80 lines) |
| `parse_replays.py` | Dispatch to doubles serializer when format is doubles (~5 lines) |

No changes to `forward.py`, `checks.py`, or `replay_state.py`.

---

## 1. `POVReplayDoubles` (in `backward.py`)

Inherits from `POVReplay`.  Overrides only `_align_states_actions`.

### What's different from singles

**Singles** (`POVReplay._align_states_actions`):
```
- pov_slot = 0                               # only slot 0
- opponent_actionlist stores 1 Action        # opponent_moves[0]
```

**Doubles** (`POVReplayDoubles._align_states_actions`):
```
- Loops over pov_slot in (0, 1)              # both active slots get cant substitution
- opponent_actionlist stores [Action|None, Action|None]   # both slots
```

### Concrete override

```python
class POVReplayDoubles(POVReplay):
    """POVReplay variant for doubles formats (two active Pokémon per side)."""

    def _align_states_actions(self, replay: ParsedReplay):
        self._povturnlist = []
        self._actionlist = []
        self._opponent_actionlist = []
        for idx, (turn_t, turn_t1) in enumerate(
            zip(replay.turnlist, replay.turnlist[1:])
        ):
            # subturns — same as singles
            for subturn in turn_t.subturns:
                if subturn.turn is not None and subturn.team == (
                    1 if self.from_p1_pov else 2
                ):
                    action = [None, None]
                    action[subturn.slot] = subturn.action
                    self._povturnlist.append(subturn.turn)
                    self._actionlist.append(action)
                    self._opponent_actionlist.append([None, None])

            self._povturnlist.append(turn_t)

            moves = turn_t1.moves_1 if self.from_p1_pov else turn_t1.moves_2
            choices = turn_t1.choices_1 if self.from_p1_pov else turn_t1.choices_2
            opponent_moves = turn_t1.moves_2 if self.from_p1_pov else turn_t1.moves_1

            actionlist = [None, None]
            for move_idx, (move, choice) in enumerate(zip(moves, choices)):
                if move is not None:
                    actionlist[move_idx] = move
                elif choice is not None:
                    actionlist[move_idx] = choice

            # ── cant substitution for BOTH slots ──
            for pov_slot in (0, 1):
                if actionlist[pov_slot] is None:
                    active = (
                        turn_t1.active_pokemon_1[pov_slot]
                        if self.from_p1_pov
                        else turn_t1.active_pokemon_2[pov_slot]
                    )
                    if active is not None:
                        reason = getattr(active, "cant_reason", None)
                        if reason is not None:
                            if reason in ("recharge", "ability: Truant"):
                                actionlist[pov_slot] = Action(
                                    name="Recharge", is_noop=True,
                                    user=active, target=None,
                                )
                            else:
                                move_objs = [m for m in active.moves.values() if m is not None]
                                if move_objs:
                                    chosen = random.choice(move_objs)
                                    actionlist[pov_slot] = Action(
                                        name=chosen.name, user=active, target=None,
                                    )
                                else:
                                    actionlist[pov_slot] = Action(
                                        name="Recharge", is_noop=True,
                                        user=active, target=None,
                                    )

            self._actionlist.append(actionlist)
            self._opponent_actionlist.append(
                [opponent_moves[0], opponent_moves[1]]
            )

        # final state
        self._povturnlist.append(turn_t1)
        self._actionlist.append([None, None])
        self._opponent_actionlist.append([None, None])
```

### Backward compatibility

`POVReplay._align_states_actions` stays exactly as-is.  `POVReplayDoubles` is a
separate subclass that only gets instantiated for doubles formats.

---

## 2. Doubles text serializer (in `text_serializer.py`)

New function `serialize_pov_replay_doubles()` mirrors `serialize_pov_replay()`.

### State block differences

| Element | Singles | Doubles |
|---------|---------|---------|
| Active Pokémon | `<active>` ×1 | `<active1>` + `<active2>` |
| Opponent | `<opponent>` ×1 | `<opponent1>` + `<opponent2>` |
| Moves | One `<begin_moves>` block | Two `<begin_moves slot="1">` / `<begin_moves slot="2">` blocks |

Example arena for doubles:
```xml
<arena>
<active1>
ninetales-alola 1.00 ice fairy lightclay snowwarning noeffect nostatus noboosts
<end_active1>
<active2>
glaceon 1.00 ice notype choicespecs snowcloak noeffect nostatus noboosts
<end_active2>
<opponent1>
cresselia 1.00 psychic notype unknownitem unknownability noeffect nostatus noboosts
<end_opponent1>
<opponent2>
diancie 1.00 rock fairy unknownitem unknownability noeffect nostatus noboosts
<end_opponent2>
<end_arena>
```

Moves per slot:
```xml
<begin_moves slot="1">
<move>
auroraveil ice status
<end_move>
<move>
blizzard ice special
<end_move>
...
<end_moves>
<begin_moves slot="2">
<move>
chillingwater water special
<end_move>
...
<end_moves>
```

### Action block differences

| Element | Singles | Doubles |
|---------|---------|---------|
| Player action | `<chosen_move>` ×1 | `<chosen_move slot="1">` + `<chosen_move slot="2">` |
| Opponent action | `<opponent_chosen_move>` ×1 | `<opponent_chosen_move slot="1">` + `<opponent_chosen_move slot="2">` |

Example:
```xml
<boa>
<turn>1<end_turn>
<chosen_move slot="1">auroraveil<end_chosen_move>
<chosen_move slot="2">chillingwater<end_chosen_move>
<opponent_chosen_move slot="1">trickroom<end_opponent_chosen_move>
<opponent_chosen_move slot="2">diamondstorm<end_opponent_chosen_move>
<eoa>
```

When a slot has no action (e.g., only one Pokémon moved):
```xml
<chosen_move slot="1">protect<end_chosen_move>
<chosen_move slot="2">unknown<end_chosen_move>
```

### Bench

Same as singles — shows all non-active POV Pokémon.  In doubles, both active
slots are excluded from bench.

### Implementation sketch

```python
def serialize_pov_replay_doubles(pov: POVReplayDoubles) -> str:
    """Serialize a doubles POVReplay to the new text format."""
    # … same team header + opponent preview as singles …
    # … same state/action interleaving loop …
    # But calls _write_state_block_doubles() and _write_action_block_doubles()

def _write_state_block_doubles(turn, pov, is_terminal=False) -> list[str]:
    """Doubles state: active1/active2/opponent1/opponent2, per-slot moves."""
    # … arena with 4 entries instead of 2 …
    # … two <begin_moves slot="N"> blocks …

def _write_action_block_doubles(turn, player_actions, opponent_actions) -> list[str]:
    """Doubles action: two <chosen_move> + two <opponent_chosen_move>."""
    # player_actions is [Action|None, Action|None]
    # opponent_actions is [Action|None, Action|None]
    # output with slot="1" / slot="2" attributes
```

---

## 3. Dispatch in `parse_replays.py`

In `save_to_disk`, detect doubles format and call the right serializer:

```python
def save_to_disk(self, replay, time_played, player_username, opponent_username):
    won = "WIN" if replay.winner else "LOSS"
    filename = f"{replay.gameid}_{replay.rating}_{player_username}_vs_{opponent_username}_{time_played.strftime('%m-%d-%Y')}_{won}"
    if self.output_dir is not None:
        path = self.output_dir
        os.makedirs(path, exist_ok=True)
        if isinstance(replay, POVReplayDoubles):
            text_output = serialize_pov_replay_doubles(replay)
        else:
            text_output = serialize_pov_replay(replay)
        with open(os.path.join(path, f"{filename}.txt"), "w", encoding="utf-8") as f:
            f.write(text_output)
```

And in `parse_replay`, detect doubles in the log (`|gametype|doubles`) and
construct `POVReplayDoubles` instead of `POVReplay`.  Or better: have
`backward_fill` (or a new `backward_fill_doubles`) return `POVReplayDoubles`.

Simplest approach: add a `backward_fill_doubles()` function identical to
`backward_fill` except it instantiates `POVReplayDoubles` instead of
`POVReplay`.  This is ~10 extra lines (a copy of the last 20 lines of
`backward_fill`).

---

## 4. Files summary

| File | Action | LOC |
|------|--------|-----|
| `backward.py` | Add `POVReplayDoubles` class + `backward_fill_doubles()` | ~65 |
| `text_serializer.py` | Add `serialize_pov_replay_doubles()` + 2 helpers | ~90 |
| `parse_replays.py` | Detect doubles → use `backward_fill_doubles` + doubles serializer | ~10 |
| `backward.py` | No changes to existing singles code | 0 |
| `text_serializer.py` | No changes to existing singles code | 0 |

**Total: ~165 lines of new code. Zero lines of existing code modified.**
