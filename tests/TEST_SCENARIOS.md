# Test Scenarios for Synthetic Raw Replay Files

This document lists test scenarios for synthetic 1v1 (or minimal) raw replay files.
Each scenario exercises a specific behavior of the replay parser.  Scenarios are
organized by the parser subsystem they target.

---

## Basic Structure / Metadata

| # | Scenario | What it tests |
|---|---|---|
| 1 | Minimal valid replay (turn 0 + 5+ turns, one side wins) | `UnfinishedReplayException` boundary |
| 2 | Missing Species Clause rule | `NoSpeciesClause` exception |
| 3 | Unsupported gen (`\|gen\|5`) | `SoftLockedGen` exception |
| 4 | `\|teamsize\|p1\|4` — smaller team | Team list sizing, warning behavior |
| 5 | Both players `Unrated` | Ratings recorded as `"Unrated"` |

## Moves — Basic Execution

| # | Scenario | What it tests |
|---|---|---|
| 6A | P1 attacks P2, deals 30% damage | `\|move\|`, `\|-damage\|`, HP tracking, `last_used_move`, PP decrement |
| 6B | P1 attacks P2, deals 30% damage, P2 move misses, P1 attacks P2 deals 10% damage with a different move, | `\|move\|`, `\|-damage\|`, HP tracking, `last_used_move`, PP decrement, pp tracking |
| 7 | P1 attacks, misses (`[miss]` flag) | `[miss]` flag, no damage, PP still consumed |
| 8 | P1 attacks, KOs P2 (`0 fnt`) | `\|faint\|`, `current_hp=0`, `status=FNT` |
| 9 | P1 uses Hyper Beam → next turn must recharge | `\|move\|Recharge`, `\|cant\|`, `is_noop=True`, no-op action index 0 |
| 10 | P1 runs out of PP → Struggle | Struggle action index 0, no PP consumed |
| 11 | P1 uses multi-turn move (Outrage) — 2–3 turns | `[still]` flag, no additional PP on auto-repeats, confusion after |
| 12 | P1 uses charge move (Solar Beam) — charge turn + attack | Charge turn tracking, two-turn move suppression |
| 13A | P1 uses Fly — semi-invulnerable then attack | Two-turn semi-invulnerable move |
| 13B | P2 uses Dig — semi-invulnerable then attack | Two-turn semi-invulnerable move |
| 13C | P2 uses Dig — P1 uses earthquake | dig was affected by earthquake move |
| 13C | P1 uses Protect — P2 attacks | no damage done by P2 |

## Damage / HP Edge Cases

| # | Scenario | What it tests |
|---|---|---|
| 14 | `\|-heal\|` — P1 recovers HP | `\|-heal\|` message, HP restored correctly |
| 15 | `\|-sethp\|` — Pain Split / Endeavor / Super Fang | Direct HP set vs. damage subtraction |
| 16 | Damage from Life Orb (`[from] item: Life Orb`) | Item revealed via damage tag |
| 17 | Damage from ability (`[from] ability: Rough Skin`) | Ability revealed via damage tag |
| 18 | Recoil damage (Double-Edge) | Self-damage on same Pokémon |

## Status Conditions

| # | Scenario | What it tests |
|---|---|---|
| 19 | P2 puts P1 to sleep (`\|-status\| p1a: ... \|slp`) | `\|-status\|` handler, status field, `\|cant\|` on subsequent turn |
| 20 | Same, then P1 wakes up (`\|-curestatus\| p1a: ... \|slp`) | `\|-curestatus\|` handler |
| 21 | P1 paralyzed, can't move (`\|cant\| p1a: ... \|par`) | `\|cant\|` → action left as `None` / missing (index -1) |
| 22 | P1 burned, takes burn damage each turn | Recurring `\|-damage\| [from] brn` |
| 23 | P1 badly poisoned (Toxic), increasing counter | `\|-status\| ... \|tox`, stacking damage counters |
| 24 | P1 frozen, then thaws | `\|-status\| frz`, `\|-curestatus\|` |

## Stat Boosts

| # | Scenario | What it tests |
|---|---|---|
| 25 | P1 uses Swords Dance (`\|-boost\| p1a: ... \|atk\|2`) | Single stat boost parsed into `boosts` dict |
| 26 | P1 uses Dragon Dance (`\|-boost\| atk\|1`, then `\|-boost\| spe\|1`) | Two stats boosted in one move |
| 27 | P2 uses Charm on P1 (`\|-unboost\| p1a: ... \|atk\|2`) | `\|-unboost\|` handler |
| 28 | P1 uses Haze (`\|-clearallboost\|`) | All boosts cleared for both sides |
| 29 | P1 uses Heart Swap (`\|-swapboost\| p1a: ... \|p2a: ...`) | Boost swaps between two Pokémon |
| 30 | P1 uses Psych Up (`\|-copyboost\| p1a: ... \|p2a: ...`) | Copy opponent's boosts |
| 31 | P1 uses Belly Drum (`\|-setboost\| atk\|6` from `-1` → sets to `6`) | `\|-setboost\|` handler |
| 32 | P1 uses Topsy-Turvy on P2 (`\|-invertboost\| p2a: ...`) | Invert all boosts |

## Weather

| # | Scenario | What it tests |
|---|---|---|
| 33 | P1 sets Rain Dance (`\|-weather\|RainDance`) | Weather set/change |
| 34 | Rain expires / clear (`\|-weather\|none`) | Weather cleared to `NO_WEATHER` |
| 35A | P1 sets Sandstorm, residual damage each turn | Sandstorm `\|-damage\|` on non-Rock/Steel/Ground types |
| 35B | P1 sets Hail, residual damage each turn | Hail `\|-damage\|` on non-Ice types (confirm this) |

## Field Conditions

| # | Scenario | What it tests |
|---|---|---|
| 36 | Electric Terrain (`\|-fieldstart\|Electric Terrain`) | `\|-fieldstart\|` handler, `battle_field` dict |
| 37 | Terrain expires (`\|-fieldend\|Electric Terrain`) | n`\|-sfieldend\|` handler |

## Side Conditions (entry hazards, screens)

| # | Scenario | What it tests |
|---|---|---|
| 38 | Stealth Rock on P2's side (`\|-sidestart\| p2:... \|move: Stealth Rock`) | Side condition tracked in `conditions_1`/`conditions_2` |
| 39 | Spikes — 3 layers (`\|-sidestart\|` × 3) | Stackable condition counter |
| 40 | Reflect on P1's side (`\|-sidestart\| p1:... \|Reflect`) | Screen set, then `\|-sideend\| p1:... \|Reflect` |
| 41 | Rapid Spin clears hazards (`\|-sideend\| p1:... \|move: Spikes`) | Side condition removal |
| 42 | P1 sets Tailwind (`\|-sidestart\| p1:... \|move: Tailwind`) | Turn-limited side condition |
| 42B | P1 sets Spikes and P2 switches in new pokemon () | Turn-limited side condition |

## Abilities (Gen 3+)

| # | Scenario | What it tests |
|---|---|---|
| 43 | Intimidate on switch-in (`\|-ability\| p1a: ... \|Intimidate\|boost`, then `\|-unboost\| p2a: ... \|atk\|1`) | Ability activation, auto-reveal of `had_ability` |
| 44 | Single-ability species — ability known from Pokédex alone | Auto-revealed via `update_pokedex_info` |
| 45 | Mummy overrides ability (`\|-ability\| p1a: ... \|Mummy\| [from] ability: Mummy [of] p2a: ...`) | `ABILITY_OVERWRITES_ABILITY` |
| 46 | Trace copies ability (`\|-ability\| p1a: ... \|Trace\| [from] ability: Trace [of] p2a: ...`) | `active_ability` vs `had_ability` distinction |
| 47A | `\|-endability\| p1a: ...` | Ability deactivation (e.g. Gastro Acid, Neutralizing Gas) |
| 47B | Test abilities which trigger due to weather effects | i.e Dry Skin ability causes a heal in rain, but takes damage in Sun  |
| 47C | Test abilities which trigger due to weather effects | i.e Swift Swim double speed in Sun, Sand Rush doubles speed in Sandstorm  |
| 47D | Test abilities which bring weather into play | i.e Groudon  |
| 47D | Test abilities which overwrite existing weather | i.e Rayquaza  |

## Items (Gen 2+)

| # | Scenario | What it tests |
|---|---|---|
| 48 | Leftovers recovery (`\|-item\| p1a: ... \|Leftovers`, then `\|-heal\| ... [from] item: Leftovers`) | Item reveal, `active_item`/`had_item` set |
| 49 | Focus Sash consumed (`\|-enditem\| p1a: ... \|Focus Sash`) | Item consumption, `active_item` cleared |
| 50 | Knock Off removes item (`\|-enditem\| p2a: ... \|Leftovers [from] move: Knock Off [of] p1a: ...`) | Knock Off handler |
| 51 | Trick swaps items | `tricking` pointer, items swapped between two Pokémon |

## Volatile Effects

| # | Scenario | What it tests |
|---|---|---|
| 52 | Leech Seed on P2 (`\|-start\| p2a: ... \|Leech Seed`, then drain each turn) | Volatile effect in `effects` dict, recurring drain |
| 53 | P1 becomes confused (`\|-start\| p1a: ... \|confusion`) | Confusion tracking |
| 54 | P1 hurts itself in confusion (`\|-damage\| ... [from] confusion`) | Self-damage from volatile |
| 55 | Taunt on P2 (`\|-start\| p2a: ... \|Taunt`) | Taunt volatile |
| 56 | Encore on P2 (`\|-start\| p2a: ... \|Encore`) | Encore volatile |
| 57 | Effect expires (`\|-end\| p2a: ... \|Leech Seed`) | `\|-end\|` handler, effect removed |
| 58 | Nightmare on sleeping P2 (`\|-start\| p2a: ... \|Nightmare`, then `\|-damage\|`) | Nested status + volatile |
| 59 | Curse (Ghost-type) — P1 cuts own HP (`\|-start\| p1a: ... \|Curse`, then `\|-damage\| [from] Curse`) | Self-inflicted volatile |

## Forced Switches

| # | Scenario | What it tests |
|---|---|---|
| 60 | P1 uses U-turn, deals damage, switches to benched mon | `mark_forced_switch`, subturn creation + fill, `is_force_switch`, check subturn used |
| 61 | P1 uses Volt Switch, blocked by Lightning Rod | `remove_empty_subturn`, failed forced switch |
| 62 | P2 uses Roar on P1 (`\|drag\| p1a: ... \|...\|HP`) | `\|drag\|` message, forced switch tracking, check subturn |
| 63 | Eject Button activates on hit (`\|-enditem\| ... \|Eject Button`) | Item-triggered forced switch, check subturn |
| 64 | Red Card forces attacker out | Red Card forced switch, check subturn |

## Faint / Replace

| # | Scenario | What it tests |
|---|---|---|
| 65 | P1 faints from an attack, P1 sends replacement | `\|faint\|` → forced switch subturn → replacement fills it |
| 66 | Explosion — both Pokémon faint simultaneously | Double faint, `\|faint\|` for both |
| 67 | P1 uses Destiny Bond, P2 KOs P1 → P2 also faints | Destiny Bond faint |

## Transform (Gen 1+)

| # | Scenario | What it tests |
|---|---|---|
| 68 | Ditto uses Transform on P2 | `transformed_into` set, `transformed_this_turn`, copied moves with 5 PP, the ditto should have the moves that the target pokemon has |
| 69 | Ditto switches out after Transform → transformation ends | `transformed_into` cleared, moves restored |
| 69B | Ditto with the automatic transformation ability | `transformed_into` happens without needing to do a move |
| 69B | Two dittos in play doing transform (this is an extreme edge case) | search smogon or bulbapedia about transform logic |

## Zoroark Illusion (Gen 5+)

| # | Scenario | What it tests |
|---|---|---|
| 70 | Zoroark disguised as another Pokémon switches in, then `\|replace\|` after zoroark gets hit | `\|replace\|` handler, `WarningFlags.ZOROARK`, disguise rewound |
| 71 | Zoroark's Illusion breaks after taking damage | `\|replace\|` from damage break |
| 71B | Zoroark's Illusion doesnt break due to no hit landing or no damage causing moves being used | zoroark disguise still in play |
you need to search up the zoroark and zoura disguise ability and how it works

## Mimic / Mirror Move / Metronome

| # | Scenario | What it tests |
|---|---|---|
| 72 | P1 uses Mimic on P2's last move | `mimic` handler, `move_change_to_from`, PP tracking |
| 73 | P1 uses Metronome → calls random move → foreign move suppressed | `pending_foreign_move`, move NOT added to `had_moves` |
| 74 | P1 uses Metronome → calls Outrage → multi-turn sequence suppressed | Foreign-called consecutive move, cross-turn suppression |
| 75 | P1 uses Sleep Talk → calls own move | `MOVE_OVERRIDE_BUT_REVEAL_ANYWAY`, move IS added to `had_moves` |

## Protect / Fail / Immune

| # | Scenario | What it tests |
|---|---|---|
| 76 | P2 uses Protect → P1's move fails (`\|-fail\|`) | `\|-fail\|` → switch-out cancellation |
| 77A | P1 uses Thunder Wave on Ground-type (`\|-immune\|`) | `\|-immune\|` → switch-out cancellation |
| 77B | normal type move on ghost type pokemon |
| 77C | poison type move on steel type pokemon |

## Choice Messages

| # | Scenario | What it tests |
|---|---|---|
| 78 | `\|choice\|move Ice Beam\|move Earthquake` | Named choice parsing → `choices_1`/`choices_2` populated |
| 79 | `\|choice\|switch 2\|switch 4` | Numeric choice (should be skipped per current limitations) |
| 80 | `\|choice\|\|` | Empty choice (skipped) |
| 81 | `\|choice\|move Ice Beam` for a Pokémon with 4 known moves already | Pre-emptive choice → skipped |

## Team Preview

| # | Scenario | What it tests |
|---|---|---|
| 82 | 6 `\|poke\|` messages per side | `teampreview_1`/`teampreview_2` populated |
| 83 | `\|teamsize\|p1\|3` — 3-Pokémon team | Team size ≠ 6, warning |

## Showteam

| # | Scenario | What it tests |
|---|---|---|
| 84 | `\|showteam\|p1\|PackedTeamString` | `showteam_data` populated, used as ground truth in backward fill |

## Forme Change

| # | Scenario | What it tests |
|---|---|---|
| 85 | `\|detailschange\| p1a: ... \|Rotom-Wash` | `had_name` persists, `name` updates, Pokédex re-looked-up |

## Tera (Gen 9)

| # | Scenario | What it tests |
|---|---|---|
| 86 | P1 Terastallizes (`\|-terastallize\| p1a: ... \|Rock`) | `tera_type` set, `can_tera_1` toggled, `is_tera=True` on move, type changes |
| 87 | Tera move action index mapping (9–12) | `action_idx = 9 + move_index` |

## Gen 1 PP Rollover

| # | Scenario | What it tests |
|---|---|---|
| 88 | P1 uses Wrap repeatedly in Gen 1 → PP rolls from 0 to 63 | `GEN1_PP_ROLLOVERS`, `pp_used = -63` |

## Multi-Pokémon (minimal beyond 1v1)

| # | Scenario | What it tests |
|---|---|---|
| 89 | P1 switches out to a benched Pokémon (normal switch) | `\|switch\|` message, action `is_switch=True`, `available_switches` |
| 90 | P2 faints, P2 sends out replacement | Faint + replacement switch sequence |
| 91 | Revival Blessing revives a fainted ally (Gen 9) | `is_revival=True`, fainted Pokémon status → `NO_STATUS` |
| 92 | P1 hits supereffiective move for P2, P2 faints, new P2 moves in, that P2 is also hit with the same supper effective move and faints | check that turns and pokemon are captured correctly |

---

## Priority ordering — core 20

If building incrementally, start with these to cover every broad message-handler category:

**1, 6, 8, 9, 17, 19, 21, 25, 33, 36, 38, 43, 48, 60, 65, 68, 70, 73, 78, 82**

---

# Testing Guide: Inline Synthetic Logs

## Philosophy

Do **not** create 91 separate raw replay JSON files on disk.  The existing pattern
in this codebase (see ``tests/test_forward_edge_cases.py``) constructs synthetic
logs as Python ``list[list[str]]`` inline in the test code.  This keeps each
scenario self-contained and makes the expected behavior visible right next to
the input.

Assertions are **inline** (not golden files).  Synthetic 1v1 logs produce tiny
``ParsedReplay`` objects — you assert directly on the fields you care about.
Golden files are reserved for real replay regression tests where the output is
too large to manually specify.

## Skeleton builder

Every synthetic log needs the same ~15-line preamble.  Extract it into a shared
helper to keep tests focused on the scenario-specific messages:

```python
# tests/synthetic_helpers.py

import datetime


def make_skeleton(
    gen: int = 1,
    tier: str = "[Gen 1] OU",
    format: str = "gen1ou",
    p1: str = "Alice",
    p2: str = "Bob",
    teamsize1: int = 6,
    teamsize2: int = 6,
    p1_pokes: list[str] | None = None,
    p2_pokes: list[str] | None = None,
    p1_lead: tuple[str, str] | None = None,  # (species, hp_string)
    p2_lead: tuple[str, str] | None = None,
) -> list[list[str]]:
    """Return the common log prefix shared by all synthetic replays.

    Defaults produce a Gen 1 OU 6v6 with 12 distinct Pokemon.
    Override *p1_pokes* / *p2_pokes* to set specific teams.
    Override *p1_lead* / *p2_lead* to set specific leads
    (defaults to the first Pokemon in each list).
    """
    if p1_pokes is None:
        p1_pokes = ["Charizard", "Blastoise", "Venusaur", "Pikachu", "Gengar", "Snorlax"]
    if p2_pokes is None:
        p2_pokes = ["Alakazam", "Golem", "Starmie", "Dragonite", "Machamp", "Jolteon"]
    if p1_lead is None:
        p1_lead = (p1_pokes[0], "100/100")
    if p2_lead is None:
        p2_lead = (p2_pokes[0], "100/100")

    log = [
        ["gen", str(gen)],
        ["tier", tier],
        ["rule", "Species Clause: Limit one of each Pokémon"],
        ["player", "p1", p1, ""],
        ["player", "p2", p2, ""],
        ["teamsize", "p1", str(teamsize1)],
        ["teamsize", "p2", str(teamsize2)],
    ]
    for poke in p1_pokes:
        log.append(["poke", "p1", poke, ""])
    for poke in p2_pokes:
        log.append(["poke", "p2", poke, ""])
    log.append(["start", ""])
    log.append(["switch", f"p1a: {p1_lead[0]}", p1_lead[0], p1_lead[1]])
    log.append(["switch", f"p2a: {p2_lead[0]}", p2_lead[0], p2_lead[1]])
    return log


def make_turn(turn_number: int) -> list[str]:
    """Return a ``|turn|N`` message."""
    return ["turn", str(turn_number)]


def make_winner(player_name: str) -> list[list[str]]:
    """Return a winning sequence that passes the >= 5 turns check."""
    return [
        ["turn", "2"],
        ["turn", "3"],
        ["turn", "4"],
        ["turn", "5"],
        ["win", player_name],
    ]


def make_faint_winner(winner: str, loser_active: str) -> list[list[str]]:
    """Return turns for a KO + win, satisfying >= 5 turns."""
    return [
        ["turn", "2"],
        ["turn", "3"],
        ["turn", "4"],
        ["turn", "5"],
        ["faint", loser_active],
        ["win", winner],
    ]


def build_parsed_replay(log: list[list[str]], gameid: str = "test", format: str = "gen1ou"):
    """Run forward_fill on a synthetic log. Returns ParsedReplay.

    Imports are local to avoid circular dependencies at module level.
    """
    from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay
    replay = ParsedReplay(
        gameid=gameid,
        format=format,
        time_played=datetime.datetime(2020, 1, 1),
    )
    return forward_fill(replay, log)
```

## Running a single scenario

```python
def test_attack_deals_damage():
    log = make_skeleton() + [
        make_turn(1),
        ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
        ["-damage", "p2a: Alakazam", "70/100"],
        ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
        ["-damage", "p1a: Charizard", "70/100"],
    ] + make_faint_winner("Alice", "p2a: Alakazam")

    replay = build_parsed_replay(log)

    turn1 = replay.turnlist[1]
    assert turn1.active_pokemon_2[0].current_hp == 70
    assert turn1.active_pokemon_1[0].last_used_move.name == "Flamethrower"
    assert turn1.moves_1[0].name == "Flamethrower"
    assert turn1.moves_1[0].is_switch is False
    assert turn1.moves_1[0].target.unique_id == turn1.active_pokemon_2[0].unique_id
```

## Parametrizing related scenarios

Groups of closely related scenarios work well with ``@pytest.mark.parametrize``.
Each parametrized case appends its specific messages to a shared skeleton:

```python
@pytest.mark.parametrize("scenario_id, extra_log, expected_status", [
    (19, [
        make_turn(1),
        ["move", "p1a: Charizard", "Flamethrower", "p2a: Alakazam"],
        ["-damage", "p2a: Alakazam", "70/100"],
        ["move", "p2a: Alakazam", "Hypnosis", "p1a: Charizard"],
        ["-status", "p1a: Charizard", "slp"],
    ], "SLP"),
    (21, [
        make_turn(1),
        ["move", "p1a: Charizard", "Thunder Wave", "p2a: Alakazam"],
        ["-status", "p2a: Alakazam", "par"],
        ["move", "p2a: Alakazam", "Psychic", "p1a: Charizard"],
        ["-damage", "p1a: Charizard", "70/100"],
        make_turn(2),
        ["cant", "p2a: Alakazam", "par"],
    ], "PAR"),
])
def test_status_conditions(scenario_id, extra_log, expected_status):
    log = make_skeleton() + extra_log + make_faint_winner("Alice", "p2a: Alakazam")
    replay = build_parsed_replay(log)

    turn1 = replay.turnlist[1]
    assert turn1.active_pokemon_1[0].status.name == expected_status

    # After being fully paralyzed, the player has no move recorded
    if expected_status == "PAR":
        turn2 = replay.turnlist[2]
        assert turn2.moves_2[0] is None
```

## Assertion patterns by scenario type

| What you're testing | Typical assertion |
|---|---|
| HP change | ``turn.active_pokemon_N[0].current_hp == expected`` |
| Status | ``turn.active_pokemon_N[0].status.name == "SLP"`` |
| Move revealed | ``"move_name" in turn.active_pokemon_N[0].moves`` |
| PP decrement | ``turn.active_pokemon_N[0].moves["move_name"].pp == expected`` |
| Boost | ``turn.active_pokemon_N[0].boosts.atk == 2`` |
| Item | ``turn.active_pokemon_N[0].active_item == "Leftovers"`` |
| Ability | ``turn.active_pokemon_N[0].had_ability == "Intimidate"`` |
| Action properties | ``turn.moves_1[0].is_switch == True`` |
| Action target | ``turn.moves_1[0].target.unique_id == uuid`` |
| Forced switch | ``turn.is_force_switch == True`` |
| Subturn | ``len(turn.subturns) == 1`` |
| Weather | ``turn.weather.name == "RAINDANCE"`` |
| Side condition | ``"STEALTHROCK" in turn.conditions_2`` |
| Field | ``"ELECTRICTERRAIN" in turn.battle_field`` |
| Transform | ``turn.active_pokemon_1[0].transformed_into is not None`` |
| Check warnings | ``WarningFlags.TRANSFORM in replay.check_warnings`` |
| No-op action | ``turn.moves_1[0].is_noop == True`` |
| Missing action | ``turn.moves_1[0] is None`` |
| Team preview | ``len(turn.teampreview_1) == 6`` |
| Choice | ``turn.choices_1[0].name == "Ice Beam"`` |
| Exception raised | ``pytest.raises(NoSpeciesClause)`` |

## Recommended file layout

```
tests/
├── synthetic_helpers.py              # make_skeleton(), make_turn(), make_winner(), etc.
├── test_synthetic_forward.py         # Scenarios 1–88 (forward-only)
├── test_synthetic_backward.py        # POV + backfill scenarios
├── test_synthetic_e2e.py             # Full pipeline for a few key scenarios
├── TEST_SCENARIOS.md                 # This file
└── ...
```

**``test_synthetic_forward.py``** is organized by subsystem as test classes:

| Class | Scenarios |
|---|---|
| ``TestMetadata`` | 1–5 |
| ``TestMoves`` | 6–13 |
| ``TestDamage`` | 14–18 |
| ``TestStatus`` | 19–24 |
| ``TestBoosts`` | 25–32 |
| ``TestWeather`` | 33–35 |
| ``TestFieldConditions`` | 36–37 |
| ``TestSideConditions`` | 38–42 |
| ``TestAbilities`` | 43–47 |
| ``TestItems`` | 48–51 |
| ``TestVolatileEffects`` | 52–59 |
| ``TestForcedSwitches`` | 60–64 |
| ``TestFaint`` | 65–67 |
| ``TestTransform`` | 68–69 |
| ``TestZoroark`` | 70–71 |
| ``TestForeignMoves`` | 72–75 |
| ``TestProtectFailImmune`` | 76–77 |
| ``TestChoices`` | 78–81 |
| ``TestTeamPreview`` | 82–83 |
| ``TestShowteam`` | 84 |
| ``TestFormeChange`` | 85 |
| ``TestTera`` | 86–87 |
| ``TestGen1PPRollover`` | 88 |
| ``TestMultiPokemon`` | 89–91 |

## Cross-generation considerations

Some scenarios only work in specific generations.  Use the ``gen`` and ``format``
parameters of ``make_skeleton()`` to target the right gen:

| Scenarios | Minimum gen | Why |
|---|---|---|
| 1–15, 18, 19–24, 25–32 | Gen 1 | Core mechanics exist in all gens |
| 33–42 | Gen 2 | Weather / field / side conditions exist from Gen 2 |
| 16 (Life Orb), 43–47 (abilities), 48–51 (items), 52–59 (volatiles) | Gen 3 | Items + abilities introduced Gen 2/3; some volatiles are Gen 3+ |
| 60–64 (forced switches) | Gen 3 | U-turn / Volt Switch are Gen 3+ |
| 68–69 (Transform) | Gen 1 | Ditto exists in Gen 1 |
| 70–71 (Zoroark) | Gen 5 | Zoroark introduced Gen 5 |
| 72–75 (Mimic/Metronome) | Gen 1 | These moves exist in Gen 1 |
| 78–81 (choice messages) | Any | Choice messages are format-agnostic |
| 86–87 (Tera) | Gen 9 | Tera is Gen 9 only |
| 88 (PP rollover) | Gen 1 | Gen 1 partial-trapping bug only |

## Error-condition tests

Scenarios that test exception paths (1–3, some of 10) should assert
``pytest.raises(...)`` rather than inspecting the ``ParsedReplay``:

```python
def test_no_species_clause_raises():
    log = [
        ["gen", "1"],
        ["tier", "[Gen 1] OU"],
        # deliberately omit Species Clause
        ["player", "p1", "Alice", ""],
        ["player", "p2", "Bob", ""],
        ["teamsize", "p1", "6"],
        ["teamsize", "p2", "6"],
    ]
    replay = ParsedReplay(
        gameid="test-no-sc",
        format="gen1ou",
        time_played=datetime.datetime(2020, 1, 1),
    )
    with pytest.raises(NoSpeciesClause):
        forward_fill(replay, log)
```

These already exist in ``tests/test_forward_edge_cases.py`` and should stay there
or be co-located in the appropriate ``Test*`` class in ``test_synthetic_forward.py``.
