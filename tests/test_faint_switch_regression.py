"""
Regression test: Zapdos faint → Snorlax switch in gen1ou-316031019.

Turn 23 of this battle contains a faint + forced switch sub-turn:
  |move|p2a: Snorlax|Body Slam|p1a: Zapdos
  |-damage|p1a: Zapdos|0 fnt
  |faint|p1a: Zapdos
  |choice|switch 3|
  |switch|p1a: Snorlax|Snorlax|523/523

p1 (mist98895, LOSS POV) must replace the fainted Zapdos.
The correct replacement is **Snorlax** (523 HP).
"""

import os
import datetime
import pytest
import orjson

from metamon.backend.replay_parser.forward import forward_fill, ParsedReplay
from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.replay_parser.backward import backward_fill
from metamon.backend.team_prediction.predictor import NoPredictor


# ── Path to the specific raw replay ──────────────────────────────────────────

RAW_PATH = os.path.join(
    os.environ.get(
        "METAMON_CACHE_DIR",
        os.path.expanduser("~/Repositories/poke-datasets"),
    ),
    "raw-replays",
    "gen1",
    "ou",
    "gen1ou-316031019.json",
)

# Known facts about this battle at turn 23
EXPECTED_FAINTED = "Zapdos"
EXPECTED_REPLACEMENT = "Snorlax"
# In the raw log, p1 chose "switch 3" (1-indexed Showdown choice).
# The actual Pokémon switched in was Snorlax (523 HP).


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_raw_replay():
    """Load the raw replay JSON for gen1ou-316031019."""
    if not os.path.exists(RAW_PATH):
        pytest.skip(f"Raw replay not found: {RAW_PATH}")
    with open(RAW_PATH, "rb") as f:
        return orjson.loads(f.read())


def _run_forward(raw_data: dict) -> ParsedReplay:
    """Run the forward fill on the raw replay."""
    log = ReplayParser.clean_log(raw_data)
    replay = ParsedReplay(
        gameid=raw_data["id"],
        format=raw_data.get("formatid", raw_data.get("format", "unknown")),
        time_played=datetime.datetime.fromtimestamp(int(raw_data["uploadtime"])),
    )
    return forward_fill(replay, log)


def _find_turn_by_number(replay: ParsedReplay, turn_number: int):
    """Return the Turn object with the given turn_number, or None."""
    for turn in replay.turnlist:
        if turn.turn_number == turn_number:
            return turn
    return None


# ── Forward pass tests ───────────────────────────────────────────────────────


class TestForwardFaintSwitch:
    """Verify the forward fill correctly reconstructs the faint + switch."""

    @pytest.fixture(scope="class")
    def parsed_replay(self):
        raw = _load_raw_replay()
        return _run_forward(raw)

    def test_turn_23_exists(self, parsed_replay):
        """Turn 23 should be present in the forward fill output."""
        turn23 = _find_turn_by_number(parsed_replay, 23)
        assert turn23 is not None, "Turn 23 not found in forward fill output"

    def test_zapdos_fainted(self, parsed_replay):
        """After turn 23, Zapdos should be fainted (0 HP)."""
        # Look through all Pokémon on both sides for the fainted Zapdos
        for turn in parsed_replay.turnlist:
            if turn.turn_number is not None and turn.turn_number >= 23:
                for p in turn.pokemon_1 + turn.pokemon_2:
                    if p is None:
                        continue
                    if p.name == EXPECTED_FAINTED and p.current_hp == 0:
                        return  # found it
        pytest.fail(
            f"Fainted {EXPECTED_FAINTED} (0 HP) not found in any turn >= 23"
        )

    def test_snorlax_switched_in(self, parsed_replay):
        """After turn 23, Snorlax should appear on p1's team with >0 HP."""
        found = False
        for turn in parsed_replay.turnlist:
            if turn.turn_number is not None and turn.turn_number >= 23:
                for p in turn.pokemon_1:
                    if p is None:
                        continue
                    if p.name == EXPECTED_REPLACEMENT and p.current_hp > 0:
                        # Verify it's active (in the active slot for p1)
                        for active in turn.active_pokemon_1:
                            if active is not None and active.unique_id == p.unique_id:
                                found = True
                                break
        assert found, (
            f"{EXPECTED_REPLACEMENT} not found active on p1's team after turn 23"
        )

    def test_zapdos_not_active_after_turn_23(self, parsed_replay):
        """After turn 23, Zapdos should no longer be in an active slot."""
        for turn in parsed_replay.turnlist:
            if turn.turn_number is not None and turn.turn_number > 23:
                for active in turn.active_pokemon_1:
                    if active is not None and active.name == EXPECTED_FAINTED:
                        pytest.fail(
                            f"Fainted {EXPECTED_FAINTED} still active at turn "
                            f"{turn.turn_number}"
                        )

    def test_switch_action_targets_snorlax(self, parsed_replay):
        """The switch action at turn 23 should target Snorlax, not Tauros."""
        turn23 = _find_turn_by_number(parsed_replay, 23)
        assert turn23 is not None

        # The switch appears as a subturn (forced switch mid-turn)
        # or as a replacement in replacements_1
        switch_target = None
        for subturn in turn23.subturns:
            if subturn.turn is not None and subturn.action is not None:
                if subturn.action.is_switch and subturn.action.target is not None:
                    switch_target = subturn.action.target.name
                    break

        # Also check turn 24's moves_1 for the switch
        turn24 = _find_turn_by_number(parsed_replay, 24)
        if turn24 is not None and switch_target is None:
            for action in turn24.moves_1:
                if action is not None and action.is_switch and action.target is not None:
                    switch_target = action.target.name
                    break

        assert switch_target is not None, (
            "Could not find switch action target after turn 23"
        )
        assert switch_target == EXPECTED_REPLACEMENT, (
            f"Switch target is '{switch_target}', expected '{EXPECTED_REPLACEMENT}'. "
            f"This is the bug where the action targets Tauros instead of Snorlax!"
        )


# ── Backward pass tests ──────────────────────────────────────────────────────


class TestBackwardFaintSwitch:
    """Verify the backward pass correctly indexes the faint + switch action."""

    @pytest.fixture(scope="class")
    def pov_replays(self):
        """Run forward + backward pass with NoPredictor for isolation."""
        raw = _load_raw_replay()
        log = ReplayParser.clean_log(raw)
        replay = ParsedReplay(
            gameid=raw["id"],
            format=raw.get("formatid", raw.get("format", "unknown")),
            time_played=datetime.datetime.fromtimestamp(int(raw["uploadtime"])),
        )
        replay = forward_fill(replay, log)
        pov_p1, pov_p2 = backward_fill(replay, team_predictor=NoPredictor())
        return pov_p1, pov_p2

    @pytest.fixture(scope="class")
    def parsed_replay(self):
        """Forward-only for context."""
        raw = _load_raw_replay()
        return _run_forward(raw)

    def test_loss_pov_has_faint_state(self, pov_replays):
        """The LOSS POV (mist98895, p1) should have a state where Zapdos
        is at 0 HP and forced_switch is True."""
        pov_loss, _ = pov_replays  # p1 = LOSS POV, p2 = WIN POV
        found_faint = False
        for turn in pov_loss.povturnlist:
            for active in turn.active_pokemon_1:
                if active is not None and active.name == EXPECTED_FAINTED and active.current_hp == 0:
                    if turn.is_force_switch:
                        found_faint = True
                        break
        assert found_faint, (
            f"No forced-switch state found with {EXPECTED_FAINTED} at 0 HP"
        )

    def test_loss_pov_switch_action_idx(self, pov_replays):
        """The action index for the forced switch should match the
        available_switches order in the *serialized* state.

        BUG: ``UniversalAction.from_ReplayAction`` computes the index
        against ``consistent_pokemon_order()``, but ``UniversalState.from_ReplayState``
        serializes switches in a different order.  This means the stored
        ``action_idx`` can point to the wrong Pokemon when read back.

        For this specific battle, ``consistent_pokemon_order`` puts Snorlax
        at index 3, so ``action_idx = 7``.  But the serialized switches are
        ``[Jynx, Chansey, Exeggutor, Tauros, Snorlax]`` where Snorlax is
        at index 4, so the correct ``action_idx`` should be **8**.
        """
        from metamon.interface import UniversalAction, consistent_pokemon_order

        pov_loss, _ = pov_replays  # p1 = LOSS POV

        parser = ReplayParser(
            replay_output_dir=None,
            team_output_dir=None,
            verbose=False,
            team_predictor=NoPredictor(),
            compress=False,
        )
        states, actions = parser.povreplay_to_state_action(pov_loss)

        found = False
        for state, action in zip(states, actions):
            if state.active_pokemon is None:
                continue
            if state.active_pokemon.name != EXPECTED_FAINTED:
                continue
            if state.active_pokemon.current_hp != 0:
                continue
            if not state.force_switch:
                continue

            found = True

            # The action target is definitely Snorlax (verified by forward test)
            assert action is not None
            assert action.target is not None
            assert action.target.name == EXPECTED_REPLACEMENT, (
                f"Forward pass says switch target is {action.target.name}, "
                f"expected {EXPECTED_REPLACEMENT}"
            )

            # The fix makes both the action index AND the serialized state
            # use consistent_pokemon_order, so they match.
            consistent_order = consistent_pokemon_order(state.available_switches)
            consistent_idx = None
            for j, sw in enumerate(consistent_order):
                if sw.unique_id == action.target.unique_id:
                    consistent_idx = j
                    break
            assert consistent_idx is not None
            expected_action_idx = 4 + consistent_idx

            # ---- What UniversalAction.from_ReplayAction actually returns ----
            ua = UniversalAction.from_ReplayAction(state=state, action=action)
            assert ua is not None
            actual_idx = ua.action_idx

            # After the fix, both should agree
            assert actual_idx == expected_action_idx, (
                f"ACTION INDEX BUG: action_idx={actual_idx} "
                f"should be {expected_action_idx} for {EXPECTED_REPLACEMENT} "
                f"(consistent_order[{consistent_idx}])"
            )
            break

        assert found, (
            f"Could not find the fainted {EXPECTED_FAINTED} + forced_switch state"
        )

    def test_loss_pov_next_state_is_snorlax(self, pov_replays):
        """The state immediately after the forced switch should have
        Snorlax as the active Pokémon."""
        pov_loss, _ = pov_replays  # p1 = LOSS POV

        prev_was_faint = False
        for turn in pov_loss.povturnlist:
            active = turn.active_pokemon_1[0] if turn.active_pokemon_1 else None
            if active is None:
                continue

            if prev_was_faint:
                assert active.name == EXPECTED_REPLACEMENT, (
                    f"After faint, expected {EXPECTED_REPLACEMENT} active, "
                    f"got {active.name}"
                )
                assert active.current_hp > 0, (
                    f"After faint, {EXPECTED_REPLACEMENT} has 0 HP"
                )
                return  # success

            if active.name == EXPECTED_FAINTED and active.current_hp == 0:
                prev_was_faint = True

        pytest.fail("Could not find the faint → switch transition")

    def test_win_pov_sees_faint(self, pov_replays, parsed_replay):
        """The WIN POV (typhlosion10919, p2) sees the opponent's
        Zapdos get replaced by Snorlax.  The backward pass does NOT
        create a separate forced-switch state for the *opponent's*
        mid-turn faint (only for the POV player's own team), so we
        check that Snorlax appears in the opponent's active slot
        after turn 22."""
        _, pov_win = pov_replays  # p1 = LOSS, p2 = WIN

        # Snorlax should appear as the opponent's active after turn 22
        found_snorlax = False
        for turn in pov_win.povturnlist:
            if turn.turn_number is not None and turn.turn_number >= 23:
                for opp in turn.active_pokemon_2:  # p2's opponent = p1
                    if opp is not None and opp.name == EXPECTED_REPLACEMENT:
                        found_snorlax = True
                        break
        assert found_snorlax, (
            f"WIN POV never observed {EXPECTED_REPLACEMENT} as opponent "
            f"after turn 22"
        )
