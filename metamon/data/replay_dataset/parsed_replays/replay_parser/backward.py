import copy
import time
import re
from functools import lru_cache
from typing import List, Optional

from metamon.data.replay_dataset.parsed_replays.replay_parser import checks, forward
from metamon.data.replay_dataset.parsed_replays.replay_parser.exceptions import *
from metamon.data.replay_dataset.parsed_replays.replay_parser.replay_state import (
    Action,
    Boosts,
    Move,
    Nothing,
    Pokemon,
    Turn,
    Winner,
    get_pokedex_and_moves,
    unknown,
)
from metamon.data.team_builder import PokemonStatsLookupError, TeamBuilder


@lru_cache
def get_team_builder(format: str) -> Optional[TeamBuilder]:
    team_builder = TeamBuilder(
        format,
        ps_path="/home/jake/pokemon-showdown/",
        verbose=False,
        remove_banned=False,
        inclusive=True,
    )
    if len(team_builder.stat.movesets) == 0:
        return None
    return team_builder


def fill_missing_team_info(
    battle_format: str, poke_list: List[Pokemon]
) -> List[Pokemon]:
    """
    Returns a new (out-of-place) version of a team with all missing information filled in.

    Info that cannot be known is inferred by sampling from human usage data.
    """
    poke_list = copy.deepcopy(poke_list)

    # Generate sample team
    cleaned_format = re.sub(r"\[|\]| ", "", battle_format).lower()
    gen = int(cleaned_format[3])
    pokemon_names = [p.name for p in poke_list if p is not None]
    try:
        team_builder = get_team_builder(cleaned_format)
    except:
        raise BackwardException(
            f"Could not load TeamBuilder for format {cleaned_format}"
        )
    else:
        if team_builder is None:
            raise BackwardException(
                f"Could not load TeamBuilder for format {cleaned_format}"
            )
    try:
        sample_team = team_builder.generate_new_team(pokemon_names)
    except PokemonStatsLookupError as e:
        raise BackwardException(str(e))
    sample_team_dict = {x["name"]: x for x in sample_team}

    # Fill in missing pokemon
    while None in poke_list and sample_team:
        generated = sample_team.pop()
        if generated["name"] not in pokemon_names:
            new_pokemon = Pokemon(name=generated["name"], lvl=100, gen=gen)
            poke_list[poke_list.index(None)] = new_pokemon
    if None in poke_list:
        raise BackwardException(
            f"Could not fill in all missing pokemon for {poke_list} with {sample_team}"
        )

    # Fill missing attributes
    for p in poke_list:
        sample_p = sample_team_dict[p.name]

        # quick cleanup of generated items/abilities
        sample_item = sample_p["item"]
        if sample_item == "No Item" or not sample_item.strip():
            sample_item = Nothing.NO_ITEM
        sample_ability = sample_p["ability"]
        if sample_ability == "No Ability" or not sample_ability.strip():
            sample_ability = Nothing.NO_ABILITY

        # the initial state of each pokemon is in the `had` attrs. if we still
        # don't know something, fill it in. then the backfill will move this
        # info into the `active_` attrs.
        if unknown(p.had_ability):
            p.had_ability = sample_ability
        if unknown(p.had_item):
            p.had_item = sample_item
        possible_moves_to_add = set(sample_p["moves"]) - set(p.had_moves.keys())
        while len(p.had_moves.keys()) < 4 and possible_moves_to_add:
            new_move = Move(name=possible_moves_to_add.pop(), gen=gen)
            p.had_moves[new_move.name] = new_move

        if p.max_hp is None:
            assert p.current_hp is None
            # if pokemon was never damaged or switch in, it would have unknown HP.
            # we can safely set to 100/100 without worrying about base stats, EVs, IVs,
            # because the hp is only ever shown to the agent as a fraction.
            p.max_hp = 100
            p.current_hp = 100

    return poke_list


class POVReplay:
    def __init__(
        self,
        replay: forward.ParsedReplay,
        filled_replay: forward.ParsedReplay,
        from_p1_pov: bool,
    ):
        if replay.gameid != filled_replay.gameid:
            raise ValueError("Using replays of different games to construct POVReplay")

        self.from_p1_pov = from_p1_pov

        # copy replay metadata
        self.replay_url = filled_replay.replay_url
        self.gameid = filled_replay.gameid
        self.format = filled_replay.format
        self.gen = filled_replay.gen
        self.time_played = filled_replay.time_played
        self.rules = filled_replay.rules

        # rating and winner from POV
        self.rating = filled_replay.ratings[0 if from_p1_pov else 1]
        self.winner = filled_replay.winner == (
            Winner.PLAYER_1 if from_p1_pov else Winner.PLAYER_2
        )

        self.fill_one_side(replay, filled_replay)
        self.align_states_actions(replay)

    def fill_one_side(self, replay, filled_replay):
        # take spectator replay and reveal one entire team from filled_replay
        assert len(replay.flattened_turnlist) == len(filled_replay.flattened_turnlist)
        for turn, filled_turn in zip(
            replay.flattened_turnlist, filled_replay.flattened_turnlist
        ):
            if self.from_p1_pov:
                turn.pokemon_1 = filled_turn.pokemon_1
                turn.active_pokemon_1 = filled_turn.active_pokemon_1
            else:
                turn.pokemon_2 = filled_turn.pokemon_2
                turn.active_pokemon_2 = filled_turn.active_pokemon_2

    def align_states_actions(self, replay: forward.ParsedReplay):
        self.povturnlist = []
        self.actionlist = []
        for idx, (turn_t, turn_t1) in enumerate(
            zip(replay.turnlist, replay.turnlist[1:])
        ):
            # subturns freeze the sim midturn, which we currently
            # only use to replicate forced switches.
            for subturn in turn_t.subturns:
                if subturn.turn is not None and subturn.team == (
                    1 if self.from_p1_pov else 2
                ):
                    action = [None, None]
                    action[subturn.slot] = subturn.action
                    self.povturnlist.append(subturn.turn)
                    self.actionlist.append(action)

            self.povturnlist.append(
                turn_t
            )  # turn_t holds the state at the very end of the turn
            # and the action we clicked between turns is held in the next turn
            moves = turn_t1.moves_1 if self.from_p1_pov else turn_t1.moves_2
            choices = turn_t1.choices_1 if self.from_p1_pov else turn_t1.choices_2
            actionlist = [None, None]
            for move_idx, (move, choice) in enumerate(zip(moves, choices)):
                if move is not None:
                    # we default to the original system of *used* moves
                    actionlist[move_idx] = move
                elif choice is not None:
                    # if the move was missing, but a `choice` message was parsed,
                    # we can fall back to that.
                    actionlist[move_idx] = choice
            self.actionlist.append(actionlist)

        # add final state
        self.povturnlist.append(turn_t1)
        self.actionlist.append([None, None])


def add_filled_final_turn(replay: forward.ParsedReplay) -> forward.ParsedReplay:
    # add an extra turn to a replay with all missing information guessed
    # by sampling from the TeamBuilder. this extra turn can then be moved
    # backwards through the replay and discareded.
    filled_turn = replay[-1].create_next_turn()
    filled_turn.on_end_of_turn()
    filled_turn.pokemon_1 = fill_missing_team_info(replay.format, replay[-1].pokemon_1)
    filled_turn.pokemon_2 = fill_missing_team_info(replay.format, replay[-1].pokemon_2)
    replay.turnlist.append(filled_turn)
    return replay


def resolve_transforms(replay):
    base_stats = {
        pokemon.unique_id: pokemon.base_stats for pokemon in replay[0].all_pokemon
    }
    types = {pokemon.unique_id: pokemon.type for pokemon in replay[0].all_pokemon}

    for prev_turn, turn in zip(replay.turnlist, replay.turnlist[1:]):
        for pokemon in turn.all_pokemon:
            prev_ids = prev_turn.id2pokemon
            on_prev_turn = prev_ids[pokemon.unique_id]

            if pokemon.transformed_into is not None:
                transformed_on_prev = prev_ids[pokemon.transformed_into.unique_id]

                if on_prev_turn.transformed_into != pokemon.transformed_into:
                    # first turn of transformation
                    for move_name in transformed_on_prev.moves.keys():
                        # when we got the -transform message on the forward pass
                        # we copied the moves we already knew about it. now we
                        # add the ones we didn't discover until the backward pass.
                        if move_name not in pokemon.moves:
                            tformed_move = Move(move_name, gen=replay.gen)
                            tformed_move.pp, tformed_move.maximum_pp = 5, 5
                            pokemon.moves[tformed_move.name] = tformed_move

                    # on the forward pass we avoided changing things that would
                    # not reset on switch out. now we make those changes manually.
                    # poke-env does NOT switch the `species` property on transform,
                    # so we don't change the `name`.
                    pokemon.base_stats = transformed_on_prev.base_stats
                    # ...except for hp
                    pokemon.base_stats["hp"] = base_stats[pokemon.unique_id]["hp"]
                    pokemon.active_ability = transformed_on_prev.active_ability
                    pokemon.type = transformed_on_prev.type

                elif pokemon.transformed_into == on_prev_turn.transformed_into:
                    # continued transformation
                    for move_name, move in on_prev_turn.moves.items():
                        # we may have discovered more of the opponents moveset
                        # during the transformation, so the moves we added above
                        # may already be in `moves` with correct PP tracking.
                        if move_name not in pokemon.moves:
                            assert move.maximum_pp == 5
                            pokemon.moves[move_name] = move

                    pokemon.base_stats = on_prev_turn.base_stats
                    pokemon.active_ability = on_prev_turn.active_ability
                    pokemon.type = on_prev_turn.type

            elif on_prev_turn.transformed_into is not None:
                # forward pass logic switches most things back
                pokemon.base_stats = base_stats[pokemon.unique_id]
                pokemon.active_ability = pokemon.had_ability
                pokemon.type = types[pokemon.unique_id]


def backward_fill(replay: forward.ParsedReplay) -> tuple[POVReplay, POVReplay]:
    cleaned_format = re.sub(r"\[|\]| ", "", replay.format).lower()
    pokedex, _ = get_pokedex_and_moves(cleaned_format)

    # fill in missing team info at the end of the forward pass
    replay_filled = add_filled_final_turn(copy.deepcopy(replay))

    # copy that info across the trajectory
    flat_turnlist = replay_filled.flattened_turnlist
    for turn_t, turn_t1 in zip(flat_turnlist[-2::-1], flat_turnlist[::-1]):
        prev_ids = turn_t.id2pokemon
        # first we move information backwards from the current team roster
        # to the previous timestep
        for prev_team, team in (
            (turn_t.pokemon_1, turn_t1.pokemon_1),
            (turn_t.pokemon_2, turn_t1.pokemon_2),
        ):
            for pokemon in team:
                if pokemon.unique_id in prev_ids:
                    prev_pokemon = prev_ids[pokemon.unique_id]
                    prev_pokemon.backfill_info(pokemon)
                else:
                    # pokemon discovered in turn_t1 enters turn_t "fresh"
                    assert None in prev_team
                    prev_team[prev_team.index(None)] = pokemon.fresh_like()

    # chop off the extra filled turn
    replay_filled.turnlist = replay_filled.turnlist[:-1]
    resolve_transforms(replay_filled)
    checks.check_info_filled(replay_filled)

    from_p1 = POVReplay(
        copy.deepcopy(replay), copy.deepcopy(replay_filled), from_p1_pov=True
    )
    checks.check_action_alignment(from_p1)
    from_p2 = POVReplay(replay, replay_filled, from_p1_pov=False)
    checks.check_action_alignment(from_p2)
    return from_p1, from_p2
