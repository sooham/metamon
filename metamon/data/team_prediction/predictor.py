import copy
import os
import random
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import deque
from typing import Set, List, Tuple
import numpy as np

from metamon.data.team_builder.team_builder import TeamBuilder, PokemonStatsLookupError
from metamon.data.team_builder.stat_reader import PreloadedSmogonStat
from metamon.data.team_prediction.team import TeamSet, PokemonSet, Roster


class TeamPredictor(ABC):
    def __init__(self):
        pass

    def predict(self, team: TeamSet) -> TeamSet:
        copy_team = copy.deepcopy(team)
        return self.fill_team(copy_team)

    @abstractmethod
    def fill_team(self, team: TeamSet):
        raise NotImplementedError


@lru_cache(maxsize=16)
def get_legacy_teambuilder(format: str):
    return TeamBuilder(format, verbose=False, remove_banned=False, inclusive=True)


class NaiveUsagePredictor(TeamPredictor):
    def fill_team(self, team: TeamSet):
        team_builder = get_legacy_teambuilder(team.format)
        gen = int(team.format.split("gen")[1][0])
        pokemon = [team.lead] + team.reserve
        # use legacy team builder to generate a team of 6 Pokémon based on the ones we already
        # have and their common teammates. Teammates and all moves/items/abilities are sampled
        # from a premade set of Showdown usage statistics.
        existing_names = [p.name for p in pokemon if p.name != PokemonSet.MISSING_NAME]
        try:
            sample_team = team_builder.generate_new_team(existing_names)
        except PokemonStatsLookupError as e:
            raise e

        # convert from the output of the old team builder to the new PokemonSet format
        cleaned_dict = {}
        for poke in sample_team:
            if poke["name"] == PokemonSet.MISSING_NAME:
                raise ValueError("Missing name in sample team")
            spread = poke["spread"]
            nature, evs = spread.split(":")
            nature = nature.strip()
            evs = [int(x) for x in evs.split("/")]
            ivs = [int(x) for x in poke["IVs"].split("/")]
            item = poke["item"].strip()
            ability = poke["ability"].strip()
            if ability == "No Ability":
                ability = PokemonSet.NO_ABILITY
            if not item:
                item = PokemonSet.NO_ITEM
            cleaned_dict[poke["name"]] = {
                "moves": poke["moves"],
                "gen": gen,
                "nature": nature,
                "evs": evs,
                "ivs": ivs,
                "ability": ability,
                "item": item,
            }

        sample_team = []
        for key, val in cleaned_dict.items():
            as_pokemon_dict = val | {"name": key}
            try:
                sample_team.append(PokemonSet.from_dict(as_pokemon_dict))
            except Exception as e:
                raise e

        # Build a mapping from name to PokemonSet for fast lookup
        sample_team_map = {p.name: p for p in sample_team}
        # Prepare a queue of new Pokemon to fill missing slots
        new_pokemon = deque([p for p in sample_team if p.name not in existing_names])
        merged_team = []
        for p in team.pokemon:
            if p.name == PokemonSet.MISSING_NAME:
                new_choice = new_pokemon.popleft()
                merged_team.append(copy.deepcopy(new_choice))
            else:
                new_p = sample_team_map.get(p.name)
                filled_p = copy.deepcopy(p)
                filled_p.fill_from_PokemonSet(new_p)
                merged_team.append(filled_p)
        final_team = TeamSet(
            lead=merged_team[0], reserve=merged_team[1:], format=team.format
        )
        return final_team


@lru_cache(maxsize=4)
def load_replay_stats_by_format(format: str):
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "consistent_pokemon_sets",
            f"{format}_pokemon.json",
        ),
        "r",
    ) as f:
        pokemon_sets = json.load(f)
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "consistent_pokemon_sets",
            f"{format}_team_rosters.json",
        ),
        "r",
    ) as f:
        team_rosters = json.load(f)["rosters"]

    rosters = []
    for rw in team_rosters:
        r = Roster.from_dict(rw["team"])
        w = rw["weight"]
        rosters.append((r, w))

    sets = {}
    for name, all_sets in pokemon_sets.items():
        sets[name] = [
            (PokemonSet.from_dict(p | {"name": name}), w) for p, w in all_sets
        ]

    return sets, rosters


class ReplayPredictor(NaiveUsagePredictor):
    """
    NaiveUsagePredictor *independently* samples missing details from PS usage stats.
    The flaw with this is that it creates unlikely combinations of pokemon and movesets.
    For example, if there are 2 common movesets for a particular Pokemon, it will sample
    a very unlikely combination of moves from both sets.

    ReplayPredictor instead matches the current revealed team to a set of candidates
    discovered from every replay in the dataset. It then samples from these candidates,
    falling back to NaiveUsagePredictor for any remaining info.
    """

    def __init__(self, top_k_teams: int = 100, top_k_movesets: int = 10):
        self.stat_format = None
        self.top_k_teams = top_k_teams
        self.top_k_movesets = top_k_movesets

    def _load_data(self, format: str):
        self.smogon_stat = PreloadedSmogonStat(format, verbose=False, inclusive=True)
        self.pokemon_sets, self.team_rosters = load_replay_stats_by_format(format)

    def score_roster(self, current_roster: Roster, candidate_roster: Roster) -> float:
        score = 1.0
        eps = 1e-6
        assert current_roster.lead != PokemonSet.MISSING_NAME

        new_pokemon = current_roster.additional_details(candidate_roster)
        if len(new_pokemon) < 6 - len(current_roster):
            score *= eps

        for p in new_pokemon:
            if p in self.smogon_stat[current_roster.lead]["teammates"]:
                score *= self.smogon_stat[current_roster.lead]["teammates"][p]
            else:
                score *= eps

        return score

    def sample_roster_from_candidates(
        self, current_roster: Roster, candidates: List[Tuple[Roster, float]]
    ):
        weights = [self.score_roster(current_roster, r) for r, _ in candidates]
        weights = [w / sum(weights) for w in weights]
        roster = random.choices(candidates, weights=weights, k=1)[0][0]
        return copy.deepcopy(roster)

    def score_pokemon(
        self, current_pokemon: PokemonSet, candidate_pokemon: PokemonSet
    ) -> float:
        """
        Scores a diff between a current Pokemon and a candidate predicted Pokemon.

        The basic problem we need to address is the difference between a moveset that is frequently *possible* vs. a moveset that is frequently *used*.
        Our replay stats track the number of real movesets that *could be* each candidate, but in reality some of these are very unlikely.

        For example, in gen1ou, it is common for Tauros to reveal {Earthquake, Body Slam, Hyper Beam} during a battle.
        Our candidates computed by the replay stats will find us all the movesets that have these three moves.
        There are many choices for the 4th move that have appeared at some point. The "weights" assigned to each candidate
        from the replay stats will give options that look something like this:
            Blizzard, weight=6812
            Fire Blast, weight=5574
            Rest, weight=5507
            Stomp, weight=5507
            ...
            Thunder, weight=4709
            Swords Dance, weight=3333
        Stomp/Thunder/Swords Dance are dramatically overrepresented because many replays are consistent with these movesets,
        but in reality that are much more rarely used.

        This function adjusts candidates based on the usage stats of their suggested additions, e.g. P(Fire Blast|Tauros) >> P(Thunder|Tauros).
        """
        eps = 1e-6
        if current_pokemon.name == PokemonSet.MISSING_NAME:
            breakpoint()

        diff = current_pokemon.additional_details(candidate_pokemon)
        if diff is None:
            breakpoint()
            return eps
        name = current_pokemon.name

        try:
            smogon_stat = self.smogon_stat[name]
        except KeyError:
            breakpoint()
            return eps

        score = 1.0

        new_moves = diff.get("moves", [])
        if len(new_moves) < (
            len(current_pokemon.moves) - current_pokemon.revealed_moves
        ):
            # if the candidate does not reveal all the missing moves, downweight
            score *= eps

        if (
            current_pokemon.ability == PokemonSet.MISSING_ABILITY
            and "ability" not in diff
        ):
            # if the candidate does not reveal the missing ability, downweight
            score *= eps

        if current_pokemon.item == PokemonSet.MISSING_ITEM and "item" not in diff:
            # if the candidate does not reveal the missing item, downweight
            score *= eps

        if "moves" in diff:
            move_stats = smogon_stat["moves"]
            for new_move in diff["moves"]:
                if new_move not in move_stats:
                    score *= eps
                else:
                    score *= move_stats[new_move]

        if "ability" in diff:
            if diff["ability"] not in smogon_stat["abilities"]:
                score *= eps
            else:
                score *= smogon_stat["abilities"][diff["ability"]]

        if "item" in diff:
            if diff["item"] not in smogon_stat["items"]:
                score *= eps
            else:
                score *= smogon_stat["items"][diff["item"]]

        return score

    def sample_pokemon_from_candidates(
        self,
        current_pokemon: PokemonSet,
        candidates: List[PokemonSet],
    ):
        weights = [self.score_pokemon(current_pokemon, p) for p, _ in candidates]
        weights = [w / sum(weights) for w in weights]
        pokemon = random.choices(candidates, weights=weights, k=1)[0][0]
        return copy.deepcopy(pokemon)

    def fill_team(self, team: TeamSet) -> TeamSet:
        if self.stat_format != team.format:
            self._load_data(team.format)
        og_team = copy.deepcopy(team)

        lead_name = team.lead.name
        reserve_names = [p.name for p in team.reserve]
        roster = Roster(lead=lead_name, reserve=reserve_names)

        candidate_rosters = [
            (r, w) for r, w in self.team_rosters if roster.is_consistent_with(r)
        ]
        if candidate_rosters:
            top_k_candidate_rosters = sorted(
                candidate_rosters, key=lambda x: x[1], reverse=True
            )[: self.top_k_teams]
            roster = self.sample_roster_from_candidates(roster, top_k_candidate_rosters)
            assert roster.lead == lead_name
            team.fill_from_Roster(roster)

        if any(pokemon.name == PokemonSet.MISSING_NAME for pokemon in team.pokemon):
            breakpoint()

        for pokemon in team.pokemon:
            candidate_sets = [
                (p, w)
                for p, w in self.pokemon_sets[pokemon.name]
                if pokemon.is_consistent_with(p)
            ]
            if candidate_sets:
                top_k_candidate_sets = sorted(
                    candidate_sets, key=lambda x: x[1], reverse=True
                )[: self.top_k_movesets]
                new_pokemon = self.sample_pokemon_from_candidates(
                    pokemon, top_k_candidate_sets
                )
                pokemon.fill_from_PokemonSet(new_pokemon)

        # fall back to old method for any remaining info
        return super().fill_team(team)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict team from partial information"
    )
    parser.add_argument(
        "format", type=str, help="Showdown battle format (e.g., gen1ou)"
    )
    args = parser.parse_args()

    team_files = os.listdir(
        os.path.join(os.environ["METAMON_TEAMFILE_PATH"], f"{args.format}_teams")
    )
    team_file = os.path.join(
        os.environ["METAMON_TEAMFILE_PATH"],
        f"{args.format}_teams",
        random.choice(team_files),
    )
    print(f"Using team file: {team_file}")

    naive_predictor = NaiveUsagePredictor()
    og_team = copy.deepcopy(TeamSet.from_showdown_file(team_file, args.format))
    naive_team = naive_predictor.predict(og_team)
    consistent = og_team.is_consistent_with(naive_team)
    print(f"Consistent: {consistent}")
    improved_predictor = ReplayPredictor()
    improved_team = improved_predictor.predict(og_team)
    consistent = og_team.is_consistent_with(improved_team)
    print(f"Consistent: {consistent}")
    # Print teams side by side
    og_lines = og_team.to_str().split("\n")
    naive_lines = naive_team.to_str().split("\n")
    improved_lines = improved_team.to_str().split("\n")

    # Pad lines to equal length
    max_len = max(len(og_lines), len(naive_lines), len(improved_lines))
    og_lines += [""] * (max_len - len(og_lines))
    naive_lines += [""] * (max_len - len(naive_lines))
    improved_lines += [""] * (max_len - len(improved_lines))

    # Print header
    print(f"{'Original Team':<40}{'Naive Prediction':<40}{'Improved Prediction':<40}")
    print("-" * 120)

    # Print lines side by side
    for og, naive, improved in zip(og_lines, naive_lines, improved_lines):
        print(f"{og:<40}{naive:<40}{improved:<40}")
