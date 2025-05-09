import copy
import os
import random
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import deque
from typing import Set, List, Tuple

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


@lru_cache(maxsize=32)
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
    def __init__(self, max_teams: int = 10_000):
        self.stat_format = None
        self.max_teams = max_teams

    def _load_data(self, format: str):
        self.smogon_stat = PreloadedSmogonStat(format, verbose=False, inclusive=True)
        self.pokemon_sets, self.team_rosters = load_replay_stats_by_format(format)

    def sample_roster_from_candidates(
        self, current_roster: Roster, candidates: List[Tuple[Roster, float]]
    ):
        weights = [w for _, w in candidates]
        weights = [w / sum(weights) for w in weights]
        roster = random.choices(candidates, weights=weights, k=1)[0][0]
        return roster

    def sample_pokemon_from_candidates(
        self, current_pokemon: PokemonSet, candidates: List[PokemonSet]
    ):
        weights = [w for _, w in candidates]
        weights = [w / sum(weights) for w in weights]
        pokemon = random.choices(candidates, weights=weights, k=1)[0][0]
        return pokemon

    def fill_team(self, team: TeamSet) -> TeamSet:
        if self.stat_format != team.format:
            self._load_data(team.format)
        og_team = copy.deepcopy(team)

        lead_name = team.lead.name
        reserve_names = [p.name for p in team.reserve]
        roster = Roster(lead=lead_name, reserve=reserve_names)

        candidates = [
            (r, w) for r, w in self.team_rosters if roster.is_consistent_with(r)
        ]
        if candidates:
            roster = self.sample_roster_from_candidates(roster, candidates)
            assert roster.lead == lead_name
            candidates = [p for p in roster.reserve if p not in reserve_names]

        for pokemon in team.pokemon:
            if pokemon.name == PokemonSet.MISSING_NAME and candidates:
                pokemon.name = candidates.pop()

        for pokemon in team.pokemon:
            candidates = [
                (p, w)
                for p, w in self.pokemon_sets[pokemon.name]
                if pokemon.is_consistent_with(p)
            ]
            if candidates:
                new_pokemon = self.sample_pokemon_from_candidates(pokemon, candidates)
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
