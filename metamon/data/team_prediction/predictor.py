import copy
import os
import random
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import deque, defaultdict
import tqdm
from torch.utils.data import DataLoader

from metamon.data.team_builder.team_builder import TeamBuilder, PokemonStatsLookupError
from metamon.data.team_builder.stat_reader import PreloadedSmogonStat
from metamon.data.team_prediction.team import TeamSet, PokemonSet
from metamon.data.team_prediction.dataset import TeamDataset


class TeamPredictor(ABC):
    def __init__(self):
        pass

    @property
    def num_processes(self):
        if not hasattr(self, "_num_processes"):
            self._num_processes = 1
        return self._num_processes

    @num_processes.setter
    def num_processes(self, value: int):
        self._num_processes = value

    @abstractmethod
    def predict(self, team: TeamSet) -> TeamSet:
        raise NotImplementedError


@lru_cache(maxsize=32)
def get_legacy_teambuilder(format: str):
    return TeamBuilder(format, verbose=False, remove_banned=False, inclusive=True)


class NaiveUsagePredictor(TeamPredictor):
    def predict(self, team: TeamSet):
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


@lru_cache(maxsize=1)
def load_replay_teams(format: str, max_teams: int, num_workers: int = 0):
    team_path = os.environ.get("METAMON_TEAMFILE_PATH", None)
    if team_path is None:
        print("METAMON_TEAMFILE_PATH environment variable is not set")
        exit()
    team_path = os.path.join(team_path, f"{format}_teams")

    # Initialize containers
    teams = []
    pokemon_sets = defaultdict(list)
    team_rosters = []

    # Get list of team files and shuffle
    filenames = [f for f in os.listdir(team_path) if f.endswith("team")]
    random.shuffle(filenames)
    filenames = filenames[:max_teams]

    print(f"Loading up to {len(filenames)} team files")

    dataset = TeamDataset(filenames, team_path, format)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=lambda x: x,
    )

    # Process in parallel with progress bar
    for batch in tqdm.tqdm(dataloader, desc="Loading Team Files", colour="blue"):
        for result in batch:
            if result is None:
                continue
            team, pokemon_dict, team_roster = result
            teams.append(team)
            for name, pokemon in pokemon_dict.items():
                pokemon_sets[name].append(pokemon)
            team_rosters.append(team_roster - {PokemonSet.MISSING_NAME})

    if not teams:
        raise ValueError(f"No valid team files found in {team_path}")

    print(f"Loaded {len(teams)} teams with {len(pokemon_sets)} unique Pokémon")
    return teams, pokemon_sets, team_rosters


class ReplayPredictor(NaiveUsagePredictor):
    def __init__(self, max_teams: int = 100_000):
        self.stat_format = None
        self.max_teams = max_teams

    def _load_data(self, format: str):
        self.smogon_stat = PreloadedSmogonStat(format, verbose=False, inclusive=True)
        workers = 0 if self.num_processes > 1 else 8
        self.teams, self.pokemon_sets, self.team_rosters = load_replay_teams(
            format, self.max_teams, workers
        )
        self.stat_format = format

    def _fill_pokemon_by_consistency(self, pokemon: PokemonSet):
        candidates = self.pokemon_sets[pokemon.name]
        consistent = [
            p for p in candidates if (pokemon.is_consistent_with(p) and pokemon != p)
        ]
        while len(consistent) > 1:
            pokemon = consistent.pop(random.randint(0, len(consistent) - 1))
            consistent = [
                p
                for p in consistent
                if (pokemon.is_consistent_with(p) and pokemon != p)
            ]
        return pokemon

    def _fill_team_by_consistency(self, team: TeamSet):
        consistent = [t for t in self.teams if team.is_consistent_with(t)]
        if consistent:
            team = consistent.pop(random.randint(0, len(consistent) - 1))
            while len(consistent) > 1:
                team = consistent.pop(random.randint(0, len(consistent) - 1))
                consistent = [t for t in consistent if team.is_consistent_with(t)]
        return team

    def _fill_roster_by_consistency(self, team: TeamSet):
        current_team_roster = set(
            p.name for p in team.pokemon if p.name != PokemonSet.MISSING_NAME
        )
        candidates = [t for t in self.team_rosters if current_team_roster < t]
        if candidates:
            while candidates:
                expanded_roster = candidates.pop(random.randint(0, len(candidates) - 1))
                candidates = [t for t in candidates if expanded_roster < t]
            new_pokemon = expanded_roster - current_team_roster
            if not len(new_pokemon) == sum(
                p.name == PokemonSet.MISSING_NAME for p in team.pokemon
            ):
                breakpoint()
            for p in team.pokemon:
                if p.name == PokemonSet.MISSING_NAME:
                    p.name = new_pokemon.pop()
        return team

    def predict(self, team: TeamSet) -> TeamSet:
        if self.stat_format != team.format:
            self._load_data(team.format)

        og_team = copy.deepcopy(team)
        # step 1: expand the team as far as possible based on other replays that are consistent with the current team
        team = self._fill_team_by_consistency(team)

        # step 2: fill the names of missing Pokemon based on other replays
        if any(p.name == PokemonSet.MISSING_NAME for p in team.pokemon):
            team = self._fill_roster_by_consistency(team)

        # step 3: fill each individual Pokemon with a self-consistent set
        filled_team = []
        for p in team.pokemon:
            if p.name != PokemonSet.MISSING_NAME:
                filled_team.append(self._fill_pokemon_by_consistency(p))
            else:
                filled_team.append(p)

        most_complete_team = TeamSet(
            lead=filled_team[0], reserve=filled_team[1:], format=team.format
        )
        if not og_team.is_consistent_with(most_complete_team):
            breakpoint()
        # step 4: fall back to old method for any remaining info
        return super().predict(most_complete_team)


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
    og_team = TeamSet.from_showdown_file(team_file, args.format)
    naive_team = naive_predictor.predict(og_team)
    consistent = og_team.is_consistent_with(naive_team)
    print(f"Consistent: {consistent}")
    edited_team = copy.deepcopy(naive_team)
    edited_team.lead.moves[0] = "Tackle"
    consistent = og_team.is_consistent_with(edited_team)
    print(f"Consistent: {consistent}")
    improved_predictor = ImprovedUsagePredictor()
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
