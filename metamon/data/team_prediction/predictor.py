import copy
from abc import ABC, abstractmethod
from functools import lru_cache

from metamon.data.team_builder.team_builder import TeamBuilder
from metamon.data.team_prediction.team import TeamSet, PokemonSet


class TeamPredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, team: TeamSet) -> TeamSet:
        raise NotImplementedError("Subclass must implement this method")


@lru_cache(maxsize=32)
def get_legacy_teambuilder(format: str):
    return TeamBuilder(format, verbose=False, remove_banned=False, inclusive=True)


class NaiveUsagePredictor(TeamPredictor):

    def predict(self, team: TeamSet):
        team_builder = get_legacy_teambuilder(team.format)
        pokemon = [team.lead] + team.reserve
        existing_names = [p.name for p in pokemon if p.name != PokemonSet.MISSING_NAME]
        sample_team = team_builder.generate_new_team(existing_names)
        cleaned_dict = {}
        for poke in sample_team:
            if poke["name"] == PokemonSet.MISSING_NAME:
                breakpoint()
            spread = poke["spread"]
            nature, evs = spread.split(":")
            nature = nature.strip()
            evs = [int(x) for x in evs.split("/")]
            ivs = [int(x) for x in poke["IVs"].split("/")]
            cleaned_dict[poke["name"]] = {
                "moves": poke["moves"],
                "nature": nature,
                "evs": evs,
                "ivs": ivs,
                "ability": poke["ability"],
                "item": poke["item"],
            }
        sample_team = [
            PokemonSet.from_dict(val | {"name": key})
            for key, val in cleaned_dict.items()
        ]
        new_pokemon = [p for p in sample_team if p.name not in existing_names]
        merged_team = []
        for p in team.pokemon:
            if p.name == PokemonSet.MISSING_NAME:
                new_choice = new_pokemon.pop(0)
                merged_team.append(copy.deepcopy(new_choice))
                continue
            for new_p in sample_team:
                if new_p.name == p.name:
                    filled_p = copy.deepcopy(p)
                    filled_p.fill_from_pokemon(new_p)
                    merged_team.append(filled_p)
                    break
            else:
                breakpoint()
        return TeamSet(lead=merged_team[0], reserve=merged_team[1:], format=team.format)
