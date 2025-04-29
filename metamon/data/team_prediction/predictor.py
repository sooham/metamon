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
        gen = int(team.format.split("gen")[1][0])
        pokemon = [team.lead] + team.reserve
        # use legacy team builder to generate a team of 6 Pokémon based on the ones we already
        # have and their common teammates. Teammates and all moves/items/abilities are sampled
        # from a premade set of Showdown usage statistics.
        existing_names = [p.name for p in pokemon if p.name != PokemonSet.MISSING_NAME]
        sample_team = team_builder.generate_new_team(existing_names)

        # convert from the output of the old team builder to the new PokemonSet format
        cleaned_dict = {}
        for poke in sample_team:
            if poke["name"] == PokemonSet.MISSING_NAME:
                breakpoint()
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
        try:
            sample_team = [
                PokemonSet.from_dict(val | {"name": key})
                for key, val in cleaned_dict.items()
            ]
        except Exception as e:
            breakpoint()
            raise e

        # fill missing information in the revelealed replay time with plausible values
        # from the generated team.
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
                    filled_p.fill_from_PokemonSet(new_p)
                    merged_team.append(filled_p)
                    break
        return TeamSet(lead=merged_team[0], reserve=merged_team[1:], format=team.format)
