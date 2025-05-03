import random
import re
import os
import copy
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Optional
import unicodedata

from metamon.data.replay_dataset.parsed_replays.replay_parser.replay_state import (
    Pokemon,
    Nothing,
)

from metamon.data.team_builder.stat_reader import PreloadedSmogonStat


@lru_cache
def get_preloaded_stat(gen: int):
    return PreloadedSmogonStat(f"gen{gen}ubers", inclusive=True, verbose=False)


def moveset_size(pokemon_name: str, gen: int) -> int:
    stat = get_preloaded_stat(gen)
    try:
        moves = len(
            set(stat.get_from_inclusive(pokemon_name)["moves"].keys()) - {"Nothing"}
        )
    except KeyError:
        print(f"KeyError for {pokemon_name} in gen {gen}")
        return 4
    if moves < 4:
        breakpoint()
    moveset = min(moves, 4)
    return moveset


@dataclass
class PokemonSet:
    name: str
    gen: int
    moves: List[str]
    ability: str
    item: str
    nature: str
    evs: List[int]
    ivs: List[int]

    NO_MOVE = "<nomove>"
    NO_ABILITY = "<noability>"
    NO_ITEM = "<noitem>"
    NO_NATURE = "<nonature>"

    MISSING_NAME = "$missing_name$"
    MISSING_MOVE = "$missing_move$"
    MISSING_ABILITY = "$missing_ability$"
    MISSING_ITEM = "$missing_item$"
    MISSING_EV = "$missing_ev$"
    MISSING_IV = "$missing_iv$"
    MISSING_NATURE = "$missing_nature$"

    def __post_init__(self):
        if self.name != self.MISSING_NAME:
            assert len(self.moves) == moveset_size(self.name, self.gen)
        assert len(self.evs) == 6
        assert len(self.ivs) == 6
        assert self.nature is not None
        assert self.item is not None
        assert self.ability is not None
        self.missing_strings = [
            self.MISSING_MOVE,
            self.MISSING_ABILITY,
            self.MISSING_ITEM,
            self.MISSING_NATURE,
        ]
        self.missing_regex = re.compile("|".join(map(re.escape, self.missing_strings)))

    @classmethod
    def default_moves(cls, name: str, gen: int):
        if name == cls.MISSING_NAME:
            return [cls.MISSING_MOVE] * 4
        m = moveset_size(name, gen)
        # we are trying to catch cases where a full moveset would actually be less than 4 moves
        return [cls.MISSING_MOVE] * m + [cls.NO_MOVE] * (4 - m)

    @classmethod
    def default_ivs(cls, gen: int):
        # mirroring Showdown logic where IVs are assumed to be 31
        return [31] * 6

    @classmethod
    def default_evs(cls, gen: int):
        return [252] * 6 if gen <= 2 else [cls.MISSING_EV] * 6

    @classmethod
    def default_nature(cls, gen: int):
        # the trick in gens before nature is for the backend to pick a neutral nature.
        # here we set to NO_NATURE and then ignore it in the output string.
        return cls.NO_NATURE if gen <= 2 else cls.MISSING_NATURE

    @classmethod
    def default_item(cls, gen: int):
        return cls.NO_ITEM if gen <= 1 else cls.MISSING_ITEM

    @classmethod
    def default_ability(cls, gen: int):
        return cls.NO_ABILITY if gen <= 2 else cls.MISSING_ABILITY

    @classmethod
    def from_ReplayPokemon(cls, pokemon: Optional[Pokemon], gen: int):
        if pokemon is None:
            return cls.missing_pokemon(gen=gen)
        moves = [m.name for m in pokemon.had_moves.values()]
        while len(moves) < moveset_size(pokemon.name, pokemon.gen):
            moves.append(cls.MISSING_MOVE)

        # maintaining the replay parser's distinction between "known to be None" and "unrevealed"
        if pokemon.gen == 1 or pokemon.had_item == Nothing.NO_ITEM:
            item = cls.NO_ITEM
        elif pokemon.had_item is None:
            item = cls.MISSING_ITEM
        else:
            item = pokemon.had_item
        if pokemon.gen <= 2 or pokemon.had_ability == Nothing.NO_ABILITY:
            ability = cls.NO_ABILITY
        elif pokemon.had_ability is None:
            ability = cls.MISSING_ABILITY
        else:
            ability = pokemon.had_ability
        return cls(
            name=pokemon.name,
            gen=pokemon.gen,
            moves=moves,
            ability=ability,
            item=item,
            nature=cls.default_nature(gen),
            evs=cls.default_evs(gen),
            ivs=cls.default_ivs(gen),
        )

    def fill_from_PokemonSet(self, other):
        """
        Used to merge the results of a team prediction into existing team info.
        """
        if not isinstance(other, PokemonSet):
            raise ValueError("other must be a PokemonSet")
        if not self.name == other.name:
            raise ValueError("other must have the same name")
        new_moves = list(set(other.moves) - set(self.moves))
        for move in self.moves:
            if move == self.MISSING_MOVE:
                if new_moves:
                    new_move = new_moves.pop()
                    self.moves[self.moves.index(move)] = new_move
        if self.ability == self.MISSING_ABILITY:
            self.ability = other.ability
        if self.item == self.MISSING_ITEM:
            self.item = other.item
        if self.nature == self.MISSING_NATURE:
            self.nature = other.nature
        for idx, ev in enumerate(self.evs):
            if ev == self.MISSING_EV:
                self.evs[idx] = other.evs[idx]
        for idx, iv in enumerate(self.ivs):
            if iv == self.MISSING_IV:
                self.ivs[idx] = other.ivs[idx]

    def to_str(self):
        evs = "EVs: "
        for desc, ev_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.evs):
            evs += f"{ev_val} {desc}"
            if desc != "Spe":
                evs += " / "
        if self.nature != self.NO_NATURE:
            evs += f"\n{self.nature} Nature"
        ivs = "IVs: "
        for desc, iv_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.ivs):
            ivs += f"{iv_val} {desc}"
            if desc != "Spe":
                ivs += " / "

        start = f"{self.name} @ {self.item}"
        moves = "\n".join([f"- {move}" for move in self.moves])
        return start + f"\nAbility: {self.ability}\n{evs}\n{ivs}\n{moves}"

    @classmethod
    def from_showdown_block(cls, block: str, gen: int):
        block = block.replace("\u200b", "")
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]

        # Parse first line: handle nickname, gender, and item
        first = lines[0]
        if "@" in first:
            name_part, item_part = first.split("@", 1)
            item = item_part.strip() or cls.default_item(gen)
        else:
            name_part = first
            item = cls.default_item(gen)
        name_raw = name_part.strip()
        # Extract parenthetical contents and pick the species (ignore gender flags)
        contents = re.findall(r"\(([^)]+)\)", name_raw)
        species = None
        for content in contents:
            c = content.strip()
            if c.upper() in ("M", "F"):
                continue
            species = c
            break
        if species:
            name = species
        else:
            # Remove any parentheses (nicknames or gender)
            name = re.sub(r"\s*\([^)]*\)", "", name_raw).strip()

        # Set defaults based on gen
        evs = cls.default_evs(gen)
        ivs = cls.default_ivs(gen)
        nature = cls.default_nature(gen)
        ability = cls.default_ability(gen)
        moves = cls.default_moves(name, gen)

        for line in lines[1:]:
            if line.startswith("Ability:"):
                if gen > 2:
                    ability = line.split(":", 1)[1].strip()
            elif line.startswith("EVs:"):
                evs = [0] * 6 if gen > 2 else [252] * 6
                for part in line[4:].split("/"):
                    stat = part.strip().split(" ")
                    if len(stat) == 2:
                        val, stat_name = stat
                        idx = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"].index(stat_name)
                        if val != cls.MISSING_EV:
                            evs[idx] = int(val)
                        else:
                            evs[idx] = cls.MISSING_EV
            elif line.startswith("IVs:"):
                for part in line[4:].split("/"):
                    stat = part.strip().split(" ")
                    if len(stat) == 2:
                        val, stat_name = stat
                        idx = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"].index(stat_name)
                        if val != cls.MISSING_IV:
                            ivs[idx] = int(val)
                        else:
                            ivs[idx] = cls.MISSING_IV
            elif line.endswith("Nature"):
                if gen >= 3:
                    nature = line.split()[0].strip()
            elif line.startswith("- "):
                move_raw = line[2:].strip()
                # if multiple options, take the first option
                if "/" in move_raw:
                    move_raw = move_raw.split("/", 1)[0].strip()
                # normalize Hidden Power by dropping any specific type
                if move_raw.startswith("Hidden Power"):
                    move_raw = "Hidden Power"
                try:
                    moves[moves.index(cls.MISSING_MOVE)] = move_raw
                except ValueError:
                    breakpoint()
        return cls(
            name=name,
            gen=gen,
            moves=moves,
            ability=ability,
            item=item,
            evs=evs,
            ivs=ivs,
            nature=nature,
        )

    def to_dict(self):
        return {
            "moves": self.moves,
            "gen": self.gen,
            "ability": self.ability,
            "item": self.item,
            "evs": self.evs,
            "ivs": self.ivs,
            "nature": self.nature,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            name=d["name"],
            gen=d["gen"],
            ability=d["ability"],
            item=d["item"],
            nature=d["nature"],
            moves=d["moves"],
            evs=d["evs"],
            ivs=d["ivs"],
        )

    def to_seq(self, include_stats: bool = True):
        seq = [
            f"Mon: {self.name}",
            f"Ability: {self.ability}",
            f"Item: {self.item}",
        ]
        moves = [f"Move: {move}" for move in self.moves]
        seq += moves
        if include_stats:
            nature = f"Nature: {self.nature}"
            evs = [f"EVs: {ev}" for ev in self.evs]
            ivs = [f"IV: {iv}" for iv in self.ivs]
            seq += [nature] + evs + ivs
        mask = [bool(self.missing_regex.search(word)) for word in seq]
        return seq, mask

    @classmethod
    def from_seq(cls, seq: List[str], gen: int, include_stats: bool = True):
        name = seq[0].split(":")[1].strip()
        ability = seq[1].split(":")[1].strip()
        item = seq[2].split(":")[1].strip()
        moves = [move.split(":")[1].strip() for move in seq[3:7]]
        if include_stats:
            nature = seq[7].split(":")[1].strip()
            evs = [ev.split(":")[1].strip() for ev in seq[8:14]]
            ivs = [iv.split(":")[1].strip() for iv in seq[14:20]]
        else:
            nature = cls.default_nature(gen)
            evs = cls.default_evs(gen)
            ivs = cls.default_ivs(gen)
        return cls(
            name=name,
            gen=gen,
            ability=ability,
            item=item,
            nature=nature,
            moves=moves,
            evs=evs,
            ivs=ivs,
        )

    @classmethod
    def missing_pokemon(cls, gen: int):
        return cls(
            name=PokemonSet.MISSING_NAME,
            gen=gen,
            ability=PokemonSet.default_ability(gen),
            item=PokemonSet.default_item(gen),
            nature=PokemonSet.default_nature(gen),
            moves=[PokemonSet.MISSING_MOVE] * 4,
            evs=PokemonSet.default_evs(gen),
            ivs=PokemonSet.default_ivs(gen),
        )

    def masked(self, mask_attrs_prob: float = 0.1):
        data = self.to_dict()
        data["name"] = self.name
        # Mask nature, item, ability
        if random.random() < mask_attrs_prob:
            data["nature"] = self.MISSING_NATURE
        if random.random() < mask_attrs_prob:
            data["item"] = self.MISSING_ITEM
        if random.random() < mask_attrs_prob:
            data["ability"] = self.MISSING_ABILITY
        # Mask moves
        masked_moves = []
        for move in data["moves"]:
            if random.random() < mask_attrs_prob:
                masked_moves.append(self.MISSING_MOVE)
            else:
                masked_moves.append(move)
        data["moves"] = masked_moves
        # Mask EVs/IVs --- it only makes sense to do this as all-or-nothing
        if random.random() < mask_attrs_prob:
            data["evs"] = [self.MISSING_EV] * 6
        if random.random() < mask_attrs_prob:
            data["ivs"] = [self.MISSING_IV] * 6
        return type(self).from_dict(data)


@dataclass
class TeamSet:
    lead: PokemonSet
    reserve: List[PokemonSet]
    format: str

    @property
    def pokemon(self):
        return [self.lead] + self.reserve

    def to_str(self):
        out = f"{self.lead.to_str()}"
        for p in self.reserve:
            out += f"\n\n{p.to_str()}"
        return out

    @classmethod
    def from_showdown_file(cls, path: str, format: str):
        with open(path, "r") as f:
            content = f.read()
        gen = int(format.split("gen")[1][0])
        blocks = [block for block in content.strip().split("\n\n") if block.strip()]
        pokemons = [PokemonSet.from_showdown_block(block, gen=gen) for block in blocks]
        if not pokemons:
            raise ValueError("No Pokémon found in file.")
        lead = pokemons[0]
        reserve = pokemons[1:]
        return cls(lead=lead, reserve=reserve, format=format)

    def write_to_file(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_str())

    def to_dict(self):
        out = {self.lead.name: self.lead.to_dict()}
        for p in self.reserve:
            out[p.name] = p.to_dict()
        return out

    def to_seq(self, include_stats: bool = True):
        lead_seq, lead_mask = self.lead.to_seq(include_stats=include_stats)
        reserve_seq, reserve_mask = [], []
        for p in self.reserve:
            p_seq, p_mask = p.to_seq(include_stats=include_stats)
            reserve_seq.append(p_seq)
            reserve_mask.append(p_mask)

        # add format to the beginning (which never needs to be predicted)
        seq = [f"Format: {self.format}"] + lead_seq
        mask = [False] + lead_mask
        for reserve_seq, reserve_mask in zip(reserve_seq, reserve_mask):
            seq += reserve_seq
            mask += reserve_mask
        return seq, mask

    @classmethod
    def from_seq(cls, seq: List[str], include_stats: bool = True):
        format = seq[0].split(":")[1].strip()
        gen = int(format.split("gen")[1][0])
        poke_seq_len = 20 if include_stats else 7
        lead = PokemonSet.from_seq(
            seq[1 : poke_seq_len + 1], gen=gen, include_stats=include_stats
        )
        idx = poke_seq_len + 1
        reserve = []
        while idx < len(seq):
            reserve.append(
                PokemonSet.from_seq(
                    seq[idx : idx + poke_seq_len], gen=gen, include_stats=include_stats
                )
            )
            idx += poke_seq_len
        return cls(lead=lead, reserve=reserve, format=format)

    def shuffle(self):
        random.shuffle(self.reserve)
        for p in [self.lead] + self.reserve:
            random.shuffle(p.moves)
        return self

    def to_prediction_pair(
        self, mask_pokemon_prob: float = 0.1, mask_attrs_prob: float = 0.1
    ):
        gen = int(self.format.split("gen")[1][0])
        y = copy.deepcopy(self)
        y.shuffle()
        x = copy.deepcopy(y)
        masked_lead = x.lead.masked(mask_attrs_prob=mask_attrs_prob)
        masked_reserve = []
        for p in x.reserve:
            if random.random() < mask_pokemon_prob:
                masked_reserve.append(PokemonSet.missing_pokemon(gen=gen))
            else:
                masked_reserve.append(p.masked(mask_attrs_prob=mask_attrs_prob))
        x.reserve = masked_reserve
        x.lead = masked_lead
        return x, y


if __name__ == "__main__":
    import os

    TEAM_DIR = os.path.join(os.path.dirname(__file__), "..", "teams")
    # TEAM_DIR = "/mnt/data1/shared_pokemon_project/metamon_team_files"
    TEAM_DIR = os.path.join(TEAM_DIR, "gen1", "ou", "competitive")
    team_files = []
    for root, dirs, files in os.walk(TEAM_DIR):
        for file in files:
            if file.endswith("team") or file.startswith("team"):
                team_files.append(os.path.join(root, file))

    print(f"Found {len(team_files)} team files.")

    random.shuffle(team_files)
    for path in team_files:
        print(f"\nLoading team from: {path}")
        with open(path, "r") as f:
            txt = f.read()
            print(txt)
        team = TeamSet.from_showdown_file(path, "gen1ou")
        print("---------------------------------------------------------")
        x, y = team.to_prediction_pair()
        print(x.to_str())
        print("---------------------------------------------------------")
        print(y.to_str())
        print(x.to_seq(include_stats=False))
        print(y.to_seq(include_stats=False))
        assert len(x.to_seq(include_stats=False)) == len(y.to_seq(include_stats=False))
        y_copy = TeamSet.from_seq(y.to_seq(include_stats=True), include_stats=True)
        if y_copy.to_str() != y.to_str():
            breakpoint()
