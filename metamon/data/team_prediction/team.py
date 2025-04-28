from typing import List, Optional
import random
from dataclasses import dataclass
import os
import copy


@dataclass
class PokemonSet:
    name: str
    moves: List[str]
    ability: str
    item: str
    nature: str
    evs: List[int]
    ivs: List[int]

    NO_MOVE = "<nomove>"
    NO_ABILITY = "<noability>"
    NO_ITEM = "<noitem>"

    MISSING_NAME = "$missing_name$"
    MISSING_MOVE = "$missing_move$"
    MISSING_ABILITY = "$missing_ability$"
    MISSING_ITEM = "$missing_item$"
    MISSING_EV = "$missing_ev$"
    MISSING_IV = "$missing_iv$"
    MISSING_NATURE = "$missing_nature$"

    def __post_init__(self):
        assert len(self.moves) == 4
        assert len(self.evs) == 6
        assert len(self.ivs) == 6

    def to_str(self):
        evs = "EVs: "
        for desc, ev_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.evs):
            evs += f"{ev_val if ev_val is not None else self.MISSING_EV} {desc}"
            if desc != "Spe":
                evs += " / "
        evs += f"\n{self.nature or self.MISSING_NATURE} Nature"
        ivs = "IVs: "
        for desc, iv_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.ivs):
            ivs += f"{iv_val if iv_val is not None else self.MISSING_IV} {desc}"
            if desc != "Spe":
                ivs += " / "

        start = f"{self.name or self.MISSING_NAME} @ {self.item or self.MISSING_ITEM}"
        moves = "\n".join([f"- {move}" for move in self.moves])
        return (
            start
            + f"\nAbility: {self.ability or self.MISSING_ABILITY}\n{evs}\n{ivs}\n{moves}"
        )

    @classmethod
    def from_showdown_block(cls, block: str, format: str):
        gen = int(format.split("gen")[1][0])

        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]

        if "@" in lines[0]:
            name, item = [l.strip() for l in lines[0].split("@")]
            if not item:
                item = cls.MISSING_ITEM if gen > 1 else cls.NO_ITEM
        else:
            name = lines[0].strip()
            item = cls.MISSING_ITEM if gen > 1 else cls.NO_ITEM

        # Set defaults based on gen
        if gen == 1 or gen == 2:
            evs = [252] * 6
            ivs = [31] * 6
            nature = "Hardy"  # Nature doesn't exist, but placeholder
            ability = cls.NO_ABILITY
        else:
            evs = [cls.MISSING_EV] * 6
            ivs = [cls.MISSING_IV] * 6
            nature = cls.MISSING_NATURE
            ability = cls.MISSING_ABILITY

        moves = [cls.MISSING_MOVE] * 4
        for line in lines[1:]:
            if line.startswith("Ability:"):
                if gen > 2:
                    ability = line.split(":", 1)[1].strip()
            elif line.startswith("EVs:"):
                # Only parse if not Gen 1/2
                if gen >= 3:
                    evs = [0] * 6
                    for part in line[4:].split("/"):
                        stat = part.strip().split(" ")
                        if len(stat) == 2:
                            val, stat_name = stat
                            idx = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"].index(
                                stat_name
                            )
                            evs[idx] = int(val)
            elif line.startswith("IVs:"):
                # Only parse if not Gen 1/2
                if gen >= 3:
                    ivs = [31] * 6
                    for part in line[4:].split("/"):
                        stat = part.strip().split(" ")
                        if len(stat) == 2:
                            val, stat_name = stat
                            idx = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"].index(
                                stat_name
                            )
                            ivs[idx] = int(val)
            elif line.endswith("Nature"):
                if gen >= 3:
                    nature = line.split()[0]
            elif line.startswith("- "):
                moves[moves.index(cls.MISSING_MOVE)] = line[2:].strip()
        return cls(
            name=name,
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
            ability=d["ability"],
            item=d["item"],
            nature=d["nature"],
            moves=d["moves"],
            evs=d["evs"],
            ivs=d["ivs"],
        )

    def to_seq(self):
        return [
            self.name,
            self.ability,
            self.item,
            self.nature,
            *self.moves,
            *self.evs,
            *self.ivs,
        ]

    @classmethod
    def from_seq(cls, seq: List[str]):
        return cls(
            name=seq[0],
            ability=seq[1],
            item=seq[2],
            nature=seq[3],
            moves=seq[4:8],
            evs=seq[8:14],
            ivs=seq[14:20],
        )

    @classmethod
    def missing_pokemon(cls):
        return cls(
            name=PokemonSet.MISSING_NAME,
            ability=PokemonSet.MISSING_ABILITY,
            item=PokemonSet.MISSING_ITEM,
            nature=PokemonSet.MISSING_NATURE,
            moves=[PokemonSet.MISSING_MOVE] * 4,
            evs=[PokemonSet.MISSING_EV] * 6,
            ivs=[PokemonSet.MISSING_IV] * 6,
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

    def to_str(self):
        out = f"{self.lead.to_str()}"
        for p in self.reserve:
            out += f"\n\n{p.to_str()}"
        return out

    @classmethod
    def from_showdown_file(cls, path: str, format: str):
        with open(path, "r") as f:
            content = f.read()
        blocks = [block for block in content.strip().split("\n\n") if block.strip()]
        pokemons = [PokemonSet.from_showdown_block(block, format) for block in blocks]
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

    def to_seq(self, shuffle: bool = True):
        seq = [self.format] + self.lead.to_seq(shuffle_moves=shuffle)
        for reserve in self.reserve:
            seq += reserve.to_seq(shuffle_moves=shuffle)
        return seq

    @classmethod
    def from_seq(cls, seq: List[str]):
        format = seq[0]
        lead = PokemonSet.from_seq(seq[1:21])
        idx = 21
        reserve = []
        while idx < len(seq):
            reserve.append(PokemonSet.from_seq(seq[idx : idx + 20]))
            idx += 20
        return cls(lead=lead, reserve=reserve, format=format)

    def shuffle(self):
        random.shuffle(self.reserve)
        for p in [self.lead] + self.reserve:
            random.shuffle(p.moves)
        return self

    def to_prediction_pair(
        self, mask_pokemon_prob: float = 0.1, mask_attrs_prob: float = 0.1
    ):
        y = copy.deepcopy(self)
        y.shuffle()
        x = copy.deepcopy(y)
        masked_lead = x.lead.masked(mask_attrs_prob=mask_attrs_prob)
        masked_reserve = []
        for p in x.reserve:
            if random.random() < mask_pokemon_prob:
                masked_reserve.append(PokemonSet.missing_pokemon())
            else:
                masked_reserve.append(p.masked(mask_attrs_prob=mask_attrs_prob))
        x.reserve = masked_reserve
        x.lead = masked_lead
        return x, y


if __name__ == "__main__":
    import os

    TEAM_DIR = os.path.join(os.path.dirname(__file__), "..", "teams")
    TEAM_DIR = os.path.join(TEAM_DIR, "gen4", "ou", "competitive")
    team_files = []
    for root, dirs, files in os.walk(TEAM_DIR):
        for file in files:
            if file.endswith(".txt") or file.startswith("team"):
                team_files.append(os.path.join(root, file))

    print(f"Found {len(team_files)} team files.")

    for path in team_files:
        print(f"\nLoading team from: {path}")
        with open(path, "r") as f:
            txt = f.read()
            print(txt)
        team = TeamSet.from_showdown_file(path, "gen4ou")
        print("---------------------------------------------------------")
        x, y = team.to_prediction_pair()
        print(x.to_str())
        print("---------------------------------------------------------")
        print(y.to_str())
        input()
