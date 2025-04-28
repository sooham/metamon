from typing import List, Optional
import random
from dataclasses import dataclass
import os


@dataclass
class PokemonSet:
    name: str
    moves: List[str]
    ability: str
    item: str
    evs: List[int]
    ivs: List[int]
    nature: str

    def to_str(self):
        evs = "EVs: "
        for desc, ev_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.evs):
            evs += f"{ev_val} {desc}"
            if desc != "Spe":
                evs += " / "
        evs += f"\n{self.nature} Nature"
        ivs = "IVs: "
        for desc, iv_val in zip(["HP", "Atk", "Def", "SpA", "SpD", "Spe"], self.ivs):
            ivs += f"{iv_val} {desc}"
            if desc != "Spe":
                ivs += " / "

        start = f"{self.name}"
        if self.item != "No Item":
            start += f" @ {self.item}"
        moves = "\n".join([f"- {move}" for move in self.moves if move != "No Move"])
        return start + f"\nAbility: {self.ability}\n{evs}\n{ivs}\n{moves}"

    @classmethod
    def from_showdown_block(cls, block: str, gen: int):
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if "@" in lines[0]:
            name, item = lines[0].split("@")
            name = name.strip()
            item = item.strip()
            if not item:
                item = "No Item"
        else:
            name = lines[0].strip()
            item = "No Item"
        # Set defaults based on gen
        if gen == 1 or gen == 2:
            evs = [252] * 6
            ivs = [15] * 6
            nature = "Hardy"  # Nature doesn't exist, but placeholder
        else:
            evs = [0] * 6
            ivs = [31] * 6
        moves = ["No Move"] * 4
        ability = "No Ability"
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
                moves[moves.index("No Move")] = line[2:].strip()
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


@dataclass
class TeamSet:
    lead: PokemonSet
    reserve: List[PokemonSet]
    gen: int

    def to_str(self):
        out = f"{self.lead.to_str()}"
        for p in self.reserve:
            out += f"\n\n{p.to_str()}"
        return out

    @classmethod
    def from_showdown_file(cls, path: str, gen: int):
        with open(path, "r") as f:
            content = f.read()
        blocks = [block for block in content.strip().split("\n\n") if block.strip()]
        pokemons = [PokemonSet.from_showdown_block(block, gen) for block in blocks]
        if not pokemons:
            raise ValueError("No Pokémon found in file.")
        lead = pokemons[0]
        reserve = pokemons[1:]
        return cls(lead=lead, reserve=reserve, gen=gen)

    def write_to_file(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_str())

    def to_dict(self, shuffle_reserve: bool = False):
        out = {self.lead.name: self.lead.to_dict()}
        order = lambda x: (
            x if not shuffle_reserve else lambda x: random.sample(x, k=len(x))
        )
        for p in order(self.reserve):
            out[p.name] = p.to_dict()
        return out


if __name__ == "__main__":
    import os

    TEAM_DIR = os.path.join(os.path.dirname(__file__), "..", "teams")
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
        gen = int(path.split("/gen")[1][0])
        team = TeamSet.from_showdown_file(path, gen)
        print("---------------------------------------------------------")
        print(team.to_str())
        print(team.to_dict())
        input()
