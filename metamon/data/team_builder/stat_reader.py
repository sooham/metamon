import os
import re
import json
from metamon.data import DATA_PATH
from metamon.data.team_builder.format_rules import get_valid_pokemon, Tier

from poke_env.data import to_id_str

TIER_MAP = {
    "ubers": Tier.UBERS,
    "ou": Tier.OU,
    "uu": Tier.UU,
    "ru": Tier.RU,
    "nu": Tier.NU,
    "pu": Tier.PU,
    "lc": Tier.LC,
}


class SmogonStat:
    def __init__(
        self, format, raw_stats_dir: str, date=None, rank=None, verbose: bool = True
    ) -> None:
        if date and type(date) == str:
            dates = [date]
        elif date and type(date) == list:
            dates = date
        else:
            dates = os.listdir(raw_stats_dir)

        self.data_paths = [os.path.join(raw_stats_dir, date) for date in dates]
        self.format = format
        self.rank = rank
        self.verbose = verbose

        self._movesets = {}
        self._inclusive = {}
        self._usage = None
        self._load()
        self._name_conversion = {
            to_id_str(pokemon): pokemon for pokemon in self._movesets.keys()
        }

    def _load(self):
        moveset_paths = [
            os.path.join(data_path, "moveset") for data_path in self.data_paths
        ]
        if self.verbose:
            print(f"Loading Smogon data from {len(moveset_paths)} paths")
        _movesets = []
        for moveset_path in moveset_paths:
            format_data = [
                x for x in os.listdir(moveset_path) if x.startswith(self.format + "-")
            ]

            if self.rank is not None:
                format_data = [x for x in format_data if self.rank in x]
            _movesets += [
                parse_pokemon_moveset(os.path.join(moveset_path, x))
                for x in format_data
            ]
        self._movesets = {to_id_str(k): v for k, v in merge_movesets(_movesets).items()}

    def pretty_print(self, key):
        data = self[key]
        print(f"------ {key} {data['count']} ------\n")
        print("\tMoveset:")
        sorted_moves = sorted(data["moves"].items(), key=lambda x: x[1], reverse=True)
        for i, (move, usage) in enumerate(sorted_moves[:10]):
            print(f"\t\t{i+1}. {move} ({usage * 100: .1f}%)")
        print("\n\tTeammates:")
        sorted_mates = sorted(
            data["teammates"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (mate, usage) in enumerate(sorted_mates[:5]):
            print(f"\t\t{i+1}. {mate} ({usage * 100: .1f}%)")
        print("\n\tChecks:")
        sorted_checks = sorted(data["checks"].items(), key=lambda x: x[1], reverse=True)
        for i, (check, usage) in enumerate(sorted_checks[:5]):
            print(f"\t\t{i+1}. {check} ({usage * 100: .1f}%)")

    def remove_banned_pm(self, ps_path):
        valid_pm_dict = get_valid_pokemon(ps_path, self.format[:4])
        tier = TIER_MAP[self.format[4:]]
        valid_pm = []
        # get all pokemon valid in this tier
        for t in valid_pm_dict.keys():
            if t >= tier:
                valid_pm.extend(valid_pm_dict[t])

        if self.verbose:
            print(f"Total {len(valid_pm)} valid pokemon for {self.format}")
        # remove pokemon that not in this tier
        for pm in list(self._movesets.keys()):
            if re.sub(r"[^a-zA-Z0-9]", "", pm) not in valid_pm:
                if self.verbose:
                    print(f"Remove {pm} from {self.format}")
                del self._movesets[pm]

    def __getitem__(self, key):
        try:
            return self.get_from_movesets(key)
        except KeyError as e:
            if self._inclusive:
                return self.get_from_inclusive(key)
            raise e

    def get_from_movesets(self, key):
        # Name in ps stat
        if key in self._movesets:
            return self._movesets[key]
        # Name in ps id
        clean_name = to_id_str(key)
        if clean_name in self._name_conversion:
            key = self._name_conversion[clean_name]
            return self._movesets[key]
        # Handle format like "Gastrodon-East"
        key = key.split("-")[0]
        if key in self._movesets:
            return self._movesets[key]

        raise KeyError(f"Pokemon {key} not found in {self.format}")

    def get_from_inclusive(self, key):
        raise KeyError(f"Please use PreloadedStat for inclusive search")

    @property
    def movesets(self):
        return self._movesets

    @property
    def usage(self):
        if self._usage is None:
            # create a list of pokemon names, sorted by count
            self._usage = list(
                sorted(
                    self._movesets.keys(),
                    key=lambda x: self._movesets[x]["count"],
                    reverse=True,
                )
            )
        return self._usage


class PreloadedSmogonStat(SmogonStat):
    def __init__(self, format, verbose: bool = True, inclusive=False):
        self.format = format
        self.rank = None
        self.verbose = verbose
        self._usage = None

        gen, tier = format[:4], format[4:]
        file_path = os.path.join(
            DATA_PATH, f"movesets_data/{gen}/{tier}/alltime_allrank.json"
        )
        inclusive_file_path = os.path.join(
            DATA_PATH, f"movesets_data/{gen}/inclusive.json"
        )
        if self.verbose:
            print(f"Loaded precomputed Smogon data from {file_path}")
        with open(file_path, "r") as file:
            self._movesets = json.load(file)
        if inclusive:
            with open(inclusive_file_path, "r") as file:
                self._inclusive = json.load(file)
        self._name_conversion = {
            to_id_str(pokemon): pokemon for pokemon in self._movesets.keys()
        }

    def _load(self):
        pass

    def get_from_inclusive(self, key):
        if self.verbose:
            print(f"Using inclusive search for {key}")
        # Name in ps stat
        if key in self._inclusive:
            return self._inclusive[key]
        # Name in ps id
        clean_name = to_id_str(key)
        if clean_name in self._name_conversion:
            key = self._name_conversion[clean_name]
            return self._inclusive[key]
        # Handle format like "Gastrodon-East"
        key = key.split("-")[0]
        if key in self._inclusive:
            return self._inclusive[key]

        raise KeyError(f"Pokemon {key} not found in {self.format}")


def parse_pokemon_moveset(file_path):
    moveset_data_list = {
        "name": [],
        "count": [],
        "abilities": [],
        "items": [],
        "spreads": [],
        "moves": [],
        "teammates": [],
        "checks": [],
    }

    def p_name(data_cache):
        name = data_cache[0][2:-2].strip()
        moveset_data_list["name"].append(name)
        return moveset_data_list

    def p_count(data_cache):
        count = int(data_cache[0][2:-2].split(":")[1].strip())
        moveset_data_list["count"].append(count)
        return moveset_data_list

    def p_abilities(data_cache):
        _abilities = {}
        assert "Abilities" in data_cache[0], "Abilities not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            # percent is in string format of xx%
            _abilities[name] = float(percent[:-1]) / 100
        moveset_data_list["abilities"].append(_abilities)
        # assert _abilities, "abilities is empty"
        return moveset_data_list

    def p_items(data_cache):
        _items = {}
        assert "Items" in data_cache[0], "Items not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            # percent is in string format of xx%
            _items[name] = float(percent[:-1]) / 100
        moveset_data_list["items"].append(_items)
        # assert _items, "items is empty"
        return moveset_data_list

    def p_spreads(data_cache):
        _spreads = {}
        assert "Spreads" in data_cache[0], "Spreads not found"
        for line in data_cache[1:]:
            nature_ev, percent = line[2:-2].strip().split()
            # percent is in string format of xx%
            _spreads[nature_ev] = float(percent[:-1]) / 100
        moveset_data_list["spreads"].append(_spreads)
        # assert _spreads, "spreads is empty"
        return moveset_data_list

    def p_moves(data_cache):
        _moves = {}
        assert "Moves" in data_cache[0], "Moves not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            _moves[name] = float(percent[:-1]) / 100
        # assert _moves, "moves is empty"
        moveset_data_list["moves"].append(_moves)
        return moveset_data_list

    def p_teammates(data_cache):
        _teammates = {}
        assert "Teammates" in data_cache[0], "Teammates not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            if percent.startswith("+") or percent.startswith("-"):
                percent = percent[1:]
            # sometimes the data will have a duplicate line and cause error, just skip that line
            try:
                _teammates[name] = float(percent[:-1]) / 100
            except:
                continue
        # assert _teammates, "teammates is empty"
        moveset_data_list["teammates"].append(_teammates)
        return moveset_data_list

    def p_checks(data_cache):
        _checks = {}
        assert "Checks and Counters" in data_cache[0], "Checks and Counters not found"
        for i in range(1, len(data_cache), 2):
            line = data_cache[i]
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-2])
            percent = line_split[-2]
            # percent is in string format of xx%
            _checks[name] = float(percent) / 100
        moveset_data_list["checks"].append(_checks)
        # assert _checks, "checks is empty"
        return moveset_data_list

    with open(file_path, "r") as file:
        lines = file.readlines()

    section_order = [
        p_name,
        p_count,
        p_abilities,
        p_items,
        p_spreads,
        p_moves,
        p_teammates,
        p_checks,
        lambda _: None,
    ]
    current_section = -1

    data_cache = []
    for line in lines:
        # print(line)
        if line.startswith(" +-"):
            section_order[current_section](data_cache)
            current_section = (current_section + 1) % len(section_order)
            data_cache = []
        elif line.startswith(" |"):
            data_cache.append(line.strip())

    n = len(moveset_data_list["name"])
    moveset_data = {}
    for i in range(n):
        _name = moveset_data_list["name"][i]
        _count = moveset_data_list["count"][i]
        _abilities = moveset_data_list["abilities"][i]
        _items = moveset_data_list["items"][i]
        _spreads = moveset_data_list["spreads"][i]
        _moves = moveset_data_list["moves"][i]
        _teammates = moveset_data_list["teammates"][i]
        _checks = moveset_data_list["checks"][i]
        moveset_data[_name] = {
            "count": _count,
            "abilities": _abilities,
            "items": _items,
            "spreads": _spreads,
            "moves": _moves,
            "teammates": _teammates,
            "checks": _checks,
        }

    return moveset_data


def merge_movesets(movesets):
    result = {}
    for moveset in movesets:
        for pokemon, data in moveset.items():
            # add up count first
            count = data["count"]
            if pokemon not in result:
                result[pokemon] = {}
                result[pokemon]["count"] = 0
            for key, value in data.items():
                if key == "count":
                    result[pokemon][key] += value
                    continue
                # key is the entry name, e.g. "abilities", "items", "spreads", "moves", "teammates", "checks"
                if key not in result[pokemon]:
                    result[pokemon][key] = {}
                # k is the counted thing, v is the percentage
                for k, v in value.items():
                    if k not in result[pokemon][key]:
                        result[pokemon][key][k] = 0
                    result[pokemon][key][k] += v * count

    # recalculating the percentage
    for pokemon, data in result.items():
        count = data["count"]
        for key, value in data.items():
            if key != "count":
                for k, v in value.items():
                    result[pokemon][key][k] /= count

    return result


if __name__ == "__main__":
    stats = PreloadedSmogonStat("gen9ou")
    print(len(stats.usage))
    stats.remove_banned_pm("/home/xieleo/pokemon-showdown")
    print(len(stats.usage))
    for mon in sorted(
        stats.movesets.keys(), key=lambda m: stats[m]["count"], reverse=True
    )[:5]:
        stats.pretty_print(mon)
        print()
