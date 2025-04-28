import os
import subprocess
import re
import json
from enum import Enum


class Tier(Enum):
    UBERS = 0
    OU = 1
    UU = 2
    RU = 3
    NU = 4
    PU = 5
    ZU = 6
    NFE = 7
    LC = 8

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


CAP_TIER_MAP = {
    "(OU)": Tier.OU,
    "OU": Tier.OU,
    "Uber": Tier.UBERS,
    "UU": Tier.UU,
    "RU": Tier.RU,
    "NU": Tier.NU,
    "PU": Tier.PU,
    "ZU": Tier.ZU,
    "NFE": Tier.NFE,
    "LC": Tier.LC,
    "OUBL": Tier.UBERS,
    "UUBL": Tier.OU,
    "RUBL": Tier.UU,
    "NUBL": Tier.RU,
    "PUBL": Tier.NU,
    "ZUBL": Tier.PU,
}


def ts_to_dict(ts_file):
    with open(ts_file) as f:
        lines = f.readlines()
        lines[0] = "{"
        lines[-1] = "}"
    content = "".join(lines)
    # fix the : in Type: Null
    content = content.replace("Type: Null", "Type Null")
    content = re.sub(r"(\w+):", r'"\1":', content)

    # Remove single-line comments
    content = re.sub(r"//.*", "", content)
    # Remove multi-line comments
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

    content = re.sub(r",(?=\s*[}\]])", "", content)
    content.replace("Type Null", "Type: Null")

    data = json.loads(content)
    return data


def find_global_npm_package_path(package_name):
    # Get the global node_modules path
    npm_root = subprocess.check_output(["npm", "root", "-g"], text=True).strip()
    breakpoint()
    package_path = os.path.join(npm_root, package_name)
    if os.path.exists(package_path):
        return package_path
    else:
        return None


def get_valid_pokemon(format):
    ps_path = os.path.join(
        os.path.abspath(__file__).split("metamon")[0],
        "metamon",
        "server",
        "pokemon-showdown",
    )
    if not os.path.exists(ps_path):
        raise ImportError("Cannot find path to pokemon-showdown")

    if "gen9" in format:
        formats_data = "formats-data.ts"

    else:
        generation = format[:4]
        formats_data = f"mods/{generation}/formats-data.ts"

    dex_data = "pokedex.ts"
    data_file = os.path.join(ps_path, "data", formats_data)
    dex_file = os.path.join(ps_path, "data", dex_data)
    formats_data = ts_to_dict(data_file)
    dex_data = ts_to_dict(dex_file)

    valid_pokemon = {}
    for pm, tiers in formats_data.items():
        if "tier" not in tiers:
            continue
        if pm not in dex_data:
            continue
        tier = tiers["tier"]
        if tier not in CAP_TIER_MAP:
            continue
        tier = CAP_TIER_MAP[tier]
        name = dex_data[pm]["name"]
        name = re.sub(r"[^a-zA-Z0-9]", "", name)
        if tier not in valid_pokemon:
            valid_pokemon[tier] = []
        valid_pokemon[tier].append(name)
    return valid_pokemon
