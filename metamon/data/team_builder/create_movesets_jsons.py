import os
import json
from metamon.data.team_builder.format_rules import Tier, get_valid_pokemon
from metamon.data.team_builder.stat_reader import SmogonStat

VALID_TIERS = [Tier.UBERS, Tier.OU, Tier.UU, Tier.RU, Tier.NU, Tier.PU]


def create_for_all_dates():
    for i in range(1, 10):
        valid_pm_dict = get_valid_pokemon("/home/xieleo/pokemon-showdown", f"gen{i}")
        for format in VALID_TIERS:
            valid_pm = []
            for tier in valid_pm_dict.keys():
                if tier >= format:
                    valid_pm.extend(valid_pm_dict[tier])
            print(f"Total {len(valid_pm)} valid pokemon for {format.name} in gen{i}")
            format_name = f"gen{i}{format.name.lower()}"
            stat = SmogonStat(format_name)
            # remove pokemon that not in format
            # for pm in list(stat.movesets.keys()):
            #     if pm not in valid_pm:
            #         del stat.movesets[pm]
            if stat.movesets:
                # recursively create the path
                path = f"movesets_data/gen{i}/{format.name.lower()}"
                os.makedirs(path, exist_ok=True)
                # dump to json
                with open(f"{path}/alltime_allrank.json", "w") as f:
                    json.dump(stat.movesets, f)


if __name__ == "__main__":
    create_for_all_dates()
