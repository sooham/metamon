import os
import json
import argparse
from metamon.data import DATA_PATH
from metamon.data.legacy_team_builder.format_rules import Tier, get_valid_pokemon
from metamon.data.legacy_team_builder.stat_reader import SmogonStat, merge_movesets

VALID_TIERS = [Tier.UBERS, Tier.OU, Tier.UU, Tier.RU, Tier.NU, Tier.PU]


def create_for_all_dates(args):
    for i in range(1, 10):
        valid_pm_dict = get_valid_pokemon(args.ps_path, f"gen{i}")
        valid_movesets = []
        for format in VALID_TIERS:
            valid_pm = []
            for tier in valid_pm_dict.keys():
                if tier >= format:
                    valid_pm.extend(valid_pm_dict[tier])
            print(f"Total {len(valid_pm)} valid pokemon for {format.name} in gen{i}")
            format_name = f"gen{i}{format.name.lower()}"
            stat = SmogonStat(format_name, args.smogon_stat_dir)
            if stat.movesets:
                # recursively create the path
                path = os.path.join(
                    DATA_PATH, f"movesets_data/gen{i}/{format.name.lower()}"
                )
                os.makedirs(path, exist_ok=True)
                # dump to json
                with open(f"{path}/alltime_allrank.json", "w") as f:
                    json.dump(stat.movesets, f)
                valid_movesets.append(stat.movesets)
        # merge all movesets
        inclusive_movesets = merge_movesets(valid_movesets)
        # dump to json
        path = os.path.join(DATA_PATH, f"movesets_data/gen{i}/inclusive.json")
        os.makedirs(path, exist_ok=True)
        with open(path, "w") as f:
            json.dump(inclusive_movesets, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create movesets jsons")
    parser.add_argument(
        "--smogon_stat_dir",
        type=str,
        help="Path to the scraped smogon stat directory",
    )
    parser.add_argument(
        "--ps_path",
        type=str,
        help="Path to the pokemon showdown directory",
    )
    args = parser.parse_args()
    create_for_all_dates(args)
