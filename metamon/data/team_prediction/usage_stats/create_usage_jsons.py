import os
import json
import argparse
from tqdm import tqdm

from metamon.data.team_prediction.usage_stats.format_rules import (
    Tier,
)
from metamon.data.team_prediction.usage_stats.stat_reader import (
    SmogonStat,
    merge_movesets,
)

VALID_TIERS = [Tier.UBERS, Tier.OU, Tier.UU, Tier.RU, Tier.NU, Tier.PU]


def main(args):
    total_iterations = 9 * 12 * 12 * len(VALID_TIERS)
    with tqdm(total=total_iterations, desc="Processing movesets") as pbar:
        for gen in range(1, 10):
            for year in range(2014, 2026):
                for month in range(1, 13):
                    stat_dir = os.path.join(args.smogon_stat_dir)
                    valid_movesets = []
                    for format in VALID_TIERS:
                        format_name = f"gen{gen}{format.name.lower()}"
                        stat = SmogonStat(
                            format_name,
                            raw_stats_dir=stat_dir,
                            date=f"{year}-{month:02d}",
                        )
                        if stat.movesets:
                            # if we find data for this, save it
                            path = os.path.join(
                                args.save_dir,
                                "movesets_data",
                                f"gen{gen}",
                                f"{format.name.lower()}",
                                f"{year}-{month:02d}.json",
                            )
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, "w") as f:
                                json.dump(stat.movesets, f)
                            valid_movesets.append(stat.movesets)

                            check_cheatsheet = {}
                            for mon in stat.movesets.keys():
                                checks = stat.movesets[mon]["checks"]
                                check_cheatsheet[mon] = checks
                            path = os.path.join(
                                args.save_dir,
                                "checks_data",
                                f"gen{gen}",
                                f"{format.name.lower()}",
                                f"{year}-{month:02d}.json",
                            )
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path, "w") as f:
                                json.dump(check_cheatsheet, f)
                        pbar.update(1)
                if valid_movesets:
                    # merge all the tiers. used to lookup rare Pokémon choices, i.e. fooling around
                    # with low-tier Pokémon in OverUsed
                    inclusive_movesets = merge_movesets(valid_movesets)
                    path = os.path.join(
                        args.save_dir,
                        "movesets_data",
                        f"gen{gen}",
                        "all_tiers",
                        f"{year}-{month:02d}.json",
                    )
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(inclusive_movesets, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create usage jsons")
    parser.add_argument(
        "--smogon_stat_dir",
        type=str,
        help="Path to the scraped smogon stat directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to the save directory",
    )
    args = parser.parse_args()
    main(args)
