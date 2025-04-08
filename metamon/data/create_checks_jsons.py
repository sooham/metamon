import json
import os
import argparse

from metamon.data import DATA_PATH
from metamon.data.team_builder.stat_reader import SmogonStat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create checks jsons")
    parser.add_argument(
        "--smogon_stat_dir",
        type=str,
        help="Path to the scraped smogon stat directory",
    )
    args = parser.parse_args()
    for gen in range(1, 10):
        for format in ["ou", "nu", "uu", "ubers"]:
            stats = SmogonStat(f"gen{gen}{format}", args.smogon_stat_dir)
            check_cheatsheet = {}
            for mon in stats.movesets.keys():
                checks = stats.movesets[mon]["checks"]
                check_cheatsheet[mon] = checks
            save_dir = os.path.join(
                DATA_PATH, "checks_data"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, f"gen{gen}{format}.json"), "w") as f:
                json.dump(check_cheatsheet, f)
