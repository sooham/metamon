import json
import os

from metamon.data.team_builder.stat_reader import SmogonStat


if __name__ == "__main__":
    for gen in range(1, 10):
        for format in ["ou", "nu", "uu", "ubers"]:
            stats = SmogonStat(f"gen{gen}{format}", date=None)
            check_cheatsheet = {}
            for mon in stats.movesets.keys():
                checks = stats.movesets[mon]["checks"]
                check_cheatsheet[mon] = checks
            save_path = os.path.join(os.path.dirname(__file__), "checks_data")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, f"gen{gen}{format}.json"), "w") as f:
                json.dump(check_cheatsheet, f)
