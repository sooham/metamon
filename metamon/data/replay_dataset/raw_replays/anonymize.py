import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Set
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from metamon.data.replay_dataset.raw_replays.usernames import (
    UsernameMap,
    OFFICIAL_USERNAMES,
)


# Commands that should be removed (chat and non-battle messages)
REMOVE_COMMANDS = {
    "c",
    "c:",
    "chat",
    "chatmsg",
    "chatmsg-raw",
    "j",
    "J",
    "join",
    "l",
    "L",
    "leave",
    "n",
    "name",
    "inactive",
    "inactiveoff",
    "raw",
    "unlink",
    "html",
    "uhtml",
    "bigerror",
    "debug",
    "seed",
    "error",
    "message",
    "-message",
    "t:",
    "callback",
    "expire",
}


class ReplayCleaner:
    PLAYER_PATTERN = re.compile(r"p[12][ab]?:")

    def __init__(self, username_map: UsernameMap):
        self.username_map = username_map
        self.player_names = [None, None]

    def clean_line(self, line: str) -> str:
        if not line:
            return line

        parts = line.split("|")
        if len(parts) < 2:
            return line

        cmd = parts[1]
        if cmd in REMOVE_COMMANDS:
            return ""

        cleaned_line = line
        if cmd == "player":
            if len(parts) >= 4:
                player_id = parts[2]
                original_name = parts[3]
                base = f"|player|{player_id}|"
                if original_name:
                    anon_name = self.username_map.real_to_anonymous(original_name)
                    if anon_name is None:
                        raise ValueError(
                            f"Username {original_name} has not been identified; skipping"
                        )
                    else:
                        # save player names to help sort through battle messages
                        if "1" in player_id:
                            self.player_names[0] = original_name
                        elif "2" in player_id:
                            self.player_names[1] = original_name
                        else:
                            raise ValueError(
                                f"Player ID {player_id} not found in ReplayCleaner"
                            )
                    remaining = "|".join(parts[4:])
                    remaining = remaining.replace(original_name, anon_name)
                    base += f"{anon_name}|{remaining}"
                cleaned_line = base

        elif cmd == "win" and len(parts) >= 3:
            original_name = parts[2]
            anon_name = self.username_map.real_to_anonymous(original_name)
            if anon_name is None:
                raise ValueError(
                    f"Username {original_name} has not been identified; skipping"
                )
            cleaned_line = f"|win|{anon_name}"

        elif len(parts) >= 3:
            player_part, rest = parts[2], "|".join(parts[3:])
            if self.PLAYER_PATTERN.match(player_part):
                prefix, name = player_part.split(":", 1)
                name = name.strip()
                if name in self.player_names:
                    anon_name = self.username_map.real_to_anonymous(name)
                    cleaned_line = f"|{cmd}|{prefix}: {anon_name}|{rest}"

        return cleaned_line

    def clean_replay(self, log_content: str) -> str:
        cleaned_lines = []
        for line in log_content.split("\n"):
            cleaned = self.clean_line(line)
            if cleaned:
                cleaned_lines.append(cleaned)
        return "\n".join(cleaned_lines) + "\n"

    def clean_replay_file(self, input_path: str, output_path: str) -> None:
        with open(input_path, "r") as f:
            data = json.load(f)

        if "log" in data:
            data["log"] = self.clean_replay(data["log"])

        # while there is a data["players"] entry in the metadata, we go with the spelling that appears in the replay
        # log, which has been saved during cleaning to `self.player_names`
        clean_players = [
            self.username_map.real_to_anonymous(player) for player in self.player_names
        ]
        if None in clean_players:
            breakpoint()
        data["players"] = clean_players

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def process_file(file_info: Tuple[Path, Path], username_map_file: str) -> None:
    input_path, output_path = file_info
    username_map = UsernameMap.load_from_file(username_map_file)
    cleaner = ReplayCleaner(username_map)
    try:
        cleaner.clean_replay_file(str(input_path), str(output_path))
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def get_file_pairs(input_dir: Path, output_dir: Path) -> List[Tuple[Path, Path]]:
    file_pairs = []
    for input_path in input_dir.rglob("*.json"):
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        file_pairs.append((input_path, output_path))
    return file_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize Pokemon Showdown replay files"
    )
    parser.add_argument("input_dir", help="Directory containing replay JSON files")
    parser.add_argument("output_dir", help="Directory to save cleaned replay files")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers. Default: 1"
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=OFFICIAL_USERNAMES,
        help="Path to username mapping file. Defaults to pre-made list based on metamon official replay dataset.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_pairs = get_file_pairs(input_dir, output_dir)
    total_files = len(file_pairs)

    if args.workers > 1:
        print(f"Processing {total_files} files using {args.workers} workers...")
        with mp.Pool(args.workers) as pool:
            list(
                tqdm(
                    pool.imap(
                        partial(process_file, username_map_file=args.mapping_file),
                        file_pairs,
                    ),
                    total=total_files,
                    desc="Anonymizing replays",
                )
            )
    else:
        print(f"Processing {total_files} files sequentially...")
        for file_pair in tqdm(file_pairs, desc="Cleaning replays"):
            process_file(file_pair, args.mapping_file)


if __name__ == "__main__":
    main()
