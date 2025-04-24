import os
import re
import json
import argparse
import random
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# creating anonymous but readable names for the players
ADJECTIVES = [
    "swift",
    "brave",
    "quiet",
    "calm",
    "bold",
    "timid",
    "jolly",
    "hasty",
    "naive",
    "lax",
    "mild",
    "rash",
    "lazy",
    "hardy",
    "docile",
]
NOUNS = [
    "trainer",
    "player",
    "battler",
    "rival",
    "ace",
    "coach",
    "master",
    "rookie",
    "expert",
    "champion",
    "student",
    "teacher",
    "leader",
]

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
    def __init__(self):
        self.player_name_map = {}
        self.player_names = set()

    def generate_random_name(self, original_name: str) -> str:
        """Generate a random but consistent name for a player."""
        if original_name in self.player_name_map:
            return self.player_name_map[original_name]

        # Generate a new random name
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        new_name = f"{adj}{noun}"

        # Ensure no duplicate names in this replay
        while new_name in self.player_name_map.values():
            adj = random.choice(ADJECTIVES)
            noun = random.choice(NOUNS)
            new_name = f"{adj}{noun}"

        self.player_name_map[original_name] = new_name
        return new_name

    def clean_line(self, line: str) -> str:
        """Clean a single line of the replay log."""
        if not line:
            return line

        parts = line.split("|")
        if len(parts) < 2:
            return line

        cmd = parts[1]

        # Remove non-battle messages
        if cmd in REMOVE_COMMANDS:
            return ""

        # Handle player-related commands
        elif cmd == "player":
            # Format: |player|PLAYER|USERNAME|AVATAR|RATING
            if len(parts) >= 4:
                player_id = parts[2]  # p1 or p2
                original_name = parts[3]
                base = f"|player|{player_id}|"
                if original_name:
                    anon_name = self.generate_random_name(original_name)
                    self.player_names.add(original_name)
                    remaining = "|".join(parts[4:])
                    remaining = remaining.replace(original_name, anon_name)
                    base += f"{anon_name}|{remaining}"
                cleaned_line = base
            else:
                cleaned_line = line
            return cleaned_line

        # Handle win/tie messages
        elif cmd == "win" and len(parts) >= 3:
            original_name = parts[2]
            anon_name = self.generate_random_name(original_name)
            cleaned_line = f"|win|{anon_name}"

        elif len(parts) >= 3:
            player_part, rest = parts[2], "|".join(parts[3:])
            if re.match(r"p[12][ab]?:", player_part):
                prefix, name = player_part.split(":", 1)
                name = name.strip()
                if name in self.player_names:
                    anon_name = self.generate_random_name(name)
                    cleaned_line = f"|{cmd}|{prefix}: {anon_name}|{rest}"
                else:
                    cleaned_line = line
            else:
                cleaned_line = line

        else:
            cleaned_line = line

        return cleaned_line

    def clean_replay(self, log_content: str) -> str:
        """Clean a replay log by removing chat and anonymizing player names."""
        self.player_name_map.clear()  # Reset for new replay

        cleaned_lines = []
        for line in log_content.split("\n"):
            cleaned = self.clean_line(line)
            if cleaned:  # Only keep non-empty lines
                cleaned_lines.append(cleaned)
        return "\n".join(cleaned_lines) + "\n"

    def clean_replay_file(self, input_path: str, output_path: str) -> None:
        """Clean a replay JSON file and save the result."""
        with open(input_path, "r") as f:
            data = json.load(f)

        # Clean the log content
        if "log" in data:
            data["log"] = self.clean_replay(data["log"])

        # Save cleaned replay
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def process_file(file_info: Tuple[Path, Path]) -> None:
    """Process a single replay file. Used for parallel processing."""
    input_path, output_path = file_info
    cleaner = ReplayCleaner()  # Create new instance for each file
    cleaner.clean_replay_file(str(input_path), str(output_path))


def get_file_pairs(input_dir: Path, output_dir: Path) -> List[Tuple[Path, Path]]:
    """Get list of (input_path, output_path) pairs for all JSON files."""
    file_pairs = []
    for input_path in input_dir.rglob("*.json"):
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        file_pairs.append((input_path, output_path))
    return file_pairs


def main():
    parser = argparse.ArgumentParser(description="Clean Pokemon Showdown replay files")
    parser.add_argument("input_dir", help="Directory containing replay JSON files")
    parser.add_argument("output_dir", help="Directory to save cleaned replay files")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers. Default: 1"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Get list of all files to process
    file_pairs = get_file_pairs(input_dir, output_dir)
    total_files = len(file_pairs)

    if args.workers > 1:
        print(f"Processing {total_files} files using {args.workers} workers...")
        with mp.Pool(args.workers) as pool:
            list(
                tqdm(
                    pool.imap(process_file, file_pairs),
                    total=total_files,
                    desc="Cleaning replays",
                )
            )
    else:
        # Process files sequentially
        print(f"Processing {total_files} files sequentially...")
        for file_pair in tqdm(file_pairs, desc="Cleaning replays"):
            process_file(file_pair)


if __name__ == "__main__":
    main()
