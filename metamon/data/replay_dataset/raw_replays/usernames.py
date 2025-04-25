import os
import json
import random
import argparse
import multiprocessing as mp
from tqdm import tqdm
import time
from typing import Optional
from functools import lru_cache

from metamon.data.tokenizer import get_tokenizer

POKEMON_WORDS = [
    word for word in get_tokenizer("allreplays-v3").all_words if "<" not in word
]

OFFICIAL_USERNAMES = os.path.join(
    os.path.dirname(__file__), "global_username_reference.json"
)


class UsernameMap:
    def __init__(self, real_to_anon: dict[str, str]):
        self.real_to_anon = real_to_anon
        self.anon_to_real = {v: k for k, v in self.real_to_anon.items()}
        self.used_anon_names = set(self.real_to_anon.keys())

    @classmethod
    @lru_cache(maxsize=4)
    def load_from_file(cls, real_to_anon_json: str):
        with open(real_to_anon_json, "r", encoding="utf-8") as f:
            real_to_anon = json.load(f)
        return cls(real_to_anon)

    def __len__(self):
        return len(self.real_to_anon)

    def real_to_anonymous(self, username: str) -> str:
        """Get anonymous name for a real username."""
        if not username:
            return None
        normalized = self.consistent_username(username)
        if normalized in self.real_to_anon:
            return self.real_to_anon[normalized]
        return None

    def anonymous_to_real(self, username: str) -> str:
        """Get real name for an anonymous username."""
        if not username:
            return username
        if username in self.anon_to_real:
            return self.anon_to_real[username]
        return username

    def add_username(self, real_username: str):
        real_username = self.consistent_username(real_username)
        if real_username in self.real_to_anon:
            return
        anon_username = self.generate_username()
        while anon_username in self.used_anon_names:
            anon_username = self.generate_username()
        self.used_anon_names.add(anon_username)
        self.real_to_anon[real_username] = anon_username
        self.anon_to_real[anon_username] = real_username

    def add_mapping(self, real_username: str, anon_username: str):
        """Add a new mapping, updating both dictionaries."""
        real_username = self.consistent_username(real_username)
        self.real_to_anon[real_username] = anon_username
        self.anon_to_real[anon_username] = real_username

    def merge_with_mapping(self, mapping):
        """Merge with another mapping."""
        assert isinstance(mapping, UsernameMap)
        self.real_to_anon.update(mapping.real_to_anon)
        self.anon_to_real.update(mapping.anon_to_real)

    def save_mapping(self, output_file: str):
        """Save the current mapping to file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.real_to_anon, f, indent=2, ensure_ascii=False)

    def consistent_username(self, username: str) -> str:
        # pokemon showdown usernames are space and case sensitive on the backend,
        # and appear inconsistently in replays.
        return username.lower().replace(" ", "")

    def generate_username(self) -> str:
        """Generate a random anonymous username using Pokemon words."""
        return f"{random.choice(POKEMON_WORDS)}_{random.randint(0, 99999)}"[-18:]


def extract_usernames_from_replay(replay_path: str) -> set[str]:
    usernames = set()
    try:
        with open(replay_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {replay_path}: {e}")
        return usernames

    if not data.get("log"):
        return usernames

    log_lines = data["log"].split("\n")
    for line in log_lines:
        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        cmd = parts[1]
        if cmd == "player" and len(parts) >= 4:
            username = parts[3].strip()
            if username:
                usernames.add(username)
        elif cmd == "win" and len(parts) >= 3:
            username = parts[2].strip()
            if username:
                usernames.add(username)
    return usernames


def process_batch(file_batch: list[str]) -> set[str]:
    all_usernames = set()
    for replay_path in file_batch:
        usernames = extract_usernames_from_replay(replay_path)
        all_usernames.update(usernames)
    return all_usernames


def find_replay_files(directory: str) -> list[str]:
    replay_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                replay_files.append(os.path.join(root, file))
    return replay_files


def main():
    parser = argparse.ArgumentParser(
        description="Build username mapping from replay files"
    )
    parser.add_argument("input_dir", help="Directory containing replay JSON files")
    parser.add_argument("output_file", help="Path to save username mapping JSON")
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count() - 1,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process per batch (default: 100)",
    )
    args = parser.parse_args()

    # Find all replay files
    print("Finding replay files...")
    replay_files = find_replay_files(args.input_dir)
    total_files = len(replay_files)
    print(f"Found {total_files} replay files")

    if total_files == 0:
        print("No replay files found!")
        return

    batch_size = min(args.batch_size, max(1, total_files // (args.workers * 10)))
    batches = [
        replay_files[i : i + batch_size]
        for i in range(0, len(replay_files), batch_size)
    ]

    print(f"Processing files using {args.workers} workers...")
    all_usernames = set()
    with mp.Pool(args.workers) as pool:
        for batch_usernames in tqdm(
            pool.imap_unordered(process_batch, batches),
            total=len(batches),
            desc="Processing batches",
        ):
            all_usernames.update(batch_usernames)

    print(f"\nFound {len(all_usernames)} unique usernames")
    print("Building username mapping...")
    mapping = UsernameMap()
    for username in sorted(all_usernames):
        mapping.add_username(username)

    if os.path.exists(args.output_file):
        existing_mapping = UsernameMap.load_from_file(args.output_file)
        mapping.merge_with_mapping(existing_mapping)
        print(
            f"Merging {len(existing_mapping)} existing usernames with {len(mapping)} discovered usernames for a total of {len(mapping)} usernames"
        )

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    print(f"Saving mapping to {args.output_file}")
    mapping.save_mapping(args.output_file)
    print("Done! ✅")


if __name__ == "__main__":
    main()
