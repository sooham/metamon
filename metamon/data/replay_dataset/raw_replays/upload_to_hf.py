"""
Upload Pokémon Showdown replays to HuggingFace datasets.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


def extract_replay_info(replay_data: dict) -> dict:
    approved_entries = {
        "id",
        "format",
        "players",
        "log",
        "uploadtime",
        "formatid",
        "rating",
    }
    for key in list(replay_data.keys()):
        if key not in approved_entries:
            del replay_data[key]
    for key in approved_entries:
        if key not in replay_data:
            replay_data[key] = "MISSING"
        else:
            replay_data[key] = str(replay_data[key])
    return replay_data


def process_replay_files(input_dir: str) -> Dataset:
    """Process all replay files in the input directory into a HuggingFace dataset."""
    input_path = Path(input_dir)
    replay_files = list(input_path.rglob("*.json"))
    all_data = defaultdict(list)
    print(f"Processing {len(replay_files)} replay files...")
    for replay_file in tqdm(replay_files):
        try:
            with open(replay_file, "r") as f:
                replay_data = json.load(f)
            replay_data = extract_replay_info(replay_data)
            for key, val in replay_data.items():
                all_data[key].append(val)
        except Exception as e:
            print(f"Error processing {replay_file}: {e}")
            continue
    l = len(all_data["log"])
    for k, v in all_data.items():
        assert len(v) == l
    return Dataset.from_dict(all_data)


def main():
    parser = argparse.ArgumentParser(
        description="Upload anonymized replays to HuggingFace"
    )
    parser.add_argument(
        "input_dir", help="Directory containing anonymized replay JSONs"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument("--commit-message", required=True, help="Commit message")
    args = parser.parse_args()

    dataset = process_replay_files(args.input_dir)

    print("\nDataset statistics:")
    print(f"Number of replays: {len(dataset)}")
    print("\nColumn statistics:")
    for col in dataset.column_names:
        n_null = sum(1 for x in dataset[col] if x is None)
        print(f"{col}: {len(dataset) - n_null} non-null values")

    print(f"\nUploading dataset to HuggingFace ({args.repo_id})")

    # Push to HuggingFace
    dataset.push_to_hub(
        args.repo_id,
        commit_message=args.commit_message,
    )
    print("Done! ✅")


if __name__ == "__main__":
    main()
