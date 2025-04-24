#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import re


def get_available_formats(dataset) -> set:
    return {normalize_format(battle["gamemode"]) for battle in dataset}


def is_valid_format(format_str: str) -> bool:
    pattern = r"^gen[1-9](ou|uu|nu|ubers|lc|doubles|monotype|random)$"
    return bool(re.match(pattern, format_str.lower()))


def normalize_format(gamemode: str) -> str:
    return gamemode.lower()


def extract_game_id(battle_id: str) -> str:
    return battle_id.split("-")[0] if battle_id else ""


def convert_battle_to_replay(
    battle_text: str, battle_id: str, format_str: str, date: str
) -> dict:
    # Convert month format from 'September2024' to timestamp
    try:
        date = datetime.strptime(date, "%Y-%m-%d")
        upload_time = int(date.timestamp())
    except ValueError:
        upload_time = 0

    # Parse the battle text into lines
    lines = battle_text.split("\n")

    # Create the replay JSON
    replay = {
        "id": battle_id,
        "format": format_str,
        "uploadtime": upload_time,
        "log": battle_text,
    }

    return replay


def process_dataset(output_dir: str, split: str = "train", formats: set = None) -> None:
    # Load the dataset
    print(f"Loading {split} split from Hugging Face dataset...")
    dataset = load_dataset("milkkarten/pokechamp", split=split)

    # Get available formats if none specified
    if formats is None:
        formats = get_available_formats(dataset)
        print(f"\nFound {len(formats)} formats in dataset:")
        for fmt in sorted(formats):
            if is_valid_format(fmt):
                print(f"  {fmt}")
        return

    # Validate requested formats
    invalid_formats = [fmt for fmt in formats if not is_valid_format(fmt)]
    if invalid_formats:
        print(f"\nError: Invalid format(s): {', '.join(invalid_formats)}")
        print(
            "Format should be genXyz where X is 1-9 and yz is ou, uu, nu, ubers, etc."
        )
        return

    # Check which formats are available
    available_formats = get_available_formats(dataset)
    missing_formats = formats - available_formats
    if missing_formats:
        print(
            f"\nWarning: The following formats were not found in the dataset: {', '.join(missing_formats)}"
        )
        formats = formats - missing_formats

    if not formats:
        print("\nNo valid formats to process!")
        return

    print(f"\nProcessing {len(formats)} format(s): {', '.join(sorted(formats))}")

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    # Process each battle
    print("Processing battles...")
    format_counts = {fmt: 0 for fmt in formats}
    skipped_count = 0

    for battle in tqdm(dataset):
        gamemode = normalize_format(battle["gamemode"])
        if gamemode not in formats:
            continue

        # Extract game ID
        game_id = extract_game_id(battle["battle_id"])
        if not game_id:
            skipped_count += 1
            continue

        date = "-".join(battle["battle_id"].split("-")[1:])
        # Convert battle to replay format
        replay = convert_battle_to_replay(
            battle_text=battle["text"],
            battle_id=f"{gamemode}-{battle['battle_id'].split('-')[0]}",
            format_str=gamemode,
            date=date,
        )

        # Save replay JSON using game ID
        replay_path = os.path.join(output_dir, f"{game_id}.json")
        with open(replay_path, "w") as f:
            json.dump(replay, f)

        format_counts[gamemode] += 1

    # Print summary
    print("\nSummary:")
    print(f"Output directory: {output_dir}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} battles due to invalid battle IDs")
    print("\nReplays by format:")
    for fmt, count in sorted(format_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fmt}: {count:,} replays")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face dataset to replay JSONs"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save replay JSONs"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which dataset split to process (default: train)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=[
            f"gen{i}{f}" for i in range(1, 5) for f in ["ou", "uu", "nu", "ubers"]
        ],
        help="Specific formats to process (e.g., gen4ou gen5ou). If not provided, lists available formats.",
    )
    args = parser.parse_args()

    # Convert formats to set and normalize
    formats = {normalize_format(fmt) for fmt in args.formats} if args.formats else None
    print(formats)

    # Process the dataset
    process_dataset(args.output_dir, args.split, formats)


if __name__ == "__main__":
    main()
