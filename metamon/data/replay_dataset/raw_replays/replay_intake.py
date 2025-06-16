import os
import json
import argparse
import shutil
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp
from functools import partial
import re
import tqdm
from tqdm.contrib.concurrent import process_map
import orjson


def load_json(f):
    return orjson.loads(f.read())


def find_json_files(directory: str) -> List[str]:
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def extract_game_id(json_data: dict) -> str:
    return str(json_data.get("id", ""))


def normalize_format(format_str: str) -> str:
    format_str = format_str.lower()
    gen_num = None
    gen_bracket_match = re.search(r"\[gen\s*(\d+)\]", format_str)
    if gen_bracket_match:
        gen_num = gen_bracket_match.group(1)
    if not gen_num:
        gen_direct_match = re.search(r"gen(\d+)", format_str)
        if gen_direct_match:
            gen_num = gen_direct_match.group(1)
    if not gen_num:
        return format_str
    tier_match = re.search(r"(ou|uu|nu|ubers)", format_str)
    tier = tier_match.group(1) if tier_match else None
    if not tier:
        return format_str
    return f"gen{gen_num}{tier}"


def get_existing_game_ids(target_dir: str) -> Dict[str, int]:
    """
    Get dictionary mapping game IDs to upload times for replays that already exist in target directory.
    Returns Dict[game_id, upload_time].
    """
    existing_games = {}
    if not os.path.exists(target_dir):
        return existing_games

    for gen_dir in os.listdir(target_dir):
        gen_path = os.path.join(target_dir, gen_dir)
        if not os.path.isdir(gen_path):
            continue
        for tier_dir in os.listdir(gen_path):
            tier_path = os.path.join(gen_path, tier_dir)
            if not os.path.isdir(tier_path):
                continue
            for json_file in os.listdir(tier_path):
                if json_file.endswith(".json"):
                    json_path = os.path.join(tier_path, json_file)
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            game_id = extract_game_id(data)
                            upload_time = int(data.get("uploadtime", 0))
                            if game_id:  # Only store if we have a valid game ID
                                existing_games[game_id] = upload_time
                    except (json.JSONDecodeError, IOError):
                        continue
    return existing_games


def get_format_dirs(format_str: str) -> Tuple[str, str]:
    """Extract generation and tier directories from format string."""
    gen_match = re.search(r"gen(\d+)", format_str)
    if not gen_match:
        return None, None

    gen_dir = f"gen{gen_match.group(1)}"
    tier = format_str.replace(gen_dir, "").strip()
    return gen_dir, tier


def copy_to_target(
    source_path: str,
    game_id: str,
    format_str: str,
    upload_time: int,
    target_base: str,
    existing_games: Dict[str, int],
) -> bool:
    """
    Copy a JSON file to the appropriate target directory.
    Returns True if file was copied, False if skipped (already exists).
    """
    if game_id in existing_games:
        return False

    gen_dir, tier = get_format_dirs(format_str)
    if not gen_dir or not tier:
        return False

    # Create directory structure
    tier_path = os.path.join(target_base, gen_dir, tier)
    os.makedirs(tier_path, exist_ok=True)

    # Define target file path using game ID
    target_path = os.path.join(tier_path, f"{game_id}.json")

    # Copy the file
    shutil.copy2(source_path, target_path)
    return True


def copy_file_task(task_data: Dict) -> bool:
    """
    Wrapper function for parallel file copying.
    Returns True if file was copied successfully.
    """
    try:
        gen_dir, tier = get_format_dirs(task_data["normalized_format"])
        if not gen_dir or not tier:
            return False

        # Create directory structure
        tier_path = os.path.join(task_data["target_dir"], gen_dir, tier)
        os.makedirs(tier_path, exist_ok=True)

        # Define target file path using game ID
        target_path = os.path.join(tier_path, f"{task_data['game_id']}.json")

        # Copy the file
        shutil.copy2(task_data["source_path"], target_path)
        return True
    except Exception as e:
        # In case of any error, return False
        return False


def process_single_file(json_path: str, allowed_formats: set) -> Dict:
    """Process a single JSON file and return its statistics if valid."""
    try:
        with open(json_path, "r") as f:
            json_data = load_json(f)

        # Basic validation checks
        log = json_data.get("log")
        if not log:
            return {"valid": False}

        # Check for game ID
        game_id = str(json_data.get("id", ""))
        if not game_id:
            return {"valid": False}

        # Check for required metadata first (fastest checks)
        has_format = "format" in json_data
        uploadtime = json_data.get("uploadtime", 0)
        has_upload_time = uploadtime and int(uploadtime) > 0

        if not (has_format and has_upload_time):
            return {"valid": False}

        # Validate format early to skip processing if not needed
        raw_format = json_data["format"]
        normalized_format = normalize_format(raw_format)
        if allowed_formats and normalized_format not in allowed_formats:
            return {"valid": False}

        # Single pass through log lines for all checks
        has_win = False
        has_gen = False
        has_tier = False
        turn_count = 0

        # Use string operations instead of splitting into lines when possible
        if "|win|" in log:
            has_win = True
        if "|gen|" in log:
            has_gen = True
        if "|tier|" in log:
            has_tier = True

        if has_win and has_gen and has_tier:
            turn_count = log.count("|turn|")
        has_min_turns = turn_count >= 5

        # Combine all validation checks
        is_valid = has_win and has_gen and has_tier and has_min_turns

        if not is_valid:
            return {"valid": False}

        # Only collect stats for valid replays
        return {
            "valid": True,
            "normalized_format": normalized_format,
            "upload_time": int(uploadtime),
            "game_id": game_id,
            "source_path": json_path,
        }

    except Exception as e:
        return {"valid": False}


def combine_stats(results: List[Dict]) -> Dict:
    """Combine statistics from valid replays only."""
    combined = {
        "total_processed": len(results),
        "valid_replays": 0,
        "invalid_replays": 0,
        "normalized_formats": defaultdict(int),
        "unique_games": set(),  # Set of unique game IDs
        "earliest_replay": float("inf"),
        "latest_replay": 0,
        "replays_by_year": defaultdict(int),
    }

    for stats in results:
        if not stats["valid"]:
            combined["invalid_replays"] += 1
            continue

        combined["valid_replays"] += 1
        combined["normalized_formats"][stats["normalized_format"]] += 1

        game_id = stats["game_id"]
        upload_time = stats["upload_time"]

        # Track unique games by ID only
        combined["unique_games"].add(game_id)

        if upload_time > 0:
            combined["earliest_replay"] = min(combined["earliest_replay"], upload_time)
            combined["latest_replay"] = max(combined["latest_replay"], upload_time)
            year = datetime.fromtimestamp(upload_time).year
            combined["replays_by_year"][year] += 1

    return combined


def generate_time_series(results: List[Dict]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Generate monthly time series data for each format.
    Returns Dict[format, List[Tuple[month_str, count]]] where month_str is YYYY-MM.
    """
    # Track monthly counts for each format
    format_monthly_counts = defaultdict(lambda: defaultdict(int))

    # Process each valid replay
    for result in results:
        if not result["valid"]:
            continue

        format_str = result["normalized_format"]
        timestamp = result["upload_time"]
        if timestamp > 0:
            # Convert timestamp to YYYY-MM format
            month_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m")
            format_monthly_counts[format_str][month_str] += 1

    # Convert to sorted lists of (month, count) tuples
    time_series = {}
    for format_str, monthly_counts in format_monthly_counts.items():
        # Sort by month string (YYYY-MM format ensures chronological order)
        sorted_months = sorted(monthly_counts.items())
        time_series[format_str] = sorted_months

    return time_series


def save_time_series(
    time_series: Dict[str, List[Tuple[str, int]]], output_path: str
) -> None:
    """Save time series data to a JSON file."""
    # Convert tuples to lists for JSON serialization
    json_data = {
        format_str: [{"month": month, "count": count} for month, count in monthly_data]
        for format_str, monthly_data in time_series.items()
    }

    # Add metadata
    json_data["_metadata"] = {
        "created_at": datetime.now().isoformat(),
        "total_formats": len(json_data) - 1,  # Subtract 1 for metadata
        "date_range": {
            "start": min(
                data[0]["month"]
                for data in json_data.values()
                if isinstance(data, list) and data
            ),
            "end": max(
                data[-1]["month"]
                for data in json_data.values()
                if isinstance(data, list) and data
            ),
        },
    }
    # Save to file
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)


def analyze_replays(
    directory: str,
    allowed_formats: set,
    target_dir: str = None,
    num_processes: int = None,
    time_series_path: str = None,
) -> None:
    """Analyze all replay JSONs in a directory using parallel processing."""
    json_files = find_json_files(directory)
    total_files = len(json_files)
    print(f"\nFound {total_files} JSON files")

    if total_files == 0:
        print("No JSON files found in the specified directory")
        return

    # get existing game IDs and upload times if target directory is specified
    existing_games = get_existing_game_ids(target_dir) if target_dir else {}
    if target_dir:
        print(f"Found {len(existing_games)} existing replays in target directory")

    if num_processes is None:
        num_processes = mp.cpu_count()
    else:
        num_processes = max(1, min(num_processes, mp.cpu_count()))
    print(f"Processing files using {num_processes} processes...")

    # Process files in parallel
    # Create a partial function with the allowed_formats
    process_func = partial(process_single_file, allowed_formats=allowed_formats)
    # Use process_map for better progress tracking with multiprocessing
    # Optimize chunksize based on total files and workers
    results = process_map(
        process_func,
        json_files,
        max_workers=num_processes,
        desc="Processing replays",
        unit="files",
        chunksize=max(1, len(json_files) // num_processes),
    )

    # Generate and save time series if requested
    if time_series_path:
        time_series = generate_time_series(results)
        save_time_series(time_series, time_series_path)
        print(f"\nSaved time series data to: {time_series_path}")

    # Combine results from all processes
    stats = combine_stats(results)

    # Copy valid files if target directory specified
    if target_dir:
        # Deduplicate results by game ID, keeping most recent version
        deduplicated_results = {}
        for result in results:
            if not result["valid"]:
                continue
            game_id = result["game_id"]
            upload_time = result["upload_time"]

            # Keep this result if it's the first time we've seen this game_id
            # or if it's more recent than the previous one we've seen
            if (
                game_id not in deduplicated_results
                or upload_time > deduplicated_results[game_id]["upload_time"]
            ):
                deduplicated_results[game_id] = result

        print(
            f"\nDeduplicated {len(results)} replays to {len(deduplicated_results)} unique games"
        )

        # Prepare data for parallel file copying
        copy_tasks = []
        skipped_count = 0

        for result in deduplicated_results.values():
            game_id = result["game_id"]
            upload_time = result["upload_time"]

            # Skip if game already exists in target
            if game_id in existing_games:
                skipped_count += 1
                continue

            copy_tasks.append(
                {
                    "source_path": result["source_path"],
                    "game_id": game_id,
                    "normalized_format": result["normalized_format"],
                    "upload_time": upload_time,
                    "target_dir": target_dir,
                }
            )

        if copy_tasks:
            print(f"Copying {len(copy_tasks)} files in parallel...")
            # Parallel file copying
            copy_results = process_map(
                copy_file_task,
                copy_tasks,
                max_workers=min(num_processes, len(copy_tasks)),
                desc="Copying files",
                unit="files",
                chunksize=max(1, len(copy_tasks) // num_processes),
            )
            copied_count = sum(copy_results)
        else:
            copied_count = 0

    # Print summary
    print("\nAnalysis Summary:")
    if allowed_formats:
        print(f"Filtering for formats: {', '.join(sorted(allowed_formats))}")
    print(f"Total files processed: {stats['total_processed']}")
    print(f"Valid replays: {stats['valid_replays']}")
    print(f"Invalid replays: {stats['invalid_replays']}")
    print(f"Unique games (by ID only): {len(stats['unique_games'])}")

    if target_dir:
        print(f"\nFile Operations:")
        print(f"  Copied to target: {copied_count}")
        print(f"  Skipped (already exists): {skipped_count}")

    if stats["earliest_replay"] != float("inf"):
        print("\nTimestamp Range (valid replays only):")
        earliest = datetime.fromtimestamp(stats["earliest_replay"])
        latest = datetime.fromtimestamp(stats["latest_replay"])
        print(f"  Earliest: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Latest: {latest.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nReplays by Year:")
        for year in sorted(stats["replays_by_year"].keys()):
            print(f"  {year}: {stats['replays_by_year'][year]}")

    print("\nFormat distribution:")
    sorted_formats = sorted(
        stats["normalized_formats"].items(), key=lambda x: x[1], reverse=True
    )
    for format_id, count in sorted_formats:
        print(f"  {format_id}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Pokemon Showdown replay JSON files"
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing replay JSON files"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="*",
        default=[
            f"gen{i}{f}" for i in range(1, 5) for f in ["ou", "uu", "nu", "ubers"]
        ],
        help="Only process replays matching these formats (e.g., gen1ou gen1ubers). Default: process all formats",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory to copy valid replays into an organized structure",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes to use for parallel processing. Default: number of CPU cores minus 1",
    )
    parser.add_argument(
        "--save_time_series",
        type=str,
        required=True,
        help="Path to save monthly time series data for each format (as JSON)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    # Convert formats list to set for faster lookup
    allowed_formats = set(args.formats) if args.formats else set()
    analyze_replays(
        args.directory,
        allowed_formats,
        args.target_dir,
        args.processes,
        args.save_time_series,
    )


if __name__ == "__main__":
    main()
