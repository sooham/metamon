import os
import json
import datetime
import tarfile
from collections import defaultdict

from huggingface_hub import hf_hub_download

METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", None)
VERSION_REFERENCE_PATH = os.path.join(METAMON_CACHE_DIR, "version_reference.json")


LATEST_RAW_REPLAY_REVISION = "v1"
LATEST_PARSED_REPLAY_REVISION = "v1"
LATEST_TEAMS_REVISION = "v0"


def _update_version_reference(key: str, name: str, version: str):
    """
    Maintains a version_reference.json file in the METAMON_CACHE_DIR
    that records the version of each dataset that is currently active.
    """
    version_reference = defaultdict(dict)
    if os.path.exists(VERSION_REFERENCE_PATH):
        with open(VERSION_REFERENCE_PATH, "r") as f:
            existing_version_reference = json.load(f)
        version_reference.update(existing_version_reference)

    version_reference[key][
        name
    ] = f"version {version}, downloaded {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    with open(VERSION_REFERENCE_PATH, "w") as f:
        json.dump(dict(version_reference), f)


def get_active_dataset_versions() -> dict:
    """
    Get the current version of a dataset
    """
    with open(VERSION_REFERENCE_PATH, "r") as f:
        version_reference = json.load(f)
    return version_reference


def download_parsed_replays(
    battle_format: str,
    version: str = LATEST_PARSED_REPLAY_REVISION,
    force_download: bool = False,
) -> str:
    """
    Download the parsed replays for a given battle format.

    Args:
        battle_format: Showdown battle format (e.g. "gen1ou")
        version: Version of the dataset to download. Corresponds to revisions on the Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version already exists in the cache.

    Returns:
        The path to the dataset on disk.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")

    parsed_replay_dir = os.path.join(METAMON_CACHE_DIR, "parsed-replays")
    tar_path = os.path.join(parsed_replay_dir, f"{battle_format}.tar.gz")
    extract_path = os.path.join(parsed_replay_dir, battle_format)
    if os.path.exists(extract_path) and not force_download:
        return extract_path

    hf_hub_download(
        cache_dir=os.path.join(METAMON_CACHE_DIR, "parsed-replays"),
        repo_id="jakegrigsby/metamon-parsed-replays",
        filename=f"{battle_format}.tar.gz",
        local_dir=os.path.join(METAMON_CACHE_DIR, "parsed-replays"),
        revision=version,
        repo_type="dataset",
    )
    with tarfile.open(tar_path) as tar:
        print(f"Extracting {tar_path}...")
        tar.extractall(path=extract_path)
    os.remove(tar_path)
    _update_version_reference("parsed-replays", battle_format, version)
    return extract_path


def download_teams(
    battle_format: str,
    set_name: str,
    version: str = LATEST_TEAMS_REVISION,
    force_download: bool = False,
) -> str:
    """
    Download the teams for a given battle format and set name.

    Args:
        battle_format: Showdown battle format (e.g. "gen1ou")
        set_name: Name of the team set to download.
        version: Version of the dataset to download. Corresponds to revisions on the Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version already exists in the cache.
    """
    if METAMON_CACHE_DIR is None:
        raise ValueError("METAMON_CACHE_DIR environment variable is not set")

    teams_dir = os.path.join(METAMON_CACHE_DIR, "teams", set_name)
    tar_path = os.path.join(teams_dir, f"{battle_format}.tar.gz")
    extract_path = os.path.join(teams_dir, battle_format)
    if os.path.exists(extract_path) and not force_download:
        return extract_path

    hf_hub_download(
        cache_dir=os.path.join(METAMON_CACHE_DIR, "teams", set_name),
        repo_id="jakegrigsby/metamon-teams",
        filename=f"{set_name}/{battle_format}.tar.gz",
        local_dir=os.path.join(METAMON_CACHE_DIR, "teams"),
        revision=version,
        repo_type="dataset",
    )
    with tarfile.open(tar_path) as tar:
        print(f"Extracting {tar_path}...")
        tar.extractall(path=os.path.dirname(extract_path))
    os.remove(tar_path)
    _update_version_reference("teams", f"{set_name}/{battle_format}", version)
    return extract_path


def download_replay_stats(
    version: str = LATEST_PARSED_REPLAY_REVISION, force_download: bool = False
) -> str:
    """
    Download the "replay stats" for a given version. Replay stats are json statistics generated from the revealed teams of the current replay dataset. They are used to predict team sets.

    Args:
        version: Version of the dataset to download. Corresponds to revisions on the Hugging Face Hub. Defaults to the latest version.
        force_download: If True, download the dataset even if a previous version already exists in the cache.
    """
    return download_parsed_replays("replay_stats", version, force_download)


def download_revealed_teams(
    version: str = LATEST_PARSED_REPLAY_REVISION, force_download: bool = False
) -> str:
    return download_parsed_replays("revealed_teams", version, force_download)


def download_raw_replays(version: str = LATEST_RAW_REPLAY_REVISION) -> str:
    """
    Download the "raw" (unpprocessed) replays. We maintain a dataset of replays downloaded from Pokémon Showdown for convenience. Our versions are also stripped of player usernames and in-game chat logs.

    Args:
        version: Version of the dataset to download. Corresponds to revisions on the Hugging Face Hub. Defaults to the latest version.
    """
    from metamon.data.replay_dataset.raw_replays.download_from_hf import process_dataset

    process_dataset(
        dataset_id="jakegrigsby/metamon-raw-replays",
        output_dir=os.path.join(METAMON_CACHE_DIR, "raw-replays"),
        revision=version,
    )
    _update_version_reference("raw-replays", "raw-replays", version)
    return os.path.join(METAMON_CACHE_DIR, "raw-replays")


def print_version_tree(version_dict: dict, indent: int = 0):
    for key, value in sorted(version_dict.items()):
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_version_tree(value, indent + 4)
        else:
            print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    import argparse
    from termcolor import colored

    parser = argparse.ArgumentParser(
        description=f"""
Metamon Dataset Downloader

This tool downloads and manages Metamon datasets from Hugging Face Hub.
Available datasets include raw-replays, parsed-replays, revealed-teams, replay-stats, and teams.

Examples:
    # Download all team files for Gen 1-4 OU
    python -m metamon.download teams --formats gen1ou gen2ou gen3ou gen4ou

    # Download parsed replays for Gen 1 UU
    python -m metamon.download parsed-replays --formats gen1uu

    # Download (anonymized) Showdown replay logs (all formats)
    python -m metamon.download raw-replays

Note: Requires METAMON_CACHE_DIR environment variable to be set.

The cache directory is currently: {METAMON_CACHE_DIR}. See `get_active_dataset_versions()` for the current versions of each dataset. Or run `python -m metamon.download --tree` to print this information and see if updates are needed.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        default=None,
        choices=[
            "raw-replays",
            "parsed-replays",
            "revealed-teams",
            "replay-stats",
            "teams",
        ],
        help="""
Dataset to download:
    raw-replays: Unprocessed Showdown replays (stripped of usernames/chat)
    parsed-replays: RL-compatible version of replays with reconstructed player actions
    revealed-teams: Teams that were revealed during battles
    replay-stats: Statistics generated from revealed teams. Used to predict team sets.
    teams: Various team sets (competitive, paper_variety, paper_replays)
""",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        type=str,
        default=[
            f"gen{i}{tier}" for i in range(1, 5) for tier in ["ou", "nu", "uu", "ubers"]
        ],
        help="""
Battle formats to download. Defaults to all Gen 1-4 formats (OU, UU, NU, Ubers).
Examples:
    --formats gen1ou gen2ou    # Only Gen 1-2 OU
    --formats gen3uu gen4uu    # Only Gen 3-4 UU
""",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="""
Specific version to download. Defaults to latest version.
Available versions:
    raw-replays: v1
    parsed-replays: v0 (deprecated), v1
    teams: v0
""",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print version information as a tree structure instead of downloading",
    )
    args = parser.parse_args()

    if args.dataset == "raw-replays":
        version = args.version or LATEST_RAW_REPLAY_REVISION
        download_raw_replays(version=version)
    elif args.dataset == "parsed-replays":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        if args.formats is None:
            raise ValueError("Must specify at least one battle format (e.g., gen1ou)")
        for format in args.formats:
            download_parsed_replays(format, version=version, force_download=True)
    elif args.dataset == "revealed-teams":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        download_revealed_teams(version=version, force_download=True)
    elif args.dataset == "replay-stats":
        version = args.version or LATEST_PARSED_REPLAY_REVISION
        download_replay_stats(version=version, force_download=True)
    elif args.dataset == "teams":
        version = args.version or LATEST_TEAMS_REVISION
        if args.formats is None:
            raise ValueError("Must specify at least one set name (e.g., gen1ou)")
        set_names = ["competitive", "paper_variety", "paper_replays"]
        for set_name in set_names:
            for format in args.formats:
                if "ou" not in format and set_name == "paper_replays":
                    # only the OU tiers have paper_replay sets
                    continue
                download_teams(
                    battle_format=format,
                    set_name=set_name,
                    version=version,
                    force_download=True,
                )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(colored("\nActive dataset versions:", "red"))
    print(f"Cache Dir: {METAMON_CACHE_DIR}\n")
    print_version_tree(get_active_dataset_versions())
    exit(0)
