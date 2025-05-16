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


def download_parsed_replays(
    battle_format: str,
    version: str = LATEST_PARSED_REPLAY_REVISION,
    force_download: bool = False,
) -> str:
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
    return download_parsed_replays("replay_stats", version, force_download)


def download_revealed_teams(
    version: str = LATEST_PARSED_REPLAY_REVISION, force_download: bool = False
) -> str:
    return download_parsed_replays("revealed_teams", version, force_download)


def download_raw_replays(version: str = LATEST_RAW_REPLAY_REVISION) -> str:
    from metamon.data.replay_dataset.raw_replays.download_from_hf import process_dataset

    process_dataset(
        dataset_id="jakegrigsby/metamon-raw-replays",
        output_dir=os.path.join(METAMON_CACHE_DIR, "raw-replays"),
        revision=version,
    )
    _update_version_reference("raw-replays", "raw-replays", version)
    return os.path.join(METAMON_CACHE_DIR, "raw-replays")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=[
            "raw-replays",
            "parsed-replays",
            "revealed-teams",
            "replay-stats",
            "teams",
        ],
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        type=str,
        default=[
            f"gen{i}{tier}" for i in range(1, 5) for tier in ["ou", "nu", "uu", "ubers"]
        ],
    )
    parser.add_argument("--version", type=str, default=None)
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
                download_teams(
                    battle_format=format,
                    set_name=set_name,
                    version=version,
                    force_download=True,
                )
