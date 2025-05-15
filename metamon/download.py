import os
import tarfile

from huggingface_hub import hf_hub_download

METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", None)


LATEST_RAW_REPLAY_REVISION = "v1"
LATEST_PARSED_REPLAY_REVISION = "v1"
LATEST_TEAMS_REVISION = "v0"


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
    extract_path = teams_dir
    if os.path.exists(extract_path) and not force_download:
        return os.path.join(teams_dir, battle_format)

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
        tar.extractall(path=extract_path)
    os.remove(tar_path)
    return os.path.join(teams_dir, battle_format)


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
    return os.path.join(METAMON_CACHE_DIR, "raw-replays")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=["raw-replays", "parsed-replays", "revealed-teams", "replay-stats"],
    )
    parser.add_argument("--formats", type=str, default=None)
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
