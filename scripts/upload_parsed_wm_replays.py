import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import orjson
from huggingface_hub import CommitOperationAdd, HfApi
from tqdm import tqdm

from metamon.config import METAMON_CACHE_DIR, SUPPORTED_BATTLE_FORMATS


DEFAULT_REPO_ID = "sooham34/metamon-parsed-wm-replays"
MIN_FREE_BYTES_AFTER_STAGING = 512 * 1024 * 1024


def _run_git(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_commit(repo_root: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], repo_root)


def _ensure_clean_git(repo_root: Path) -> None:
    status = _run_git(["status", "--porcelain", "--untracked-files=all"], repo_root)
    if not status:
        return
    raise RuntimeError(
        "Refusing to upload parsed-wm-replays because the git repository is dirty. "
        "Commit, stash, or remove local changes first.\n"
        f"\nDirty paths:\n{status}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parsed_files(parsed_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in parsed_dir.iterdir()
        if path.is_file() and path.suffix == ".txt"
    )


def _check_staging_space(staging_parent: Path, total_uncompressed_bytes: int) -> None:
    free_bytes = shutil.disk_usage(staging_parent).free
    required_bytes = total_uncompressed_bytes + MIN_FREE_BYTES_AFTER_STAGING
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough free disk space to package parsed-wm-replays. "
            f"Need at least {required_bytes / 1e9:.2f} GB free, "
            f"found {free_bytes / 1e9:.2f} GB."
        )


def _write_index(
    index_path: Path,
    parsed_files: list[Path],
    format_name: str,
) -> tuple[int, int]:
    file_count = 0
    total_uncompressed_bytes = 0
    with gzip.open(index_path, "wt", encoding="utf-8") as f:
        for path in tqdm(parsed_files, desc="Indexing parsed replays", unit="file"):
            size_bytes = path.stat().st_size
            total_uncompressed_bytes += size_bytes
            file_count += 1
            row = {
                "path": f"{format_name}/{path.name}",
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
            }
            f.write(orjson.dumps(row).decode("utf-8"))
            f.write("\n")
    return file_count, total_uncompressed_bytes


def _write_archive(archive_path: Path, parsed_files: list[Path], format_name: str) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in tqdm(parsed_files, desc="Writing tar archive", unit="file"):
            tar.add(path, arcname=f"{format_name}/{path.name}", recursive=False)


def package_format(
    format_name: str,
    parsed_replay_root: Path,
    staging_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path, Path, dict]:
    parsed_dir = parsed_replay_root / format_name
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"No parsed replay directory found: {parsed_dir}")

    parsed_files = _parsed_files(parsed_dir)
    if not parsed_files:
        raise FileNotFoundError(f"No parsed replay .txt files found in: {parsed_dir}")

    total_bytes = sum(path.stat().st_size for path in parsed_files)
    _check_staging_space(staging_dir, total_bytes)
    print(
        f"Packaging {len(parsed_files)} parsed replay files "
        f"({total_bytes / 1e9:.2f} GB uncompressed) from {parsed_dir}"
    )

    archive_path = staging_dir / f"{format_name}.tar.gz"
    index_path = staging_dir / f"{format_name}.jsonl.gz"
    manifest_path = staging_dir / f"{format_name}.json"

    file_count, total_uncompressed_bytes = _write_index(
        index_path=index_path,
        parsed_files=parsed_files,
        format_name=format_name,
    )
    _write_archive(
        archive_path=archive_path,
        parsed_files=parsed_files,
        format_name=format_name,
    )

    manifest = {
        "format": format_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repo_commit": _source_commit(repo_root),
        "source_repo_dirty": False,
        "file_count": file_count,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "archive_sha256": _sha256_file(archive_path),
        "index_sha256": _sha256_file(index_path),
        "parser_output_dir": str(parsed_dir),
        "upload_command": " ".join(
            [
                "uv",
                "run",
                "python",
                "scripts/upload_parsed_wm_replays.py",
                "--format",
                format_name,
            ]
        ),
    }
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return archive_path, index_path, manifest_path, manifest


def upload_format(
    repo_id: str,
    revision: str,
    private: bool,
    archive_path: Path,
    index_path: Path,
    manifest_path: Path,
    manifest: dict,
) -> None:
    api = HfApi()
    print(f"Ensuring Hugging Face dataset repo exists: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"archives/{manifest['format']}.tar.gz",
            path_or_fileobj=str(archive_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"indexes/{manifest['format']}.jsonl.gz",
            path_or_fileobj=str(index_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"manifests/{manifest['format']}.json",
            path_or_fileobj=str(manifest_path),
        ),
    ]
    upload_bytes = (
        archive_path.stat().st_size
        + index_path.stat().st_size
        + manifest_path.stat().st_size
    )
    print(
        f"Uploading archive, index, and manifest "
        f"({upload_bytes / 1e9:.2f} GB total) to {repo_id}@{revision}"
    )
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        operations=operations,
        commit_message=(
            f"Upload parsed wm replays for {manifest['format']} "
            f"({manifest['file_count']} files, source {manifest['source_repo_commit'][:12]})"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package and upload parser text outputs for world-model training."
    )
    parser.add_argument("--format", required=True, choices=SUPPORTED_BATTLE_FORMATS)
    parser.add_argument(
        "--parsed_replay_root",
        default=(
            os.path.join(METAMON_CACHE_DIR, "parsed-replays")
            if METAMON_CACHE_DIR is not None
            else None
        ),
        help="Root containing parsed replay format directories.",
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Package files and print manifest, but do not create or upload to the Hub.",
    )
    args = parser.parse_args()

    if args.parsed_replay_root is None:
        raise ValueError(
            "METAMON_CACHE_DIR is not set; pass --parsed_replay_root explicitly."
        )

    repo_root = _repo_root()
    _ensure_clean_git(repo_root)

    with tempfile.TemporaryDirectory(prefix="metamon-parsed-wm-") as tmp:
        staging_dir = Path(tmp)
        archive_path, index_path, manifest_path, manifest = package_format(
            format_name=args.format,
            parsed_replay_root=Path(args.parsed_replay_root),
            staging_dir=staging_dir,
            repo_root=repo_root,
        )
        print(orjson.dumps(manifest, option=orjson.OPT_INDENT_2).decode("utf-8"))
        if args.dry_run:
            print("Dry run complete; no Hugging Face upload performed.")
            return
        upload_format(
            repo_id=args.repo_id,
            revision=args.revision,
            private=args.private,
            archive_path=archive_path,
            index_path=index_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        print(
            f"Uploaded {args.format} parsed-wm-replays to {args.repo_id}@{args.revision}."
        )


if __name__ == "__main__":
    main()
