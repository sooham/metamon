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

import lz4.frame
import orjson
from huggingface_hub import CommitOperationAdd, HfApi
from tqdm import tqdm

from metamon.config import METAMON_CACHE_DIR, SUPPORTED_BATTLE_FORMATS


DEFAULT_REPO_ID = "sooham34/metamon-wm-dataset"
MANIFEST_EXTENSION = ".metadata"
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


def _ensure_clean_git(repo_root: Path) -> None:
    status = _run_git(["status", "--porcelain", "--untracked-files=all"], repo_root)
    if not status:
        return
    raise RuntimeError(
        "Refusing to upload wm-dataset because the git repository is dirty. "
        "Commit, stash, or remove local changes first.\n"
        f"\nDirty paths:\n{status}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _dataset_files(output_dir: Path) -> list[Path]:
    ignored_names = {".DS_Store"}
    return sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in ignored_names
    )


def _validate_output_dir(output_dir: Path, format_name: str) -> dict:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"No world-model output directory found: {output_dir}")
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing generated metadata file: {metadata_path}")
    metadata = _load_json(metadata_path)
    formats = metadata.get("formats") or []
    if formats != [format_name]:
        raise ValueError(
            "World-model dataset upload expects a single-format generated dataset. "
            f"Requested {format_name!r}, but metadata.json has formats={formats!r}."
        )
    for split in ["train", "val"]:
        split_dir = output_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing generated split directory: {split_dir}")
        if not list(split_dir.glob("paired_shard_*.npz")):
            raise FileNotFoundError(f"No paired_shard_*.npz files found in {split_dir}")
    return metadata


def _check_staging_space(staging_parent: Path, total_bytes: int) -> None:
    free_bytes = shutil.disk_usage(staging_parent).free
    required_bytes = total_bytes + MIN_FREE_BYTES_AFTER_STAGING
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough free disk space to package wm-dataset. "
            f"Need at least {required_bytes / 1e9:.2f} GB free, "
            f"found {free_bytes / 1e9:.2f} GB."
        )


def _write_index(index_path: Path, files: list[Path], output_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    with gzip.open(index_path, "wt", encoding="utf-8") as f:
        for path in tqdm(files, desc="Indexing wm dataset", unit="file"):
            rel = path.relative_to(output_dir).as_posix()
            size_bytes = path.stat().st_size
            total_bytes += size_bytes
            file_count += 1
            row = {
                "path": rel,
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
            }
            f.write(orjson.dumps(row).decode("utf-8"))
            f.write("\n")
    return file_count, total_bytes


def _write_archive(archive_path: Path, files: list[Path], output_dir: Path) -> None:
    with lz4.frame.open(archive_path, "wb") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for path in tqdm(files, desc="Writing wm tar.lz4", unit="file"):
                rel = path.relative_to(output_dir).as_posix()
                tar.add(path, arcname=rel, recursive=False)


def _write_dataset_card(readme_path: Path, repo_id: str, manifest: dict) -> None:
    metadata = manifest.get("generator_metadata", {})
    card = f"""---
license: mit
task_categories:
- reinforcement-learning
language:
- en
pretty_name: Metamon World-Model Dataset
---

# Metamon World-Model Dataset

This dataset contains paired-POV world-model shards produced by
`scripts/generate_world_model_data.py`, usually via `make wm-dataset`.

## Storage layout

- `archives/<format>.tar.lz4`: generated world-model dataset root for one format.
- `indexes/<format>.jsonl.gz`: one row per archived file with `path`,
  `size_bytes`, and `sha256`.
- `manifests/<format>.metadata`: packaging metadata for the corresponding
  archive and index.

The archive extracts directly to the layout expected by JEPA training:
`train/paired_shard_*.npz`, `val/paired_shard_*.npz`, `metadata.json`, and
auxiliary summary files.

## Current upload

- Dataset repo: `{repo_id}`
- Format: `{manifest["format"]}`
- Source Metamon commit: `{manifest["source_repo_commit"]}`
- Created at: `{manifest["created_at"]}`
- File count: `{manifest["file_count"]}`
- Total uncompressed bytes: `{manifest["total_uncompressed_bytes"]}`
- Archive SHA256: `{manifest["archive_sha256"]}`
- Index SHA256: `{manifest["index_sha256"]}`
- Schema version: `{metadata.get("schema_version", "unknown")}`
- Tokenizer version: `{metadata.get("tokenizer_version", "unknown")}`
- Rollout length: `{metadata.get("rollout_len", "unknown")}`
- Seed: `{metadata.get("seed", "unknown")}`

## Download

```bash
uv run python -m metamon.data.download wm-dataset --formats {manifest["format"]}
```

The downloader extracts files into `$METAMON_CACHE_DIR/world-model-samples`.
"""
    readme_path.write_text(card, encoding="utf-8")


def package_format(
    format_name: str,
    output_dir: Path,
    staging_dir: Path,
    repo_root: Path,
    repo_id: str,
) -> tuple[Path, Path, Path, Path, dict]:
    metadata = _validate_output_dir(output_dir, format_name)
    files = _dataset_files(output_dir)
    if not files:
        raise FileNotFoundError(f"No files found under {output_dir}")

    total_bytes = sum(path.stat().st_size for path in files)
    _check_staging_space(staging_dir, total_bytes)
    print(
        f"Packaging {len(files)} wm dataset files "
        f"({total_bytes / 1e9:.2f} GB uncompressed) from {output_dir}"
    )

    archive_path = staging_dir / f"{format_name}.tar.lz4"
    index_path = staging_dir / f"{format_name}.jsonl.gz"
    manifest_path = staging_dir / f"{format_name}{MANIFEST_EXTENSION}"
    readme_path = staging_dir / "README.md"

    file_count, total_uncompressed_bytes = _write_index(
        index_path=index_path,
        files=files,
        output_dir=output_dir,
    )
    _write_archive(archive_path=archive_path, files=files, output_dir=output_dir)

    manifest = {
        "format": format_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repo_commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "source_repo_dirty": False,
        "file_count": file_count,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "archive_sha256": _sha256_file(archive_path),
        "index_sha256": _sha256_file(index_path),
        "output_dir": str(output_dir),
        "generator_metadata": metadata,
        "upload_command": " ".join(
            [
                "uv",
                "run",
                "python",
                "scripts/upload_wm_dataset.py",
                "--format",
                format_name,
            ]
        ),
    }
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    _write_dataset_card(readme_path=readme_path, repo_id=repo_id, manifest=manifest)
    return archive_path, index_path, manifest_path, readme_path, manifest


def upload_format(
    repo_id: str,
    revision: str,
    private: bool,
    archive_path: Path,
    index_path: Path,
    manifest_path: Path,
    readme_path: Path,
    manifest: dict,
) -> None:
    api = HfApi()
    print(f"Ensuring Hugging Face dataset repo exists: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"archives/{manifest['format']}.tar.lz4",
            path_or_fileobj=str(archive_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"indexes/{manifest['format']}.jsonl.gz",
            path_or_fileobj=str(index_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"manifests/{manifest['format']}{MANIFEST_EXTENSION}",
            path_or_fileobj=str(manifest_path),
        ),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path)),
    ]
    upload_bytes = (
        archive_path.stat().st_size
        + index_path.stat().st_size
        + manifest_path.stat().st_size
        + readme_path.stat().st_size
    )
    print(
        f"Uploading archive, index, manifest, and dataset card "
        f"({upload_bytes / 1e9:.2f} GB total) to {repo_id}@{revision}"
    )
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        operations=operations,
        commit_message=(
            f"Upload wm dataset for {manifest['format']} "
            f"({manifest['file_count']} files, source {manifest['source_repo_commit'][:12]})"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package and upload generated world-model dataset shards."
    )
    parser.add_argument("--format", required=True, choices=SUPPORTED_BATTLE_FORMATS)
    parser.add_argument(
        "--output_dir",
        default=(
            os.path.join(METAMON_CACHE_DIR, "world-model-samples")
            if METAMON_CACHE_DIR is not None
            else None
        ),
        help="Generated world-model dataset root from make wm-dataset.",
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
        help="Package files and print manifest, but do not upload to the Hub.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        raise ValueError("METAMON_CACHE_DIR is not set; pass --output_dir explicitly.")

    repo_root = _repo_root()
    _ensure_clean_git(repo_root)

    with tempfile.TemporaryDirectory(prefix="metamon-wm-dataset-") as tmp:
        staging_dir = Path(tmp)
        archive_path, index_path, manifest_path, readme_path, manifest = package_format(
            format_name=args.format,
            output_dir=Path(args.output_dir),
            staging_dir=staging_dir,
            repo_root=repo_root,
            repo_id=args.repo_id,
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
            readme_path=readme_path,
            manifest=manifest,
        )
        print(f"Uploaded {args.format} wm-dataset to {args.repo_id}@{args.revision}.")


if __name__ == "__main__":
    main()
