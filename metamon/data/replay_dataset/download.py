from huggingface_hub import hf_hub_download
import argparse
import os
import tarfile
from typing import List
from pathlib import Path

REPO_ID = "jakegrigsby/metamon-parsed-replays"
ALL_FORMATS = [
    "gen1nu",
    "gen1ou",
    "gen1uu",
    "gen1ubers",
    "gen2nu",
    "gen2ou",
    "gen2uu",
    "gen2ubers",
    "gen3nu",
    "gen3ou",
    "gen3uu",
    "gen3ubers",
    "gen4nu",
    "gen4ou",
    "gen4uu",
    "gen4ubers",
]


def download_and_extract(formats: List[str], output_dir: str, keep_tar: bool = False):
    """Download and extract specified formats from the HF dataset.

    Args:
        formats: List of format names to download (e.g. ["gen1ou", "gen2ou"])
        output_dir: Directory to extract files to
        keep_tar: If True, keep the downloaded .tar.gz files
    """
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        print(f"Downloading {fmt}...")
        tar_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"data/{fmt}.tar.gz",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        print(f"Extracting {fmt}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

        if not keep_tar:
            os.remove(tar_path)
            print(f"Removed {tar_path}")

        print(f"✓ {fmt} complete")


def main():
    parser = argparse.ArgumentParser(description="Download Metamon Replay Datasets")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=ALL_FORMATS + ["all"],
        default=["all"],
        help="Which formats to download (e.g. gen1ou gen2ou). Use 'all' for all formats.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the extracted files",
    )
    parser.add_argument(
        "--keep-tar", action="store_true", help="Keep the downloaded .tar.gz files"
    )
    args = parser.parse_args()

    formats = ALL_FORMATS if "all" in args.formats else args.formats

    print(f"Will download: {', '.join(formats)}")
    print(f"Output directory: {args.output_dir}")

    download_and_extract(formats, args.output_dir, args.keep_tar)
    print("\n✅ Download complete!")


if __name__ == "__main__":
    main()
