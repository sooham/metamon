import os
import tqdm
from multiprocessing import Pool, cpu_count
from itertools import islice

from metamon.backend.team_prediction.dataset import FilteredTeamsFromReplaysDataset
from metamon.backend.team_prediction.predictor import ReplayPredictor


def chunk_dataset(dataset, chunk_size):
    iterator = iter(dataset)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


def process_chunk(args):
    chunk, format_name, output_dir, offset = args
    predictor = ReplayPredictor()
    success_count = 0

    for i, result in tqdm.tqdm(enumerate(chunk), total=len(chunk)):
        try:
            team, *_ = result
            predicted_team = predictor.predict(team)
            output = os.path.join(output_dir, f"team_{i + offset}.{format_name}_team")
            predicted_team.write_to_file(output)
            success_count += 1
        except Exception as e:
            print(f"Error generating teamset: {e}")
            continue

    return success_count


def main(args):
    for format in args.formats:
        os.makedirs(os.path.join(args.base_output_dir, format), exist_ok=True)
        team_dataset = FilteredTeamsFromReplaysDataset(
            replay_teamfile_dir=args.replay_teamfile_dir,
            min_date=args.min_date,
            max_date=args.max_date,
            min_rating=args.min_rating,
            format=format,
        )
        total_teams = len(team_dataset)
        print(f"Found {total_teams} teams for format {format}")

        num_processes = min(args.num_processes, total_teams)
        chunk_size = max(1, total_teams // (num_processes * 4))
        chunks = list(chunk_dataset(team_dataset, chunk_size))
        chunk_args = [
            (chunk, format, os.path.join(args.base_output_dir, format), i * chunk_size)
            for i, chunk in enumerate(chunks)
        ]
        with Pool(processes=num_processes) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(process_chunk, chunk_args),
                    total=len(chunks),
                    desc=f"Generating teamsets for {format}",
                )
            )

        total_success = sum(results)
        print(
            f"Successfully generated {total_success}/{total_teams} teamsets for {format}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate teamsets from replay data")
    parser.add_argument(
        "--replay_teamfile_dir",
        type=str,
        required=True,
        help="Path to the directory containing the replay teamfiles",
    )
    parser.add_argument(
        "--min_date",
        type=str,
        required=False,
        default=None,
        help="Minimum date of replays to include (MM-DD-YYYY)",
    )
    parser.add_argument(
        "--max_date",
        type=str,
        required=False,
        default=None,
        help="Maximum date of replays to include (MM-DD-YYYY)",
    )
    parser.add_argument(
        "--min_rating",
        type=int,
        required=False,
        default=None,
        help="Minimum rating threshold for included replays",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=[
            f"gen{i}{tier}" for i in range(1, 5) for tier in ["ou", "ubers", "nu", "uu"]
        ],
        help="List of formats to include",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the generated teamsets",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=False,
        default=cpu_count(),
        help="Number of processes to use for parallel processing",
    )
    args = parser.parse_args()
    main(args)
