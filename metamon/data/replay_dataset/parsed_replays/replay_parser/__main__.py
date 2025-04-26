import os
import glob
import random
import tqdm

from .parse_replays import ReplayParser

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--gen", type=int, choices=list(range(1, 10)), required=True)
    parser.add_argument(
        "--raw_replay_dir", required=True, help="Path to raw replay dataset folder."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["ou", "uu", "nu", "ubers"],
        required=True,
    )
    parser.add_argument("--max", type=int, help="Parse up to this many replays.")
    parser.add_argument(
        "--filter_by_code",
        help="Skip to a specific game id. For example: `gen4ubers-1101300080`",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start parsing from this index of the dataset (skip replays you've already checked)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Prints the raw replay stream during parsing (useful for debugging)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel parser processes to run",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for output .npz files. `None` runs w/o saving to disk. Data will be saved to {--output_dir}/gen{gen}{format}",
    )
    args = parser.parse_args()

    invalid_format_set: set[str] = set()
    path = os.path.join(args.raw_replay_dir, f"gen{args.gen}", args.format)
    filenames = glob.glob(f"{path}/**/*.json", recursive=True)
    random.shuffle(filenames)
    if args.filter_by_code is not None:
        filenames = [f for f in filenames if args.filter_by_code in f]
    if args.start_from is not None:
        filenames = filenames[args.start_from :]
    if args.max is not None:
        filenames = filenames[: args.max]

    output_dir = (
        os.path.join(args.output_dir, f"gen{args.gen}{args.format}")
        if args.output_dir
        else None
    )
    parser = ReplayParser(
        output_dir=output_dir,
        verbose=args.verbose,
    )
    if args.processes > 1:
        random.shuffle(filenames)
        parser.parse_parallel(filenames, args.processes)
    else:
        for filename in tqdm.tqdm(filenames):
            parser.parse_replay(filename)
        errors = parser.summarize_errors()
        for fb, sub in errors.items():
            print(f"{fb} Errors:")
            for i, (err, c) in enumerate(sub.items()):
                print(f"\t{i + 1}. {err}: {c}")
