import os

from metamon.data.team_prediction.dataset import FilteredTeamsFromReplaysDataset
from metamon.data.team_prediction.predictor import ReplayPredictor


def main(args):
    os.makedirs(args.base_output_dir, exist_ok=True)
    predictor = ReplayPredictor()
    for format in args.formats:
        team_dataset = FilteredTeamsFromReplaysDataset(
            replay_teamfile_dir=args.replay_teamfile_dir,
            min_date=args.min_date,
            max_date=args.max_date,
            min_rating=args.min_rating,
            format=format,
        )
        print(f"Found {len(team_dataset)} teams for format {format}")
        for i, (team, *_) in enumerate(team_dataset):
            predicted_team = predictor.predict(team)
            output = os.path.join(args.base_output_dir, f"team_{i}.{format}_team")
            predicted_team.write_to_file(output)


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
        help="Minimum date to include replays from (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max_date",
        type=str,
        required=False,
        default=None,
        help="Maximum date to include replays from (YYYY-MM-DD)",
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
    args = parser.parse_args()
    main(args)
