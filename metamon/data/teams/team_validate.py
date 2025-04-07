if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from metamon.data import DATA_PATH
    import tqdm

    parser = ArgumentParser()
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--format", type=str, default="ou")
    parser.add_argument("--team_split", type=str, default="train")
    args = parser.parse_args()

    team_dir = os.path.join(
        DATA_PATH, "teams", f"gen{args.gen}", args.format, args.team_split
    )
    team_list = os.listdir(team_dir)

    # valid each team
    failures = []
    for t in tqdm.tqdm(team_list):
        with open(os.path.join(team_dir, t), "r") as f:
            team = f.read()
        valid = os.popen(
            f"echo '{team}' | ~/pokemon-showdown/pokemon-showdown validate-team gen{args.gen}{args.format}"
        )
        if valid.close() is not None:
            failures.append(t)
            print(f"Team {t} failure.")
    print(f"All Failures: {failures}")
