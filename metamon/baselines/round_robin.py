import multiprocessing as mp
from itertools import combinations

from metamon.baselines import BASELINES_BY_GEN
from metamon.baselines.compete import head2head
from metamon.task_distributions import TASK_DISTRIBUTIONS


if __name__ == "__main__":
    from argparse import ArgumentParser
    from prettytable import PrettyTable, DOUBLE_BORDER

    parser = ArgumentParser()
    parser.add_argument("--gen", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--format", required=True, choices=["ou", "uu", "ubers"])
    parser.add_argument("--teams_per_matchup", type=int, default=10)
    parser.add_argument("--concurrent_matchups", type=int, default=20)
    parser.add_argument("--battles_per_team", type=int, default=5)
    parser.add_argument("--save_results_to")
    args = parser.parse_args()

    mp.set_start_method("spawn")

    baselines = BASELINES_BY_GEN[args.gen][args.format]
    matchups = [sorted((a.__name__, b.__name__)) for a, b in combinations(baselines, 2)]
    for baseline in baselines:
        matchups.append((baseline.__name__, baseline.__name__))

    task_dist_name = (
        f"Gen{args.gen}{'Ubers' if args.format == 'ubers' else args.format.upper()}"
    )
    matchup_args = [
        (*matchup, task_dist_name, args.teams_per_matchup, args.battles_per_team, True)
        for matchup in matchups
    ]
    results = []
    try:
        with mp.Pool(args.concurrent_matchups) as p:
            results.append(p.starmap(head2head, matchup_args))
    except KeyboardInterrupt:
        p.terminate()
    except Exception as e:
        print(e)
        p.terminate()
        exit(1)
    finally:
        p.join()

    table = PrettyTable()
    table.set_style(DOUBLE_BORDER)
    columns = sorted([b.__name__ for b in baselines])
    table.field_names = [
        f"{task_dist_name} Win Rates (%) (Sample of {args.teams_per_matchup} Teams x {args.battles_per_team} Games)"
    ] + [f"vs. {name}" for name in columns]
    for player in columns:
        row = [player]
        for opponent in columns:
            for result in results[0]:
                if (result.player_name, result.opponent_name) == (player, opponent):
                    row.append(int(result.win_pct))
                    break
                elif (result.opponent_name, result.player_name) == (player, opponent):
                    row.append(int(result.lose_pct))
                    break
        table.add_row(row)

    print(table)
    if args.save_results_to:
        with open(args.save_results_to, "w") as f:
            f.write(table.get_csv_string())
