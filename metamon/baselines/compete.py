import tqdm
import uuid
from dataclasses import dataclass
from poke_env import AccountConfiguration
import asyncio

from metamon.baselines import ALL_BASELINES
from metamon.task_distributions import get_task_distribution


@dataclass
class MatchupResult:
    player_name: str
    opponent_name: str
    task_dist_name: str
    win_pct: float
    lose_pct: float
    tie_pct: float


async def _run_battles(player, opponent, n_battles=5):
    player.reset_battles()
    opponent.reset_battles()
    await player.battle_against(opponent, n_battles=n_battles)
    return player.win_rate, player.lose_rate, player.tie_rate


def head2head(
    player_name: str,
    opponent_name: str,
    task_dist_name: str,
    n_tasks: int,
    max_concurrent: int = 5,
    verbose: bool = False,
) -> MatchupResult:
    task_dist = get_task_distribution(task_dist_name)(k_shot_range=[0, 0])
    iter_ = range(n_tasks)
    if verbose:
        iter_ = tqdm.tqdm(iter_)
    all_win_rates, all_lose_rates, all_tie_rates = [], [], []
    for _ in iter_:
        task = task_dist.generate_task()
        player = ALL_BASELINES[player_name](
            battle_format=task.battle_format,
            team=task.player_teambuilder,
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(
                username=f"p{str(uuid.uuid4())[:15]}", password=None
            ),
        )
        opponent = ALL_BASELINES[opponent_name](
            battle_format=task.battle_format,
            team=task.opponent_teambuilder,
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(
                username=f"p{str(uuid.uuid4())[:15]}", password=None
            ),
        )
        player.randomize()
        opponent.randomize()
        win_rate, lose_rate, tie_rate = asyncio.get_event_loop().run_until_complete(
            _run_battles(player, opponent)
        )
        all_win_rates.append(win_rate)
        all_lose_rates.append(lose_rate)
        all_tie_rates.append(tie_rate)

    avg = lambda x: 100 * (sum(x) / len(x))
    return MatchupResult(
        player_name=player_name,
        opponent_name=opponent_name,
        task_dist_name=task_dist_name,
        win_pct=avg(all_win_rates),
        lose_pct=avg(all_lose_rates),
        tie_pct=avg(all_tie_rates),
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()
    parser.add_argument("--task_dist")
    parser.add_argument("--player")
    parser.add_argument("--opponent")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()

    result = head2head(
        args.player,
        args.opponent,
        args.task_dist,
        n_tasks=args.tasks,
        max_concurrent=args.concurrent,
        verbose=True,
    )

    print(
        f"{result.player_name} vs {result.opponent_name} on {result.task_dist_name}: Wins {result.win_pct: .2f}% / Loses {result.lose_pct : .2f}% / Ties {result.tie_pct : .2f}%"
    )
