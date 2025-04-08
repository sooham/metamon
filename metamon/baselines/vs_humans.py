import asyncio
import os

from poke_env import ShowdownServerConfiguration, AccountConfiguration
from metamon.baselines import ALL_BASELINES
from metamon.task_distributions import UniformRandomTeambuilder, fixed_seed_teambuilder
from metamon.data import DATA_PATH


async def player_vs_humans(args):
    player = ALL_BASELINES[args.player](
        battle_format=f"gen{args.gen}{args.format.lower()}",
        team=UniformRandomTeambuilder(
            gen=args.gen, format=args.format, split=args.team_split
        ),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration(
            username=args.username, password=args.password
        ),
        server_configuration=ShowdownServerConfiguration,
        start_timer_on_battle_start=True,
        avatar=args.avatar,
        save_replays=f"gen{args.gen}{args.format}_{args.player}_vs_humans_replays",
    )
    await player.ladder(args.battles)
    win_loss = [b.won for b in player.battles.values()]
    opponent_rating = [b.opponent_rating for b in player.battles.values()]
    rating = [b.rating for b in player.battles.values()]
    print(f"Win Rate: {sum(win_loss) / len(win_loss)}")
    print(f"Rating : {rating}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--format", default="ou", choices=["ubers", "ou", "uu", "nu"])
    parser.add_argument("--player", default="GymLeader")
    parser.add_argument("--team_split", default="competitive")
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--avatar", type=str, default="clown")
    args = parser.parse_args()

    if args.avatar is not None:
        with open(os.path.join(DATA_PATH, "avatar_names.txt"), "r") as f:
            options = [l.strip() for l in f.readlines()]
        if args.avatar not in options:
            raise ValueError(
                f"Avatar {args.avatar} is not valid. See https://play.pokemonshowdown.com/sprites/trainers/ for a list of options."
            )

    asyncio.get_event_loop().run_until_complete(player_vs_humans(args))
