import os
import random
from functools import partial
from typing import Callable, Iterable, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass

from poke_env.teambuilder import ConstantTeambuilder, Teambuilder

from metamon.data import DATA_PATH
from metamon import baselines
from metamon.baselines import BASELINES_BY_GEN, Baseline
from metamon.interface import RewardFunction, DefaultShapedReward


class seeded_random:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args, **kwargs):
        random.setstate(self.old_state)


TASK_DISTRIBUTIONS = {}


def register_task_dist(name: str):
    def _register(make_dist: Callable):
        TASK_DISTRIBUTIONS[name] = make_dist
        return make_dist

    return _register


class FixedTeambuilder(ConstantTeambuilder):
    def __init__(self, team: str, team_name: str):
        super().__init__(team)
        self.team_name = team_name


def select_random_team(gen: int, format: str, split: str) -> str:
    assert split in [
        "train",
        "test",
        "competitive",
        "train-competitive",
        "replays",
        "random_lead",
    ]
    if split == "train-competitive":
        split = random.choice(["train", "competitive"])
    path = os.path.join(DATA_PATH, "teams", f"gen{gen}", format, split)
    if not os.path.exists(path):
        raise ValueError(
            f"Cannot locate valid team directory for format [gen{gen}{format}]"
        )
    print(path)
    choice = random.choice(os.listdir(path))
    path_to_choice = os.path.join(path, choice)
    return path_to_choice


def fixed_seed_teambuilder(team_path: str) -> FixedTeambuilder:
    with open(team_path, "r") as f:
        team_data = f.read()
    fixed = FixedTeambuilder(team=team_data, team_name=os.path.basename(team_path))
    return fixed


class UniformRandomTeambuilder(Teambuilder):
    def __init__(self, gen: int, format: str, split: str):
        self.gen = gen
        self.format = format
        self.split = split

    def yield_team(self):
        team = select_random_team(self.gen, self.format, self.split)
        with open(os.path.join(team), "r") as f:
            team_data = f.read()
        self.team_name = os.path.basename(team)
        return self.join_team(self.parse_showdown_team(team_data))


@dataclass
class Task:
    battle_format: str
    opponent_type: Type[Baseline]
    k_shots: int
    player_teambuilder: Teambuilder
    opponent_teambuilder: Teambuilder
    reward_function: RewardFunction

    def __repr__(self):
        return f"{self.battle_format}_{self.player_teambuilder.team_name}_vs_{self.opponent_type.__name__}-{self.opponent_teambuilder.team_name}_k{self.k_shots}-rf{self.reward_function.__name__}"


class TaskDistribution(ABC):
    def __init__(
        self,
        opp_account_config=None,
        opp_server_config=None,
        k_shot_range: tuple[int, int] = [2, 2],
    ):
        self.opp_account_config = opp_account_config
        self.opp_server_config = opp_server_config
        self.k_range = k_shot_range

    @abstractmethod
    def generate_task(self, test_set: bool = False) -> Task:
        pass


class _RandomByFormat(TaskDistribution):
    def __init__(
        self,
        format: str,
        specify_opponents: Optional[Iterable[Baseline]] = None,
        reward_functions: Optional[Iterable[RewardFunction]] = None,
        player_split: str = "train",
        opponent_split: str = "train",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gen = int(format[3])
        self.format = format[4:]
        self.battle_format = format
        self.player_split = player_split
        self.opponent_split = opponent_split
        self.opponent_choices = (
            specify_opponents or BASELINES_BY_GEN[self.gen][self.format]
        )
        self.reward_functions = reward_functions or [DefaultShapedReward()]

    def add_opponent(self, opp_type):
        self.opponent_choices.append(opp_type)

    def generate_task(self, test_set: bool = False):
        opponent = random.choice(self.opponent_choices)
        opp_team = fixed_seed_teambuilder(
            select_random_team(self.gen, self.format, split=self.opponent_split)
        )
        player_team = fixed_seed_teambuilder(
            select_random_team(self.gen, self.format, split=self.player_split)
        )
        k_shots = random.randint(*self.k_range)
        return Task(
            battle_format=self.battle_format,
            opponent_type=opponent,
            k_shots=k_shots,
            player_teambuilder=player_team,
            opponent_teambuilder=opp_team,
            reward_function=random.choice(self.reward_functions),
        )


class FixedGenOpponentDistribution(TaskDistribution):
    def __init__(
        self,
        format: str,
        opponent: Baseline,
        reward_function: Optional[RewardFunction] = None,
        player_split: str = "train",
        opponent_split: str = "train",
        opp_account_config=None,
        opp_server_config=None,
    ):
        super().__init__(
            opp_account_config=opp_account_config,
            opp_server_config=opp_server_config,
            k_shot_range=[0, 0],
        )
        self.gen = int(format[3])
        self.format = format[4:]
        self.battle_format = format
        self.player_split = player_split
        self.opponent_split = opponent_split
        self.opponent = opponent
        self.reward_function = reward_function or DefaultShapedReward()

    def generate_task(self, test_set=False):
        opp_team = UniformRandomTeambuilder(self.gen, self.format, self.opponent_split)
        player_team = UniformRandomTeambuilder(self.gen, self.format, self.player_split)
        return Task(
            battle_format=self.battle_format,
            opponent_type=self.opponent,
            k_shots=0,
            player_teambuilder=player_team,
            opponent_teambuilder=opp_team,
            reward_function=self.reward_function,
        )


class UniformMixFormats(TaskDistribution):
    def __init__(
        self,
        formats: list[str],
        specify_opponents: Optional[Iterable[Baseline]] = None,
        reward_functions: Optional[Iterable[RewardFunction]] = None,
        player_split: str = "train",
        opponent_split: str = "train",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dists = {
            format: _RandomByFormat(
                format,
                specify_opponents=specify_opponents,
                reward_functions=reward_functions,
                player_split=player_split,
                opponent_split=opponent_split,
                *args,
                **kwargs,
            )
            for format in formats
        }
        self.opponent_choices = {
            format: self.dists[format].opponent_choices for format in formats
        }

    def add_opponent(self, opp_type):
        for dist in self.dists.values():
            dist.add_opponent(opp_type)

    def generate_task(self, seed: int | None = None, test_set: bool = False):
        seed = seed or random.randint(0, 1_000_000)
        with seeded_random(seed):
            dist = random.choice([d for d in self.dists.values()])
            return dist.generate_task(test_set=test_set)


for gen in range(1, 5):
    for format in ["OU", "UU", "NU", "Ubers"]:
        # easier to treat these as a "mix" of a single format
        register_task_dist(f"Gen{gen}{format}")(
            partial(UniformMixFormats, [f"gen{gen}{format.lower()}"])
        )

for gen in range(1, 5):
    register_task_dist(f"Gen{gen}All")(
        partial(
            UniformMixFormats,
            [f"gen{gen}{format}" for format in ["ou", "uu", "nu", "ubers"]],
        )
    )

for format in ["OU", "UU", "NU", "Ubers"]:
    register_task_dist(f"AllGen{format}")(
        partial(UniformMixFormats, [f"gen{gen}{format.lower()}" for gen in range(1, 5)])
    )

register_task_dist("AllGenAllFormat")(
    partial(
        UniformMixFormats,
        [
            f"gen{gen}{format}"
            for gen in range(1, 5)
            for format in ["ou", "uu", "nu", "ubers"]
        ],
    )
)


@register_task_dist("Tutorial")
class Tutorial(TaskDistribution):
    def generate_task(self, test_set: bool = False):
        path_to_team = os.path.join(DATA_PATH, "teams", "gen1", "tutorial_team")
        opp_team = fixed_seed_teambuilder(path_to_team)
        player_team = fixed_seed_teambuilder(path_to_team)
        return Task(
            battle_format="gen1nu",
            opponent_type=baselines.heuristic.basic.RandomBaseline,
            k_shots=0,
            player_teambuilder=player_team,
            opponent_teambuilder=opp_team,
            reward_function=DefaultShapedReward(),
        )


@register_task_dist("MultiOpponentTutorial")
class MultiOpponentTutorial(TaskDistribution):
    def generate_task(self, test_set: bool = False):
        path_to_team = os.path.join(DATA_PATH, "teams", "gen1", "tutorial_team")
        opp_team = fixed_seed_teambuilder(path_to_team)
        player_team = fixed_seed_teambuilder(path_to_team)
        opp = random.choice(
            [
                baselines.heuristic.basic.RandomBaseline,
                baselines.heuristic.basic.Gen1Trainer,
                baselines.heuristic.basic.Gen1GoodAI,
                baselines.heuristic.kaizo.EmeraldKaizo,
                baselines.heuristic.basic.PokeEnvHeuristic,
            ]
        )
        return Task(
            battle_format="gen1nu",
            opponent_type=opp,
            k_shots=2,
            player_teambuilder=player_team,
            opponent_teambuilder=opp_team,
            reward_function=DefaultShapedReward(),
        )


def get_task_distribution(name: str) -> Type[TaskDistribution]:
    if name not in TASK_DISTRIBUTIONS:
        # try checking for the more standard lowercase version of the name
        if name.lower().startswith("gen"):
            gen = int(name[3])
            format = name[4:].lower()
            format = "Ubers" if format == "ubers" else format.upper()
            name = f"Gen{gen}{format}"

    if name not in TASK_DISTRIBUTIONS:
        raise ValueError(
            f"Task Distribution `{name}` not found. Options are: {list(TASK_DISTRIBUTIONS.keys())}"
        )
    return TASK_DISTRIBUTIONS[name]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--list_default_opponents", choices=TASK_DISTRIBUTIONS.keys())
    parser.add_argument("--list_distributions", action="store_true")
    args = parser.parse_args()

    if args.list_distributions:
        print("---- Metamon Task Distributions ----")
        all_dists = sorted(TASK_DISTRIBUTIONS)
        for i, task_dist in enumerate(all_dists):
            print(f"{i + 1}. {task_dist}")

    if args.list_default_opponents:
        dist = TASK_DISTRIBUTIONS[args.list_default_opponents]()
        try:
            for i in range(20):
                task = dist.generate_task()

            breakpoint()
            task_23 = dist.generate_task(seed=23)
            task_23_again = dist.generate_task(seed=23)
            assert str(task_23) == str(task_23_again)
            task_24 = dist.generate_task(seed=24)
            assert str(task_23) != str(task_24)
        except Exception as e:
            print(
                f"Failed to generate task distribution {args.list_default_opponents} with error: {e}"
            )
            exit(1)
        else:
            print(f"{args.list_default_opponents} Loaded Successfully")
            if isinstance(dist, UniformMixFormats):
                print(
                    f"---- {dist.__class__.__name__} ({args.list_default_opponents}) ----"
                )
                formats = sorted(dist.opponent_choices)
                for format in formats:
                    opponents = dist.opponent_choices[format]
                    print(f"{format} Opponents:")
                    opponents = sorted([opp.__name__ for opp in opponents])
                    for i, opponent in enumerate(opponents):
                        print(f"\t{i + 1}. {opponent}")
