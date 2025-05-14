import time
import random
import os
import copy
import json
from datetime import datetime
from typing import Optional, Type

import numpy as np
import gymnasium as gym
from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
)
from poke_env.environment import Battle
from poke_env.player import OpenAIGymEnv, Player
from poke_env.teambuilder import Teambuilder

from metamon.interface import (
    UniversalState,
    action_idx_to_battle_order,
    RewardFunction,
    DefaultShapedReward,
    ObservationSpace,
    DefaultObservationSpace,
)
from metamon.data import DATA_PATH


TEAM_PATH = os.path.join(os.path.dirname(__file__), "teams")


class TeamSet(Teambuilder):

    def __init__(self, team_file_dir: str, battle_format: str):
        super().__init__()
        self.team_file_dir = team_file_dir
        self.battle_format = battle_format
        self.team_files = self._find_team_files()

    def _find_team_files(self):
        team_files = []
        for root, _, files in os.walk(self.team_file_dir):
            for file in files:
                if file.endswith(f".{self.battle_format}_team"):
                    team_files.append(os.path.join(root, file))
        return team_files

    def yield_team(self):
        file = random.choice(self.team_files)
        with open(file, "r") as f:
            team_data = f.read()
        return self.join_team(self.parse_showdown_team(team_data))


def get_metamon_teams(battle_format: str, split: str) -> TeamSet:
    gen = int(battle_format[3])
    tier = battle_format[4:]
    path = os.path.join(TEAM_PATH, f"gen{gen}", tier, split)
    if not os.path.exists(path):
        raise ValueError(
            f"Cannot locate valid team directory for format [gen{gen}{tier} at path {path}]"
        )
    return TeamSet(path, battle_format)


def _check_avatar(avatar: str):
    if avatar is None:
        return
    with open(os.path.join(DATA_PATH, "avatar_names.txt"), "r") as f:
        options = [l.strip() for l in f.readlines()]
    if avatar not in options:
        raise ValueError(
            f"Avatar {avatar} is not valid. See https://play.pokemonshowdown.com/sprites/trainers/ for a list of options."
        )
    else:
        return avatar


class PokeEnvWrapper(OpenAIGymEnv):
    """
    A thin wrapper around poke-env's OpenAIGymEnv that handles the observation space, action
    space, and reward function while adding some basic conveniences.
    """

    def __init__(
        self,
        battle_format: str,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        player_team_set: TeamSet,
        opponent_type: Optional[Type[Player]] = None,
        opponent_team_set: Optional[TeamSet] = None,
        player_username: Optional[str] = None,
        player_password: Optional[str] = None,
        opponent_username: Optional[str] = None,
        player_avatar: Optional[str] = None,
        opponent_avatar: Optional[str] = None,
        start_challenging: bool = True,
        start_timer_on_battle_start: bool = False,
        turn_limit: int = 1000,
        save_trajectories_to: Optional[str] = None,
    ):
        opponent_team_set = opponent_team_set or copy.deepcopy(player_team_set)
        random_username = (
            lambda: f"MM-{''.join(str(random.randint(0, 9)) for _ in range(10))}"
        )
        self.player_username = player_username or random_username()
        self.opponent_username = opponent_username or random_username()

        player_account_configuration = AccountConfiguration(
            self.player_username, player_password
        )
        opponent_account_configuration = AccountConfiguration(
            self.opponent_username, None
        )
        if save_trajectories_to is not None:
            self.save_trajectories_to = os.path.join(
                save_trajectories_to, battle_format
            )
            os.makedirs(self.save_trajectories_to, exist_ok=True)
        else:
            self.save_trajectories_to = None

        if opponent_type is not None:
            self.metamon_opponent_name = opponent_type.__name__
            self._current_opponent = opponent_type(
                battle_format=battle_format,
                team=opponent_team_set,
                account_configuration=opponent_account_configuration,
                server_configuration=self.server_configuration,
                avatar=_check_avatar(opponent_avatar),
                ping_timeout=None,
            )
        else:
            self._current_opponent = None
            self.metamon_opponent_name = "Ladder"

        self.reward_function = reward_function
        self.metamon_obs_space = observation_space
        self.turn_limit = turn_limit
        self.metamon_battle_format = battle_format
        super().__init__(
            battle_format=battle_format,
            server_configuration=self.server_configuration,
            account_configuration=player_account_configuration,
            team=player_team_set,
            avatar=_check_avatar(player_avatar),
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_challenging=start_challenging,
            ping_timeout=None,
        )

    @property
    def server_configuration(self):
        return LocalhostServerConfiguration

    def get_opponent(self):
        return self._current_opponent

    def action_space_size(self):
        return 9

    def on_invalid_order(self, battle: Battle):
        return self.choose_random_move(battle)

    def reset(self, *args, **kwargs):
        self.invalid_action_counter = 0
        self.valid_action_counter = 0
        self.turn_counter = 0
        self.battle_reference = self.agent.n_won_battles
        self.trajectory = {"states": [], "actions": []}
        return super().reset(*args, **kwargs)

    def action_to_move(self, action: int, battle: Battle):
        order = action_idx_to_battle_order(battle, action)
        if order is None:
            self.invalid_action_counter += 1
            return self.on_invalid_order(battle)
        else:
            self.valid_action_counter += 1
            return order

    def describe_embedding(self) -> gym.spaces.Space:
        return self.metamon_obs_space.gym_space

    def calc_reward(self, last_battle: Battle, current_battle: Battle) -> float:
        last_state = UniversalState.from_Battle(last_battle)
        state = UniversalState.from_Battle(current_battle)
        reward = self.reward_function(last_state, state)
        return reward

    def embed_battle(self, battle: Battle):
        universal_state = UniversalState.from_Battle(battle)
        if self.save_trajectories_to is not None:
            self.trajectory["states"].append(universal_state)
        return self.metamon_obs_space.state_to_obs(universal_state)

    def step(self, action):
        self.turn_counter += 1
        next_state, reward, terminated, truncated, info = super().step(action)
        if self.save_trajectories_to is not None:
            self.trajectory["actions"].append(int(action))

        # enforce simple turn limit
        hit_time_limit = self.turn_counter > self.turn_limit
        terminated |= hit_time_limit
        truncated |= hit_time_limit
        if terminated or truncated:
            # logging info
            info["valid_action_count"] = self.valid_action_counter
            info["invalid_action_count"] = self.invalid_action_counter
            info["won"] = self.agent.n_won_battles > self.battle_reference
            self.battle_reference = self.agent.n_won_battles

            if self.save_trajectories_to is not None:
                # build a long filename that matches the format of the parsed replay dataset
                result = "WIN" if info["won"] == 1 else "LOSS"
                battle_id = "".join(str(random.randint(0, 9)) for _ in range(10))
                timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
                filename = f"metamon-{self.metamon_battle_format}-{battle_id}_Unrated_{self.player_username}_vs_{self.metamon_opponent_name}_{timestamp}_{result}.json"
                # matches the format of the parsed replay dataset
                output_json = {
                    "states": [s.to_dict() for s in self.trajectory["states"]],
                    "actions": self.trajectory["actions"],
                }
                # conservative file writing to avoid partial writes on shutdown or interruption
                # when launching multiple environments in parallel
                path = os.path.join(self.save_trajectories_to, filename)
                temp_path = path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(output_json, f)
                os.rename(temp_path, path)

        return next_state, reward, terminated, truncated, info


class BattleAgainstBaseline(PokeEnvWrapper):
    """
    Battle against a specified opponent.

    Can be used to battle any opponent that connects via the poke-env interface
    (e.g., any baseline in the `baselines` module, or any other custom `Player`)
    """

    def __init__(
        self,
        battle_format: str,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        team_set: TeamSet,
        opponent_type: Type[Player],
        turn_limit: int = 200,
        save_trajectories_to: Optional[str] = None,
    ):
        super().__init__(
            battle_format=battle_format,
            observation_space=observation_space,
            reward_function=reward_function,
            player_team_set=team_set,
            opponent_team_set=team_set,
            opponent_type=opponent_type,
            turn_limit=turn_limit,
            save_trajectories_to=save_trajectories_to,
        )


class QueueOnLocalLadder(PokeEnvWrapper):
    """
    Battle against an opponent by queueing for ladder matches on the local server.

    Can be used to battle any opponent that plays with the Showdown API
    (e.g. humans, your own ML baselines, third-party heuristic bots, etc.).

    Create the environment, start laddering with the opponent, and the battle(s) will begin
    when both players are connected.
    """

    # increases time to launch opponent envs before ladder loop times out ("Agent is not challenging")
    _INIT_RETRIES = 1000

    def __init__(
        self,
        battle_format: str,
        num_battles: int,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        player_team_set: TeamSet,
        player_username: str,
        player_avatar: Optional[str] = None,
        start_timer_on_battle_start: bool = True,
        save_trajectories_to: Optional[str] = None,
    ):

        super().__init__(
            battle_format=battle_format,
            observation_space=observation_space,
            reward_function=reward_function,
            player_team_set=player_team_set,
            player_username=player_username,
            player_avatar=player_avatar,
            start_timer_on_battle_start=start_timer_on_battle_start,
            opponent_type=None,
            start_challenging=False,
            turn_limit=float("inf"),
            save_trajectories_to=save_trajectories_to,
        )
        self.start_laddering(n_challenges=num_battles)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)
        self.render()
        return next_state, reward, terminated, truncated, info


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.baselines.heuristic.basic import GymLeader

    parser = ArgumentParser()
    parser.add_argument("--battle_format", type=str, default="gen1ou")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--team_split", type=str, default="replays")
    args = parser.parse_args()

    env = BattleAgainstBaseline(
        battle_format=args.battle_format,
        team_set=get_metamon_teams(args.battle_format, args.team_split),
        opponent_type=GymLeader,
        observation_space=DefaultObservationSpace(),
        reward_function=DefaultShapedReward(),
    )

    start = time.time()
    counter = 0
    for ep in range(args.episodes):
        print(f"Episode {ep}")
        inner_start = time.time()
        state, info = env.reset()
        done = False
        return_ = 0.0
        timesteps = 0
        while not done:
            env.render()
            state, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            return_ += reward
            done = terminated or truncated
            timesteps += 1
            counter += 1
        print(
            f"Episode {ep}:: Timesteps: {timesteps}, Total Return: {return_ : .2f}, FPS: {timesteps / (time.time() - inner_start) : .2f}, Invalid Action: {info['invalid_action_count']}, Valid Actions: {info['valid_action_count']}"
        )

    end = time.time()
    print(f"{counter / (end - start) : .2f} Steps Per Second")
