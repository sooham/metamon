import time
import random
import os
import copy
import json
from datetime import datetime
from typing import Optional, Type

import numpy as np
import lz4.frame
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
from metamon.download import download_teams


class TeamSet(Teambuilder):
    """Sample from a directory of Showdown team files.

    A simple wrapper around poke-env's Teambuilder that randomly samples a team from a
    directory of team files.

    Args:
        team_file_dir: The directory containing the team files (searched recursively).
            Team files are just text files in the standard Showdown export format. See
            https://pokepast.es/syntax.html for details.
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers",
            etc.). Note that we assume files have a matching extension (e.g.
            "any_name.gen1ou_team").
    """

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


def get_metamon_teams(battle_format: str, set_name: str) -> TeamSet:
    """
    Download a set of teams from huggingface (if necessary) and return a TeamSet.

    Args:
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers", etc.).
        set_name: The name of the set of teams to download. See the README for options.
    """
    path = download_teams(battle_format, set_name=set_name)
    if not os.path.exists(path):
        raise ValueError(
            f"Cannot locate valid team directory for format {battle_format} at path {path}"
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
    """A thin wrapper around poke-env's OpenAIGymEnv that handles the observation space,
    action space, and reward function while adding some basic conveniences.

    Args:
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers",
            etc.).
        observation_space: The observation space to use. Must be an instance of
            `interface.ObservationSpace`.
        reward_function: The reward function to use. Must be an instance of
            `interface.RewardFunction`.
        player_team_set: The team set to use for the player. Must be an instance of
            `TeamSet` or a poke-env TeamBuilder.
        opponent_type: The type of opponent to use. Must be a callable that creates a
            `poke_env.player.Player`. All the `metamon.baselines` implement this.
        opponent_team_set: The team set to use for the opponent. Must be an instance of
            `TeamSet` or a poke-env TeamBuilder. If the opponent set is not specified,
            the opponent plays with the same team set (but an independently sampled team)
            as the player.
        player_username: The username to use for the player. If not specified, a random
            username will be generated. It is important for this to be unique across
            parallel environments.
        player_password: The password to use for the player. This is usually not needed
            on the local server.
        opponent_username: The username to use for the opponent. If not specified, a
            random username will be generated. It is important for this to be unique
            across parallel environments.
        player_avatar: The avatar for the player when viewing battles in your browser.
            See https://play.pokemonshowdown.com/sprites/trainers/ for a list of
            options.
        opponent_avatar: The avatar for the opponent when viewing battles in your
            browser. See https://play.pokemonshowdown.com/sprites/trainers/ for a list
            of options.
        start_challenging: Whether to start challenging the opponent immediately. This
            is a poke-env detail handled by the other wrappers.
        start_timer_on_battle_start: Start the time increment controls that prevent
            infinite battles from inactive players.
        turn_limit: The maximum number of turns in a battle. Note that Showdown already
            enforces a limit of 1000 regardless of this setting.
        save_trajectories_to: The directory to save the trajectories to. Data is saved
            in the same format as the parsed replay dataset.
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
            # TODO: need to re-check these settings for online RL
            ping_interval=None,
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
        self.metamon_obs_space.reset()
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
                    # NOTE: the replay parser leaves a blank (-1) final action; matched here
                    "actions": self.trajectory["actions"] + [-1],
                }
                # conservative file writing to avoid partial writes on shutdown or interruption
                # when launching multiple environments in parallel
                path = os.path.join(self.save_trajectories_to, filename)
                temp_path = path + ".tmp"
                with lz4.frame.open(temp_path, "wb") as f:
                    f.write(json.dumps(output_json).encode("utf-8"))
                os.rename(temp_path, path)

        return next_state, reward, terminated, truncated, info

    def take_long_break(self):
        self.close(purge=False)
        self.reset_battles()

    def resume_from_break(self):
        self.start_challenging()


class BattleAgainstBaseline(PokeEnvWrapper):
    """
    Battle against a specified opponent.

    Can be used to battle any opponent that connects via the poke-env interface
    (e.g., any baseline in the `baselines` module, or any other custom `Player`)

    Assumes the player and opponent are both sampling from the same set of team files.
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
        print(f"Laddering for {num_battles} battles")
        self.start_laddering(n_challenges=num_battles)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)
        self.render()
        return next_state, reward, terminated, truncated, info


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.baselines.heuristic.basic import GymLeader
    from metamon.interface import TokenizedObservationSpace, DefaultPlusObservationSpace
    from metamon.tokenizer import get_tokenizer

    parser = ArgumentParser()
    parser.add_argument("--battle_format", type=str, default="gen1ou")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--team_set", type=str, default="paper_replays")
    args = parser.parse_args()

    env = BattleAgainstBaseline(
        battle_format=args.battle_format,
        team_set=get_metamon_teams(args.battle_format, args.team_set),
        opponent_type=GymLeader,
        observation_space=TokenizedObservationSpace(
            DefaultPlusObservationSpace(),
            tokenizer=get_tokenizer("DefaultObservationSpace-v0"),
        ),
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
