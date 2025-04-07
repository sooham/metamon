import time
import random
import os
import gc
import string
import uuid
from typing import Optional, Type

import numpy as np
import gymnasium as gym
from poke_env import (
    AccountConfiguration,
    ShowdownServerConfiguration,
    LocalhostServerConfiguration,
)
from poke_env.environment import Battle, Status
from poke_env.player import OpenAIGymEnv, Player


from metamon.interface import (
    UniversalState,
    action_idx_to_battle_order,
    RewardFunction,
    DefaultShapedReward,
)
from metamon.task_distributions import (
    TaskDistribution,
    UniformRandomTeambuilder,
    FixedGenOpponentDistribution,
)
from metamon.data.tokenizer import get_tokenizer
from metamon.data import DATA_PATH


class _ShowdownEnv(OpenAIGymEnv):
    def __init__(
        self, opponent: Player, reward_function: RewardFunction, *args, **kwargs
    ):
        self._current_baseline_opponent = opponent
        self.reward_function = reward_function
        super().__init__(*args, **kwargs)

    def action_space_size(self):
        return 9

    def on_invalid_order(self, battle: Battle):
        return self.choose_random_move(battle)

    def reset(self, *args, **kwargs):
        self.invalid_action_counter = 0
        self.valid_action_counter = 0
        return super().reset(*args, **kwargs)

    def action_to_move(self, action: int, battle: Battle):
        order = action_idx_to_battle_order(battle, action)
        if order is None:
            self.invalid_action_counter += 1
            return self.on_invalid_order(battle)
        else:
            self.valid_action_counter += 1
            return order

    def set_opponent(self, opponent: Player):
        self._current_baseline_opponent = opponent

    def get_opponent(self) -> Player:
        return self._current_baseline_opponent

    def describe_embedding(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                "numbers": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(48,),
                    dtype=np.float32,
                ),
                "text": gym.spaces.Text(
                    max_length=900,
                    min_length=800,
                    charset=set(string.ascii_lowercase)
                    | set(str(n) for n in range(0, 10))
                    | {"<", ">"},
                ),
            }
        )

    def calc_reward(self, last_battle: Battle, current_battle: Battle) -> float:
        last_state = UniversalState.from_Battle(last_battle)
        state = UniversalState.from_Battle(current_battle)
        reward = self.reward_function(last_state, state)
        return reward

    def embed_battle(self, battle: Battle):
        universal_state = UniversalState.from_Battle(battle)
        return universal_state.to_numpy()


class MetaShowdown(gym.Env):
    def __init__(
        self,
        task_distribution: TaskDistribution,
        test_set: bool = False,
        new_task_every: int = 50,
        turn_limit: int = 200,
    ):
        self.task_distribution = task_distribution
        self.current_task = self.task_distribution.generate_task(test_set=test_set)
        self.inner_env = None
        self.battle_format = None
        self.test_set = test_set
        self.opponent = None
        self.reset_counter = 0
        self.num_won_battles = 0
        self.new_task_every = new_task_every
        self.action_space = gym.spaces.Discrete(9)
        self.turn_limit = turn_limit
        self.observation_space = gym.spaces.Dict(
            {
                "numbers": gym.spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(48,),
                    dtype=np.float32,
                ),
                "text": gym.spaces.Text(
                    max_length=900,
                    min_length=800,
                    charset=set(string.ascii_lowercase)
                    | set(str(n) for n in range(0, 10))
                    | {"<", ">"},
                ),
                "meta": gym.spaces.Box(
                    low=0.0, high=float("inf"), shape=(2,), dtype=np.float32
                ),
            }
        )

    def new_task(self):
        if isinstance(self.task_distribution, FixedGenOpponentDistribution):
            return self.new_task_quick_reset()
        else:
            return self.new_task_full_reset()

    def new_task_quick_reset(self):
        assert (
            self.new_task_every == 1
        ), "The fast reset distributions will switch tasks every reset whether this is intended or not..."
        if self.inner_env is None:
            self.current_task = self.task_distribution.generate_task(
                test_set=self.test_set
            )
            self.battle_format = self.current_task.battle_format
            self.opponent = self.current_task.opponent_type(
                battle_format=self.current_task.battle_format,
                server_configuration=self.task_distribution.opp_server_config,
                team=self.current_task.opponent_teambuilder,
                account_configuration=AccountConfiguration(
                    username=f"o{str(uuid.uuid4())[:15]}", password=None
                ),
                ping_timeout=None,
            )
            self.inner_env = _ShowdownEnv(
                battle_format=self.current_task.battle_format,
                server_configuration=self.task_distribution.opp_server_config,
                account_configuration=AccountConfiguration(
                    username=f"p{str(uuid.uuid4())[:15]}", password=None
                ),
                opponent=self.opponent,
                reward_function=self.current_task.reward_function,
                start_challenging=True,
                team=self.current_task.player_teambuilder,
                ping_timeout=None,
            )
            self._baseline_win_count = self.inner_env.agent.n_won_battles

    def new_task_full_reset(self):
        self.current_task = self.task_distribution.generate_task(test_set=self.test_set)
        if self.opponent is not None:
            self.finish_all_battles(self.opponent)
        self.opponent = self.current_task.opponent_type(
            battle_format=self.current_task.battle_format,
            server_configuration=self.task_distribution.opp_server_config,
            team=self.current_task.opponent_teambuilder,
            account_configuration=AccountConfiguration(
                username=f"o{str(uuid.uuid4())[:15]}", password=None
            ),
            ping_timeout=None,
        )

        if self.inner_env is not None:
            self.inner_env.close()
            self.finish_all_battles(self.inner_env)
        self.inner_env = _ShowdownEnv(
            battle_format=self.current_task.battle_format,
            server_configuration=self.task_distribution.opp_server_config,
            account_configuration=AccountConfiguration(
                username=f"p{str(uuid.uuid4())[:15]}", password=None
            ),
            opponent=self.opponent,
            reward_function=self.current_task.reward_function,
            start_challenging=True,
            team=self.current_task.player_teambuilder,
            ping_timeout=None,
        )
        self.battle_format = self.current_task.battle_format
        # trying hard to keep track of wins in case of crashes or some other
        # swap of the inner PS/poke-env env.
        self._baseline_win_count = self.inner_env.agent.n_won_battles

    def close(self, *args, **kwargs):
        if self.inner_env is not None:
            self.inner_env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.inner_env.render(*args, **kwargs)

    def finish_all_battles(self, player: Player):
        del player
        gc.collect()

    def reset(self, *args, **kwargs):
        if self.reset_counter % self.new_task_every == 0:
            self.new_task()
        self.reset_counter += 1
        self.current_ep = 0
        self._reset_last = False
        self.win_loss_history = []
        obs, info = self.inner_env.reset()
        self.turn_counter = 0
        return self.add_meta_info(obs), info

    def add_meta_info(self, obs):
        obs["meta"] = np.array(
            [self.current_task.k_shots, self.current_ep], dtype=np.float32
        )
        return obs

    def step(self, action):
        if self._reset_last:
            next_state, info = self.inner_env.reset()
            self.turn_counter = 0
            reward = 0.0
            self._reset_last = False
        else:
            (
                next_state,
                reward,
                inner_terminated,
                inner_truncated,
                info,
            ) = self.inner_env.step(action)
            self.turn_counter += 1
            inner_truncated = inner_truncated or self.turn_counter > self.turn_limit
            # we will reset the env on the next `step`
            self._reset_last = inner_terminated or inner_truncated
            if self._reset_last:
                # advance episode here instead of during soft reset to avoid creating a new
                # game at the end of real server matchups
                self.current_ep += 1
                # win tracking system is overly conservative because we're worried about poke-env crashes/errors.
                battles_won = (
                    self.inner_env.agent.n_won_battles - self._baseline_win_count
                )
                assert battles_won in [1, 0]
                self.num_won_battles += battles_won
                self._baseline_win_count = self.inner_env.agent.n_won_battles
                self.win_loss_history.append(battles_won)

        terminated = self.current_ep > self.current_task.k_shots
        if terminated:
            info["win_loss_history"] = self.win_loss_history
            info["win_rate"] = sum(self.win_loss_history) / len(self.win_loss_history)
            info[
                "metamon_task_name"
            ] = f"{self.current_task.battle_format}_vs_{self.current_task.opponent_type.__name__}"
            info["valid_action_count"] = self.inner_env.valid_action_counter
            info["invalid_action_count"] = self.inner_env.invalid_action_counter
        return self.add_meta_info(next_state), reward, terminated, False, info


def check_avatar(avatar: str):
    with open(os.path.join(DATA_PATH, "avatar_names.txt"), "r") as f:
        options = [l.strip() for l in f.readlines()]
    if avatar not in options:
        raise ValueError(
            f"Avatar {avatar} is not valid. See https://play.pokemonshowdown.com/sprites/trainers/ for a list of options."
        )


class PSLadder(_ShowdownEnv):
    def __init__(
        self,
        username: str,
        password: str,
        gen: int,
        format: str,
        avatar: Optional[str] = None,
        reward_function: Optional[Type[RewardFunction]] = None,
        team_split: str = "competitive",
    ):
        if avatar is not None:
            check_avatar(avatar)
        super().__init__(
            reward_function=reward_function or DefaultShapedReward(),
            opponent=None,
            battle_format=f"gen{gen}{format.lower()}",
            server_configuration=ShowdownServerConfiguration,
            account_configuration=AccountConfiguration(
                username=username, password=password
            ),
            team=UniformRandomTeambuilder(gen=gen, format=format, split=team_split),
            start_challenging=False,
            start_timer_on_battle_start=True,
            ping_timeout=2000.0,
            avatar=avatar,
        )

    def on_invalid_order(self, battle: Battle):
        print("invalid action!")
        if battle.available_moves and random.random() < 0.75:
            # the most common case that leads to invalid action choices is PP stalls.
            return self.create_order(random.choice(battle.available_moves))
        elif battle.available_switches:
            return self.create_order(random.choice(battle.available_switches))
        return self.choose_random_move(battle)


class LocalLadder(_ShowdownEnv):
    def __init__(
        self,
        username: str,
        gen: int,
        format: str,
        team_split: str,
        avatar: Optional[str] = None,
        reward_function: Optional[Type[RewardFunction]] = None,
    ):
        if avatar is not None:
            check_avatar(avatar)
        super().__init__(
            reward_function=reward_function or DefaultShapedReward(),
            opponent=None,
            battle_format=f"gen{gen}{format.lower()}",
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(username, None),
            team=UniformRandomTeambuilder(gen=gen, format=format, split=team_split),
            start_challenging=False,
            start_timer_on_battle_start=True,
        )

    def reset(self, *args, **kwargs):
        self.invalid_action_counter = 0
        self.valid_action_counter = 0
        self.battle_reference = self.agent.n_won_battles
        return super().reset(*args, **kwargs)

    def step(self, action):
        next_state, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            info["valid_action_count"] = self.valid_action_counter
            info["invalid_action_count"] = self.invalid_action_counter
            info["win_rate"] = self.agent.n_won_battles - self.battle_reference
            self.battle_reference = self.agent.n_won_battles
            assert info["win_rate"] in [0, 1]
        self.render()
        return next_state, reward, terminated, truncated, info


class TokenizedEnv(gym.ObservationWrapper):
    def __init__(self, env: MetaShowdown, tokenizer=get_tokenizer("allreplays-v3")):
        super().__init__(env)
        self.tokenizer = tokenizer
        obs_space = {
            "tokens": gym.spaces.Box(
                low=-1, high=len(tokenizer), shape=(87,), dtype=np.int32
            ),
            "numbers": env.observation_space["numbers"],
        }
        self.add_meta = "meta" in env.observation_space
        if self.add_meta:
            obs_space["meta"] = env.observation_space["meta"]
        self.observation_space = gym.spaces.Dict(obs_space)

    def observation(self, obs):
        tokens = self.tokenizer.tokenize(obs["text"].tolist())
        obs_dict = {"tokens": tokens, "numbers": obs["numbers"]}
        if self.add_meta:
            obs_dict["meta"] = obs["meta"]
        return obs_dict

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from metamon.task_distributions import (
        get_task_distribution,
        FixedGenOpponentDistribution,
    )
    import metamon

    parser = ArgumentParser()
    parser.add_argument("--task_dist", default="Gen1OU")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--team_split", type=str, default="train")
    args = parser.parse_args()

    def make_env():
        task_dist = FixedGenOpponentDistribution(
            "gen1ou",
            opponent=metamon.baselines.heuristic.basic.GymLeader,
            player_split="train",
            opponent_split="train",
        )
        env = MetaShowdown(task_dist, new_task_every=1)
        env = TokenizedEnv(env)
        return env

    env = make_env()
    start = time.time()
    counter = 0
    for ep in range(args.episodes):
        state, info = env.reset()
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
    print(env.win_loss_history)
