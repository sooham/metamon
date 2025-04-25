import glob
import random
import tqdm
import multiprocessing
import json
import os
import warnings
from datetime import datetime

import numpy as np
from metamon import interface
from metamon.data.replay_dataset.parsed_replays.replay_parser import backward, forward
from metamon.data.replay_dataset.parsed_replays.replay_parser.exceptions import (
    BackwardException,
    ForwardException,
    InvalidActionIndex,
    ToNumpyError,
)
from metamon.data.replay_dataset.parsed_replays.replay_parser.replay_state import Action, ReplayState


class ReplayParser:
    def __init__(
        self,
        output_dir: str,
        train_test_split: float = 0.8,
        verbose: bool = False,
        sleep_on_handled_exception: int = 0.1,
        reward_function: interface.RewardFunction = interface.DefaultShapedReward(),
        observation_space: interface.ObservationSpace = interface.DefaultObservationSpace(),
    ):
        self.output_dir = output_dir
        self.verbose = verbose
        assert 0.0 <= train_test_split <= 1.0
        self.train_test_split = train_test_split
        self.sleep_on_handled_exception = sleep_on_handled_exception
        self.error_history = {"Forward": {}, "Backward": {}}
        self.reward_function = reward_function
        self.observation_space = observation_space

    def summarize_errors(self):
        return {
            forw_back: {err: len(paths) for err, paths in records.items()}
            for forw_back, records in self.error_history.items()
        }

    @staticmethod
    def clean_log(raw_replay_json):
        log = [
            [x.strip() for x in line.split("|")[1:]]
            for line in raw_replay_json["log"].split("\n")
            if line.replace("|", "").strip() != ""
        ]
        return log

    def povreplay_to_state_action(self, replay: backward.POVReplay):
        # TODO for future reference: here is where we start intentionally
        # dropping the doubles format. most but not all of the code before
        # this should work with doubles (in theory... no replays scraped to test)
        p1 = replay.from_p1_pov
        states, actions = [], []
        for turn, action in zip(replay.povturnlist, replay.actionlist):
            # flip the observation around
            active_mon = (turn.active_pokemon_1 if p1 else turn.active_pokemon_2)[0]
            opponent_mon = (turn.active_pokemon_2 if p1 else turn.active_pokemon_1)[0]
            opponent_team = turn.pokemon_2 if p1 else turn.pokemon_1
            switches = turn.available_switches_1 if p1 else turn.available_switches_2
            player_conditions = turn.conditions_1 if p1 else turn.conditions_2
            opponent_conditions = turn.conditions_2 if p1 else turn.conditions_1

            # fill a ReplayState
            states.append(
                ReplayState(
                    format=replay.format,
                    force_switch=turn.is_force_switch,
                    active_pokemon=active_mon,
                    opponent_active_pokemon=opponent_mon,
                    opponent_team=opponent_team,
                    available_switches=switches,
                    player_prev_move=active_mon.last_used_move,
                    opponent_prev_move=opponent_mon.last_used_move,
                    player_conditions=player_conditions,
                    opponent_conditions=opponent_conditions,
                    weather=turn.weather,
                    battle_field=turn.battle_field,
                    battle_won=False,
                    battle_lost=False,
                )
            )
            actions.append(action[0])

        states[-1].battle_won = replay.winner
        states[-1].battle_lost = not replay.winner

        return states, actions

    def state_action_to_obs_action_reward(
        self, states: list[ReplayState], actions: list[Action]
    ):
        universal_states = []
        obs_seq = {key: [] for key in self.observation_space.gym_space.keys()}
        action_idxs = []

        for state, action in zip(states, actions):
            universal_state = interface.UniversalState.from_ReplayState(state)
            action_idx = interface.replaystate_action_to_idx(state, action)
            obs = self.observation_space.state_to_obs(universal_state)
            for obs_key, obs_val in obs.items():
                obs_seq[obs_key].append(obs_val)
            if action_idx is None:
                raise InvalidActionIndex(obs, action)
            action_idxs.append(action_idx)
            universal_states.append(universal_state)

        rewards = [0.0]
        for prev_state, state in zip(universal_states, universal_states[1:]):
            rewards.append(self.reward_function(prev_state, state))

        obs_seq = {key: np.array(val) for key, val in obs_seq.items()}
        action_idxs = np.array(action_idxs, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        return obs_seq, action_idxs, rewards

    def povreplay_to_seq(self, replay: backward.POVReplay):
        states, actions = self.povreplay_to_state_action(replay)
        obs, action_idxs, rewards = self.state_action_to_obs_action_reward(
            states, actions
        )
        return obs, action_idxs, rewards

    def save_to_disk(
        self,
        replay: backward.POVReplay,
        to_train_set: bool,
        time_played: datetime,
        player_username: str,
        opponenent_username: str,
    ):
        obs_seq, actions, rewards = self.povreplay_to_seq(replay)
        if self.output_dir is not None:
            won = "WIN" if replay.winner else "LOSS"
            filename = f"{replay.gameid}_{replay.rating}_{player_username}_vs_{opponenent_username}_{time_played.strftime('%m-%d-%Y')}_{won}.npz"
            split = "train" if to_train_set else "val"
            path = os.path.join(self.output_dir, split)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, filename), "wb") as f:
                np.savez_compressed(
                    f,
                    **obs_seq,
                    actions=actions,
                    rewards=rewards,
                )

    def add_exception_to_history(self, e):
        if isinstance(e, ForwardException):
            e_dict = self.error_history["Forward"]
        elif isinstance(e, BackwardException):
            e_dict = self.error_history["Backward"]
        else:
            raise e
        err_key = type(e).__name__
        if err_key in e_dict:
            e_dict[err_key].append(path)
        else:
            e_dict[err_key] = [path]

    def parse_parallel(self, paths: list[str], pool_size: int = 8):
        pool = multiprocessing.Pool(pool_size)
        for _ in tqdm.tqdm(
            pool.imap_unordered(self.parse_replay, paths), total=len(filenames)
        ):
            pass

    def parse_replay(self, path: str):
        # read replay data from disk
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                warnings.warn(f"Skipping replay {path} due to known exception: {e}.")
                return

        # prepare data
        p1_username, p2_username = data["players"]
        time_played = datetime.fromtimestamp(int(data["uploadtime"]))
        replay = forward.ParsedReplay(
            gameid="-".join(path.split("-")[-2:]).replace(".json", ""),
            format=data["format"],
            time_played=time_played,
        )
        log = self.clean_log(data)

        try:
            # forward fill
            replay = forward.forward_fill(replay, log, verbose=self.verbose)

            # backward fill
            replay_from_p1, replay_from_p2 = backward.backward_fill(replay)

            # save as IL/RL experience
            to_train_set = random.random() < self.train_test_split
            self.save_to_disk(
                replay_from_p1,
                to_train_set,
                time_played=time_played,
                player_username=p1_username,
                opponenent_username=p2_username,
            )
            self.save_to_disk(
                replay_from_p2,
                to_train_set,
                time_played=time_played,
                player_username=p2_username,
                opponenent_username=p1_username,
            )

        except (ForwardException, BackwardException) as e:
            self.add_exception_to_history(e)
            warnings.warn(f"Skipping replay {path} due to known exception: {e}.")


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
