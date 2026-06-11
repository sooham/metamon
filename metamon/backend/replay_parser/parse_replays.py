import multiprocessing
import orjson
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Optional

import tqdm
import termcolor
import lz4.frame

from metamon import interface
from metamon.backend.replay_parser import backward, forward, checks
from metamon.backend.replay_parser.exceptions import (
    BackwardException,
    CustomRulesException,
    ForwardException,
    InvalidActionIndex,
    WarningFlags,
)
from metamon.backend.replay_parser.replay_state import (
    Action,
    ReplayState,
)
from metamon.backend.replay_parser.pe_datatypes import (
    PEStatus,
)
from metamon.backend.team_prediction.predictor import TeamPredictor, NaiveUsagePredictor


class ReplayParser:
    def __init__(
        self,
        replay_output_dir: Optional[str] = None,
        team_output_dir: Optional[str] = None,
        verbose: bool = False,
        sleep_on_handled_exception: int = 0.1,
        team_predictor: Optional[TeamPredictor] = None,
        compress: bool = True,
        pretty: bool = False,
    ):
        self.output_dir = replay_output_dir
        self.team_output_dir = team_output_dir
        self.verbose = verbose
        self.sleep_on_handled_exception = sleep_on_handled_exception
        self.error_history = {"Forward": {}, "Backward": {}}
        self.team_predictor = team_predictor or NaiveUsagePredictor()
        self.compress = compress
        self.pretty = pretty

    def summarize_errors(self):
        return {
            forw_back: {err: len(paths) for err, paths in records.items()}
            for forw_back, records in self.error_history.items()
        }

    @staticmethod
    def clean_log(raw_replay_json):
        """
        Nice cleaning function which turns ugly one line battle logs 
        i.e |player|p1|mist98895|209\n|player|p2|typhlosion10919|#typhlosion10919\n|game ... 
        into a list of list structure 
        i.e [["player", "p1", "mist98895", "209"], ...] 
        """
        log = [
            [x.strip() for x in line.split("|")[1:]]
            for line in raw_replay_json["log"].split("\n")
            if line.replace("|", "").strip() != ""
        ]
        return log

    @staticmethod
    def _detect_capture_format(log):
        """Detect capture/tournament formats where |poke| messages appear
        after |start|, meaning Pokemon are added mid-battle (e.g. defeated
        Pokemon are captured by the opponent).  Normal battles only emit
        |poke| during team preview, before |start|."""
        started = False
        for line in log:
            if not line:
                continue
            if line[0] == "start":
                started = True
            elif started and line[0] == "poke":
                return True
        return False

    def povreplay_to_state_action(self, replay: backward.POVReplay):
        # TODO for future reference: here is where we start intentionally
        # dropping the doubles format. most but not all of the code before
        # this should work with doubles (in theory... no replays scraped to test)
        p1 = replay.from_p1_pov
        states, actions = [], []
        for turn, slot_actions in zip(replay.povturnlist, replay.actionlist):
            # flip the observation around
            action = slot_actions[0]
            active_mon = (turn.active_pokemon_1 if p1 else turn.active_pokemon_2)[0]
            opponent_mon = (turn.active_pokemon_2 if p1 else turn.active_pokemon_1)[0]
            opponent_team = turn.pokemon_2 if p1 else turn.pokemon_1
            player_team = turn.pokemon_1 if p1 else turn.pokemon_2
            if action and action.is_revival:
                switches = [
                    p
                    for p in player_team
                    if p.status == PEStatus.FNT and p != active_mon
                ]
            else:
                switches = (
                    turn.available_switches_1 if p1 else turn.available_switches_2
                )
            player_conditions = turn.conditions_1 if p1 else turn.conditions_2
            opponent_conditions = turn.conditions_2 if p1 else turn.conditions_1
            can_tera = turn.can_tera_1 if p1 else turn.can_tera_2
            opponent_teampreview = turn.teampreview_2 if p1 else turn.teampreview_1

            # fill a ReplayState
            states.append(
                ReplayState(
                    format=replay.format,
                    force_switch=turn.is_force_switch,
                    active_pokemon=active_mon,
                    opponent_active_pokemon=opponent_mon,
                    opponent_team=opponent_team,
                    player_team=player_team,
                    available_switches=switches,
                    player_prev_move=active_mon.last_used_move,
                    opponent_prev_move=opponent_mon.last_used_move,
                    player_conditions=player_conditions,
                    opponent_conditions=opponent_conditions,
                    weather=turn.weather,
                    battle_field=turn.battle_field,
                    battle_won=False,
                    battle_lost=False,
                    can_tera=can_tera,
                    opponent_teampreview=opponent_teampreview,
                )
            )
            actions.append(action)

        states[-1].battle_won = replay.winner
        states[-1].battle_lost = not replay.winner

        return states, actions

    def state_action_to_obs_action_reward(
        self, states: list[ReplayState], actions: list[Action],
        allow_missing: bool = False,
    ):
        universal_states = []
        action_idxs = []

        if self.verbose:
            print()
        for state, action in zip(states, actions):
            universal_state = interface.UniversalState.from_ReplayState(state)
            universal_action = interface.UniversalAction.from_ReplayAction(
                state=state, action=action
            )
            if universal_action is None:
                if allow_missing:
                    action_idxs.append(-1)
                    universal_states.append(universal_state)
                    continue
                raise InvalidActionIndex(state, action)
            if self.verbose:
                print(
                    f"forced: {universal_state.forced_switch}; {universal_state.player_active_pokemon.name} {universal_state.player_active_pokemon.status} vs. {universal_state.opponent_active_pokemon.name} {universal_state.opponent_active_pokemon.status}; {action} --> {universal_action.action_idx}"
                )
            action_idxs.append(universal_action.action_idx)
            universal_states.append(universal_state)

        return universal_states, action_idxs

    def povreplay_to_seq(self, replay: backward.POVReplay):
        states, actions = self.povreplay_to_state_action(replay)
        has_zoroark = WarningFlags.ZOROARK in replay.check_warnings
        universal_states, action_idxs = self.state_action_to_obs_action_reward(
            states, actions, allow_missing=has_zoroark,
        )
        if not has_zoroark:
            checks.check_action_idxs(universal_states, actions, action_idxs, gen=replay.gen)
        return universal_states, action_idxs

    def save_to_disk(
        self,
        replay: backward.POVReplay,
        time_played: datetime,
        player_username: str,
        opponenent_username: str,
    ):
        universal_states, action_idxs = self.povreplay_to_seq(replay)
        won = "WIN" if replay.winner else "LOSS"
        filename = f"{replay.gameid}_{replay.rating}_{player_username}_vs_{opponenent_username}_{time_played.strftime('%m-%d-%Y')}_{won}"
        if self.output_dir is not None:
            path = self.output_dir
            os.makedirs(path, exist_ok=True)
            output_json = {
                "states": [state.to_dict() for state in universal_states],
                "actions": action_idxs,
            }
            if self.pretty:
                payload = json.dumps(output_json, indent=2, ensure_ascii=False).encode("utf-8")
            else:
                payload = orjson.dumps(output_json)
            if self.compress:
                with lz4.frame.open(os.path.join(path, f"{filename}.json.lz4"), "wb") as f:
                    f.write(payload)
            else:
                with open(os.path.join(path, f"{filename}.json"), "wb") as f:
                    f.write(payload)

        if self.team_output_dir is not None:
            path = self.team_output_dir
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, f"{filename}.{replay.format}_team"), "w") as f:
                f.write(replay.revealed_team.to_str())

    def add_exception_to_history(self, e, path):
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

    def parse_parallel(self, file_paths: list[str], pool_size: int = 8):
        pool = multiprocessing.Pool(pool_size)
        # Write the progress bar to stdout so it doesn't compete with
        # warnings.warn() output from worker processes (which goes to stderr).
        for _ in tqdm.tqdm(
            pool.imap_unordered(self.parse_replay, file_paths),
            total=len(file_paths),
            file=sys.stdout,
        ):
            pass
        pool.close()
        pool.join()

    def parse_replay(self, path: str):
        # read replay data from disk
        gameid = os.path.basename(path).replace(".json", "")
        with open(path, "r") as f:
            try:
                data = orjson.loads(f.read())
            except orjson.JSONDecodeError as e:
                warnings.warn(
                    f"Skipping replay {gameid} "
                    f"({path}) due to known exception: {e}."
                )
                return

        # prepare data
        p1_username, p2_username = data["players"]
        time_played = datetime.fromtimestamp(int(data["uploadtime"]))
        # Some raw replays have formatid="MISSING" (malformed upload).
        # Fall back to the 'format' field, then to parsing gen from the log.
        formatid = data.get("formatid", "")
        if formatid == "MISSING" or not formatid:
            formatid = data.get("format", "")
        if not formatid or formatid == "MISSING":
            # Last resort: extract gen from the |gen|N line in the log
            import re
            m = re.search(r"\|gen\|(\d+)", data.get("log", ""))
            formatid = f"gen{m.group(1)}ou" if m else "MISSING"
        replay = forward.ParsedReplay(
            gameid=os.path.basename(path).replace(".json", ""),
            format=formatid,
            time_played=time_played,
        )
        log = self.clean_log(data)

        try:
            # Skip capture-format tournaments (Pokemon are added mid-battle).
            if self._detect_capture_format(log):
                raise CustomRulesException(
                    "Capture format detected (|poke| messages after |start|)"
                )

            # forward fill
            replay = forward.forward_fill(replay, log, verbose=self.verbose)

            # backward fill
            replay_from_p1, replay_from_p2 = backward.backward_fill(
                replay,
                team_predictor=self.team_predictor,
            )
            # save
            self.save_to_disk(
                replay_from_p1,
                time_played=time_played,
                player_username=p1_username,
                opponenent_username=p2_username,
            )
            self.save_to_disk(
                replay_from_p2,
                time_played=time_played,
                player_username=p2_username,
                opponenent_username=p1_username,
            )

        except (ForwardException, BackwardException) as e:
            self.add_exception_to_history(e, path)
            warning_str = f"{replay.gameid} ({path}):\n\t{e}"
            for check_warning in replay.check_warnings:
                warning_str += f"\n\t{termcolor.colored(f'Note: this replay has a {check_warning.value} warning flag, which may explain the above message.', 'yellow')}"
            warnings.warn(warning_str)
