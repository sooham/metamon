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

from metamon.backend.replay_parser import backward, forward
from metamon.backend.replay_parser.backward import POVReplayDoubles
from metamon.backend.replay_parser.exceptions import (
    BackwardException,
    CustomRulesException,
    ForwardException,
)
from metamon.backend.replay_parser.text_serializer import (
    serialize_pov_replay,
    serialize_pov_replay_doubles,
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

    def save_to_disk(
        self,
        replay: backward.POVReplay,
        time_played: datetime,
        player_username: str,
        opponenent_username: str,
    ):
        won = "WIN" if replay.winner else "LOSS"
        filename = f"{replay.gameid}_{replay.rating}_{player_username}_vs_{opponenent_username}_{time_played.strftime('%m-%d-%Y')}_{won}"
        if self.output_dir is not None:
            path = self.output_dir
            os.makedirs(path, exist_ok=True)
            if isinstance(replay, POVReplayDoubles):
                text_output = serialize_pov_replay_doubles(replay)
            else:
                text_output = serialize_pov_replay(replay)
            with open(os.path.join(path, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write(text_output)

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

            # backward fill — use doubles variant when gametype is doubles
            is_doubles = any(
                msg and msg[0] == "gametype" and len(msg) > 1 and msg[1] == "doubles"
                for msg in log
            )
            if is_doubles:
                replay_from_p1, replay_from_p2 = backward.backward_fill_doubles(
                    replay,
                    team_predictor=self.team_predictor,
                )
            else:
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
