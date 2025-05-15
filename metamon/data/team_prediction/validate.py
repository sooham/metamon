import subprocess
from typing import List, Tuple
import gc
import argparse
import os
import tqdm

from metamon.data.team_prediction.team import TeamSet
from poke_env.teambuilder import ConstantTeambuilder
from metamon.env import BattleAgainstBaseline
from metamon.baselines.heuristic.basic import RandomBaseline
from metamon.interface import (
    TokenizedObservationSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
)
from metamon.tokenizer import get_tokenizer


def validate_showdown_team(
    team_str: str,
    format_id: str = "gen1ou",
    cmd: List[str] = ["npx", "pokemon-showdown", "validate-team"],
) -> Tuple[bool, List[str]]:
    full_cmd = cmd + [format_id]

    proc = subprocess.run(full_cmd, input=team_str, text=True, capture_output=True)

    if proc.returncode == 0:
        return True
    else:
        # output = proc.stdout.strip().splitlines() + proc.stderr.strip().splitlines()
        # print(output)
        return False


def env_verify_team(team_str: str, format_id: str = "gen1ou") -> bool:
    team_set = ConstantTeambuilder(team_str)
    obs_space = TokenizedObservationSpace(
        base_obs_space=DefaultObservationSpace(),
        tokenizer=get_tokenizer("DefaultObservationSpace-v0"),
    )
    reward_fn = DefaultShapedReward()
    env = BattleAgainstBaseline(
        battle_format=format_id,
        team_set=team_set,
        opponent_type=RandomBaseline,
        observation_space=obs_space,
        reward_function=reward_fn,
    )
    env._INIT_RETRIES = 2
    env._TIME_BETWEEN_RETRIES = 0.05
    try:
        env.reset()
        env.step(env.action_space.sample())
    except Exception as e:
        del env
        gc.collect()
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and rewrite Pokemon Showdown teams."
    )
    parser.add_argument(
        "format", type=str, help="The format to process (e.g. gen1ou, gen4uu)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/mnt/nfs_client/jake/metamon_hf_staging/teams/modern_replays_cleaned",
        help="Path to input directory containing team files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/mnt/nfs_client/jake/metamon_hf_staging/teams/modern_replays_cleaned_verified",
        help="Path to output directory for verified teams",
    )

    args = parser.parse_args()
    print(f"Processing format: {args.format}")

    path = os.path.join(args.input_path, args.format)
    if os.path.isdir(path):
        for file in tqdm.tqdm(os.listdir(path)):
            if file.endswith("team"):
                filename = os.path.join(path, file)
                format = path.split("/")[-1]
                try:
                    team = TeamSet.from_showdown_file(filename, format=format)
                    team_str = team.to_str()
                except Exception as e:
                    print(team_str)
                    print(e)
                    continue

                if not env_verify_team(team_str, format):
                    print(team_str)
                    continue

                os.makedirs(os.path.join(args.output_path, format), exist_ok=True)
                with open(os.path.join(args.output_path, format, file), "w") as f:
                    f.write(team_str)
