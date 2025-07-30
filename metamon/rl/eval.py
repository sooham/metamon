import json
from typing import Optional, Dict, Any
from functools import partial

import metamon
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
    PretrainedModel,
)
from metamon.baselines import ALL_BASELINES
from metamon.rl.metamon_to_amago import make_baseline_env, make_ladder_env


def standard_eval(
    pretrained_model: PretrainedModel,
    eval_type: str,
    battle_format: str,
    player_team_set: metamon.env.TeamSet,
    n_challenges: int,
    checkpoint: Optional[int] = None,
    async_mp_context: str = "spawn",
    battle_backend: str = "poke-env",
    username: Optional[str] = None,
    avatar: Optional[str] = None,
    save_trajectories_to: Optional[str] = None,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    agent = pretrained_model.initialize_agent(checkpoint=checkpoint, log=log_to_wandb)
    # create envs
    env_kwargs = dict(
        battle_format=battle_format,
        player_team_set=player_team_set,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        save_trajectories_to=save_trajectories_to,
        battle_backend=battle_backend,
    )
    if eval_type == "heuristic":
        agent.async_env_mp_context = async_mp_context
        make_envs = [
            partial(make_baseline_env, **env_kwargs, opponent_type=ALL_BASELINES[o])
            for o in [
                "PokeEnvHeuristic",
                "Gen1BossAI",
                "Grunt",
                "GymLeader",
                "EmeraldKaizo",
                "RandomBaseline",
            ]
        ]
        make_envs *= 5
    elif eval_type == "il":
        agent.async_env_mp_context = async_mp_context
        make_envs = [
            partial(make_baseline_env, **env_kwargs, opponent_type=ALL_BASELINES[o])
            for o in ["BaseRNN"]
        ]
        make_envs *= 1
    elif eval_type == "ladder":
        agent.env_mode = "sync"
        make_envs = [
            partial(
                make_ladder_env,
                **env_kwargs,
                num_battles=n_challenges,
                username=username,
                avatar=avatar,
            )
        ]
        # disables AMAGO tqdm because we'll be rendering the poke-env battle bar
        agent.verbose = False
    else:
        raise ValueError(f"Invalid eval_type: {eval_type}")

    # evaluate
    agent.parallel_actors = len(make_envs)
    results = agent.evaluate_test(
        make_envs,
        # sets upper bound on total timesteps
        timesteps=n_challenges * 250,
        # terminates after n_challenges
        episodes=n_challenges,
    )
    return results


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--agent",
        required=True,
        choices=get_pretrained_model_names(),
        help="Choose a pretrained model to evaluate.",
    )
    parser.add_argument(
        "--gens",
        type=int,
        nargs="+",
        default=1,
        help="Specify the Pokémon generations to evaluate.",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default="ou",
        choices=["ubers", "ou", "uu", "nu"],
        help="Specify the battle format/tier.",
    )
    parser.add_argument(
        "--username",
        default="Metamon",
        help="Username for the Showdown server.",
    )
    parser.add_argument(
        "--n_challenges",
        type=int,
        default=10,
        help=(
            "Number of battles to run before returning eval stats. "
            "Note this is the total sample size across all parallel actors."
        ),
    )
    parser.add_argument(
        "--avatar",
        default="red-gen1main",
        help="Avatar to use for the battles.",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[None],
        help="Checkpoints to evaluate.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["heuristic", "il", "ladder"],
        help=(
            "Type of evaluation to perform. 'heuristic' will run against 6 "
            "heuristic baselines, 'il' will run against a BCRNN baseline, "
            "'ladder' will queue the agent for battles on your self-hosted Showdown ladder."
        ),
    )
    parser.add_argument(
        "--team_set",
        default="competitive",
        choices=["competitive", "paper_variety", "paper_replays", "modern_replays"],
        help="Team Set.",
    )
    parser.add_argument(
        "--save_trajectories_to",
        default=None,
        help="Save replays (in the parsed replay format) to a directory.",
    )
    parser.add_argument(
        "--wait_for_input",
        action="store_true",
        help="Wait for user input before starting.",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help=(
            "Method for interpreting Showdown's requests and simulator messages. "
            "poke-env is the default. metamon is an experimental option that aims to "
            "remove sim2sim gap by reusing the code that generates our huggingface "
            "replay dataset."
        ),
    )
    parser.add_argument(
        "--heuristic_async_mp_context",
        type=str,
        default="spawn",
        help="Async environment setup method. Does not apply to `--eval_type ladder`. Try 'forkserver' or 'fork' if you run into issues!",
    )
    args = parser.parse_args()

    pretrained_model = get_pretrained_model(args.agent)()
    for gen in args.gens:
        for format in args.formats:
            battle_format = f"gen{gen}{format.lower()}"
            player_team_set = metamon.env.get_metamon_teams(
                battle_format, args.team_set
            )
            for checkpoint in args.checkpoints:
                results = standard_eval(
                    pretrained_model=pretrained_model,
                    eval_type=args.eval_type,
                    battle_format=battle_format,
                    player_team_set=player_team_set,
                    n_challenges=args.n_challenges,
                    checkpoint=checkpoint,
                    async_mp_context=args.heuristic_async_mp_context,
                    battle_backend=args.battle_backend,
                    username=args.username,
                    avatar=args.avatar,
                    save_trajectories_to=args.save_trajectories_to,
                    log_to_wandb=args.log_to_wandb,
                )
                print(json.dumps(results, indent=4, sort_keys=True))
