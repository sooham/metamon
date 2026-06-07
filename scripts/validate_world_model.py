#!/usr/bin/env python3
"""Validate parsed replays against WorldModelObservationSpace format.

Usage:
    uv run python scripts/validate_world_model.py <parsed_replay.json>...
"""
import sys
import orjson
from metamon.interface import UniversalState, WorldModelObservationSpace


def validate_file(path: str) -> None:
    with open(path, "r") as f:
        data = orjson.loads(f.read())

    obs_space = WorldModelObservationSpace()
    obs_space.reset()

    for i, s in enumerate(data["states"]):
        us = UniversalState.from_dict(s)
        obs = obs_space.state_to_obs(us)
        tokens = obs["text"].tolist().split(" ")
        assert len(tokens) <= 336, f"{path} state {i}: {len(tokens)} tokens (exceeds soft max 336)"
        assert obs["numbers"].shape == (63,), f"{path} state {i}: shape {obs['numbers'].shape} (expected (63,))"
        assert len(us.opponent_bench) <= 5
        assert len(us.fainted_pokemon) <= 5
        assert len(us.opponent_fainted) <= 5

    print(f"  OK: {path} ({len(data['states'])} states)")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        validate_file(path)
    print("All validations passed.")
