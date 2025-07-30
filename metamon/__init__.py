import os
from importlib.metadata import version

__version__ = "1.3.1"

poke_env_version = version("poke-env")

if not os.environ.get("METAMON_ALLOW_ANY_POKE_ENV"):
    if poke_env_version != "0.8.3.2":
        raise ImportError(
            f"poke-env version {poke_env_version} is not officially supported.\n"
            f"Please install version '0.8.3.2', found here: https://github.com/UT-Austin-RPL/poke-env).\n"
            f"This error is here to prevent silent bugs. If you are sure you want to use a\n"
            f"different version of poke-env, set the METAMON_ALLOW_ANY_POKE_ENV environment\n"
            f"variable to True."
        )

SUPPORTED_BATTLE_FORMATS = [
    "gen1ou",
    "gen1uu",
    "gen1nu",
    "gen1ubers",
    "gen2ou",
    "gen2uu",
    "gen2nu",
    "gen2ubers",
    "gen3ou",
    "gen3uu",
    "gen3nu",
    "gen3ubers",
    "gen4ou",
    "gen4uu",
    "gen4nu",
    "gen4ubers",
    "gen9ou",
]

METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", None)

from . import data
from . import backend
from . import env
from . import il
from . import rl
from . import baselines
