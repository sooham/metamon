import os
from importlib.metadata import version

__version__ = "1.2.0"

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
