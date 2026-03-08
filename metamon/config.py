import os

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
