import os

DATA_PATH = os.path.dirname(__file__)

from .parsed_replay_dset import ParsedReplayDataset, SelfPlayDataset
from . import raw_replay_util
