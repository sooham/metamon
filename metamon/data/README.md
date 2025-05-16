# Metamon Data

> [!WARNING]
> 
> This module contains the behind-the-scenes work that makes the `ParsedReplayDataset` possible. Very little of this is intended to be used externally. Please consider yourself warned :)


<p align="center">
  <img src="../../media/replay_parser_warning.png">
</p>


`replay_dataset/raw_replays/` contains code for importing, anonymizing, uploading/downloading Pokemon Showdown replay `json` files.

`replay_dataset/parsed_replays/` contains the replay parser that converts Showdown replays to agent data. The replay parser has [its own README](replay_dataset/parsed_replays/README.md).


`legacy_team_builder/` is the original (paper) logic for procedurally generating (or filling in missing details of) Pokémon teams. It is still used to access Smogon usage statistics.

`team_prediction/` is the new system for predicting full teams from partially revealed teams in replays. It contains a lot of dead-end code for model-based team prediction --- which was almost completely set up before I decided we have too many replays to justify the complexity in Gen 1-4 Singles (for now). Instead, we are now defaulting to an improved prediction strategy based on the revealed teams of all historical replays.


The full replay process uses the scripts in an order like this:

1. `raw_replays.find_replay_links` uses the PS API to find battle ids played in a given date range.
2. `raw_replays.replay_intake` merges/deduplicates directories of downloaded PS replays into an existing set.
3. `raw_replays.usernames` maintains a consistent mapping from real PS usernames to "anonymous" usernames. We are trying to be polite to the players and remove NSFW usernames from the public huggingface dataset, but real usernames are easy to recover. This isn't a serious privacy issue --- we are working with screen names from public battles.
4. `raw_replays.anonymize` uses a username mapping to "anonymize" replays and removes player chat logs from replay files.
5. `raw_replays.upload_to_hf` pushes the anonymized replay directory to hugginface as a structured dataset.
6. `raw_replays.download_from_hf` skips prevoius steps. It downloads the current version of the raw replay dataset and puts it back in the dir structure expected by the rest of the code.
7. `raw_replays.download_from_hf` skips all previous steps and downloads our latest version of the parsed replay set from hugginface.
8. `parsed_replays.replay_parser` converts PS replays to training data using `team_prediction` as a key step.



