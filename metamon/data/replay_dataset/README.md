# Metamon Replay Dataset


<p align="center">
  <img src="../../../media/replay_parser_warning.png">
</p>


`raw_replays/` contains code for importing, anonymizing, uploading/downloading Pokemon Showdown replay `json` files.

`parsed_replays/` contains the replay parser that converts Showdown replays to agnet data. The replay parser has [its own README](parsed_replays/README.md).


The full replay process uses the scripts in an order like this:

1. `raw_replays.find_replay_links` uses the PS API to find battle ids played in a given date range.
2. `raw_replays.replay_intake` merges/deduplicates directories of downloaded PS replays into an existing set.
3. `raw_replays.usernames` maintains a consistent mapping from real PS usernames to "anonymous" usernames. We are trying to be polite to the players and remove NSFW usernames from the public huggingface dataset, but real usernames are easy to recover. This isn't a serious privacy issue --- we are working with screen names from public battles.
4. `raw_replays.anonymize` uses a username mapping to "anonymize" replays and removes player chat logs from replay files.
5. `raw_replays.upload_to_hf` pushes the anonymized replay directory to hugginface as a structured dataset.
6. `raw_replays.download_from_hf` skips prevoius steps. It downloads the current version of the raw replay dataset and puts it back in the dir structure expected by the rest of the code.
7. `raw_replays.replay_parser` converts PS replays to `.npz` training data by reconstructing the players' POV.
8. `raw_replays.download_from_hf` skips all previous steps and downloads our latest version of the parsed replay set from hugginface.



