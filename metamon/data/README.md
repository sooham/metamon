# Metamon Data

> [!WARNING]
> 
> This module contains the behind-the-scenes work that makes the `ParsedReplayDataset` possible. Very little of this is intended to be used externally. Please consider yourself warned :)


<p align="center">
  <img src="../../media/replay_parser_warning.png">
</p>


`replay_dataset/raw_replays/` now solely contains the download helper for the raw replay dataset; everything involved in maintaining the dataset has been moved elsewhere.

`replay_dataset/parsed_replays/` contains the replay parser that converts Showdown replays to agent data. The replay parser has [its own README](replay_dataset/parsed_replays/replay_parser/README.md).

`team_prediction/` predicts full teams from partially revealed teams in replays. 