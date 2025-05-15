
<div align="center">
    <img src="media/metamon_text_logo.png" alt="Metamon Text Logo" width="390">
</div>

<br>

<div align="center">
    <img src="media/metamon_banner.png" alt="Metamon Banner" width="700">
</div>

<br>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2504.04395-red)](https://arxiv.org/abs/2504.04395)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://metamon.tech)

</div>


<br>




**Metamon** enables reinforcement learning (RL) research on [Pokémon Showdown](https://pokemonshowdown.com/) by providing:

1) A standardized suite of teams and opponents for evaluation.
2) A large dataset of RL trajectories "reconstructed" from real human battles.
3) Starting points for training imitation learning (IL) and RL policies.

Currently, it is focused on singles formats of the **first four generations** (Gen1-4 OverUsed, UnderUsed, NeverUsed, and Ubers).


Metamon is the codebase behind ["Human-Level Competetitive Pokémon via Scalable Offline RL and Transformers"](https://arxiv.org/abs/2504.04395). Please check out our [project website](https://metamon.tech) for an overview of our results. This README documents the dataset, pretrained models, training, and evaluation details to help you get battling!

<br>

<div align="center">
    <img src="media/figure1.png" alt="Figure 1" width="700">
</div>



<br>
 
---

<br>

## Installation

Metamon is written and tested for unix and python 3.10+. We recommend creating a fresh virtual environment or [conda](https://docs.anaconda.com/anaconda/install/) environment:

```shell
conda create -n metamon python==3.10
conda activate metamon
```

Then, install with:

```shell
git clone git@github.com:UT-Austin-RPL/metamon.git
cd metamon
pip install -e .
```

To install [Pokémon Showdown](https://pokemonshowdown.com/) (PS), you will need a modern version of `npm` / Node.js (instructions [here](https://nodejs.org/en/download/package-manager)). This repo comes packaged with the specific commit that we used during the project (though newer versions should be fine!)

```shell
cd server/pokemon-showdown
npm install
```

Then, we will start a local PS server to handle our battle traffic. The server settings are determined by a configuration file which we'll copy from the provided example (`server/config.js`):
```shell
cp ../config.js config/
```

You will need to have the PS server running in the background while using Metamon:
```shell
# recommended: `screen`
node pokemon-showdown start --no-security # no-security removes the account login of the public website
# Press Ctrl+A+D to detach from the screen
```

You can verify that installation has gone smoothly with:
```bash
python -m metamon.env
```
Which will run a few test battles on your local server and print a progress bar to the terminal.


Metamon provides large datasets of Pokémon team files, human battles, and other statistics that will automatically download when requested. You will need to specify a path:
```bash
# add to ~/.bashrc
export METAMON_CACHE=/path/to/plenty/of/disk/space
```


> [!NOTE]
>
> `metamon` installs an old fork of [poke-env]() ([here]()). **Please do not bother the main poke-env GitHub with issues related to `poke_env` error messages you encounter while using metamon**. They've probably already fixed it!

<br>

____

<br>

## Quick Start

The RL environment is a "batteries included" wrapper of [poke-env](). We pick (1) a set of Pokémon teams to play with, and (2) an observation space & reward function.

```python
from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward


# Step 1: grab a set of human-made starter teams for Gen 1 OverUsed
team_set = get_metamon_teams("gen1ou", "competitive")

# Step 2: pick the observation space and reward function from our paper
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
```

It is easy to battle against built-in baselines (anything in `metamon.baselines` or any `poke_env.Player`):

```python 
from metamon.env import BattleAgainstBaseline
from metamon.baselines.heuristic.basic import Gen1BossAI
# see metamon.baselines.ALL_BASELINES for more options

env = BattleAgainstBaseline(
    battle_format="gen1ou",
    observation_space=obs_space,
    reward_function=reward_fn,
    team_set=team_set,
    opponent_type=Gen1BossAI,
)

# `env` is a standard `gymnasium` environment!
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The more flexible option is to request battles on your local Showdown server and battle anyone else who is online (yourself, our pretrained agents, or other Pokémon AI projects like [Foul Play]()). If it plays Showdown, you can battle against it!

```python
from metamon.env import QueueOnLocalLadder

env = QueueOnLocalLadder(
    battle_format="gen1ou",
    username="my_scary_username",
    num_battles=10,
    observation_space=obs_space,
    reward_function=reward_fn,
    team_set=team_set,
)
```


**Metamon's main feature is that it creates a dataset of "reconstructed" human demonstrations for these environments**:

```python
from metamon.datasets import ParsedReplayDataset

# will download/extract large files the first time it's called.
# converts replays to the chosen observation space / reward function.
dset = ParsedReplayDataset(
    observation_space=obs_space,
    reward_function=reward_func,
    formats=["gen1ou"],
)

obs_seq, action_seq, reward_seq, done_seq, missing_action_mask_seq = dset[0]
```

You can save your own agents' experience in the same format:

```python
env = QueueOnLocalLadder(
    battle_format="gen1ou",
    .., # rest of args
    save_trajectories_to="my_data_path",
)

dset = ParsedReplayDataset(
    dset_root="my_data_path",
    observation_space=obs_space,
    reward_function=reward_func,
    formats=["gen1ou"],
)

# interact with the environment
done = False
while not done:
    *_, terminated, truncated, _ = env.step(env.action_space.sample())
    done = terminated | truncated

# find completed battles before loading examples
dset.refresh_files()
```

You are free to use this data to train an agent however you'd like, but we provide starting points for smaller-scale IL (`python -m metamon.il.train`) and RL (`python -m metamon.rl.train`), and a large set of pretrained models from our paper.


<br>

____

<br>


## Pretrained Models

We have made every checkpoint of 18 models available on huggingface at [`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon/tree/main). Pretrained models can run without research GPUs, but you will need to install [`amago`](), which is an RL codebase by the same authors. Follow instructions [here]().


Load and run pretrained models with `metamon.rl.eval_pretrained`. For example:

```bash
python -m metamon.rl.eval_pretrained --agent SyntheticRLV2 --gens 1 --formats ou --n_challenges 50 --eval_type heuristic
```

Will run the default checkpoint of the best model for 50 battles against a set of heuristic baselines highlighted in the paper.

Or to battle against whatever is logged onto the local Showdown server (including other pretrained models that are already waiting):

```bash
python -m metamon.rl.eval_pretrained --agent SyntheticRLV2 --gens 1 --formats ou --n_challenges 50 --eval_type local-ladder --username <pick a unique username> --team_split replays
```

Some model sizes have several variants testing different RL objectives. See `metamon/rl/eval_pretrained.py` for a complete list.

| Model Name (`--agent`)                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **`SmallIL`** (2 variants)                | 15M imitation learning model trained on 1M human battles             |
| **`SmallRL`** (5 variants)                | 15M actor-critic model trained on 1M human battles                 |
| **`MediumIL`**                | 50M imitation learning model trained on 1M human battles          |
| **`MediumRL`** (3 variants)                | 50M actor-critic model trained on 1M human battles        |
| **`LargeIL`**                 | 200M imitation learning model trained on 1M human battles            |
| **`LargeRL`**                 | 200M actor-critic model trained on 1M human battles          |
| **`SyntheticRLV0`**           | 200M actor-critic model trained on 1M human + 1M diverse self-play battles          |
| **`SyntheticRLV1`**           | 200M actor-critic model trained on 1M human + 2M diverse self-play battles  |
| **`SyntheticRLV1_SelfPlay`**   | SyntheticRLV1 fine-tuned on 2M extra battles against itself                 |
| **`SyntheticRLV1_PlusPlus`**          | SyntheticRLV1 fine-tuned on 2M extra battles against diverse opponents      |
| **`SyntheticRLV2`**           | Final 200M actor-critic model with value classification trained on 1M human + 4M diverse self-play battles. |



<br>

____

<br>


## Battle Datasets

PS creates "replays" of battles that players can choose to upload to the website before they expire. We gathered all surviving historical replays for Generations 1-4 Ubers, OverUsed, UnderUsed, and NeverUsed, and now save new battles to grow the dataset.

PS replays are saved from the point-of-view of a *spectator* rather than the point-of-view of a *player*. We unlock the replay dataset for RL by "reconstructing" the point-of-view of each player. 

<div align="center">
    <img src="media/dataset.png" alt="Dataset Overview" width="700">
</div>
<br>


Datasets are stored on huggingface in three formats:

| Name |  Battles | Description |
|------|------|-------------|
|**[`metamon-raw-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays)** | 535k | Our curated set of Pokémon Showdown replay `.json` files... to save the Showdown API some download requests and to maintain an official reference of our training data. Will be regularly updated as new battles are played and collected. |
|**[`metamon-parsed-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)** | 1.05M | The RL-compatible version of the dataset as reconstructed by the [replay parser](metamon/data/replay_dataset/parsed_replays/replay_parser/). These datasets have **missing actions** (`action = -1`) where the player's choice is not revelead to spectators. Includes ~100k more trajectories than were used by the paper (because more human battles have been played). The method for predicting partially revealed teams has also been significantly improved.|
|**[`metamon-synthetic`](https://huggingface.co/datasets/jakegrigsby/metamon-synthetic)** | 5M | A set of real + self-play battles that were used to train the best models in the paper. Unfortunately, this dataset has been deprecated as part of changes that make the parsed replay dataset more future-proof. It provides `amago` trajectory files with a fixed observation space + reward function and missing actions filled by an IL model. We are working to create a new starter dataset of self-play battles.|


Parsed replays will download automatically when requested by the `ParsedReplayDataset`. Alternatively, you can download them in advance with `python -m metamon.download parsed-replays`. Use `python -m metamon.download raw-replays` to grab the unprocessed Showdown replays if needed.


<br>

 ## Team Sets

 Team sets are directories of Pokémon Showdown team files that are randomly sampled between env resets:

```python
metamon.env.get_metamon_teams(battle_format : str, set_name : str)
```


 | `set_name` | Teams Per Battle Format | Description |
|------|------|-------------|
|`"competitive"`| Varies (< 30) | Human-made teams scraped from forum threads. These are usually official "sample teams" designed by experts for beginners, but we are less selective for non-OU tiers. This is the set used for human ladder evaluations in the paper. |
|`"paper_variety"`| 900 | Procedurally generated teams with unrealistic OOD lead-off Pokémon. The paper calls this the "variety set". Movests were generated by the legacy `data.team_prediction.predictor.NaiveUsagePredictor` strategy. |
| `"paper_replays"` | 1k | *predicted* full teams based on *revealed* teams in replays. The paper calls this the "replay set". Surpassed by the "modern_replays" set below but included for reproducibilty. Uses a simple team prediction strategy that is now defined by `data.team_prediction.predictor.NaiveUsagePredictor`|
| `"modern_replays"` | Varies (300 - 60k) | *predicted* full teams based on *revealed* teams of recent replays (currently: since May 14th, 2024). In OverUsed tiers (where we can afford to be selective) we filter to ladder games played above 1200 ELO. Predictions use the `data.team_prediction.predictor.ReplayPredictor`. The result is a set representing the modern metagame with blanks filled by a mixture of historical trends. |



You can also use your own directory of team files with, for example:
```python
from metamon.env import TeamSet

team_set = TeamSet("/path/to/your/team/dir", battle_format: str) # e.g. gen3ou
```
But note that your files would need to have the extension `"{battle_format}_team"`.





<br>



<br>


## Baselines

`baselines/` contains baseline opponents that we can battle against via `BattleAgainstBasline`. `baselines/heuritics` provides more than a dozen heuristic opponents and starter code for developing new ones (or mixing ground-truth Pokémon knowledge into ML agents). `baselines/model_based` ties the simple `il` model checkpoints to `poke-env` (with CPU inference).

Compare baselines with:

```bash
python -m metamon.compete --task_dist Gen1OU --player GymLeader --opponent RandomBaseline --tasks 10
```

<br>


<br>

## Data

`metamon/data` contains utilities for handling Showdown replays (`data.replay_dataset.raw_replays`), converting replays into training data (`data.replay_dataset.parsed_replays`), predicting teams from partial information (`data.team_prediction`), and accessing Showdown usage statistics (`data.team_prediction.legacy_team_builder`). These modules generate our huggingface datasets, but may be useful for other things. More info in the [`data` README](data/README/.md).


<br>

***

<br>

## Citation

```bibtex
@misc{grigsby2025metamon,
      title={Human-Level Competitive Pok\'emon via Scalable Offline Reinforcement Learning with Transformers}, 
      author={Jake Grigsby and Yuqi Xie and Justin Sasek and Steven Zheng and Yuke Zhu},
      year={2025},
      eprint={2504.04395},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04395}, 
}
```
