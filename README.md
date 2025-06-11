
<div align="center">
    <img src="media/metamon_text_logo.png" alt="Metamon Text Logo" width="410">
</div>

<br>

<div align="center">
    <img src="media/metamon_banner.png" alt="Metamon Banner" width="720">
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


### Table of Contents

0. [Quick Start](#quick-start)
1. [Pretrained Models](#pretrained-models)
2. [Battle Datasets](#battle-datasets)
3. [Team Sets](#team-sets)
4. [Baselines](#baselines)
5. [Observation & Rewards](#observation-spaces--reward-functions)
6. [Training](#training)
7. [Extra](#extra)



<br>
 
---

<br>

## Installation

Metamon is written and tested for linux and python 3.10+. We recommend creating a fresh virtual environment or [conda](https://docs.anaconda.com/anaconda/install/) environment:

```shell
conda create -n metamon python==3.10
conda activate metamon
```

Then, install with:

```shell
git clone --recursive git@github.com:UT-Austin-RPL/metamon.git
cd metamon
pip install -e .
```

To install [Pokémon Showdown](https://pokemonshowdown.com/) (PS), you will need a modern version of `npm` / Node.js (instructions [here](https://nodejs.org/en/download/package-manager)). This repo comes packaged with the specific commit that we used during the project.

> [!IMPORTANT]
> Now updated to handle [breaking changes](https://github.com/smogon/pokemon-showdown/pull/11105) to Pokémon Showdown. Use a June 2025 Showdown commit (or later). The version that downloads with `metamon` (`metamon/server`) is always supported. Team Sets are not yet fully validated for this version. Fix in progress.

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
# in the background (`screen`, etc.)
node pokemon-showdown start --no-security
# no-security removes battle speed throttling and password requirements on your local server
```

You can verify that installation has gone smoothly with:
```bash
# run a few test battles on your local server and print a progress bar to the terminal
python -m metamon.env
```


Metamon provides large datasets of Pokémon team files, human battles, and other statistics that will automatically download when requested. You will need to specify a path:
```bash
# add to ~/.bashrc
export METAMON_CACHE_DIR=/path/to/plenty/of/disk/space
```


> [!NOTE]
>
> `metamon` relies on (and should automatically install) a fork of [poke-env](https://github.com/hsahovic/poke-env) ([here](https://github.com/UT-Austin-RPL/poke-env)). Don't let other packages update `poke-env` in metamon's venv.

<br>

____

<br>

## Quick Start

The RL environment is just a "batteries included" wrapper of [poke-env](https://github.com/hsahovic/poke-env). Pick a set of Pokémon teams to play with, an observation space, and a reward function:

```python
from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward

# Step 1: grab a set of human-made starter teams for Gen 1 OverUsed.
# every `reset` will sample a new team for your agent and your opponent.
team_set = get_metamon_teams("gen1ou", "competitive")

# Step 2: pick the observation space and reward function from the paper
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

The more flexible option is to request battles on your local Showdown server and battle anyone else who is online (yourself, our pretrained agents, or other Pokémon AI projects). If it plays Showdown, you can battle against it!

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
# examples are converted to the chosen obs space / reward function
# on-the-fly during dataloading. 
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
terminated = False
while not terminated:
    *_, terminated, _, _ = env.step(env.action_space.sample())
# find completed battles before loading examples
dset.refresh_files()
```

You are free to use this data to train an agent however you'd like, but we provide starting points for smaller-scale IL (`python -m metamon.il.train`) and RL (`python -m metamon.rl.train`), and a large set of pretrained models from our paper.


<br>

____

<br>


## Pretrained Models

We have made every checkpoint of 18 models available on huggingface at [`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon/tree/main). Pretrained models can run without research GPUs, but you will need to install [`amago`](https://github.com/UT-Austin-RPL/amago), which is an RL codebase by the same authors. Follow instructions [here](https://ut-austin-rpl.github.io/amago/installation.html).

> [!TIP]
> See the `amago` [documentation](https://ut-austin-rpl.github.io/amago/) for help with training hyperparameters and customization.

<div align="center">
    <img src="media/arch_v6_safe.png" alt="Figure 1" width="450">
</div>

<br>

Load and run pretrained models with `metamon.rl.eval_pretrained`. For example:

```bash
python -m metamon.rl.eval_pretrained --agent SyntheticRLV2 --gens 1 --formats ou --n_challenges 50 --eval_type heuristic
```

Will run the default checkpoint of the best model for 50 battles against a set of heuristic baselines highlighted in the paper.

Or to battle against whatever is logged onto the local Showdown server (including other pretrained models that are already waiting):

```bash
python -m metamon.rl.eval_pretrained --agent SyntheticRLV2 --gens 1 --formats ou --n_challenges 50 --eval_type ladder --username <pick unique username> --team_set paper_replays
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

Here is a reference of human evals for key models according to our paper:


<div align="center">
    <img src="media/human_ratings.png" alt="Figure 1" width="800">
</div>

<br>

____

<br>


## Battle Datasets

PS creates "replays" of battles that players can choose to upload to the website before they expire. We gathered all surviving historical replays for Generations 1-4 Ubers, OverUsed, UnderUsed, and NeverUsed, and now save new battles to grow the dataset.

PS replays are saved from the point-of-view of a *spectator* rather than the point-of-view of a *player*. We unlock the replay dataset for RL by "reconstructing" the point-of-view of each player. 

<div align="center">
    <img src="media/dataset.png" alt="Dataset Overview" width="800">
</div>
<br>


Datasets are stored on huggingface in three formats:

| Name |  Entries | Description |
|------|------|-------------|
|**[`metamon-raw-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays)** | 544k | Our curated set of Pokémon Showdown replay `.json` files... to save the Showdown API some download requests and to maintain an official reference of our training data. Will be regularly updated as new battles are played and collected. |
|**[`metamon-parsed-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)** | 1.07M | The RL-compatible version of the dataset as reconstructed by the [replay parser](metamon/data/replay_dataset/replay_parser/README.md). These datasets have **missing actions** (`action = -1`) where the player's choice is not revelead to spectators. Includes ~100k more trajectories than were used by the paper (because more human battles have been played). The method for predicting partially revealed teams has also been significantly improved.|
|**[`metamon-synthetic`](https://huggingface.co/datasets/jakegrigsby/metamon-synthetic)** | 5M | A set of real + self-play battles that were used to train the best models in the paper. Unfortunately, this dataset has been deprecated as part of changes that make the parsed replay dataset more future-proof. It provides `amago` trajectory files with a fixed observation space + reward function and missing actions filled by an IL model. We are working to create a new starter dataset of self-play battles.|


Parsed replays will download automatically when requested by the `ParsedReplayDataset`, but these datasets are large. See `python -m metamon.download --help` to download them in advance.

<br>

___

<br>

 ## Team Sets

 Team sets are dirs of PS team files that are randomly sampled between episodes. They are stored on huggingface at [`jakegrigsby/metamon-teams`](https://huggingface.co/datasets/jakegrigsby/metamon-teams) and can be downloaded in advance with `python -m metamon.download teams`

```python
metamon.env.get_metamon_teams(battle_format : str, set_name : str)
```

 | `set_name` | Teams Per Battle Format | Description |
|------|---------------------------|-----------------------|
|`"competitive"`| Varies (< 30) | Human-made teams scraped from forum threads. These are usually official "sample teams" designed by experts for beginners, but we are less selective for non-OU tiers. This is the set used for human ladder evaluations in the paper. |
|`"paper_variety"`| 1k | Procedurally generated teams with unrealistic OOD lead-off Pokémon. The paper calls this the "variety set". Movests were generated by the legacy `data.team_prediction.predictor.NaiveUsagePredictor` strategy. |
| `"paper_replays"` | 1k (OU Only) | *Predicted* teams from replays. The paper calls this the "replay set". Surpassed by the "modern_replays" set below but included for reproducibilty. Uses a simple team prediction strategy that is now defined by `data.team_prediction.predictor.NaiveUsagePredictor`|
| `"modern_replays"` | 8k-12k<br> (OU Only) | *Predicted* teams based on recent replays (currently: since May 20th, 2024). Predictions use the `data.team_prediction.predictor.ReplayPredictor`. The result is a set representing the recent metagame with blanks filled by a mixture of historical trends. Teams are validated against the Showdown version that comes with the repo (`metamon/server`). We plan to update to current Showdown rules soon.|

You can also use your own directory of team files with, for example:
```python
from metamon.env import TeamSet

team_set = TeamSet("/path/to/your/team/dir", battle_format: str) # e.g. gen3ou
```
But note that your files would need to have the extension `".{battle_format}_team"`.


<br>

___

<br>


## Baselines

`baselines/` contains baseline opponents that we can battle against via `BattleAgainstBasline`. `baselines/heuritics` provides more than a dozen heuristic opponents and starter code for developing new ones (or mixing ground-truth Pokémon knowledge into ML agents). `baselines/model_based` ties the simple `il` model checkpoints to `poke-env` (with CPU inference).


Here is an overview of opponents mentioned in the paper:

```python
from metamon.baselines import ALL_BASELINES
opponent = ALL_BASELINES[name]
```

 | `name` | Description |
|------|-------------|
| `BugCatcher` | An actively bad trainer that always picks the least damaging move. When forced to switch, picks the pokemon in its party with the worst type matchup vs the player.
|`RandomBaseline`| Selects a legal move (or switch) uniformly at random and measures the most basic level of learning early in training runs.|
|`Gen1BossAI`| Emulates opponents in the original Pokémon Generation 1 games. Usually chooses random moves. However, it prefers using stat-boosting moves on the second turn and “super effective” moves when available. |
| `Grunt` | A maximally offensive player that selects the move that will deal the greatest damage against the current opposing Pokémon using Pokémon’s damage equation and a type chart and selects the best matchup by type when forced to switch.|
| `GymLeader` | Improves upon Grunt by additionally taking into account factors such as health. It prioritizes using stat boosts when the current Pokémon is very healthy, and heal moves when unhealthy.|
| `PokeEnvHeuristic` | The `SimpleHeuristicsPlayer` baseline provided by `poke-env` with configurable difficulty (shortcuts like `EasyPokeEnvHeuristic`).|
| `EmeraldKaizo` | An adaptation of the AI in a Pokémon Emerald ROM hack intended to be as difficult as possible. It selects actions by scoring the available options against a rule set that includes handwritten conditional statements for a large portion of the moves in the game.|
| `BaseRNN` | A simple RNN IL policy trained on an early version of our parsed replay dataset. Runs inference on CPU.|

Compare baselines with:

```bash
python -m metamon.baselines.compete --battle_format gen2ou --player GymLeader --opponent RandomBaseline --battles 10
```

Here is a reference for the relative strength of some heuristic baselines from the paper:
<div align="center">
    <img src="media/heuristic_heatmap.png" alt="Figure 1" width="380">
</div>

<br>
<br>

___

<br>

## Observation Spaces & Reward Functions

The `DefaultObservationSpace` is the (quite high-dimensional) text/numerical observation space used in our paper. Alternatives are listed in `metamon.interface.ALL_OBSERVATION_SPACES`. Currently there is just one other built-in option (still in testing) (`DefaultPlusObservationSpace`). You may want to create a custom version with more/less features by inheriting from `metamon.interface.ObservationSpace`.

Text features have inconsistent length, which is not something most RL frameworks have a reason to support. You might want to convert to int IDs from a list
of known vocab words. The observation space is designed such that the "tokenized" version *will* have fixed length.

```python
from metamon.interface import TokenizedObservationSpace, DefaultObservationSpace
from metamon.tokenizer import get_toknenizer

base_obs = DefaultObservationSpace()
tokenized_space = TokenizedObservationSpace(
    base_obs_space=base_obs,
    tokenizer=get_tokenizer("DefaultObservationSpace-v0"),
)
```

You can read the vocabs as jsons in `metamon/tokenizer`; they are generated by tracking unique
words across the entire replay dataset, with an unknown token for rare cases we may have missed.

 | Name | Description |
|------|-------------|
| `allreplays-v3` | Legacy version for pre-release models. |
|`DefaultObservationSpace-v0`| Updated post-release vocabulary as of `metamon-parsed-replays` dataset `v2`. |

Reward functions assign scalar reward based on consecutive states (R(s, s')). `DefaultShapedReward` is the shaped reward used by the paper. `BinaryReward` removes the smaller shaping terms and simply provides +/- 100 for win/loss. Any new reward functions would be added to `metamon.interface.ALL_REWARD_FUNCTIONS`, and you can implement your own by inheriting from `metamon.interface.RewardFunction`.

---

 <br>

 ## Training

 We trained all of our main RL **& IL** models with [`amago`](https://ut-austin-rpl.github.io/amago/index.html). Everything you need to train your own model on metamon data and evaluate against Pokémon baselines is provided in **`metamon/rl/`**.

#### Configure `wandb` logging (optional):
```shell
cd metamon/rl/
export METAMON_WANDB_PROJECT="my_wandb_project_name"
export METAMON_WANDB_ENTITY="my_wandb_username"
```

#### Basic Training Run

See `python train.py --help` for options. The training script currently implements *offline RL on the human battle dataset*. We are working on reintroducing self-play datasets and extending to online RL on the local ladder. 

We might retrain the "`SmallIL`" model like this: 
```
python train.py --run_name any_name_here --model_gin_config configs/models/small_agent.gin --ckpt_dir /pick/a/ckpt/dir --train_gin_config configs/training/base_offline.gin --il --log
```
"`SmallRL`" would be the same command without `--il`.

Larger training runs take *days* to complete and [can use mulitple GPUs (link)](https://ut-austin-rpl.github.io/amago/tutorial/async.html#multi-gpu-training). We think it's likely that faster hparams can reach similar performance, and are working on it!


#### Evaluate
The easiest way to eval a new model is to go in and add a `LocalPretrainedModel` to `rl/eval_pretrained.py`. 

Let's say the training command was: `python train.py --run_name psyduck_is_ubers --model_gin_config gigantic_agent.gin --ckpt_dir /my_metamon_ckpts/`. We'd add:

```python
# metamon/rl/evaluate_pretrained.py
class PsyduckIsUbers(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            # absolute path to where amago saves the run's outputs
            amago_run_path="/my_metamon_ckpts/psyduck_is_ubers/",
            # relative path within rl/configs/ to the model hparams
            gin_config="models/gigantic_agent.gin",
            # if the training command had --il in it...
            is_il_model=False,
        )
```
And now we can evaluate it just like any of the huggingface models.

#### Customize

Customize the agent architecture by creating new `rl/configs/models/` `.gin` files. Customize the RL hyperparameters by creating new `rl/configs/training/` files. [Here is a link](https://ut-austin-rpl.github.io/amago/tutorial/configuration.html) to a lot more information about configuring training runs. `amago` is modular and you can swap just about any piece of the agent with your own ideas. [Here is a link](https://ut-austin-rpl.github.io/amago/tutorial/customization.html) to more information about custom components.


#### Standalone Toy `il` (Deprecated)

<details>

`il/` is old toy code that does basic behavior cloning with RNNs. We used it to train early learning-based baselines (`BaseRNN`, `WinsOnlyRNN`, and `MiniRNN`) that you can play against with the `BattleAgainstBaseline` env. We may add more of these as the dataset grows/improves and more architectures are tried. Playing around with this code might be an easier way to get started, but note that the main `rl/train` script can also be configured to do RNN BC... but faster and on multiple GPUs.

Get started with something like:
```shell
cd metamon/il/
python train.py --run_name any_name_will_do --model_config configs/transformer_embedding.gin  --gpu 0
```

</details>


 ---

 <br>


## Extra

`metamon/data` contains utilities for handling Showdown replays (`data.replay_dataset.raw_replays`), converting replays into training data (`data.replay_dataset.replay_parser`), predicting teams from partial information (`data.team_prediction`), and accessing Showdown usage statistics (`data.legacy_team_builder`). These modules generate our huggingface datasets, but may be useful for other things. More info in the [`data` README](metamon/data/README.md).


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
