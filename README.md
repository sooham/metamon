
<div align="center">
    <img src="media/metamon_text_logo.png" alt="Metamon Text Logo" width="380">
</div>

<br>

<div align="center">
    <img src="media/metamon_banner.png" alt="Metamon Banner" width="700">
</div>

<br>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2304.12345-red)](https://arxiv.org/abs/2504.04395)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://metamon.tech)

</div>


<br>




**Metamon** enables reinforcement learning (RL) research in Competitive Pokémon (as played on [Pokémon Showdown](https://pokemonshowdown.com/)) by providing:

- A large dataset of RL trajectories "reconstructed" from real human battles.
- Starting points for training imitation learning (IL) and offline RL policies.
- A standardized suite of teams and opponents for evaluation.

Currently, it is focused on the **first four generations of Pokémon**, which have the longset battle lengths and provide the least information about the opponent's team.


Metamon is the codebase behind ["Human-Level Competetitive Pokémon via Scalable Offline RL and Transformers"](https://arxiv.org/abs/2504.04395). Please check out our [project website](https://metamon.tech) for an overview of our results. This README documents the dataset, pretrained models, training, and evaluation details to help you get battling!


<div align="center">
    <img src="media/figure1.png" alt="Figure 1" width="700">
</div>



<br>

> The public version of this repo is very much in beta :) Please come back soon for updates!

<br>
 
---

<br>

## Installation

Metamon is written and tested for ubuntu and python 3.10+. We recommend creating a fresh virtual environment or [conda](https://docs.anaconda.com/anaconda/install/) environment:

```bash
conda create -n metamon python==3.10
conda activate metamon
```

Then, install with:

```bash
git clone git@github.com:UT-Austin-RPL/metamon.git
cd metamon
pip install -e .
```

To install [Pokémon Showdown](https://pokemonshowdown.com/) (PS), you will need a modern version of `npm` / Node.js. It's likely you already have this (check that `npm -v` is > 10.0), but if not, you can find instructions [here](https://nodejs.org/en/download/package-manager). This repo comes packaged with the specific commit that we used during the project (though newer versions should be fine!)

```bash
cd server/pokemon-showdown
npm install
```

Then, we will start a local PS server to handle our battle traffic. The server settings are determined by a configuration file which we'll copy from the provided example (`server/config.js`):
```bash
cp ../config.js config/
```
The main setting in this `config.js` file worth knowing about is `export.num_workers`, which helps handle concurrent battles.

You will need to have the PS server running in the background while using Metamon:
```bash
# recommended: `screen`
node pokemon-showdown start --no-security # no-security removes the account login of the public website
# Press Ctrl+A+D to detach from the screen
```
You should see a status message printed for each worker.


[`poke-env`](https://github.com/hsahovic/poke-env) is a python interface for interacting with the javascript PS server. **Metamon relies on a custom (and now quite out-of-sync) fork for various early-gen fixes, which should install as part of the metamon package**. If you run into issues, the repo is here:

```bash
# does not need to be in the same directory as pokemon-showdown
git clone git@github.com:jakegrigsby/poke-env.git
cd poke-env
pip install -e .
```

You can verify that installation has gone smoothly with:
```bash
python -m metamon.env
```
Which will run a few test battles on your local server and print a progress bar to the terminal.


<br>


## Battle Datasets

PS creates "replays" of battles that players can choose to upload to the website before they expire. We gathered all surviving historical replays for Generations 1-4 Ubers, OverUsed, UnderUsed, and NeverUsed, and now save active battles before expiration to accelerate dataset growth.

PS replays are saved from the point-of-view of a *spectator* rather than the point-of-view of a *player*. We unlock the replay dataset for RL by "reconstructing" the point-of-view of each player. 

<div align="center">
    <img src="media/dataset.png" alt="Dataset Overview" width="700">
</div>
<br>


Datasets are stored on huggingface in two formats:

| Name |  Battles | Description |
|------|------|-------------|
|**[`jakegrigsby/metamon-parsed-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)** | 1.05M | Real Showdown battles only! Provides the dataset in the most portable form (fresh from the [replay parser](metamon/data/replay_dataset/replay_parser/)). Observations are dicts of text and numerical features. These datasets have **missing actions** (`action = -1`) where the player's choice is not revelead to spectators. Includes ~100k more trajectories than were used by most experiments in the paper (because more human battles have been played!). |
|**`jakegrigsby/metamon-synthetic`** | 5M | Provides the dataset in the format expected by the RL trainer --- though they can still be used anywhere with a little pre-processing. Text is stored as tokenized ints based on all the words that appear in the parsed replays. Missing actions have been filled by an IL model, and we include 4M self-play trajectories generated by our RL models. This is the final version of the dataset used to train the best model in the paper. |



<br>

## Pretrained Models

We have made every checkpoint of 18 models available on huggingface at [`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon/tree/main).


Load and run pretrained models with `metamon.rl.eval_pretrained`. For example:

```bash
python -m metamon.rl.eval_pretrained --agent SyntheticRLV2 --gens 1 --formats ou --n_challenges 50 --eval_type heuristic
```

Will run the default checkpoint of the best model for 50 battles against a set of heuristic baselines.


Here is an overview. Some model sizes have several variants testing different RL objectives. See `metamon/rl/eval_pretrained.py` for a complete list.

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


<br>


## Training
**Use `--help`** for documentation of each script


**`metamon/il/`** is a basic imitation learning pipeline that might be a useful template / starting point for playing with model architectures and new datasets. This code is a remnant of an early version of this project, and the final paper's experiments only use this for a few "BC-RNN" baselines that were mostly pushed to the Appendix.

`python -m metamon.il.train`

**`metamon/rl/`** connects Metamon to [`amago`](https://github.com/UT-Austin-RPL/amago), which powers the main IL and RL experiments in the paper. 

`python -m metamon.rl.offline_from_config` is the main training script. Before training begins, we need to assemble an offline dataset from the parsed replays released above and self-play trajectoires. More on this soon.

<br>


<br>


## Baselines

`baselines/` contains baseline opponents that we can battle against via `poke-env`. `baselines/heuritics` provides more than a dozen heuristic opponents and starter code for developing new ones (or mixing ground-truth Pokémon knowledge into ML agents). `baselines/model_based` ties the simple `il` model checkpoints to `poke-env` (with CPU inference).

Compare baselines with:

```bash
python -m metamon.compete --task_dist Gen1OU --player GymLeader --opponent RandomBaseline --tasks 10
```

<br>


<br>

## Data

`data/teams`: contains sets of Pokémon teams scraped from forum discussions or procedurally generated from real replays and/or usage statistics.

`data/tokenizer`: standardizes the conversion between text observations and token ids.

`data/replay_dataset`: includes all the behind-the-scenes logic that creates the replay dataset on huggingface.

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
