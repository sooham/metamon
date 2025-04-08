# Metamon
<div align="center">
    <img src="media/metamon_banner.png" alt="Metamon Banner" width="700">
</div>

<br>


<div align="center">
    <a href="https://metamon.tech" target="_blank">
        <button style="
            background-color: #87CEEB; /* Sky Blue */
            border: 2px solid black; /* Black border */
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        ">
            Website
        </button>
    </a>
    <a href="https://arxiv.org/abs/2504.04395" target="_blank">
        <button style="
            background-color: #f44336; /* Red */
            border: 2px solid black; /* Black border */
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
        ">
            Paper
        </button>
    </a>
</div>

<br>





**Metamon** enables reinforcement learning (RL) research in Competitive Pokémon Singles (as played on [Pokémon Showdown](https://pokemonshowdown.com/)) by providing:

1. A large (and continuously growing) dataset of RL trajectories "reconstructed" from real human battles.
2. Starting points for training imitation learning (IL) and offline reinforcement learning (RL) policies.
3. A standardized suite of teams and opponents for evaluation.

Currently, it is focused on the **first four generations of Pokémon**, which have the longset battle lengths and provide the least information about the opponent's team.


Metamon is the codebase behind ["Human-Level Competetitive Pokémon via Scalable Offline RL and Transformers"](https://arxiv.org/abs/2504.04395) Please check out our project website for an overview of our results. This README documents the dataset and evaluation details to help get started in Competitive Pokémon AI.


<div align="center">
    <img src="media/figure1.png" alt="Figure 1" width="700">
</div>



<br>

> The public version of this repo is very much in beta :) Please come back soon for updates!

<br>
 
---

<br>

## Installation

Metamon is written and tested for linux and python 3.10+. We recommend creating a fresh virtual environment or [conda](https://docs.anaconda.com/anaconda/install/) environment:

```bash
conda create -n metamon python==3.10
conda activate metamon
```


### Pokémon Showdown

To install [Pokémon Showdown](https://pokemonshowdown.com/) (PS), you will need a modern version of `npm` / Node.js. It's likely you already have this (check that `npm -v` is > 10.0), but if not, you can find instructions [here](https://nodejs.org/en/download/package-manager).

This repo comes packaged with the specific commit that we used during the project (though newer versions should be fine!)

```bash
cd server/pokemon-showdown
npm install
```

Then, we will start a local PS server to handle our battle traffic. The server settings are determined by a configuration file which we'll copy from the provided example (`server/config.js`):
```bash
cp ../config.js config/
```
The main setting in this `config.js` file worth increasing is `export.num_workers`, which will help handle concurrent battles.

You will need to have the PS server running in the background while using Metamon:
```bash
# recommended: `screen`
node pokemon-showdown start --no-security # no-security removes the account login of the public website
# Press Ctrl+A+D to detach from the screen
```
You should see a status message printed for each worker.


### Poke-Env
[`poke-env`](https://github.com/hsahovic/poke-env) is a python interface for interacting with the javascript PS server. **Metamon relies on a custom (and by now quite out-of-sync) fork for various early-gen fixes, which should install as part of the metamon package below**. If you run into issues, the repo is here:

```bash
# does not need to be in the same directory as pokemon-showdown
git clone git@github.com:jakegrigsby/poke-env.git
cd poke-env
pip install -e .
```

### Metamon
Metamon closes the rest of the gap between `poke-env` and large-scale RL. It can be installed with:
```bash
git clone git@github.com:UT-Austin-RPL/metamon.git
cd metamon
pip install -e .
```

You can verify that installation has gone smoothly with:
```bash
python -m metamon.env
```
Which will run a few test battles and print a progress bar to the terminal.


### AMAGO *(Optional)*
We plan to move the pretrained agents into `metamon` in the near future. But for now, the only way to run them is to install [`amago`](https://github.com/UT-Austin-RPL/amago), which is work by the same authors, and handled all our RL training. Please follow the instructions there.

<br>


<br>


## Battle Datasets

PS creates "replays" of battles that players can choose to upload to the website before they expire. We gathered all surviving historical replays for Generations 1-4 Ubers, OverUsed, UnderUsed, and NeverUsed, and now save active battles before expiration to accelerate dataset growth.

PS replays are saved from the point-of-view of a *spectator* rather than the point-of-view of a *player*. We unlock the replay dataset for RL by "reconstructing" the point-of-view of each player. 

<div align="center">
    <img src="media/dataset.png" alt="Dataset Overview" width="700">
</div>
<br>


We release (and plan to continue releasing) several versions. They are stored on huggingface in two formats:

### [Parsed Replays](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)
Provide the dataset in the most portable form. Observations are dicts of text and numerical features and actions are ints.

These datasets have **missing actions** (`action = -1`). Missing actions are inherent to the replay reconstruction process because there are many situations where the player's choice is not revelead to spectators (e.g., their Pokémon faints before it can move, is paralyzed/asleep/frozen).

The datasets are structured as follows:

```bash
gen1nu/
    train/
        *.npz
    val/
        *.npz
gen1ou/
...
gen4uu/
```



### AMAGO Trajectories *(Coming Soon)*
Provides the dataset in the format expected by the AMAGO RL trainer --- though they can still be used anywhere with a little pre-processing. Text is stored as tokenized ints based on all the words that appear in the parsed replays.

The datasets are structured as follows:

```bash
{dset_name}/
    buffer/
        protected/
            # real human replays as .npz files
        fifo/
            # synthetic (self-play) battles as .npz files
```


| Name |  Battles | Description |
|------|------|-------------|
|**[`jakegrigsby/metamon-parsed-replays`](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays)** | 1.05M | Includes ~100k more trajectories than were used by most experiments in the paper |
|**`jakegrigsby/metamon-synthetic`** | 5M | The final version of the dataset used to train the best model in the paper. |

We are working on releasing intermediate versions of the dataset used to train various models over the long runtime of this project.

<br>


<br>

## Pretrained Models

We have made every checkpoint of 18 models available on huggingface at [`jakegrigsby/metamon`](https://huggingface.co/jakegrigsby/metamon/tree/main).


Load and run pretrained models with `metamon.rl.eval_pretrained`. For example:

```bash
python -m metamon.rl.eval_pretrained --agent SyntheticRLV2 --gens 1 --formats ou --n_challenges 50 --eval_type heuristic
```

Will run the default checkpoint of the best model for 50 battles against a set of heuristic baselines.

<br>


<br>


## Training
**Use `--help`** for documentation of each script


**`metamon/il/`** is a basic imitation learning pipeline that might be a useful template / starting point for playing with model architectures and new datasets. This code is a remnant of an early version of this project, and the final paper's experiments only use this for a few "BC-RNN" baselines that were mostly pushed to the Appendix.

`python -m metamon.il.train`

**`metamon/rl/`** connects Metamon to `amago`, which powers the main IL and RL experiments in the paper. 

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
@misc{grigsby2025humanlevelcompetitivepokemonscalable,
      title={Human-Level Competitive Pok\'emon via Scalable Offline Reinforcement Learning with Transformers}, 
      author={Jake Grigsby and Yuqi Xie and Justin Sasek and Steven Zheng and Yuke Zhu},
      year={2025},
      eprint={2504.04395},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04395}, 
}
```