import metamon
from metamon.rl import (
    pretrained_vs_pokeagent_ladder,
    LocalPretrainedModel,
    LocalFinetunedModel,
)
from metamon.rl.pretrained import SmallRL

"""
In this example, let's say we trained a new model from scratch with:

python -m metamon.rl.train \\
    --run_name gen9v3 \\
    --model_gin_config medium_multitaskagent.gin \\
    --save_dir ~/metamon_ckpts/ \\
    --train_gin_config binary_rl.gin \\
    --obs_space TeamPreviewObservationSpace \\
    --tokenizer DefaultObservationSpace-v1 \\
    --log
"""
MyCustomModel = LocalPretrainedModel(
    amago_ckpt_dir="~/metamon_ckpts/",
    model_name="gen9v3",
    model_gin_config="medium_multitaskagent.gin",
    train_gin_config="binary_rl.gin",
    default_checkpoint=40,
    action_space=metamon.interface.DefaultActionSpace(),
    observation_space=metamon.interface.TeamPreviewObservationSpace(),
    tokenizer=metamon.tokenizer.get_tokenizer("DefaultObservationSpace-v1"),
)

"""
Then let's say we finetuned SmallRL to Gen9 with:

python -m metamon.rl.finetune_from_hf \\
    --finetune_from_model SmallRL \\
    --run_name smallrlfinetune \\
    --save_dir ~/metamon_ckpts/ \\
    --steps_per_epoch 10000 \\
    --epochs 3 \\
    --eval_gens 9 \\
    --formats gen9ou \\
    --log
"""
MyFinetunedModel = LocalFinetunedModel(
    base_model=SmallRL,
    amago_ckpt_dir="~/metamon_ckpts/",
    model_name="smallrlfinetune",
    default_checkpoint=2,
)

teams = metamon.env.get_metamon_teams("gen1ou", "competitive")
# or create a custom set of teams (metamon.env.TeamSet)
results = pretrained_vs_pokeagent_ladder(
    pretrained_model=MyFinetunedModel,
    username="PAC-MyTeamName",
    password="my_password",
    battle_format="gen1ou",
    team_set=teams,
    total_battles=10,
)
print(results)
