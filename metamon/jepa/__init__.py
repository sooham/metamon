# JEPA (Joint Embedding Predictive Architecture) for Pokémon battle replays.
#
# Unlike the SL world model (metamon/sl/) which predicts next state tokens,
# JEPA learns representations directly from raw replay strings with:
#   1. Contrastive JEPA loss — masked-view encoder → predictor → unmasked-view target
#   2. β-VAE loss — reconstruction + KL divergence regularizing latents toward
#      an isotropic Gaussian prior.
#
# Files:
#   model.py       — transformers, encoders, predictor, decoder, loss functions
#   train.py       — training loop, dataset (raw replays → BPE tokens → batches)
#   configs/        — model hyperparameter YAML files
