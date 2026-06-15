# LeJEPA (Latent-Euclidean Joint Embedding Predictive Architecture)
# for Pokémon battle replay state learning.
#
# Learns an encoder that maps battle states to deterministic embeddings *e*,
# and a predictor that maps (e_prev, action) → e_next.  Embeddings are
# regularized toward an isotropic Gaussian via SIGReg (Epps-Pulley test
# over sketched random projections).  No VAE decoder, no stop-gradient,
# no teacher-student — a single hyperparameter λ.
#
# Reference: Balestriero & LeCun (2025), "LeJEPA: Provable and Scalable
# Self-Supervised Learning Without the Heuristics".
#
# Files:
#   model.py       — transformers, encoder, predictor, SIGReg, loss functions
#   train.py       — training loop, dataset (npz shards → token batches)
#   configs/        — model hyperparameter YAML files
