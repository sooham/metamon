# V/M/C simple world model

The simple world model is a staged pipeline.  Old combined checkpoints and
old cache artifacts are intentionally incompatible.

```text
V: team header + current visible state -> q(z_t)
M: team_z, z_0, own_a_0, opp_a_0, ..., z_t -> h_t
C: z_t, h_t, legal own actions -> behavior-cloning prior
```

V reconstructs only the current state.  The header is encoder context, never a
decoder target.  M retains observed temporal context in a causal latent
Transformer rather than asking a token VAE to compress a full battle history.

## Stages

```bash
# 1. Train V (50k optimizer updates by default)
uv run python -m metamon.simple_world_model.train \
  --stage v --data_root "$METAMON_CACHE_DIR/world-model-samples" \
  --formats gen1ou gen9ou --tokenizer_path "$TOKENIZER" \
  --save_dir "$METAMON_CACHE_DIR/simple-world-model-checkpoints" \
  --checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v_best.pt"

# 2. Cache full p1/p2 posterior means and log variances (check free disk first)
uv run python -m metamon.simple_world_model.cache_latents \
  --data_root "$METAMON_CACHE_DIR/world-model-samples" --formats gen1ou gen9ou \
  --v_checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v_best.pt" \
  --latent_cache_root "$METAMON_CACHE_DIR/simple-world-model-latents"

# 3. Train causal M (100k updates by default)
uv run python -m metamon.simple_world_model.train \
  --stage m --data_root "$METAMON_CACHE_DIR/world-model-samples" \
  --formats gen1ou gen9ou --tokenizer_path "$TOKENIZER" \
  --v_checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v_best.pt" \
  --latent_cache_root "$METAMON_CACHE_DIR/simple-world-model-latents" \
  --save_dir "$METAMON_CACHE_DIR/simple-world-model-checkpoints" \
  --checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/m_best.pt"

# 4. Freeze V/M and train C (50k updates by default)
uv run python -m metamon.simple_world_model.train \
  --stage c --data_root "$METAMON_CACHE_DIR/world-model-samples" \
  --formats gen1ou gen9ou --tokenizer_path "$TOKENIZER" \
  --v_checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v_best.pt" \
  --m_checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/m_best.pt" \
  --latent_cache_root "$METAMON_CACHE_DIR/simple-world-model-latents" \
  --save_dir "$METAMON_CACHE_DIR/simple-world-model-checkpoints" \
  --checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/c_best.pt"
```

The cache manifest binds the dataset metadata hash, tokenizer state hash, V
checkpoint hash, latent dimension, dtype, schema version, canonical action
vocabulary, and sidecar coverage.  M/C refuse a cache that differs from the
requested dataset or V checkpoint.

## Make targets

Run the stages in order:

```bash
make train-simple-world-model-v
make cache-simple-world-model-latents
make train-simple-world-model-m
make train-simple-world-model-c
```

Each training target launches a `simple-world-model-train` tmux session and
writes a per-stage log under `simple-world-model-checkpoints`:

```bash
tmux attach -t simple-world-model-train
tail -f "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v.log"
```

V indexes the production shards compactly before its first batch; this is a
short startup scan, not a stalled GPU job.  Its action vocabulary is deferred
to the cache stage because V has no action objective.  The cache stage performs
the one-time full observed-and-legal-action scan.  With `SIMPLE_WM_COMPILE=true`
(the default), the first V optimizer step also includes PyTorch compilation.
Use `SIMPLE_WM_COMPILE=false` for a quick diagnostic run.

At play time, C checkpoints use horizon-4, eight-rollout action evaluation by
default.  Their final score is `0.75 * risk-adjusted rollout value + 0.25 *
normalized C prior`.
