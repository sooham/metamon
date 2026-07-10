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
Before V attention, each sample's valid header and current-state tokens are
packed contiguously; this keeps posterior latents invariant to the lengths of
other samples in the batch.

## Stages

```bash
# 1. Train V (100k optimizer updates by default)
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

The first V Make run targets 100K optimizer updates. Once `v_latest.pt` exists,
each later invocation resumes model and AdamW state and adds another 50K
updates: 100K to 150K, 150K to 200K, and so on. `v_best.pt` continues to track
the best validation reconstruction. Override the interval with
`SIMPLE_WM_ADDITIONAL_UPDATES`, set an absolute target with
`SIMPLE_WM_MAX_UPDATES`, or force a fresh run with
`make train-simple-world-model-v SIMPLE_WM_RESUME=false`.

For a direct CLI resume, add:

```bash
--resume_checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v_latest.pt" \
--additional_updates 50000
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

## Weights & Biases logging

V, M, and C runs log to Weights & Biases by default. Authenticate once in the
same environment that launches training:

```bash
uv run wandb login
```

The Make targets use the `metamon-simple-world-model` project by default. Set
a project and an optional per-run name like this:

```bash
make train-simple-world-model-m \
  SIMPLE_WM_WANDB_PROJECT=my-world-models \
  SIMPLE_WM_WANDB_NAME=m-gen1-gen9
```

Training metrics are logged every `SIMPLE_WM_WANDB_LOG_INTERVAL` optimizer
updates (the console interval by default), and validation metrics are logged at
each validation step. Runs include the stage, model configuration, dataset
manifest hash, cache manifest hash for M/C, throughput, gradient norm, learning
rate, and CUDA memory metrics. To run locally without W&B, set
`SIMPLE_WM_WANDB=false`.

At play time, C checkpoints use horizon-4, eight-rollout action evaluation by
default.  Their final score is `0.75 * risk-adjusted rollout value + 0.25 *
normalized C prior`.
