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
# 1. Train V (200k optimizer updates by default)
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

The default V recipe is a fresh 200K-update run with a fixed batch size of 128,
AdamW weight decay 0.01, and gradient clipping at 1.0. The learning rate warms
linearly for 2K updates to 3e-5, then follows a cosine decay to 3e-6 at update
200K. Validation runs every 5K updates and training stops after ten
consecutive validations without an improvement in reconstruction CE.
V validation also reports CE and token accuracy (plus across-draw standard
deviations) from four fixed posterior draws by default under `val/*_mc`.
Posterior standard deviation, expected sampling-noise norm, and expected sampled
latent norm are logged alongside the deterministic mean norm. The deterministic
`z=mu` reconstruction remains the checkpoint-selection metric. Override the
draw count with `SIMPLE_WM_VAL_MC_SAMPLES`.
Validation also audits the aggregate posterior against the standard-normal
prior. `aggregate_mean_*`, `aggregate_std_*`, covariance error, and the KL of a
moment-matched full-covariance Gaussian reveal population-level drift or latent
correlation that a scalar per-example KL can obscure.

The standard-normal constraint is on posterior samples
`z = mu + exp(0.5 * logvar) * epsilon`, not on `mu` by itself. In a well-formed
VAE, variation of posterior means plus average posterior variance together make
the aggregate sampled latent approximately unit variance. Requiring both the
means and samples independently to have unit variance would over-dispense the
sampled latent. The exact average conditional KL also upper-bounds aggregate
posterior KL, while including the mutual information retained about the input.
Each validation also evaluates 2K fixed samples from the training split through
the identical deterministic and sampled-posterior path. Metrics named
`val/train_eval_*` and `val/generalization_gap_*` therefore measure a matched
generalization gap instead of comparing validation to one stochastic training
batch. Set `SIMPLE_WM_TRAIN_EVAL_SAMPLES=0` to disable this extra pass.

V uses a 512-dimensional latent with `beta_kl=0.01`, 0.02 free bits, and a
20K-update KL warmup. Encoder token masking is initially disabled. If enabled,
the configured fraction of valid content tokens is replaced by `<unk>` only in
the encoder input; structural tags and the clean reconstruction target remain
untouched. V's Transformer dropout is 0.0. Training logs both the pre-clipping
gradient norm and the fraction of recent optimizer updates whose norm exceeded
the 1.0 threshold. `train_smooth/*` metrics and the console `avgN` loss average
the latest 100 optimizer updates by default; change the window with
`SIMPLE_WM_TRAIN_METRIC_WINDOW`.

`SIMPLE_WM_MEAN_RECON_WEIGHT` defaults to `0.25` in the Make recipe, using 75%
sampled reconstruction plus 25% `z=mu` reconstruction. The interpolation keeps
the total reconstruction and KL scales constant. This keeps the deterministic
downstream representation on the decoder's training manifold. The direct CLI
default remains `0` so standard sampled-only VAE experiments must opt into the
extra decoder pass explicitly.

V no longer auto-resumes merely because `v_latest.pt` exists. Use a new save
directory for the restart so the previous artifacts remain available:

```bash
make train-simple-world-model-v \
  SIMPLE_WM_RESUME=false \
  SIMPLE_WM_SAVE_DIR="$METAMON_CACHE_DIR/simple-world-model-checkpoints-v2"
```

To continue an interrupted run made with the same model and dataset, set
`SIMPLE_WM_RESUME=true`. The default absolute target remains update 200K;
use `SIMPLE_WM_ADDITIONAL_UPDATES` only when intentionally extending a
finished schedule. `v_best.pt` tracks the best validation reconstruction.

V samples raw battles with weight `num_states ** alpha`, where `num_states`
is the average non-header state count across the paired POVs, then selects one
POV and one state within the chosen battle. The Make default is
`SIMPLE_WM_V_SAMPLING_ALPHA=0.5`, a compromise between uniform battles
(`0`) and uniform states (`1`). Override it, for example, with:

```bash
make train-simple-world-model-v SIMPLE_WM_V_SAMPLING_ALPHA=0
```

For a direct CLI resume of an interrupted matching run, add:

```bash
--resume_checkpoint "$METAMON_CACHE_DIR/simple-world-model-checkpoints/v_latest.pt" \
--max_updates 200000
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
