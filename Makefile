# Detect OS and number of CPU cores
OS := $(shell uname -s)
N_THREADS := $(shell expr \( $$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) + 1 \))

ifeq ($(OS),Darwin)
METAMON_CACHE_DIR ?= /Users/srafiz/Repositories/poke-datasets
else
METAMON_CACHE_DIR ?= /workspace/poke-datasets
endif
RAW_REPLAY_DIR ?= $(METAMON_CACHE_DIR)/raw-replays
FORMAT ?= gen1ou gen9ou
FORMATS ?= $(FORMAT)

.PHONY: parse parse-all upload-parsed-wm-replays show-battle show-rand-battle \
        wm-tokenizer \
        wm-dataset upload-wm-dataset \
        test test-quick test-forward test-backward test-e2e \
        clean show-tokenizer clean-tokenizer \
        train-jepa train-jepa-debug _train-jepa-inner \
        train-simple-world-model train-simple-world-model-debug _train-simple-world-model-inner \
        train-simple-world-model-v cache-simple-world-model-latents \
        train-simple-world-model-m train-simple-world-model-c \
        play-simple-world-model play-simple-world-model-local \
        play-jepa play-jepa-local \
        showdown ensure-showdown showdown-daemon showdown-status showdown-stop \
        showdown-install-autostart showdown-uninstall-autostart \
        save-checkpoint save-checkpoints bash-completion

# Start a local Pokemon Showdown server (no auth, port 8000)
# Requires the server/pokemon-showdown submodule to be initialized.
NODE ?= node
SHOWDOWN_WATCHDOG ?= tools/showdown_watchdog.sh
showdown:
	cd server/pokemon-showdown && $(NODE) pokemon-showdown start --no-security

# Ensure a local Showdown server is running and supervised (starts one if not).
# Used as a dependency by targets that need a local server.
ensure-showdown:
	@$(SHOWDOWN_WATCHDOG) start

# Run the local Showdown server under a watchdog in a detached tmux session.
showdown-daemon:
	@$(SHOWDOWN_WATCHDOG) start

showdown-status:
	@$(SHOWDOWN_WATCHDOG) status

showdown-stop:
	@$(SHOWDOWN_WATCHDOG) stop

# Install/remove a cron entry that restarts the tmux watchdog after reboot.
showdown-install-autostart:
	@$(SHOWDOWN_WATCHDOG) install-cron

showdown-uninstall-autostart:
	@$(SHOWDOWN_WATCHDOG) uninstall-cron

# Open a battle replay in browser + parsed output in Cursor
# Usage: make show-battle BATTLE_ID=smogtours-gen1ou-694141
BATTLE_ID ?=
show-battle:
	@open https://replay.pokemonshowdown.com/$(BATTLE_ID)
	@format=$$(echo $(BATTLE_ID) | sed -E 's/^(smogtours-)?//;s/-[0-9]+$$//'); \
	dir="$(METAMON_CACHE_DIR)/parsed-replays/$$format"; \
	if [ -d "$$dir" ]; then \
		find "$$dir" -name "$(BATTLE_ID)_*" -print0 | xargs -0 -n 1 cursor; \
	else \
		echo "No parsed directory: $$dir"; \
	fi

# Catch-all to prevent "No rule to make target" errors for positional arguments
%:
	@:

# Parse one format with default NaiveUsagePredictor
parse:
	uv run python -m metamon.backend.replay_parser \
		--format $(FORMAT) \
		--raw_replay_dir $(METAMON_CACHE_DIR)/raw-replays \
		--output_dir $(METAMON_CACHE_DIR)/parsed-replays \
		--processes $(N_THREADS) --no-compress --pretty

# Parse all supported formats
parse-all:
	@for fmt in gen1ou gen9ou; do \
		echo "=== $$fmt ==="; \
		$(MAKE) parse FORMAT=$$fmt; \
	done

PARSED_WM_REPLAY_REPO ?= sooham34/metamon-parsed-wm-replays
PARSED_WM_REPLAY_REVISION ?= main
PARSED_WM_REPLAY_PRIVATE ?= 0
PARSED_WM_REPLAY_DRY_RUN ?= 0
PARSED_WM_REPLAY_ROOT ?= $(METAMON_CACHE_DIR)/parsed-replays
# Upload online-play saves with:
#   make upload-parsed-wm-replays FORMAT=gen1ou PARSED_WM_REPLAY_ROOT=$(METAMON_CACHE_DIR)/online-play
# Multiple formats at once:
#   make upload-parsed-wm-replays FORMATS="gen1ou gen9ou"
upload-parsed-wm-replays:
	uv run python scripts/upload_parsed_wm_replays.py \
		--formats $(FORMATS) \
		--parsed_replay_root $(PARSED_WM_REPLAY_ROOT) \
		--repo_id $(PARSED_WM_REPLAY_REPO) \
		--revision $(PARSED_WM_REPLAY_REVISION) \
		$(if $(filter 1 true yes,$(PARSED_WM_REPLAY_PRIVATE)),--private,) \
		$(if $(filter 1 true yes,$(PARSED_WM_REPLAY_DRY_RUN)),--dry_run,)

# Inspect a random sample of 5 parsed battles from a format (one at a time in Cursor + browser)
# Usage: make show-rand-battle FORMAT=gen1ou
show-rand-battle:
	@dir="$(METAMON_CACHE_DIR)/parsed-replays/$(FORMAT)"; \
	if [ ! -d "$$dir" ]; then \
		echo "No parsed directory for format $(FORMAT): $$dir"; \
		exit 1; \
	fi; \
	count=$$(find "$$dir" -name '*.txt' -type f | wc -l | tr -d ' '); \
	if [ "$$count" -eq 0 ]; then \
		echo "No parsed files found in $$dir"; \
		exit 1; \
	fi; \
	echo "=== Sampling 5 of $$count battles from $(FORMAT) ==="; \
	find "$$dir" -name '*.txt' -type f | sort -R | head -5 | while IFS= read -r f; do \
		echo "--- $$(basename "$$f") ---"; \
		battle_id=$$(basename "$$f" .txt | cut -d_ -f1); \
		open "https://replay.pokemonshowdown.com/$$battle_id"; \
		cursor "$$f"; \
	done

# ── World Model Targets ──────────────────────────────────────────────

# Build a tokenizer vocabulary for WorldModelObservationSpace from parsed replays.
# Scans all replays in the parsed directory for the given formats, collects every
# unique word in the token text observations, and saves to a JSON file.
#
# Usage:
#   make tokenize-world-model FORMATS=gen1ou
#   make tokenize-world-model FORMATS="gen1ou gen9ou"
#
# Start from an existing tokenizer to only add new tokens:
#   make tokenize-world-model FORMATS=gen1ou \
#       START_TOKENS=WorldModelObservationSpace-v0 \
#       TOKENIZER_VERSION=WorldModelObservationSpace-v1
TOKENIZER_OUTPUT_DIR ?= $(METAMON_CACHE_DIR)/tokenizers
TOKENIZER_VERSION ?= WorldModelObservationSpace-v1
NUM_WORKERS ?= $(N_THREADS)
EARLY_STOP ?= 0
wm-tokenizer:
	mkdir -p $(TOKENIZER_OUTPUT_DIR)
	uv run python -m metamon.tokenizer.tokenizer \
		--parsed_replay_root $(METAMON_CACHE_DIR)/parsed-replays \
		--formats $(FORMATS) \
		--num_workers $(NUM_WORKERS) \
		--early_stop $(EARLY_STOP) \
		--save_tokens $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json

# Generate world-model training data from parsed replays.
# Automatically builds the WorldModel tokenizer if it doesn't exist yet.
# Aborts early if parsed replays are missing for any requested format.
#
# Each output .npz shard contains paired POV transition data:
#   p1_* / p2_* state + action arrays, plus aligned transition rows.
#
# Usage:
#   make wm-dataset FORMATS=gen1ou
#   make wm-dataset FORMATS="gen1ou gen9ou"
WM_OUTPUT_DIR ?= $(METAMON_CACHE_DIR)/world-model-samples
WM_PROCESSES ?= $(N_THREADS)
WM_VAL_SPLIT ?= 0.05
WM_SEED ?= 42
WM_ROLLOUT_LEN ?= 1
TOKENIZER_FILE := $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json
wm-dataset:
	@# ---- 1. Check parsed replays exist for every format ----
	@missing=""; \
	for fmt in $(FORMATS); do \
		dir="$(METAMON_CACHE_DIR)/parsed-replays/$$fmt"; \
		if [ ! -d "$$dir" ] || [ -z "$$(ls -A "$$dir" 2>/dev/null)" ]; then \
			missing="$$missing $$fmt"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "ERROR: No parsed replays found for:$$missing"; \
		echo "  Run 'make parse FORMAT=<format>' first for each format."; \
		exit 1; \
	fi
	@# ---- 2. Build tokenizer if missing ----
	@if [ ! -f "$(TOKENIZER_FILE)" ]; then \
		echo "Tokenizer $(TOKENIZER_FILE) not found — building it now..."; \
		$(MAKE) wm-tokenizer FORMATS="$(FORMATS)"; \
	fi
	@# ---- 3. Generate sharded .npz files ----
	mkdir -p $(WM_OUTPUT_DIR)
	uv run python scripts/generate_world_model_data.py \
		--parsed_replay_root $(METAMON_CACHE_DIR)/parsed-replays \
		--tokenizer_path $(TOKENIZER_FILE) \
		--output_dir $(WM_OUTPUT_DIR) \
		--formats $(FORMATS) \
		--val_split $(WM_VAL_SPLIT) \
		--seed $(WM_SEED) \
		--rollout_len $(WM_ROLLOUT_LEN) \
		--processes $(WM_PROCESSES)

WM_DATASET_REPO ?= sooham34/metamon-wm-dataset
WM_DATASET_REVISION ?= main
WM_DATASET_PRIVATE ?= 0
WM_DATASET_DRY_RUN ?= 0
# One or more formats, matching the make wm-dataset invocation:
#   make upload-wm-dataset FORMATS="gen1ou gen9ou"
# (single format: FORMATS=gen1ou still works)
upload-wm-dataset:
	uv run python scripts/upload_wm_dataset.py \
		--formats $(FORMATS) \
		--output_dir $(WM_OUTPUT_DIR) \
		--repo_id $(WM_DATASET_REPO) \
		--revision $(WM_DATASET_REVISION) \
		$(if $(filter 1 true yes,$(WM_DATASET_PRIVATE)),--private,) \
		$(if $(filter 1 true yes,$(WM_DATASET_DRY_RUN)),--dry_run,)

# SL targets removed — SL model deleted.

# ── JEPA Training ────────────────────────────────────────────────────

JEPA_DATA_ROOT ?= $(WM_OUTPUT_DIR)
JEPA_TOKENIZER ?= $(TOKENIZER_FILE)
JEPA_SAVE_DIR ?= $(METAMON_CACHE_DIR)/jepa-checkpoints

JEPA_LR ?= 5e-5
JEPA_EPOCHS ?= 10
JEPA_GRAD_CLIP ?= 1.0
JEPA_NUM_WORKERS ?= $(shell python3 -c 'import os; n=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4); print(min(12, n))')
JEPA_PREFETCH_FACTOR ?= 4
JEPA_PRINT_INTERVAL ?= 10
JEPA_CONFIG ?= metamon/jepa/configs/default.yaml
JEPA_COMPILE ?= true
JEPA_MAX_HISTORY ?= 64

# Train the paired-POV JEPA model on paired_shard_*.npz files.
# Requires paired data generated with scripts/generate_world_model_data.py --paired_pov.
#
# Automatically launches inside a tmux session named 'jepa-train' so the job
# survives SSH disconnects.  Attach/detach with:
#   tmux attach -t jepa-train    # reattach to watch progress
#   Ctrl+B, D                    # detach (leave running)
#
# Usage:
#   make train-jepa FORMATS=gen1ou
#   make train-jepa FORMATS=gen1ou JEPA_MAX_HISTORY=0  # full battle history
JEPA_PAIRED_BATCH_SIZE ?= 128
JEPA_PAIRED_GRAD_ACCUM_STEPS ?= 1
JEPA_PAIRED_CHECKPOINT ?= $(JEPA_SAVE_DIR)/paired_best_stochastic.pt
JEPA_PAIRED_MAX_STEPS ?= 0
JEPA_PAIRED_VAL_INTERVAL ?= 100
JEPA_PAIRED_VAL_MAX_BATCHES ?= 5
JEPA_CONSOLE_INTERVAL ?= 10
JEPA_WANDB_INTERVAL ?= 10
JEPA_ENCODER_CHUNK_TOKENS ?= 131072
JEPA_BELIEF_BATCH_SIZE ?= 1024
JEPA_PAIRED_EXTRA_ARGS ?=

JEPA_DEBUG_BATCH_SIZE ?= 1
JEPA_DEBUG_MAX_STEPS ?= 1
JEPA_DEBUG_TENSOR_STEPS ?= 1
JEPA_DEBUG_TENSOR_VALUES ?= 64
JEPA_DEBUG_TENSOR_SAMPLES ?= 2

train-jepa:
	@if [ -z "$$TMUX" ]; then \
		if tmux has-session -t jepa-train 2>/dev/null; then \
			echo "tmux session 'jepa-train' already exists."; \
			echo "  Attach:  tmux attach -t jepa-train"; \
			echo "  Kill:    tmux kill-session -t jepa-train"; \
			exit 1; \
		fi; \
		echo "Launching training in tmux session 'jepa-train'..."; \
		tmux new-session -d -s jepa-train "$(MAKE) _train-jepa-inner FORMAT='$(FORMAT)' FORMATS='$(FORMATS)' WANDB='$(WANDB)' WANDB_PROJECT='$(WANDB_PROJECT)' WANDB_NAME='$(WANDB_NAME)' JEPA_COMPILE='$(JEPA_COMPILE)' JEPA_MAX_HISTORY='$(JEPA_MAX_HISTORY)' JEPA_PAIRED_BATCH_SIZE='$(JEPA_PAIRED_BATCH_SIZE)' JEPA_PAIRED_GRAD_ACCUM_STEPS='$(JEPA_PAIRED_GRAD_ACCUM_STEPS)' JEPA_LR='$(JEPA_LR)' JEPA_EPOCHS='$(JEPA_EPOCHS)' JEPA_NUM_WORKERS='$(JEPA_NUM_WORKERS)' JEPA_PREFETCH_FACTOR='$(JEPA_PREFETCH_FACTOR)' JEPA_GRAD_CLIP='$(JEPA_GRAD_CLIP)' JEPA_PAIRED_MAX_STEPS='$(JEPA_PAIRED_MAX_STEPS)' JEPA_PAIRED_VAL_INTERVAL='$(JEPA_PAIRED_VAL_INTERVAL)' JEPA_PAIRED_VAL_MAX_BATCHES='$(JEPA_PAIRED_VAL_MAX_BATCHES)' JEPA_CONSOLE_INTERVAL='$(JEPA_CONSOLE_INTERVAL)' JEPA_WANDB_INTERVAL='$(JEPA_WANDB_INTERVAL)' JEPA_ENCODER_CHUNK_TOKENS='$(JEPA_ENCODER_CHUNK_TOKENS)' JEPA_BELIEF_BATCH_SIZE='$(JEPA_BELIEF_BATCH_SIZE)' JEPA_PAIRED_EXTRA_ARGS='$(JEPA_PAIRED_EXTRA_ARGS)'"; \
		echo ""; \
		echo "  Attach:  tmux attach -t jepa-train"; \
		echo "  Detach:  Ctrl+B, D"; \
		echo "  Kill:    tmux kill-session -t jepa-train"; \
	else \
		$(MAKE) _train-jepa-inner; \
	fi

# Run a small paired-POV JEPA training step with tensor debug dumps enabled.
#
# This stays in the foreground instead of tmux so the batch/model tensor logs are
# immediately visible in the current terminal.
#
# Usage:
#   make train-jepa-debug FORMATS=gen1ou
#   make train-jepa-debug JEPA_DEBUG_TENSOR_VALUES=128 JEPA_DEBUG_TENSOR_SAMPLES=1 JEPA_MAX_HISTORY=4
train-jepa-debug:
	$(MAKE) _train-jepa-inner \
		FORMAT='$(FORMAT)' \
		FORMATS='$(FORMATS)' \
		WANDB=false \
		JEPA_COMPILE=false \
		JEPA_PAIRED_BATCH_SIZE='$(JEPA_DEBUG_BATCH_SIZE)' \
		JEPA_PAIRED_GRAD_ACCUM_STEPS=1 \
		JEPA_PAIRED_MAX_STEPS='$(JEPA_DEBUG_MAX_STEPS)' \
		JEPA_PAIRED_VAL_INTERVAL=0 \
		JEPA_PAIRED_VAL_MAX_BATCHES=0 \
		JEPA_CONSOLE_INTERVAL=1 \
		JEPA_WANDB_INTERVAL=0 \
		JEPA_NUM_WORKERS=0 \
		JEPA_PREFETCH_FACTOR=2 \
		JEPA_PAIRED_EXTRA_ARGS="--debug_tensors --debug_tensor_steps $(JEPA_DEBUG_TENSOR_STEPS) --debug_tensor_values $(JEPA_DEBUG_TENSOR_VALUES) --debug_tensor_samples $(JEPA_DEBUG_TENSOR_SAMPLES) $(JEPA_PAIRED_EXTRA_ARGS)"

_train-jepa-inner:
	@if [ ! -d "$(JEPA_DATA_ROOT)" ]; then \
		echo "ERROR: No paired .npz data found at $(JEPA_DATA_ROOT)."; \
		exit 1; \
	fi
	@if [ ! -f "$(JEPA_TOKENIZER)" ]; then \
		echo "ERROR: Tokenizer not found at $(JEPA_TOKENIZER)."; \
		exit 1; \
	fi
	mkdir -p $(JEPA_SAVE_DIR)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	uv run python -m metamon.jepa.train_paired \
		--data_root $(JEPA_DATA_ROOT) \
		--formats $(FORMATS) \
		--tokenizer_path $(JEPA_TOKENIZER) \
		--save_dir $(JEPA_SAVE_DIR) \
		--batch_size $(JEPA_PAIRED_BATCH_SIZE) \
		--grad_accum_steps $(JEPA_PAIRED_GRAD_ACCUM_STEPS) \
		--lr $(JEPA_LR) \
		--epochs $(JEPA_EPOCHS) \
		--max_steps $(JEPA_PAIRED_MAX_STEPS) \
		--grad_clip $(JEPA_GRAD_CLIP) \
		--num_workers $(JEPA_NUM_WORKERS) \
		--prefetch_factor $(JEPA_PREFETCH_FACTOR) \
		--print_interval $(JEPA_CONSOLE_INTERVAL) \
		$(if $(JEPA_PAIRED_CHECKPOINT),--checkpoint $(JEPA_PAIRED_CHECKPOINT)) \
		$(if $(JEPA_CONFIG),--config $(JEPA_CONFIG)) \
		--log_interval $(JEPA_WANDB_INTERVAL) \
		$(if $(filter false,$(WANDB)),--no-wandb) \
		$(if $(WANDB_PROJECT),--wandb_project $(WANDB_PROJECT)) \
		$(if $(WANDB_NAME),--wandb_name $(WANDB_NAME)) \
		--val_interval $(JEPA_PAIRED_VAL_INTERVAL) \
		--val_max_batches $(JEPA_PAIRED_VAL_MAX_BATCHES) \
		$(if $(filter false,$(JEPA_COMPILE)),--no-compile) \
		--encoder_chunk_tokens $(JEPA_ENCODER_CHUNK_TOKENS) \
		--belief_batch_size $(JEPA_BELIEF_BATCH_SIZE) \
		--max_history_blocks $(JEPA_MAX_HISTORY) \
		$(JEPA_PAIRED_EXTRA_ARGS)

# ── Simple World Model Training ─────────────────────────────────────

SIMPLE_WM_DATA_ROOT ?= $(WM_OUTPUT_DIR)
SIMPLE_WM_TOKENIZER ?= $(TOKENIZER_FILE)
SIMPLE_WM_SAVE_DIR ?= $(METAMON_CACHE_DIR)/simple-world-model-checkpoints
SIMPLE_WM_CONFIG ?= metamon/simple_world_model/configs/default.yaml
SIMPLE_WM_STAGE ?= v
SIMPLE_WM_CACHE_ROOT ?= $(METAMON_CACHE_DIR)/simple-world-model-latents
SIMPLE_WM_V_CHECKPOINT ?= $(SIMPLE_WM_SAVE_DIR)/v_best.pt
SIMPLE_WM_M_CHECKPOINT ?= $(SIMPLE_WM_SAVE_DIR)/m_best.pt
SIMPLE_WM_C_CHECKPOINT ?= $(SIMPLE_WM_SAVE_DIR)/c_best.pt
SIMPLE_WM_CHECKPOINT ?= $(SIMPLE_WM_SAVE_DIR)/$(SIMPLE_WM_STAGE)_best.pt
SIMPLE_WM_LR ?= 5e-5
SIMPLE_WM_BATCH_SIZE ?= 32
SIMPLE_WM_GRAD_ACCUM_STEPS ?= 1
SIMPLE_WM_GRAD_CLIP ?= 1.0
SIMPLE_WM_MAX_UPDATES ?= 0
SIMPLE_WM_NUM_WORKERS ?= $(shell python3 -c 'import os; n=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4); print(min(12, n))')
SIMPLE_WM_VAL_INTERVAL ?= 5000
SIMPLE_WM_VAL_SAMPLES ?= 10000
SIMPLE_WM_CONSOLE_INTERVAL ?= 100
SIMPLE_WM_COMPILE ?= true
SIMPLE_WM_MAX_CONTEXT_TRANSITIONS ?= 32
SIMPLE_WM_EXTRA_ARGS ?=

SIMPLE_WM_DEBUG_BATCH_SIZE ?= 1
SIMPLE_WM_DEBUG_MAX_STEPS ?= 1
SIMPLE_WM_DEBUG_TENSOR_STEPS ?= 1
SIMPLE_WM_DEBUG_TENSOR_VALUES ?= 64
SIMPLE_WM_DEBUG_TENSOR_SAMPLES ?= 2

train-simple-world-model:
	@if [ -z "$$TMUX" ]; then \
		if tmux has-session -t simple-world-model-train 2>/dev/null; then \
			echo "tmux session 'simple-world-model-train' already exists."; \
			echo "  Attach:  tmux attach -t simple-world-model-train"; \
			echo "  Kill:    tmux kill-session -t simple-world-model-train"; \
			exit 1; \
		fi; \
		echo "Launching training in tmux session 'simple-world-model-train'..."; \
		tmux new-session -d -s simple-world-model-train "$(MAKE) _train-simple-world-model-inner FORMAT='$(FORMAT)' FORMATS='$(FORMATS)' SIMPLE_WM_STAGE='$(SIMPLE_WM_STAGE)' SIMPLE_WM_CACHE_ROOT='$(SIMPLE_WM_CACHE_ROOT)' SIMPLE_WM_V_CHECKPOINT='$(SIMPLE_WM_V_CHECKPOINT)' SIMPLE_WM_M_CHECKPOINT='$(SIMPLE_WM_M_CHECKPOINT)' SIMPLE_WM_CHECKPOINT='$(SIMPLE_WM_CHECKPOINT)' SIMPLE_WM_COMPILE='$(SIMPLE_WM_COMPILE)' SIMPLE_WM_MAX_CONTEXT_TRANSITIONS='$(SIMPLE_WM_MAX_CONTEXT_TRANSITIONS)' SIMPLE_WM_BATCH_SIZE='$(SIMPLE_WM_BATCH_SIZE)' SIMPLE_WM_GRAD_ACCUM_STEPS='$(SIMPLE_WM_GRAD_ACCUM_STEPS)' SIMPLE_WM_LR='$(SIMPLE_WM_LR)' SIMPLE_WM_NUM_WORKERS='$(SIMPLE_WM_NUM_WORKERS)' SIMPLE_WM_GRAD_CLIP='$(SIMPLE_WM_GRAD_CLIP)' SIMPLE_WM_MAX_UPDATES='$(SIMPLE_WM_MAX_UPDATES)' SIMPLE_WM_VAL_INTERVAL='$(SIMPLE_WM_VAL_INTERVAL)' SIMPLE_WM_VAL_SAMPLES='$(SIMPLE_WM_VAL_SAMPLES)' SIMPLE_WM_CONSOLE_INTERVAL='$(SIMPLE_WM_CONSOLE_INTERVAL)' SIMPLE_WM_EXTRA_ARGS='$(SIMPLE_WM_EXTRA_ARGS)'"; \
		echo ""; \
		echo "  Attach:  tmux attach -t simple-world-model-train"; \
		echo "  Detach:  Ctrl+B, D"; \
		echo "  Kill:    tmux kill-session -t simple-world-model-train"; \
	else \
		$(MAKE) _train-simple-world-model-inner; \
	fi

# Staged V/M/C pipeline. Run these in order; cache creation validates free
# space before writing the full fp16 posterior sidecars.
#
#   make train-simple-world-model-v
#   make cache-simple-world-model-latents
#   make train-simple-world-model-m
#   make train-simple-world-model-c
train-simple-world-model-v:
	$(MAKE) train-simple-world-model \
		SIMPLE_WM_STAGE=v \
		SIMPLE_WM_CHECKPOINT='$(SIMPLE_WM_V_CHECKPOINT)'

cache-simple-world-model-latents:
	$(MAKE) train-simple-world-model \
		SIMPLE_WM_STAGE=cache \
		SIMPLE_WM_CHECKPOINT=

train-simple-world-model-m:
	$(MAKE) train-simple-world-model \
		SIMPLE_WM_STAGE=m \
		SIMPLE_WM_CHECKPOINT='$(SIMPLE_WM_M_CHECKPOINT)'

train-simple-world-model-c:
	$(MAKE) train-simple-world-model \
		SIMPLE_WM_STAGE=c \
		SIMPLE_WM_CHECKPOINT='$(SIMPLE_WM_C_CHECKPOINT)'

train-simple-world-model-debug:
	$(MAKE) _train-simple-world-model-inner \
		FORMAT='$(FORMAT)' \
		FORMATS='$(FORMATS)' \
		SIMPLE_WM_STAGE=v \
		SIMPLE_WM_COMPILE=false \
		SIMPLE_WM_BATCH_SIZE='$(SIMPLE_WM_DEBUG_BATCH_SIZE)' \
		SIMPLE_WM_GRAD_ACCUM_STEPS=1 \
		SIMPLE_WM_MAX_UPDATES='$(SIMPLE_WM_DEBUG_MAX_STEPS)' \
		SIMPLE_WM_VAL_INTERVAL=0 \
		SIMPLE_WM_CONSOLE_INTERVAL=1 \
		SIMPLE_WM_NUM_WORKERS=0 \
		SIMPLE_WM_EXTRA_ARGS="$(SIMPLE_WM_EXTRA_ARGS)"

_train-simple-world-model-inner:
	@if [ ! -d "$(SIMPLE_WM_DATA_ROOT)" ]; then \
		echo "ERROR: No paired .npz data found at $(SIMPLE_WM_DATA_ROOT)."; \
		exit 1; \
	fi
	@if [ ! -f "$(SIMPLE_WM_TOKENIZER)" ]; then \
		echo "ERROR: Tokenizer not found at $(SIMPLE_WM_TOKENIZER)."; \
		exit 1; \
	fi
	mkdir -p $(SIMPLE_WM_SAVE_DIR)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	uv run python -m metamon.simple_world_model.train \
		--stage $(SIMPLE_WM_STAGE) \
		--data_root $(SIMPLE_WM_DATA_ROOT) \
		--formats $(FORMATS) \
		--tokenizer_path $(SIMPLE_WM_TOKENIZER) \
		--save_dir $(SIMPLE_WM_SAVE_DIR) \
		--latent_cache_root $(SIMPLE_WM_CACHE_ROOT) \
		--v_checkpoint $(SIMPLE_WM_V_CHECKPOINT) \
		--m_checkpoint $(SIMPLE_WM_M_CHECKPOINT) \
		--batch_size $(SIMPLE_WM_BATCH_SIZE) \
		--grad_accum_steps $(SIMPLE_WM_GRAD_ACCUM_STEPS) \
		--lr $(SIMPLE_WM_LR) \
		--max_updates $(SIMPLE_WM_MAX_UPDATES) \
		--grad_clip $(SIMPLE_WM_GRAD_CLIP) \
		--num_workers $(SIMPLE_WM_NUM_WORKERS) \
		--print_interval $(SIMPLE_WM_CONSOLE_INTERVAL) \
		$(if $(SIMPLE_WM_CHECKPOINT),--checkpoint $(SIMPLE_WM_CHECKPOINT)) \
		$(if $(SIMPLE_WM_CONFIG),--config $(SIMPLE_WM_CONFIG)) \
		--val_interval $(SIMPLE_WM_VAL_INTERVAL) \
		--val_samples $(SIMPLE_WM_VAL_SAMPLES) \
		$(if $(filter true,$(SIMPLE_WM_COMPILE)),--compile,--no-compile) \
		--max_context_transitions $(SIMPLE_WM_MAX_CONTEXT_TRANSITIONS) \
		$(SIMPLE_WM_EXTRA_ARGS)

SIMPLE_WM_PLAY_CHECKPOINT ?= $(SIMPLE_WM_C_CHECKPOINT)
SIMPLE_WM_PLAY_FORMAT ?= gen1randombattle
SIMPLE_WM_PLAY_USERNAME ?= simplewmbot
SIMPLE_WM_PLAY_TEAM_SET ?= competitive
SIMPLE_WM_PLAY_NUM_BATTLES ?= 30
SIMPLE_WM_PLAY_MAX_CONCURRENT_BATTLES ?= 100
SIMPLE_WM_PLAY_KEEP_LADDER_BATTLES ?= 100
SIMPLE_WM_PLAY_RANDOM_BATTLE_BOT ?= 0
SIMPLE_WM_PLAY_LADDER ?=
SIMPLE_WM_PLAY_VERBOSE_BLOCKS ?=
SIMPLE_WM_PLAY_SERVER ?= showdown
SIMPLE_WM_PLAY_PASSWORD ?= SIMPLEWM
SIMPLE_WM_PLAY_TIMER ?= 1
SIMPLE_WM_PLAY_SAVE ?= 1
SIMPLE_WM_PLAY_SAVE_DIR ?=

play-simple-world-model:
	@if [ ! -f "$(SIMPLE_WM_PLAY_CHECKPOINT)" ]; then \
		echo "ERROR: Checkpoint not found at $(SIMPLE_WM_PLAY_CHECKPOINT)."; \
		echo "  Train first: make train-simple-world-model FORMATS=$(SIMPLE_WM_PLAY_FORMAT)"; \
		exit 1; \
	fi
	uv run python -m metamon.simple_world_model.play \
		--checkpoint $(SIMPLE_WM_PLAY_CHECKPOINT) \
		--format $(SIMPLE_WM_PLAY_FORMAT) \
		--username $(SIMPLE_WM_PLAY_USERNAME) \
		--team_set $(SIMPLE_WM_PLAY_TEAM_SET) \
		$(if $(SIMPLE_WM_PLAY_KEEP_LADDER_BATTLES),,--num_battles $(SIMPLE_WM_PLAY_NUM_BATTLES)) \
		--max_concurrent_battles $(SIMPLE_WM_PLAY_MAX_CONCURRENT_BATTLES) \
		$(if $(SIMPLE_WM_PLAY_KEEP_LADDER_BATTLES),--keep_ladder_battles $(SIMPLE_WM_PLAY_KEEP_LADDER_BATTLES)) \
		$(if $(filter 0 false no,$(SIMPLE_WM_PLAY_RANDOM_BATTLE_BOT)),--no_random_battle_bot) \
		$(if $(SIMPLE_WM_PLAY_LADDER),--ladder) \
		$(if $(SIMPLE_WM_PLAY_VERBOSE_BLOCKS),--verbose_blocks) \
		$(if $(filter 0 false no,$(SIMPLE_WM_PLAY_TIMER)),--no-timer) \
		$(if $(SIMPLE_WM_PLAY_SAVE),--save) \
		$(if $(SIMPLE_WM_PLAY_SAVE_DIR),--save-dir $(SIMPLE_WM_PLAY_SAVE_DIR)) \
		--server $(SIMPLE_WM_PLAY_SERVER) \
		$(if $(SIMPLE_WM_PLAY_PASSWORD),--password $(SIMPLE_WM_PLAY_PASSWORD))

play-simple-world-model-local:
	$(MAKE) play-simple-world-model SIMPLE_WM_PLAY_SERVER=localhost SIMPLE_WM_PLAY_PASSWORD=

# ── JEPA Showdown Play
# Requires a checkpoint from train-jepa and a running Showdown server.
#
# Play the JEPA bot against human opponents on Showdown (or locally).
# The tokenizer and max-history window are loaded from the checkpoint; do not
# pass a separate tokenizer path to play targets.
#
# Usage:
#   make play-jepa                             # default: keep 100 gen1 random battles active, save parsed replays
#   make play-jepa-local
#   make play-jepa DEMO=1                         # use bundled demo checkpoint
#   make play-jepa JEPA_PLAY_FORMAT=gen1ou JEPA_PLAY_USERNAME=JEPABot
#   make play-jepa JEPA_PLAY_CHECKPOINT=/path/to/paired_best.pt
#   make play-jepa JEPA_PLAY_VERBOSE_BLOCKS=true JEPA_PLAY_LADDER=true
#   make play-jepa JEPA_PLAY_SAVE=1
#   make play-jepa JEPA_PLAY_LADDER=true JEPA_PLAY_NUM_BATTLES=100 JEPA_PLAY_MAX_CONCURRENT_BATTLES=8
#   make play-jepa JEPA_PLAY_KEEP_LADDER_BATTLES=100 JEPA_PLAY_RANDOM_BATTLE_BOT=0 JEPA_PLAY_SAVE=1
#
# REPL keys (press during battle):
#   R = raw protocol logs    P = state/action blocks
#   V = toggle verbose       O = battle overview    Q = quit REPL
JEPA_PLAY_CHECKPOINT ?= $(if $(filter 1,$(DEMO)),./checkpoints/demo_best.pt,$(JEPA_SAVE_DIR)/paired_best_stochastic.pt)
JEPA_PLAY_FORMAT ?= gen1randombattle
JEPA_PLAY_USERNAME ?= jepabot
JEPA_PLAY_TEAM_SET ?= competitive
# Used only when JEPA_PLAY_KEEP_LADDER_BATTLES is empty.
JEPA_PLAY_NUM_BATTLES ?= 30
JEPA_PLAY_MAX_CONCURRENT_BATTLES ?= 100
JEPA_PLAY_KEEP_LADDER_BATTLES ?= 100
JEPA_PLAY_RANDOM_BATTLE_BOT ?= 0
JEPA_PLAY_LADDER ?=
JEPA_PLAY_VERBOSE_BLOCKS ?=
JEPA_PLAY_SERVER ?= showdown
JEPA_PLAY_PASSWORD ?= JEPAJEPA
JEPA_PLAY_TIMER ?= 1
JEPA_PLAY_SAVE ?= 1
JEPA_PLAY_SAVE_DIR ?=
play-jepa:
	@if [ ! -f "$(JEPA_PLAY_CHECKPOINT)" ]; then \
		echo "ERROR: Checkpoint not found at $(JEPA_PLAY_CHECKPOINT)."; \
		echo "  Train first: make train-jepa FORMATS=$(JEPA_PLAY_FORMAT)"; \
		echo "  Or use demo: make play-jepa DEMO=1"; \
		exit 1; \
	fi
	uv run python -m metamon.jepa.play \
		--checkpoint $(JEPA_PLAY_CHECKPOINT) \
		--format $(JEPA_PLAY_FORMAT) \
		--username $(JEPA_PLAY_USERNAME) \
		--team_set $(JEPA_PLAY_TEAM_SET) \
		$(if $(JEPA_PLAY_KEEP_LADDER_BATTLES),,--num_battles $(JEPA_PLAY_NUM_BATTLES)) \
		--max_concurrent_battles $(JEPA_PLAY_MAX_CONCURRENT_BATTLES) \
		$(if $(JEPA_PLAY_KEEP_LADDER_BATTLES),--keep_ladder_battles $(JEPA_PLAY_KEEP_LADDER_BATTLES)) \
		$(if $(filter 0 false no,$(JEPA_PLAY_RANDOM_BATTLE_BOT)),--no_random_battle_bot) \
		$(if $(JEPA_PLAY_LADDER),--ladder) \
		$(if $(JEPA_PLAY_VERBOSE_BLOCKS),--verbose_blocks) \
		$(if $(filter 0 false no,$(JEPA_PLAY_TIMER)),--no-timer) \
		$(if $(JEPA_PLAY_SAVE),--save) \
		$(if $(JEPA_PLAY_SAVE_DIR),--save-dir $(JEPA_PLAY_SAVE_DIR)) \
		--server $(JEPA_PLAY_SERVER) \
		$(if $(JEPA_PLAY_PASSWORD),--password $(JEPA_PLAY_PASSWORD))

# Battle with the paired JEPA bot on a local Pokemon Showdown server.
# Start the server first with: make showdown
#
# DEMO=1 uses the bundled demo checkpoint (./checkpoints/demo_best.pt)
play-jepa-local: ensure-showdown
	$(MAKE) play-jepa JEPA_PLAY_SERVER=localhost JEPA_PLAY_PASSWORD=

# ── JEPA vs baseline competition ────────────────────────────────────
#
#   make test-jepa-baseline FORMAT=gen1ou BASELINE=Gen1BossAI
#   make test-jepa-baseline BASELINE=MaxBPBaseline N_BATTLES=20

JEPA_BASELINE_FORMAT ?= gen1ou
JEPA_BASELINE ?= Gen1BossAI
JEPA_BASELINE_N_BATTLES ?= 10
JEPA_BASELINE_SERVER ?= localhost

test-jepa-baseline: $(if $(filter localhost,$(JEPA_BASELINE_SERVER)),ensure-showdown)
	@if [ ! -f "$(JEPA_PLAY_CHECKPOINT)" ]; then \
		echo "ERROR: Checkpoint not found at $(JEPA_PLAY_CHECKPOINT)."; \
		echo "  Train first: make train-jepa FORMATS=$(JEPA_BASELINE_FORMAT)"; \
		echo "  Or use demo: make test-jepa-baseline DEMO=1"; \
		exit 1; \
	fi
	uv run python -m metamon.jepa.compete_baseline \
		--checkpoint $(JEPA_PLAY_CHECKPOINT) \
		--format $(JEPA_BASELINE_FORMAT) \
		--baseline $(JEPA_BASELINE) \
		--n_battles $(JEPA_BASELINE_N_BATTLES) \
		--team_set $(JEPA_PLAY_TEAM_SET) \
		--server $(JEPA_BASELINE_SERVER)

# Run JEPA against *all* registered baselines for the format.
#   make test-jepa-all-baselines FORMAT=gen1ou
#   make test-jepa-all-baselines N_BATTLES=5

test-jepa-all-baselines: $(if $(filter localhost,$(JEPA_BASELINE_SERVER)),ensure-showdown)
	@if [ ! -f "$(JEPA_PLAY_CHECKPOINT)" ]; then \
		echo "ERROR: Checkpoint not found at $(JEPA_PLAY_CHECKPOINT)."; \
		echo "  Or use demo: make test-jepa-all-baselines DEMO=1"; \
		exit 1; \
	fi
	uv run python -m metamon.jepa.compete_baseline \
		--checkpoint $(JEPA_PLAY_CHECKPOINT) \
		--format $(JEPA_BASELINE_FORMAT) \
		--all-baselines \
		--n_battles $(JEPA_BASELINE_N_BATTLES) \
		--team_set $(JEPA_PLAY_TEAM_SET) \
		--server $(JEPA_BASELINE_SERVER)

# ── Checkpoint backup ───────────────────────────────────────────────

# Copy all world-model checkpoints (JEPA and simple-world-model) to a timestamped backup
# directory under the project root so they can be committed to git.
# The original checkpoints in train save-dirs are left untouched.
#
# Usage:
#   make save-checkpoints
#   make save-checkpoints BACKUP_NAME=experiment-v2
SAVE_CHECKPOINTS_DIR ?= checkpoints
BACKUP_NAME ?=
save-checkpoint: save-checkpoints

save-checkpoints:
	@now=$$(date +%Y-%m-%d_%H%M%S); \
	commit=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown-git); \
	if [ -n "$(BACKUP_NAME)" ]; then dest="$(SAVE_CHECKPOINTS_DIR)/$${commit}_$(BACKUP_NAME)_$${now}"; \
	else dest="$(SAVE_CHECKPOINTS_DIR)/$${commit}_$${now}"; fi; \
	mkdir -p "$$dest"; \
	echo "Backing up checkpoints to $$dest"; \
	copied=0; \
	for src_dir in $(JEPA_SAVE_DIR) $(SIMPLE_WM_SAVE_DIR); do \
		if [ -d "$$src_dir" ]; then \
			label=$$(basename "$$src_dir"); \
			for f in "$$src_dir"/*.pt; do \
				if [ -f "$$f" ]; then \
					cp -v "$$f" "$$dest/$${label}_$$(basename "$$f")"; \
					copied=$$((copied + 1)); \
				fi; \
			done; \
		fi; \
	done; \
	if [ $$copied -eq 0 ]; then \
		echo "No checkpoints found to back up."; \
		rmdir "$$dest"; \
	else \
		echo "Saved $$copied checkpoints to $$dest"; \
	fi

# Run the full test suite (parallel by default via pytest-xdist)
test:
	uv run pytest tests/ -v

# Quick smoke tests only (~30s)
test-quick:
	uv run pytest tests/test_forward_smoke.py tests/test_forward_edge_cases.py -v

# Forward parsing tests only
test-forward:
	uv run pytest tests/test_forward_smoke.py tests/test_forward_structure.py tests/test_forward_pokemon.py tests/test_forward_actions.py tests/test_forward_edge_cases.py -v

# Backward fill tests only
test-backward:
	uv run pytest tests/test_backward_smoke.py tests/test_backward_structure.py tests/test_backward_consistency.py -v

# End-to-end pipeline tests only
test-e2e:
	uv run pytest tests/test_e2e_smoke.py tests/test_e2e_output.py -v

clean:
	@echo "WARNING: This will delete ALL parsed PoV replays, world-model samples, and tokenizers."
	@echo "Usage statistics (replay_stats, revealed_teams, usage-stats) and raw replays will NOT be affected."
	@read -p "Are you sure you want to continue? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "removing parsed replays (preserving replay_stats and revealed_teams)"; \
		for dir in $(METAMON_CACHE_DIR)/parsed-replays/*/; do \
			name=$$(basename "$$dir"); \
			if [ "$$name" != "replay_stats" ] && [ "$$name" != "revealed_teams" ]; then \
				rm -rf "$$dir"; \
			fi; \
		done; \
		echo "removing world model outputs"; \
		rm -rf $(WM_OUTPUT_DIR); \
		echo "removing tokenizers"; \
		rm -rf $(TOKENIZER_OUTPUT_DIR); \
	else \
		echo "Aborted."; \
	fi

show-tokenizer:
ifeq ($(OS),Darwin)
	cursor $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json
else
	vim $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json
endif

clean-tokenizer:
	rm -rf $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json

# ── Shell completion ─────────────────────────────────────────────────

# Install:  source <(make bash-completion)
# Permanent: make bash-completion >> ~/.bashrc
bash-completion:
	@echo '_make_completion() {'
	@echo '  local cur="$${COMP_WORDS[COMP_CWORD]}"'
	@echo '  COMPREPLY=($$(compgen -W "$(shell $(MAKE) -qp 2>/dev/null | grep -E '^[a-zA-Z_-]+:' | grep -v '^\.' | grep -v '^%' | grep -v '^Makefile' | cut -d: -f1 | sort -u | tr '\n' ' ')" -- "$$cur"))'
	@echo '}'
	@echo 'complete -F _make_completion make'
