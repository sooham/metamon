METAMON_CACHE_DIR ?= /workspace/poke-datasets
RAW_REPLAY_DIR ?= $(METAMON_CACHE_DIR)/raw-replays
MINI_RAW_REPLAY_DIR ?= $(METAMON_CACHE_DIR)/mini-raw-replays
FORMAT ?= gen1ou gen9ou
FORMATS ?= $(FORMAT)

.PHONY: parse-no-pred parse parse-all-no-pred parse-all battle battle-inspect inspect-replay \
        tokenize-world-model parse-world-model inspect-wm-state \
        generate-world-model-data inspect-wm-npz sample-inspect-wm-npz \
        test test-quick test-forward test-backward test-e2e \
        clean show-tokenizer clean-tokenizer sample-inspect-wm-state \
        train-sl play-sl bash-completion

# Open a battle replay in browser + parsed output in Cursor
# Usage: make battle BATTLE_ID=smogtours-gen1ou-694141
BATTLE_ID ?=
battle:
	@open https://replay.pokemonshowdown.com/$(BATTLE_ID)
	@format=$$(echo $(BATTLE_ID) | sed -E 's/^(smogtours-)?//;s/-[0-9]+$$//'); \
	dir="$(METAMON_CACHE_DIR)/parsed-replays/$$format"; \
	if [ -d "$$dir" ]; then \
		find "$$dir" -name "$(BATTLE_ID)_*" -print0 | xargs -0 -n 1 cursor; \
	else \
		echo "No parsed directory: $$dir"; \
	fi

# Inspect a replay using the interactive TUI
# Usage: make inspect-replay <battleid>
# Example: make inspect-replay gen4uu-184050323
inspect-replay:
	@battleid="$(word 2,$(MAKECMDGOALS))"; \
	if [ -z "$$battleid" ]; then \
		echo "Usage: make inspect-replay <battleid>"; \
		echo "Example: make inspect-replay gen4uu-184050323"; \
		exit 1; \
	fi; \
	echo "uv run python tools/inspect_replay.py $$battleid" --showdown; \
	uv run python tools/inspect_replay.py "$$battleid" --showdown

# Catch-all to prevent "No rule to make target" errors for positional arguments
%:
	@:

# Parse one format with player-team-only prediction.
# Player's own Pokemon get predicted moves/items/abilities from usage stats.
# Opponent info is forward-observed only (no prediction, no backfill leak).
# Now includes opponent_bench, fainted_pokemon, opponent_fainted.
parse-no-pred:
	uv run python -m metamon.backend.replay_parser \
		--format $(FORMAT) \
		--team_predictor NoPredictor \
		--raw_replay_dir $(METAMON_CACHE_DIR)/raw-replays \
		--output_dir $(METAMON_CACHE_DIR)/parsed-no-pred \
		--processes 32 --no-compress --pretty

# Parse one format with default NaiveUsagePredictor
parse:
	uv run python -m metamon.backend.replay_parser \
		--format $(FORMAT) \
		--raw_replay_dir $(METAMON_CACHE_DIR)/raw-replays \
		--output_dir $(METAMON_CACHE_DIR)/parsed-replays \
		--processes 64 --no-compress --pretty

# Parse all supported formats with NoPredictor
parse-all-no-pred:
	@for fmt in gen1ou gen1uu gen1nu gen1ubers \
	            gen2ou gen2uu gen2nu gen2ubers \
	            gen3ou gen3uu gen3nu gen3ubers \
	            gen4ou gen4uu gen4nu gen4ubers \
	            gen9ou; do \
		echo "=== $$fmt ==="; \
		$(MAKE) parse-no-pred FORMAT=$$fmt; \
	done

# Parse all supported formats with NoPredictor
parse-all:
	@for fmt in gen1ou gen1uu gen1nu gen1ubers \
	            gen2ou gen2uu gen2nu gen2ubers \
	            gen3ou gen3uu gen3nu gen3ubers \
	            gen4ou gen4uu gen4nu gen4ubers \
	            gen9ou; do \
		echo "=== $$fmt ==="; \
		$(MAKE) parse FORMAT=$$fmt; \
	done

# Inspect a random sample of 5 parsed battles from a format (one at a time in Cursor + browser)
# Usage: make battle-inspect FORMAT=gen1ou
battle-inspect:
	@dir="$(METAMON_CACHE_DIR)/parsed-replays/$(FORMAT)"; \
	if [ ! -d "$$dir" ]; then \
		echo "No parsed directory for format $(FORMAT): $$dir"; \
		exit 1; \
	fi; \
	count=$$(find "$$dir" -name '*.json' -type f | wc -l | tr -d ' '); \
	if [ "$$count" -eq 0 ]; then \
		echo "No JSON files found in $$dir"; \
		exit 1; \
	fi; \
	echo "=== Sampling 5 of $$count battles from $(FORMAT) ==="; \
	find "$$dir" -name '*.json' -type f | sort -R | head -5 | while IFS= read -r f; do \
		echo "--- $$(basename "$$f") ---"; \
		battle_id=$$(basename "$$f" .json | cut -d_ -f1); \
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
NUM_WORKERS ?= 32
EARLY_STOP ?= 0
tokenize-world-model:
	mkdir -p $(TOKENIZER_OUTPUT_DIR)
	uv run python -m metamon.tokenizer.tokenizer \
		--parsed_replay_root $(METAMON_CACHE_DIR)/parsed-replays \
		--formats $(FORMATS) \
		--obs_space WorldModelObservationSpace \
		--num_workers $(NUM_WORKERS) \
		--early_stop $(EARLY_STOP) \
		--save_tokens $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json

# Parse and validate with WorldModelObservationSpace.
# Runs the standard pred parse, then spot-checks 5 random replays per format.
# Usage:
#   make parse-world-model FORMATS=gen1ou
#   make parse-world-model FORMATS="gen1ou gen9ou"
parse-world-model:
	@for fmt in $(FORMATS); do \
		echo "=== Parsing $$fmt ==="; \
		$(MAKE) parse FORMAT=$$fmt; \
	done
	@for fmt in $(FORMATS); do \
		echo "=== Validating world-model format on 5 random $$fmt replays ==="; \
		dir="$(METAMON_CACHE_DIR)/parsed-replays/$$fmt"; \
		files=$$(find "$$dir" -name '*.json' -type f 2>/dev/null | sort -R | head -5); \
		if [ -n "$$files" ]; then \
			uv run python scripts/validate_world_model.py $$files; \
			echo "=== All $$fmt spot-checks passed ==="; \
		else \
			echo "=== No files found for $$fmt ==="; \
		fi; \
	done

# Inspect a single parsed replay through the WorldModelObservationSpace.
# Prints the tokenized text for states 0 and -1 (or specified indices).
# Usage:
#   make inspect-wm-state FILE=/path/to/replay.json
#   make inspect-wm-state FILE=/path/to/replay.json INDICES='0 5 10'
#   make inspect-wm-state FILE=/path/to/replay.json FLAGS='--pretty'
#   make inspect-wm-state FILE=/path/to/replay.json FLAGS='--pretty --show-all'
FILE ?=
INDICES ?= 0 -1
FLAGS ?=
inspect-wm-state:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make inspect-wm-state FILE=/path/to/replay.json [INDICES='0 5 10'] [FLAGS='--pretty --show-all']"; \
		exit 1; \
	fi
	uv run python scripts/inspect_world_model_state.py $(FILE) $(INDICES) $(FLAGS)

# Generate world-model training data from parsed replays.
# Automatically builds the WorldModel tokenizer if it doesn't exist yet.
# Aborts early if parsed replays are missing for any requested format.
#
# Each output .npz contains:
#   states  (seq_len, 336) int16  — token IDs for each state (padded to soft max)
#   actions (seq_len-1,)  int16  — action index for each transition
#   won     bool                  — whether POV won
# Training pairs: (states[t], actions[t], states[t+1])
#
# Usage:
#   make generate-world-model-data FORMATS=gen1ou
#   make generate-world-model-data FORMATS="gen1ou gen9ou"
WM_OUTPUT_DIR ?= $(METAMON_CACHE_DIR)/world-model-samples
WM_PROCESSES ?= 32
TOKENIZER_FILE := $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json
generate-world-model-data:
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
		$(MAKE) tokenize-world-model FORMATS="$(FORMATS)"; \
	fi
	@# ---- 3. Generate sharded .npz files ----
	mkdir -p $(WM_OUTPUT_DIR)
	uv run python scripts/generate_world_model_data.py \
		--parsed_replay_root $(METAMON_CACHE_DIR)/parsed-replays \
		--tokenizer_path $(TOKENIZER_FILE) \
		--output_dir $(WM_OUTPUT_DIR) \
		--formats $(FORMATS) \
		--processes $(WM_PROCESSES)

# ── Supervised-Learning Training ────────────────────────────────────

# Train the WorldModelTransformer on next-state prediction using .npz shards.
# Requires tokenized world-model data (run generate-world-model-data first).
#
# Usage:
#   make train-sl FORMATS="gen1ou gen9ou"
#   make train-sl FORMATS=gen9ou EPOCHS=20 BATCH_SIZE=64
#   make train-sl FORMATS="gen1ou gen9ou" WANDB=true WANDB_PROJECT=metamon WANDB_NAME=my-run
#   make train-sl FORMATS=gen9ou CHECKPOINT=/workspace/checkpoints/model.pt
SL_DATA_ROOT ?= $(WM_OUTPUT_DIR)
SL_TOKENIZER ?= $(TOKENIZER_FILE)
SL_SAVE_DIR ?= $(METAMON_CACHE_DIR)/sl-checkpoints
SL_BATCH_SIZE ?= 256
SL_LR ?= 3e-4
SL_EPOCHS ?= 10
SL_GRAD_CLIP ?= 1.0
SL_NUM_WORKERS ?= 4
SL_PRINT_INTERVAL ?= 100
SL_CONFIG ?=
CHECKPOINT ?= $(SL_SAVE_DIR)/best.pt
WANDB ?= true
WANDB_PROJECT ?=
WANDB_NAME ?=
train-sl:
	@if [ ! -d "$(SL_DATA_ROOT)" ]; then \
		echo "ERROR: No .npz data found at $(SL_DATA_ROOT)."; \
		echo "  Run: make generate-world-model-data FORMATS=\"$(FORMATS)\" first."; \
		exit 1; \
	fi
	@if [ ! -f "$(SL_TOKENIZER)" ]; then \
		echo "ERROR: Tokenizer not found at $(SL_TOKENIZER)."; \
		echo "  Run: make tokenize-world-model FORMATS=\"$(FORMATS)\" first."; \
		exit 1; \
	fi
	mkdir -p $(SL_SAVE_DIR)
	uv run python -m metamon.sl.train \
		--data_root $(SL_DATA_ROOT) \
		--formats $(FORMATS) \
		--tokenizer_path $(SL_TOKENIZER) \
		--save_dir $(SL_SAVE_DIR) \
		--batch_size $(SL_BATCH_SIZE) \
		--lr $(SL_LR) \
		--epochs $(SL_EPOCHS) \
		--grad_clip $(SL_GRAD_CLIP) \
		--num_workers $(SL_NUM_WORKERS) \
		--print_interval $(SL_PRINT_INTERVAL) \
		$(if $(filter true,$(WANDB)),--wandb) \
		$(if $(WANDB_PROJECT),--wandb_project $(WANDB_PROJECT)) \
		$(if $(WANDB_NAME),--wandb_name $(WANDB_NAME)) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT)) \
		$(if $(SL_CONFIG),--config $(SL_CONFIG)) \
		--log --log_interval 100

# ── World Model Showdown Play ─────────────────────────────────────────

# Battle with a trained WorldModelTransformer on the local Showdown server.
# Requires a checkpoint from train-sl and a running Showdown server.
#
# Usage:
#   make play-sl FORMAT=gen1ou
#   make play-sl FORMAT=gen1ou USERNAME=MyBot TEAM_SET=competitive NUM_BATTLES=10
#   make play-sl FORMAT=gen9ou CHECKPOINT=/path/to/checkpoint.pt
SL_PLAY_CHECKPOINT ?= $(SL_SAVE_DIR)/best.pt
SL_PLAY_FORMAT ?= gen1ou
SL_PLAY_USERNAME ?= WorldModelBot
SL_PLAY_TEAM_SET ?= competitive
SL_PLAY_BATTLES ?= 5
SL_PLAY_MAX_TOKENS ?= 200
play-sl:
	@if [ ! -f "$(SL_PLAY_CHECKPOINT)" ]; then \
		echo "ERROR: Checkpoint not found at $(SL_PLAY_CHECKPOINT)."; \
		echo "  Train first: make train-sl FORMATS=$(SL_PLAY_FORMAT)"; \
		exit 1; \
	fi
	uv run python -m metamon.sl.play \
		--checkpoint $(SL_PLAY_CHECKPOINT) \
		--format $(SL_PLAY_FORMAT) \
		--username $(SL_PLAY_USERNAME) \
		--team_set $(SL_PLAY_TEAM_SET) \
		--num_battles $(SL_PLAY_BATTLES) \
		--max_new_tokens $(SL_PLAY_MAX_TOKENS)

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
	cursor $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json

clean-tokenizer:
	rm -rf $(TOKENIZER_OUTPUT_DIR)/$(TOKENIZER_VERSION).json

sample-inspect-wm-state:
	make inspect-wm-state FILE=$(METAMON_CACHE_DIR)/parsed-replays/gen1ou/smogtours-gen1ou-749168_Unrated_encore90411_vs_mindplate96156_02-23-2024_WIN.json FORMAT=gen1ou FLAGS="--show-all"

# ── World Model NPZ Inspection ─────────────────────────────────────

WM_FORMAT ?= gen1ou
WM_FLAGS ?=

# Inspect a random battle from the world-model .npz training data.
# Shows token IDs and detokenized text for selected states.
# Pass all optional arguments through WM_FLAGS.
# Usage:
#   make inspect-wm-npz WM_FORMAT=gen1ou
#   make inspect-wm-npz WM_FORMAT=gen1ou WM_FLAGS='--pretty --show-all --showdown'
#   make inspect-wm-npz WM_FORMAT=gen1ou WM_FLAGS='--state-idx 0 5 10 --pretty'
#   make inspect-wm-npz WM_FORMAT=gen1ou WM_FLAGS='--shard path/to/shard.npz --battle 3 --pretty'
inspect-wm-npz:
	uv run python scripts/inspect_wm_npz.py \
		--wm_dir $(WM_OUTPUT_DIR) \
		--tokenizer_path $(TOKENIZER_FILE) \
		--format $(WM_FORMAT) \
		--parsed_replay_root $(METAMON_CACHE_DIR)/parsed-replays \
		$(WM_FLAGS)

# Convenience: inspect a random world-model npz battle with pretty output and Showdown link.
# Usage:
#   make sample-inspect-wm-npz
#   make sample-inspect-wm-npz WM_FORMAT=gen1ou
sample-inspect-wm-npz:
	make inspect-wm-npz WM_FORMAT=$(WM_FORMAT) WM_FLAGS='--pretty --show-all --showdown'

# ── Shell completion ─────────────────────────────────────────────────

# Install:  source <(make bash-completion)
# Permanent: make bash-completion >> ~/.bashrc
bash-completion:
	@echo '_make_completion() {'
	@echo '  local cur="$${COMP_WORDS[COMP_CWORD]}"'
	@echo '  COMPREPLY=($$(compgen -W "$(shell $(MAKE) -qp 2>/dev/null | grep -E '^[a-zA-Z_-]+:' | grep -v '^\.' | grep -v '^%' | grep -v '^Makefile' | cut -d: -f1 | sort -u | tr '\n' ' ')" -- "$$cur"))'
	@echo '}'
	@echo 'complete -F _make_completion make'
