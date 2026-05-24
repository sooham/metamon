METAMON_CACHE_DIR ?= $(HOME)/Repositories/poke-datasets
FORMAT ?= gen4uu

.PHONY: parse-no-pred parse parse-all-no-pred

# Parse one format without team prediction (observed info only)
parse-no-pred:
	uv run python -m metamon.backend.replay_parser \
		--format $(FORMAT) \
		--team_predictor NoPredictor \
		--raw_replay_dir $(METAMON_CACHE_DIR)/raw-replays \
		--output_dir $(METAMON_CACHE_DIR)/parsed-no-pred \
		--processes 10

# Parse one format with default NaiveUsagePredictor
parse:
	uv run python -m metamon.backend.replay_parser \
		--format $(FORMAT) \
		--raw_replay_dir $(METAMON_CACHE_DIR)/raw-replays \
		--output_dir $(METAMON_CACHE_DIR)/parsed-replays \
		--processes 10

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
