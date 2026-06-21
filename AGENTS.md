# Environments
The package manager in python here is `uv`. The development machine is a macbook pro M4 Pro with 1 TB of storage. Currently there is no production environement.

# Dependencies
If you install a new library (via `uv pip install` or similar), also add it to `pyproject.toml` under `[project].dependencies` so it is tracked declaratively.

# Storage limitations of development machine 
Before running any commands which will generate a lot of files, check if the computer has enough free storage.

# Surveying the pokemon datasets directory
The pokemon datasets is in $METAMON_CACHE_DIR . when you look at subfolders in that be mindful because
the folders contain millions of files, using common bash commands like `ls`, `find` etc. will time out.
if the user has given you the exact battle id or filename , use that smogtours-gen1ou-749168_Unrated_encore90411_vs_mindplate96156_02-23-2024_WIN.txt 
if you want to pick random battles or replay files , use `ls -f` in combination with tools like `head` or `tail` and others which are only going to read so many inodes in the folder.

# Asking the user for help
If you run into a bug i.e bad environment setups, an error you can't resolve, ambiguous references and error traces, please ask the user to clarify with more information.

# Tests and updating tests
The metamon repo has pytests tests , they can be run with `make test`. Analyzing if the test suite needs an updatee is mandatory. If you make critical changes or breaking changes you are expected to also update the tests. Your new test cases should be simple, composable and respect module and class boundaries. End2end tests  are in `uv run pytest tests/test_e2e_smoke.py tests/test_e2e_output.py -v` can combine multiple modules and classes to achive good test coverage. Mocking is done with monkeypatch if necessary.

# Performance
You should write code which if necessary and at your own discrection and determination of performance and runtime based on input size, should use parallelism such as threading , pooling, multi-process code if necessary, be mindful of shared resources and that functions being called are thread safe.  Other common perfomance optimizations include using caching in memory, writing to files for faster processing and reading from them on the next run are also good practices.

# JEPA — Joint Embedding Predictive Architecture (`metamon/jepa/`)

`metamon/jepa/` trains a paired-POV world model that learns to predict the **hidden opponent state** and the **next state latent** from the visible board state and action history. It is a self-supervised world model that encodes battle states into a deterministic latent space (via `JEPAEncoder`), encodes action text into action latents (via `JEPAActionEncoder`), and uses a temporal encoder over interleaved block embeddings to produce a history context. The model is trained on *paired* POV data — both players' perspectives of the same battle synchronized, so each side predicts the other's hidden state.

### Architecture overview

```
State block tokens ─► JEPAEncoder φ ─► state embedding (latent_dim=192)
Action text tokens  ─► JEPAActionEncoder ψ ─► action embedding (action_latent_dim=32)

[team_header, state₀, p_action₀, o_action₀, state₁, ...] ─► JEPATemporalEncoder τ ─► history context c

c + current_state_z ─► JEPAOpponentBeliefPredictor shared backbone
                    ├─ state head  ─► predicted opponent state z_opp (Gaussian)
                    └─ action head ─► predicted opponent action a_opp (Gaussian)
z + own_action + z_opp + a_opp ─► JEPANextStatePredictor ─► predicted next state z_next
z + z_opp                    ─► JEPAPairwiseRankHead ─► scalar advantage score
```

`JEPAOpponentBeliefPredictor` is the current code path and checkpoint module name; it replaces the older separate opponent-state and paired-action predictor modules with a shared representation plus two output heads.

**Losses:** diagonal Gaussian NLL for opponent state prediction, opponent action prediction, and next-state prediction; SIGReg (Epps-Pulley Gaussianity regularizer); and Bradley-Terry ranking loss from battle outcomes. During paired supervised training, the next-state predictor is conditioned on the actual paired opponent state and the actual opponent action taken between the current and next state.

### Training (`train_paired.py`)

Consumes `paired_shard_*.npz` files produced by `scripts/generate_world_model_data.py`.

```bash
uv run python scripts/generate_world_model_data.py \
    --parsed_replay_root $METAMON_CACHE_DIR/parsed-replays \
    --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \
    --output_dir $METAMON_CACHE_DIR/world-model-samples \
    --formats gen1ou \
    --battles_per_shard 1000 \
    --rollout_len 1 \
    --processes 8
```

```bash
uv run python -m metamon.jepa.train_paired \
    --data_root $METAMON_CACHE_DIR/world-model-samples \
    --formats gen1ou \
    --tokenizer_path $METAMON_CACHE_DIR/tokenizers/WorldModelObservationSpace-v1.json \
    --save_dir $METAMON_CACHE_DIR/jepa-checkpoints \
    --checkpoint $METAMON_CACHE_DIR/jepa-checkpoints/paired_best.pt \
    --batch_size 8 --grad_accum_steps 4 --lr 5e-5 --epochs 10 \
    --num_workers 4 --compile
```

Key training flags:
- `--max_history_blocks N` — window to last N state blocks (0 = unlimited, the **default**). The team header is always retained. Lower values reduce memory and speed up training.
- `--lambda_rank 0` — disable ranking loss (e.g. when outcome labels are unreliable)
- `--compile` — enable `torch.compile` on encoder + action encoder (CUDA only)
- `--checkpoint` — path for both warm-start loading AND best-checkpoint saving
- `--no-wandb` — disable Weights & Biases logging. W&B is enabled by default when the `wandb` package is installed; `--wandb` is accepted but redundant.

Training uses bf16 on CUDA, SIGReg resampled per step, and micro-batched encoding (max 65536 tokens per encoder call). The temporal encoder's `max_seq_len` defaults to 6144 to accommodate full-battle histories. The paired dataset and model preserve an explicit rollout axis: state/history tensors are `[B, K, blocks, tokens]`, action tensors are `[B, K, tokens]`, and losses reduce over both batch and rollout steps.

### Config (`metamon/jepa/configs/default.yaml`)

| Module | Key params |
|---|---|
| `encoder` | d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_seq_len=256, gradient_checkpointing=true |
| `temporal_encoder` | n_heads=6, n_layers=4, d_ff=768, max_seq_len=6144 |
| `action_encoder` | d_model=128, n_heads=4, n_layers=3, d_ff=512, max_seq_len=64 |
| Latents | `latent_dim: 192`, `action_latent_dim: 32` |
| Loss weights | `lambda_sigreg_state: 0.1`, `lambda_sigreg_action: 0.0`, `lambda_rank: 1.0`; deprecated fallback `lambda_sigreg: 0.1` |

### Data format (paired shards)

Paired shards contain both POVs' state arrays side-by-side: `p1_states` / `p2_states`, `p1_actions` / `p2_actions`, `p1_opponent_actions` / `p2_opponent_actions`, plus per-battle outcome labels (`p1_won`, `p2_won`). Action arrays store canonical action content without role delimiters: moves start with `move`, switches start with `switch`, and missing actions are `unknown unknown`.

`--rollout_len K` controls the experience-replay sample length. Each sample row stores `K` contiguous aligned transitions from one raw battle using `(num_samples, K)` index matrices such as `p1_state_idx`, `p1_next_state_idx`, and `p1_action_idx` (with matching `p2_*` arrays). Battles with fewer than `K` aligned action steps, or no contiguous K-step windows, are skipped with warning counts during generation. The train/validation split is by raw battle key, so both POV files for a battle stay in the same split. After the split, battle keys and rollout rows are shuffled to reduce batch correlation. See `docs/world_model_data_format.md` for the complete shard schema and `PairedJEPADataset` sample shape.

### Online play (`play.py`)

The JEPA model can be used as an online Pokémon Showdown bot via `metamon.jepa.play`. The bot uses JEPA latent rollouts to score legal actions and pick the best one.

#### Local play (local Pokémon Showdown server)

Requires a local Showdown server running on `localhost:8000`:

```bash
# Start a local showdown server first (if not already running):
# git clone https://github.com/smogon/pokemon-showdown.git && cd pokemon-showdown && npm install && node pokemon-showdown start --no-security

uv run python -m metamon.jepa.play \
    --checkpoint $METAMON_CACHE_DIR/jepa-checkpoints/paired_best.pt \
    --format gen1ou \
    --username JEPABot \
    --num_battles 5 \
    --heuristic max-rank \
    --server localhost
```

With `--server localhost` (the default), the bot connects to `localhost:8000` and **waits for challenges**. Another user or bot must challenge it:
```
/challenge JEPABot, gen1ou
```

To have the bot search for random ladder battles instead:
```bash
uv run python -m metamon.jepa.play \
    --checkpoint ... \
    --server localhost \
    --ladder
```

#### Online play (real Pokémon Showdown server)

Connect to `play.pokemonshowdown.com`. Requires a registered Showdown account and password:

```bash
uv run python -m metamon.jepa.play \
    --checkpoint $METAMON_CACHE_DIR/jepa-checkpoints/paired_best.pt \
    --format gen1ou \
    --username YourBotName \
    --password your_password \
    --num_battles 30 \
    --heuristic max-rank \
    --server showdown \
    --ladder
```

Without `--ladder`, the bot waits for challenges:
```bash
uv run python -m metamon.jepa.play \
    --checkpoint ... \
    --server showdown \
    --username YourBotName --password your_password
```

Then another player challenges with:
```
/challenge YourBotName, gen1ou
```

#### Heuristics (action selection)

The `--heuristic` flag controls how the bot scores legal actions from JEPA latent rollouts:

| Heuristic | What it maximizes |
|---|---|
| `max-rank` (default) | `rank_head(z_next, z_opp)` — predicted advantage after the action |
| `max-self-state-delta` | `‖z_next − z_current‖` — how much the board state changes |
| `max-opponent-state-delta` | `‖predicted_opp_next − predicted_opp_current‖` — disruption to the opponent |

#### Interactive REPL (keyboard shortcuts during battle)

Press these keys in the terminal while battles are running:
- **R** — show last 40 raw protocol messages (`|move|`, `|switch|`, etc.)
- **P** — show all state/action blocks + last JEPA diagnostics (latent norms, logvar, belief rank)
- **V** — toggle verbose block printing on/off
- **O** — overview of all active battles (turn count, Pokémon, HP)
- **Q** — stop the REPL key listener

#### Per-turn verbose output

When `--verbose` (default on), each turn prints:
```
── JEPA turn 5 (battle-gen1ou-12345) [max-rank] ──
  vs: opponent_name  |  replay: https://replay.pokemonshowdown.com/battle-gen1ou-12345
  active: Snorlax hp=0.87
  opponent: Chansey hp=0.65
  legal actions:
      0 move: body slam           delta=  0.234
      1 move: earthquake           delta=  0.112
    ...
  chosen: move: body slam
```

With `--verbose_blocks`, it also prints the full tokenized state/action blocks, latent norms, and per-action prediction details.

#### Complete CLI reference

```
uv run python -m metamon.jepa.play \
    --checkpoint PATH                 # required: .pt checkpoint file
    --format FORMAT                   # default: gen1ou
    --username NAME                   # default: JEPABot
    --num_battles N                   # default: 30
    --team_set SET                    # default: competitive (options: competitive, random, etc.)
    --server {localhost|showdown}     # default: localhost
    --password PASS                   # required for --server showdown
    --heuristic {max-rank|max-self-state-delta|max-opponent-state-delta}  # default: max-rank
    --ladder                          # search ladder instead of waiting for challenges
    --quiet                           # suppress per-turn verbose output
    --verbose_blocks                  # print full tokenized blocks each turn
    --config PATH                     # model config yaml (default: configs/default.yaml)
```

The play CLI loads tokenizer vocabulary and `max_history_blocks` from the checkpoint. There is no `--tokenizer_path` override for `metamon.jepa.play`; checkpoints without embedded `tokenizer_state` are rejected because loading a separate tokenizer can silently change token IDs.

#### Important notes

- **Checkpoint-controlled history windowing in the player** — `play.py` loads `max_history_blocks` from the checkpoint, and `player.py` applies the same windowing logic used during paired training. The team header is always retained. `max_history_blocks=0` means unlimited/full-history encoding; positive values keep only the most recent N state blocks to stay within the temporal encoder's `max_seq_len` (default 6144 positional slots).
- **Random battle mode** — the bot always launches a second instance for `gen1randombattle` in parallel, searching the ladder. This runs alongside the main OU bot.
- **Concurrent battles** — `max_concurrent_battles=30`, so the bot can handle many battles simultaneously sharing one model.
- **Model runs in eval mode with bf16** on CUDA. MPS and CPU also work.

**Further reading:** `metamon/jepa/player.py` (bot implementation, history tracking, diagnostics, REPL), `metamon/jepa/model.py` (full architecture, SIGReg, loss functions), `metamon/jepa/train_paired.py` (training loop, dataset, windowing logic), `metamon/jepa/online_serializer.py` (state/action/team block tokenization for live battles), `metamon/jepa/configs/default.yaml` (model hyperparameters).

# Showdown Dex — the static Pokémon data layer

The `Dex` class (`metamon/backend/showdown_dex/dex.py`) is the **single source of truth** for canonical Pokémon game data in the codebase. It is adapted from the [poke-env](https://github.com/hsahovic/poke-env) library but Metamon maintains its own static JSON files with corrections for early generations.

### What it loads (per generation)

`Dex.from_gen(gen)` or `Dex.from_format("gen9ou")` loads five data files from `metamon/backend/showdown_dex/static/`:

| Data file | Contents | Key fields |
|---|---|---|
| `moves/gen{gen}moves.json` | All move definitions | type, power, accuracy, pp, category, priority, flags |
| `pokemon/gen{gen}pokedex.json` | All Pokémon species | name, baseSpecies, types, baseStats, abilities, requiredItem, requiredAbility, requiredTeraType, cosmeticFormes, num |
| `typechart/gen{gen}typechart.json` | Type effectiveness matrix | raw Showdown `damageTaken` codes (0=normal, 1=weak to attacker, 2=resists attacker, 3=immune); `Dex.type_chart` converts these to multipliers (1.0, 2.0, 0.5, 0.0) |
| `natures.json` | Stat-modifying natures (Gen 3+) | increased/decreased stat |
| `learnset.json` | Move learnsets per Pokémon | which Pokémon learn which moves |

### Key API

- **`Dex.from_gen(gen: int)`** — returns a cached `Dex` instance for that generation. Uses `@lru_cache` so repeated calls are free.
- **`Dex.from_format(format: str)`** — parses the generation from a format string (e.g., `"gen9ou"` → gen 9) and returns the corresponding `Dex`.
- **`dex.get_pokedex_entry(name: str)`** — looks up a Pokémon by its canonical (normalized) name. Raises `KeyError` if not found.
- **`dex.pokedex`**, **`dex.moves`**, **`dex.type_chart`** — direct access to the loaded dicts.

### Cross-generation fallback

When looking up a Pokémon that might not exist in the current gen's Pokédex (e.g., a Gen 9 species appearing in a Gen 9 format), the code searches **progressively higher** generation dex files (gen, gen+1, gen+2, …) until a match is found. This is implemented in `Pokemon._lookup_pokedex_info()` in `replay_state.py`.

### Where it's used

| Consumer | How it uses Dex |
|---|---|
| **Replay parser** (`replay_state.py`) | Looks up species name, types, base stats, abilities, required items, and Tera types when a Pokémon is first revealed during parsing |
| **Team construction** (`pokemon_pool.py`) | Looks up dex entries for ability resolution, required items, species clause enforcement (via `num`/`baseSpecies`), and base species deduplication |
| **Team prediction / usage stats** | Resolves species from usage data for team prediction models |

### Thread safety and instantiation

The `Dex` class uses `__slots__` and the constructor raises if a `Dex` for that gen already exists in `_gen_data_per_gen`. This means you should **always** use `Dex.from_gen()` (which caches) rather than calling `Dex(gen)` directly. The class is read-only after construction, making it safe to share across threads.

**Further reading:** The static JSON files live in `metamon/backend/showdown_dex/static/`. The cross-gen fallback logic is in `Pokemon._lookup_pokedex_info()` in `metamon/backend/replay_parser/replay_state.py`. For how the dex integrates with species clause in team construction, see `build_species_clause_keys()` in `metamon/backend/team_construction/pokemon_pool.py`.

# Raw replay format and the Showdown SIM-PROTOCOL

Raw replays are JSON files with a "log" field containing the battle transcript as a newline-separated string of **pipe-delimited messages** (`|type|arg1|arg2|...`). Each line's first token after the pipe is the message type, and the rest are arguments. This format is defined by the [Showdown SIM-PROTOCOL](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md).

Key message types the parser cares about (with concrete examples from real replays):
- `|player|p1|stick27544|224` — declares a player (avatar ID and rating are optional)
- `|poke|p1|Mimikyu, M|` — reveals a team member during team preview (item may be blank)
- `|switch|p1a: Jynx|Jynx|333/333` — a Pokémon switches in (also used for leads); HP in `cur/max` format, status optionally appended (e.g. `100/100 par`)
- `|move|p1a: Jynx|Lovely Kiss|p2a: Gengar` — a Pokémon uses a move; may include flags like `[miss]` or `[still]`
- `|-damage|p2a: Chansey|87/100` — HP change; `0 fnt` means fainted; can carry `[from] item: Life Orb` or `[from] ability: ...` tags
- `|faint|p2a: Exeggutor` — a Pokémon faints (triggers forced switch tracking)
- `|turn|1` — turn boundary (also triggers turn initialization)
- `|choice|move lovelykiss|move hypnosis` — reveals both players' clicks using **named** format (move names spelled out). Also found as numeric format: `|choice|switch 2|switch 4`. Can be empty: `|choice||`. Only present in some replays.
- `|win|hustle11937` or `|tie` — battle outcome
- `|replace|p1a: Zoroark|Zoroark|87/100` — Zoroark/Zorua's Illusion breaks, revealing the real Pokémon
- `|-status|p2a: Gengar|slp` — status condition applied; `|-curestatus|p1a: Slowbro|slp` removes it
- `|-boost|p1a: Slowbro|spa|2` — stat stage change; `|-unboost|p1a: Snorlax|spd|1` is the reverse
- `|-ability|p2a: Zamazenta|Dauntless Shield|boost` — ability activation; `|-endability|p1a: ...` deactivates
- `|-item|p1a: Scizor|Life Orb` — item revealed; `|-enditem|p2a: Glimmora|Focus Sash` means consumed/removed
- `|-sidestart|p1: confusion58079|move: Toxic Spikes` — side condition (entry hazard, screen, etc.) applied; `|-sideend` removes
- `|-weather|RainDance` — weather set; `|-weather|none` clears it
- `|-fieldstart|Electric Terrain` — field condition; `|-fieldend` removes
- `|-activate|p2a: Glimmora|ability: Toxic Debris` — catch-all for ability/item effects (Trick, Mimic, Berry consumption, etc.)
- `|-terastallize|p1a: Garganacl|Rock` — Gen 9 only; reveals Tera type
- `|-transform|p1a: Ditto|p2a: Gengar` — Transform (Ditto, Mew); user copies target's species
- `|drag|p2a: Chansey|Chansey|100/100` — forced switch-in from Roar / Dragon Tail / Circle Throw
- `|teamsize|p1|6` — declares team size (usually 6, can be fewer)
- `|gen|1` — generation number
- `|tier|[Gen 1] OU` — battle format
- `|rule|Sleep Clause Mod: Limit one foe put to sleep` — ruleset entry
- `|-sethp|p1a: Snorlax|100/100` — direct HP set (e.g. from Pain Split, Endeavor)
- `|-swapboost|p1a: ...|p2a: ...|[from] move: Heart Swap` — swaps stat boosts between two Pokémon
- `|-clearboost|p1a: Snorlax` — clears all stat boosts
- `|-start|p2a: Snorlax|Reflect` — volatile effect (Reflect, Leech Seed, Curse, etc.) applied; `|-end` removes
- `|cant|p1a: Slowbro|par` — a Pokémon can't move (paralysis, sleep, flinch, etc.)
- `|c|...` or `|chat|...` or `|-message|...` — chat messages (ignored by the parser)

Messages starting with `-` are "minor" protocol messages that describe side effects (damage, status, boosts, weather, items, abilities, etc.). The parser's `SimProtocol.IGNORES` set lists message types that are intentionally skipped (animations, chat, timers, redundant info like `-crit` and `-supereffective`).

The raw replays live in `$METAMON_CACHE_DIR/raw-replays/{gen}/{tier}/*.json`. Example: `smogtours-gen1ou-235844` is a Gen 1 OU replay with real `|choice|` messages like `|choice|move lovelykiss|move hypnosis` and `|choice|switch 2|switch 4`.

**Further reading:** The authoritative reference is [pokemon-showdown/SIM-PROTOCOL.md](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md). For line-by-line real-world examples, browse a few raw replay JSON files in `$METAMON_CACHE_DIR/raw-replays/` (use `ls -f | head` to pick a handful without listing millions of files). The `SimProtocol.IGNORES` set and `interpret_message()` dispatch table in `metamon/backend/replay_parser/forward.py` are the canonical list of which messages the parser handles and which it deliberately skips.

# Parser: forward fill, backward fill, and one-sided POV conversion

The replay parser (`metamon/backend/replay_parser/`) converts a **spectator-perspective** raw replay into two **one-sided** parsed trajectory files — one from each player's point of view (WIN and LOSS). This happens in three stages:

### 1. Forward fill (`forward.py`)
`SimProtocol` walks the raw log line-by-line via `interpret_message()`, maintaining full-knowledge game state in a `ParsedReplay` object (a list of `Turn` dataclasses). Each `Turn` holds both players' teams, active Pokémon, moves, conditions, weather, etc. The forward pass tracks everything a spectator would see — both players' full teams and actions. It also handles complex mechanics like forced switches (U-turn, Volt Switch, Eject Button, Red Card, Revival Blessing), Zoroark Illusion (`|replace|` messages), Transform, Mimic, and multi-turn/consecutive moves. The forward result is a complete battle transcript with all 12 Pokémon and their revealed info.

### 2. Backward fill (`backward.py`)
After the forward pass, a "final turn" is appended with both full teams filled in using a `TeamPredictor` (usage-stats-based guessing, or exact data from `|showteam|` messages). This filled turn is then propagated **backwards** through the trajectory via `backfill_info()`: each Pokémon in turn `t+1` contributes its known item, ability, moves, stats, etc. to the same Pokémon in turn `t`. This fills gaps where information wasn't revealed until later in the battle. The final turn is then discarded.

### 3. POV conversion
`POVReplay` takes the spectator `ParsedReplay` and the backward-filled copy, then:
- Overwrites one side's team with the filled version (`_fill_one_side`)
- Resolves Transform edge cases (`_resolve_transforms`) — copies moves learned during transformation backwards through the window
- Resolves Zoroark Illusion (`_resolve_zoroark`) — fixes action targets and movesets that were misattributed to the disguise Pokémon
- Aligns states and actions (`_align_states_actions`) — flattens turns+subturns into a timeline of `(state, action)` pairs for one player, with the action being what the player clicked at that state (from `moves_1`/`moves_2`, falling back to `choices_1`/`choices_2`)

The result is serialized to a **stateful text format** and saved as two `.txt` files per raw replay — e.g. `gen1ou-370249571_Unrated_uturn10423_vs_tintedlens67414_02-23-2024_WIN.txt` and the corresponding LOSS file.

The text format is **not Markovian** — each state shows only the current battlefield (active Pokémon, HP, status, weather, side conditions) plus the POV player's bench.  The model derives cumulative knowledge by reading the state sequence from beginning to end.  Actions use explicit move names and Pokémon names with a canonical action kind (e.g. `<chosen_move>move blizzard<end_chosen_move>`, `<chosen_move>switch alakazam<end_chosen_move>`) rather than integer indices.  See `docs/new_parser_format_spec.md` for the full specification.

For doubles formats, `POVReplayDoubles` (a subclass of `POVReplay`) handles two active Pokémon per side.  The text output uses `<active1>`/`<active2>`/`<opponent1>`/`<opponent2>` tags, per-slot `<begin_moves:1>` blocks, and per-slot action entries like `<chosen_move:1>`.  See `docs/doubles_implementation_plan.md`.

### Key classes in the pipeline
- `ParsedReplay` / `Turn` / `Pokemon` / `Move` / `Action` (`replay_state.py`) — the in-memory battle state during parsing
- `SimProtocol` (`forward.py`) — the line-by-line log interpreter
- `POVReplay` / `POVReplayDoubles` (`backward.py`) — converts spectator state to one-sided POV (singles and doubles)
- `ReplayParser` (`parse_replays.py`) — orchestrates the full pipeline (forward → backward → save)
- `text_serializer.py` — serializes `POVReplay` objects to the new stateful text format (separate functions for singles and doubles)
- `UniversalState` / `UniversalAction` / `UniversalPokemon` (`interface.py`) — legacy backend-agnostic representations for JSON trajectory data; the new text serializer bypasses these for parser output

**Further reading:** The core pipeline entry point is `ReplayParser.parse_replay()` in `metamon/backend/replay_parser/parse_replays.py` — read this first for the big picture. Then trace into `forward.forward_fill()` → `SimProtocol.interpret_message()`, and `backward.backward_fill()` → `POVReplay`. The new text format is fully specified in `docs/new_parser_format_spec.md`. Tests in `tests/test_forward_actions.py`, `tests/test_backward_structure.py`, `tests/test_e2e_smoke.py`, and `tests/test_e2e_doubles.py` show the expected behavior.

# Parsed-replay text format (v2)

The new parser output is a **stateful text format** where each file is a sequence of state blocks (`<bos>`…`<eos>`) and action blocks (`<boa>`…`<eoa>`) interleaved.  See `docs/new_parser_format_spec.md` for the authoritative specification.  Key differences from the old JSON `{"states": [...], "actions": [...]}` format:

- **Actions use explicit names**, not integer indices.  Moves: `<chosen_move>move blizzard<end_chosen_move>`.  Switches: `<chosen_move>switch alakazam<end_chosen_move>`.  Unknown/missing: `<chosen_move>unknown unknown<end_chosen_move>`.  When a Pokémon is fully paralyzed / asleep and can't execute its move, the action block still shows the chosen action and the `cant` outcome appears in the following state's `<last_turn_results>`.
- **States show only current battlefield info** — active Pokémon (HP, status, boosts, effects), weather, side conditions, and the POV player's bench.  Opponent bench is NOT shown (the model infers it from the state sequence).
- **Team header** at the top of the file shows the POV player's full backward-filled team (all 6 species, types, items, abilities, gender, full 4-move movesets).
- **Opponent team preview** (Gen 5+ only) shows opponent species at file start.
- **Turn numbering** starts at 1 (the pre-battle lead-selection state is "turn 1").
- **Doubles** uses `<active1>`/`<active2>`/`<opponent1>`/`<opponent2>`, per-slot moves and actions.
- **Gender** (Gen 2+) is shown in team headers and bench entries: `M`, `F`, or `N` (unknown/genderless).  Omitted in Gen 1.
- **No XML-style closers** — all closing tags use `<end_foo>` form (never `</foo>`).
- **Value tokens are bracketless** — `noboosts`, `noeffect`, `nostatus`, `noweather`, `par`, `slp`, etc. have no angle brackets.

# Team Preview — lead prediction model

The `TeamPreviewModel` (`metamon/backend/team_preview/preview.py`) is a **Perceiver-style neural network** that predicts which Pokémon to lead with at the start of a battle. It consumes parsed replay JSON files in the legacy Universal format for training.

### Problem statement

At team preview, you see all 12 Pokémon (6 yours, 6 opponent's). You must pick one of your 6 to send out first. The model learns this from human gameplay data — for each parsed replay, the first state contains the team preview info and the player's actual lead choice (the first active Pokémon).

### Architecture

```
Inputs: 12 Pokémon tokens + optional additional info + format token
   │
   ├─ Token embeddings (nn.Embedding over PokemonTokenizer vocab)
   ├─ Positional embeddings (0–11 for the 12 Pokémon)
   ├─ Team embeddings (0=ours, 1=opponent)
   └─ Optional: additional info embeddings (moves, ability, item per our Pokémon)
   │
   ▼
Cross-Attention: latent tokens attend to the input sequence
   │
   ▼
Self-Attention: latent tokens attend to each other
   │
   ▼
LayerNorm + Flatten → Linear classifier → 6-way softmax
```

The learnable latent tokens (default 4) act as a bottleneck — they extract relevant information from the input sequence through cross-attention, then refine it through self-attention, and finally the classifier predicts over the 6 team slots.

### Input details

- **Team tokens:** 12 integers — our 6 Pokémon token IDs followed by opponent 6, both sorted alphabetically by `consistent_pokemon_order()` for consistency
- **Additional info (optional, per our Pokémon):** a 6-token vector: up to 4 moves (sorted alphabetically, padded with `<blank>`), ability token, item token. This gives the model knowledge of our own team's full sets, not just species names.
- **Format token (optional):** a single token like `<gen9ou>` to condition on the battle format

### Dataset

`TeamPreviewDataset` loads parsed replay JSON files from `$METAMON_CACHE_DIR/parsed-replays/{format}/`. For each replay it:
1. Reads the first `UniversalState` (team preview state)
2. Extracts our 6 Pokémon + opponent's 6 teampreview names
3. Tokenizes everything
4. Labels the lead index (which of our 6 sorted Pokémon is the active one)

It supports filtering by rating, result (wins/losses/both), and format.

### Training

`train_team_preview()` handles the full training loop:
- 95/5 train/val split
- Cross-entropy loss, AdamW optimizer
- Early stopping on validation accuracy (patience default 5 epochs)
- Saves `best_model.pt` and `latest_model.pt` checkpoints
- Optional W&B logging

### Inference API

```python
model = TeamPreviewModel.load_from_checkpoint("best_model.pt")
predicted_lead, probs, sorted_team = model.predict_lead(
    our_team=["Garchomp", "Rotom-Wash", ...],
    our_team_moves=[["Earthquake", "Swords Dance", ...], ...],
    our_team_abilities=["Rough Skin", ...],
    our_team_items=["Rocky Helmet", ...],
    opponent_team=["Landorus-Therian", "Ferrothorn", ...],
)
```

There's also `predict_lead_from_state(state: UniversalState)` which takes a parsed replay state directly.

Lead selection can use either **argmax** (deterministic) or **multinomial sampling** from the predicted distribution (controlled by `use_argmax`).

### Where it's used

| Consumer | How it uses TeamPreviewModel |
|---|---|
| **Standalone training** (`python -m metamon.backend.team_preview.preview`) | The module can be run directly with CLI arguments to train a new model |

**Further reading:** The model definition and training loop are in `metamon/backend/team_preview/preview.py`. The `PokemonTokenizer` (which maps species/move/ability/item names to integer tokens) is defined in `metamon/tokenizer.py`. The `consistent_pokemon_order()` and `consistent_move_order()` sorting utilities are in `metamon/interface.py`. The `CrossAttentionBlock` and `SelfAttentionBlock` used by the model are in `metamon/il/model.py`.

# Tricky battle mechanics that cause parsing headaches

Several Pokémon battle mechanics are notoriously difficult to parse correctly from spectator logs. When working on the parser, watch out for these:

### Zoroark / Zorua (Illusion)
Zoroark disguises itself as the last Pokémon in the party. The spectator sees the disguise's species in `|switch|` and `|move|` messages. When Illusion breaks, Showdown emits `|replace|POKEMON|DETAILS|HP` — the parser must rewind to before the illusion started, restore the disguise Pokémon's original state, and transfer newly-discovered moves/items/abilities from the disguise window to the real Zoroark. The backward pass (`_resolve_zoroark`) fixes action targets that pointed to the disguise and copies Zoroark's real moveset to the disguise Pokémon for action validation. Replays with Zoroark are flagged with `WarningFlags.ZOROARK` and have relaxed validation.

### Foreign-summoned moves (Metronome, Sleep Talk, etc.)
`MOVE_OVERRIDE` moves (Metronome, Mirror Move, Copycat, Assist, Nature Power, Me First, Magic Coat, Snatch) call random or opponent moves that the user does **not** actually know. The parser must suppress these from being added to `had_moves`. `MOVE_OVERRIDE_BUT_REVEAL_ANYWAY` (Sleep Talk) is the exception — it draws from the user's own moveset, so the revealed move IS a real move. The parser uses `pending_foreign_move` to track cross-turn foreign move sequences and suppress follow-up turns.

### Consecutive / multi-turn moves (Outrage, Thrash, Petal Dance, Rollout, etc.)
`CONSECUTIVE_MOVES` lock the user in for 2–3 turns. When called by Metronome, the parser must suppress all turns. The flag `pending_foreign_move` with a charge-move counter (`_pending_foreign_charge_remaining`) handles the nested case of foreign-called charge moves (e.g. Metronome → Solar Beam).

### Gen 1 PP rollover
Partial trapping moves (Wrap, Bind, Fire Spin, Clamp) in Gen 1 cause PP to roll over from 0 to 63 after the first use — a well-known RBY bug. The parser handles this with `GEN1_PP_ROLLOVERS` and a special `pp_used = -63` assignment.

### Transform and Mimic
Transform copies the opponent's species, types, stats, and moves. The parser tracks `transformed_into` and `transformed_this_turn`. During backward fill, `_resolve_transforms` propagates moves the transformed opponent had back through the transformation window, so the dataset sees a full moveset. Mimic temporarily copies one move; `PEEffect.MIMIC` and `|-start|` / `|-activate|` messages reveal which move was copied. Both Transform and Mimic can cause movesets to exceed 4 moves, which is handled with truncation at the interface level.

### Forced switches (U-turn, Volt Switch, Eject Button, Red Card, Roar, Dragon Tail, Revival Blessing, etc.)
When a move or item forces a switch, the parser creates a **subturn** — a frozen mid-turn state where the forced switch action happens. Subturns must be matched with actual switch-in messages. Edge cases where the forced switch *fails* (e.g. U-turn into Protect, Volt Switch blocked by Lightning Rod) leave unfilled subturns that produce warnings but don't crash the parser.

### Skill Swap and ability overwriting
Abilities that overwrite other abilities (Lingering Aroma, Mummy, Wandering Spirit) require care with `[from] ability: [of]` message parsing. Skill Swap can fail against certain abilities (Wonder Guard, Multitype, Illusion, etc.). The parser has an explicit list (`SKILL_SWAP_FAILS`) but the failure case raises `ForwardException("Detected Skill Swap failure with patch TODO")` — it's not fully handled.

### Item manipulation (Trick, Switcheroo, Thief, Covet, Knock Off, Fling)
Trick and Switcheroo swap items — tracked via `pokemon.tricking`. Thief and Covet steal from a named target. Knock Off and Fling remove the target's item. Corrosive Gas is also in `ITEM_APPROVED_SKIP`. The `[from] move: [of] pokemon` messages in `|-item|` / `|-enditem|` must be carefully parsed to determine whose item changed.

### Choice messages with numeric format
When `|choice|` uses numeric format (`move 1`, `switch 3`), the parser currently **cannot** use it because the mapping from numbers to specific move/switch names is unknown without the Showdown request messages (which are only present in the online env, not in raw replays). Only named choices (`move Ice Beam`) are processed.

**Further reading:** The `_parse_choice` method in `metamon/backend/replay_parser/forward.py` shows the current (limited) choice handling logic. The `_parse_move` method (~300 lines) is where all the special-case move handling lives — study the [from] effect parsing, `MOVE_OVERRIDE` suppression, and `pending_foreign_move` tracking. The `_parse_replace` method handles Zoroark. All known exception types are catalogued in `metamon/backend/replay_parser/exceptions.py` — grep for any exception class name to find where it's raised. The `check_forward_consistency` and `check_forced_switching` functions in `checks.py` enforce invariants and often surface edge cases that weren't handled. Specific tricky replays: Gen 1 Wrap/Bind PP rollover battles, any Gen 9 replay with Revival Blessing, and replays containing Zoroark or Ditto (Transform) — search `$METAMON_CACHE_DIR/parsed-replays/` for files that triggered `WarningFlags` by checking the parser's error history.
