from metamon.tokenizer.tokenizer import (
    PokemonTokenizer,
    get_tokenizer,
)

# Module-level sentinel kept for backward compatibility with RL / IL /
# baselines code that predates the per-instance unknown/pad tokens.
# The world-model pipeline uses tokenizer.unknown_token_id and
# tokenizer.pad_token_id instead.
UNKNOWN_TOKEN: int = 0
PADDING_TOKEN: int = 0
