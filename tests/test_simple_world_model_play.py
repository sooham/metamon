import numpy as np

from metamon.jepa.player import BattleHistory
from metamon.simple_world_model.model import SimpleWorldModel
from metamon.simple_world_model.player import SimpleWorldModelPlayer
from metamon.tokenizer import PokemonTokenizer


def _tokenizer():
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens({
        "snorlax": 1,
        "tackle": 2,
        "normal": 3,
        "1.00": 4,
        "523": 5,
        "move": 6,
        "switch": 7,
    })
    return tokenizer


def _tiny_model(vocab_size: int, pad_id: int) -> SimpleWorldModel:
    return SimpleWorldModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        latent_dim=8,
        v_cfg={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "max_seq_len": 16,
            "gradient_checkpointing": False,
        },
        action_encoder_cfg={
            "action_dim": 8,
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "max_seq_len": 4,
            "gradient_checkpointing": False,
        },
        m_cfg={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "num_mixtures": 2,
            "gradient_checkpointing": False,
        },
        controller_cfg={
            "hidden_dim": 16,
            "dropout": 0.0,
        },
    )


def test_simple_world_model_online_diagnostics_use_team_plus_current_state():
    tokenizer = _tokenizer()
    player = SimpleWorldModelPlayer.__new__(SimpleWorldModelPlayer)
    player._tokenizer = tokenizer
    player._swm = _tiny_model(len(tokenizer), tokenizer.pad_token_id)
    player._max_history_blocks = 1

    team = np.array([
        tokenizer["<begin_team>"],
        tokenizer["snorlax"],
        tokenizer["523"],
        tokenizer["normal"],
        tokenizer["<end_team>"],
    ], dtype=np.int16)
    state = np.array([
        tokenizer["<bos>"],
        tokenizer["snorlax"],
        tokenizer["1.00"],
        tokenizer["523"],
        tokenizer["523"],
        tokenizer["normal"],
        tokenizer["<eos>"],
    ], dtype=np.int16)
    hist = BattleHistory(state_blocks=[team, state])

    diag = player._simple_world_model_diagnostics(
        hist,
        {
            0: "move: tackle",
            4: "switch: snorlax",
        },
    )

    assert diag["team_token_count"] == len(team)
    assert diag["current_state_token_count"] == len(state)
    # with no prior actions, the interleaved seen history is team || current state
    assert diag["history_token_count"] == len(team) + len(state)
    assert len(diag["rows"]) == 2
    assert {"score", "p_win", "p_loss", "p_ongoing", "next_norm"} <= set(diag["rows"][0])
