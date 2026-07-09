import numpy as np

from metamon.jepa.player import BattleHistory
from metamon.simple_world_model.action_vocab import ActionVocabulary
from metamon.simple_world_model.model import SimpleWorldModel
from metamon.simple_world_model.player import SimpleWorldModelPlayer
from metamon.tokenizer import PokemonTokenizer


def _tokenizer():
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens({
        "team": 1, "snorlax": 2, "state": 3, "move": 4, "tackle": 5,
        "switch": 6, "bench": 7,
    })
    return tokenizer


def _tiny_model(vocab_size: int, pad_id: int, action_vocab_size: int) -> SimpleWorldModel:
    return SimpleWorldModel(
        vocab_size=vocab_size, pad_id=pad_id, action_vocab_size=action_vocab_size, latent_dim=8,
        v_cfg={"d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "max_seq_len": 16,
               "max_state_tokens": 8, "gradient_checkpointing": False},
        action_encoder_cfg={"action_dim": 8},
        m_cfg={"d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "num_mixtures": 2,
               "max_context_transitions": 2, "gradient_checkpointing": False},
        controller_cfg={"hidden_dim": 16},
    )


def test_online_diagnostics_blends_rollouts_and_bc_prior():
    tokenizer = _tokenizer()
    action_vocab = ActionVocabulary.build([
        ("move tackle", "gen1ou"),
        ("switch bench", "gen1ou"),
    ])
    player = SimpleWorldModelPlayer.__new__(SimpleWorldModelPlayer)
    player._tokenizer = tokenizer
    player._fmt = "gen1ou"
    player._action_vocabulary = action_vocab
    player._swm = _tiny_model(len(tokenizer), tokenizer.pad_token_id, len(action_vocab)).eval()
    player._max_context_transitions = 2
    player._rollout_horizon = 1
    player._rollouts_per_action = 1

    team = np.array([tokenizer["team"], tokenizer["snorlax"]], dtype=np.int16)
    state = np.array([tokenizer["state"], tokenizer["snorlax"]], dtype=np.int16)
    hist = BattleHistory(state_blocks=[team, state])
    diag = player._simple_world_model_diagnostics(
        hist, {0: "move: tackle", 4: "switch: bench"}
    )

    assert diag["scorer"] == "rollout_plus_bc"
    assert diag["team_token_count"] == len(team)
    assert diag["current_state_token_count"] == len(state)
    assert len(diag["rows"]) == 2
    assert {"score", "bc_score", "rollout_value", "p_win", "p_loss"} <= set(diag["rows"][0])
