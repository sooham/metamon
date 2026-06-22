import os
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.generate_world_model_data as wm_data
from metamon.jepa.model import (
    JEPAActionProjector,
    JEPAActionValueHead,
    JEPADecisionStateEncoder,
    JEPAValueHead,
    PairedJEPAModel,
    compute_paired_losses,
)

from scripts.generate_world_model_data import (
    PairedBattle,
    PairedShardAccumulator,
    TokenizedPOV,
    _contiguous_rollout_windows,
    _paired_transition_rows,
    raw_battle_key,
    split_groups,
    group_txt_files,
)
from metamon.jepa.train_paired import PairedJEPADataset, collate_paired_fn


def test_raw_battle_grouping_keeps_win_loss_together():
    files = [
        "/tmp/gen1ou/battle_a_WIN.txt",
        "/tmp/gen1ou/battle_a_LOSS.txt",
        "/tmp/gen1ou/gen1ou-2405104611_Unrated_voltorb80670_vs_synthesis81182_07-18-2025_LOSS.txt",
        "/tmp/gen1ou/gen1ou-2405104611_Unrated_synthesis81182_vs_voltorb80670_07-18-2025_WIN.txt",
        "/tmp/gen1ou/smogtours-gen1ou-749168_Unrated_encore90411_vs_mindplate96156_02-23-2024_WIN.txt",
        "/tmp/gen1ou/battle_b_WIN.txt",
        "/tmp/gen1ou/standalone.txt",
    ]

    assert raw_battle_key(files[0]) == "battle_a"
    assert raw_battle_key(files[1]) == "battle_a"
    assert raw_battle_key(files[2]) == "gen1ou-2405104611"
    assert raw_battle_key(files[3]) == "gen1ou-2405104611"
    assert raw_battle_key(files[4]) == "smogtours-gen1ou-749168"
    assert raw_battle_key(files[-1]) == "standalone"

    groups = group_txt_files(files)
    rng = np.random.default_rng(0)
    train_keys, val_keys, train_files, val_files = split_groups(groups, 0.5, rng)

    assert set(train_keys).isdisjoint(val_keys)
    for split_files in (train_files, val_files):
        keys = {raw_battle_key(path) for path in split_files}
        if "battle_a" in keys:
            assert "/tmp/gen1ou/battle_a_WIN.txt" in split_files
            assert "/tmp/gen1ou/battle_a_LOSS.txt" in split_files


def test_terminal_regex_handles_stateful_text_whitespace():
    text = """
<terminal>
won
<end_terminal>
"""

    match = wm_data._TERMINAL_RE.search(text)

    assert match is not None
    assert match.group(1) == "won"


def test_paired_shard_accumulator_aligns_common_immediate_subturns(tmp_path):
    p1 = TokenizedPOV(
        state_token_arrays=[
            np.array([90], dtype=np.int16),
            np.array([101], dtype=np.int16),
            np.array([102], dtype=np.int16),
            np.array([103], dtype=np.int16),
            np.array([104], dtype=np.int16),
        ],
        player_action_arrays=[
            np.array([10], dtype=np.int16),
            np.array([11], dtype=np.int16),
            np.array([12], dtype=np.int16),
        ],
        opponent_action_arrays=[
            np.array([20], dtype=np.int16),
            np.array([21], dtype=np.int16),
            np.array([22], dtype=np.int16),
        ],
        turn_numbers=[1, 2, 3, 4],
        won=True,
        path="p1.txt",
    )
    p2 = TokenizedPOV(
        state_token_arrays=[
            np.array([91], dtype=np.int16),
            np.array([201], dtype=np.int16),
            np.array([202], dtype=np.int16),
            np.array([299], dtype=np.int16),  # extra turn-2 subturn
            np.array([203], dtype=np.int16),
            np.array([204], dtype=np.int16),
        ],
        player_action_arrays=[
            np.array([20], dtype=np.int16),
            np.array([98], dtype=np.int16),
            np.array([97], dtype=np.int16),
            np.array([22], dtype=np.int16),
        ],
        opponent_action_arrays=[
            np.array([10], dtype=np.int16),
            np.array([88], dtype=np.int16),
            np.array([87], dtype=np.int16),
            np.array([12], dtype=np.int16),
        ],
        turn_numbers=[1, 2, 2, 3, 4],
        won=False,
        path="p2.txt",
    )
    rows = _paired_transition_rows(p1, p2)
    assert [(r.p1_action_idx, r.p2_action_idx) for r in rows] == [(0, 0), (2, 3)]
    windows = _contiguous_rollout_windows(rows, rollout_len=1)

    acc = PairedShardAccumulator(fmt="gen1ou", fmt_id=0)
    acc.append(PairedBattle("battle-1", p1, p2, rows, windows))
    stats = acc.write(str(tmp_path), shard_idx=0)
    assert stats["transitions"] == 2
    assert stats["rollout_samples"] == 2
    assert stats["rollout_len"] == 1

    data = np.load(tmp_path / "paired_shard_0000.npz")
    np.testing.assert_array_equal(data["p1_state_idx"], np.array([[1], [3]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_next_state_idx"], np.array([[2], [4]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_action_idx"], np.array([[0], [2]], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_state_idx"], np.array([[1], [4]], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_next_state_idx"], np.array([[2], [5]], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_action_idx"], np.array([[0], [3]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_battle_start"], np.array([0, 5], dtype=np.int64))
    np.testing.assert_array_equal(data["p2_battle_start"], np.array([0, 6], dtype=np.int64))
    np.testing.assert_array_equal(data["p1_battle_action_start"], np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(data["p2_battle_action_start"], np.array([0, 4], dtype=np.int64))
    np.testing.assert_array_equal(data["rank_valid"], np.array([True], dtype=bool))
    assert data["p1_legal_actions"].shape[:2] == (3, 1)
    assert data["p2_legal_actions"].shape[:2] == (4, 1)
    np.testing.assert_array_equal(data["p1_chosen_legal_action_idx"], np.zeros(3, dtype=np.int16))
    np.testing.assert_array_equal(data["p2_chosen_legal_action_idx"], np.zeros(4, dtype=np.int16))

    struct_ids = {
        "unknown": 99,
    }
    dataset = PairedJEPADataset(
        [str(tmp_path / "paired_shard_0000.npz")],
        struct_ids,
        shuffle_shards=False,
    )
    samples = list(dataset)
    assert len(samples) == 2
    assert samples[0]["rank_valid"] == [True]
    assert samples[0]["p1_is_terminal"] == [False]
    assert samples[1]["p1_is_terminal"] == [True]
    np.testing.assert_array_equal(
        samples[0]["p1_next_legal_actions"][0],
        np.array([[11]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        samples[0]["p1_next_legal_action_mask"][0],
        np.array([True], dtype=bool),
    )
    assert samples[1]["p1_next_legal_actions"][0].shape == (0, 1)
    assert not bool(samples[1]["p1_next_legal_action_mask"][0].any())
    second = samples[1]
    assert len(second["p1_player_hist_T"]) == 1
    assert len(second["p1_player_hist_T"][0]) == 2
    assert len(second["p1_player_hist_T1"][0]) == 3
    assert len(second["p2_player_hist_T"][0]) == 3
    assert len(second["p2_player_hist_T1"][0]) == 4
    np.testing.assert_array_equal(
        second["p1_player_hist_T"][0][-1],
        np.array([11], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p1_player_hist_T1"][0][-1],
        np.array([12], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_player_hist_T"][0][-1],
        np.array([97], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_player_hist_T1"][0][-1],
        np.array([22], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p1_action"][0],
        np.array([12], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_action"][0],
        np.array([22], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["actual_p2_action_from_p1_perspective"][0],
        np.array([22], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["actual_p1_action_from_p2_perspective"][0],
        np.array([12], dtype=np.int16),
    )

    batch = collate_paired_fn(samples, pad_id=0)
    assert batch["p1_state_T"].ndim == 4
    assert batch["p1_state_T"].shape[:2] == (2, 1)
    assert batch["p1_action"].shape[:2] == (2, 1)
    assert batch["p1_legal_actions"].shape[:3] == (2, 1, 1)
    assert batch["p1_legal_action_mask"].shape == (2, 1, 1)
    assert batch["p1_next_legal_actions"].shape[:3] == (2, 1, 1)
    assert batch["p1_next_legal_action_mask"].shape == (2, 1, 1)
    assert batch["p1_next_legal_action_mask"][0, 0, 0]
    assert not batch["p1_next_legal_action_mask"][1, 0, 0]
    assert torch.equal(
        batch["p1_is_terminal"],
        torch.tensor([[False], [True]], dtype=torch.bool),
    )
    assert batch["p1_chosen_legal_action_idx"].shape == (2, 1)
    assert batch["rank_valid"].shape == (2, 1)

    capped_dataset = PairedJEPADataset(
        [str(tmp_path / "paired_shard_0000.npz")],
        struct_ids,
        shuffle_shards=False,
        max_history_blocks=2,
    )
    capped_second = list(capped_dataset)[1]
    capped_p1_state_T = capped_second["p1_state_T"][0]
    capped_p1_state_T1 = capped_second["p1_state_T1"][0]
    assert len(capped_p1_state_T) == 3
    assert len(capped_p1_state_T1) == 3
    np.testing.assert_array_equal(
        capped_p1_state_T[0],
        np.array([90], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_state_T[1],
        np.array([102], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_state_T[2],
        np.array([103], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_state_T1[0],
        np.array([90], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_state_T1[1],
        np.array([103], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_state_T1[2],
        np.array([104], dtype=np.int16),
    )
    assert len(capped_second["p1_player_hist_T"][0]) == 1
    assert len(capped_second["p1_player_hist_T1"][0]) == 1
    np.testing.assert_array_equal(
        capped_second["p1_player_hist_T"][0][0],
        np.array([11], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_player_hist_T1"][0][0],
        np.array([12], dtype=np.int16),
    )


def test_paired_shard_accumulator_writes_k_step_rollout_samples(tmp_path):
    p1 = TokenizedPOV(
        state_token_arrays=[
            np.array([90], dtype=np.int16),
            np.array([101], dtype=np.int16),
            np.array([102], dtype=np.int16),
            np.array([103], dtype=np.int16),
            np.array([104], dtype=np.int16),
        ],
        player_action_arrays=[
            np.array([10], dtype=np.int16),
            np.array([11], dtype=np.int16),
            np.array([12], dtype=np.int16),
        ],
        opponent_action_arrays=[
            np.array([20], dtype=np.int16),
            np.array([21], dtype=np.int16),
            np.array([22], dtype=np.int16),
        ],
        turn_numbers=[1, 2, 3, 4],
        won=True,
        path="p1.txt",
    )
    p2 = TokenizedPOV(
        state_token_arrays=[
            np.array([91], dtype=np.int16),
            np.array([201], dtype=np.int16),
            np.array([202], dtype=np.int16),
            np.array([203], dtype=np.int16),
            np.array([204], dtype=np.int16),
        ],
        player_action_arrays=[
            np.array([20], dtype=np.int16),
            np.array([21], dtype=np.int16),
            np.array([22], dtype=np.int16),
        ],
        opponent_action_arrays=[
            np.array([10], dtype=np.int16),
            np.array([11], dtype=np.int16),
            np.array([12], dtype=np.int16),
        ],
        turn_numbers=[1, 2, 3, 4],
        won=False,
        path="p2.txt",
    )
    rows = _paired_transition_rows(p1, p2)
    windows = _contiguous_rollout_windows(rows, rollout_len=3)
    assert len(windows) == 1

    acc = PairedShardAccumulator(fmt="gen1ou", fmt_id=0, rollout_len=3)
    acc.append(PairedBattle("battle-1", p1, p2, rows, windows))
    stats = acc.write(str(tmp_path), shard_idx=0)
    assert stats["rollout_samples"] == 1
    assert stats["transitions"] == 3

    data = np.load(tmp_path / "paired_shard_0000.npz")
    assert int(data["rollout_len"]) == 3
    np.testing.assert_array_equal(data["p1_state_idx"], np.array([[1, 2, 3]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_next_state_idx"], np.array([[2, 3, 4]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_action_idx"], np.array([[0, 1, 2]], dtype=np.int32))

    struct_ids = {
        "unknown": 99,
    }
    dataset = PairedJEPADataset(
        [str(tmp_path / "paired_shard_0000.npz")],
        struct_ids,
        shuffle_shards=False,
    )
    sample = next(iter(dataset))
    assert len(sample["p1_state_T"]) == 3
    assert len(sample["p1_action"]) == 3
    assert sample["p1_won"] == [True, True, True]
    assert sample["p1_is_terminal"] == [False, False, True]
    np.testing.assert_array_equal(sample["p1_action"][2], np.array([12], dtype=np.int16))
    batch = collate_paired_fn([sample], pad_id=0)
    assert batch["p1_state_T"].shape[:2] == (1, 3)
    assert batch["p1_action"].shape[:2] == (1, 3)
    assert batch["p1_won"].shape == (1, 3)
    assert torch.equal(
        batch["p1_is_terminal"],
        torch.tensor([[False, False, True]], dtype=torch.bool),
    )


def test_paired_jepa_dataset_history_window_can_be_capped():
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        state_end=4,
        action_base=0,
        max_hist=0,
    ) == (0, 0, 2)
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        state_end=5,
        action_base=0,
        max_hist=0,
    ) == (0, 0, 3)
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        state_end=4,
        action_base=0,
        max_hist=2,
    ) == (2, 1, 2)
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        state_end=5,
        action_base=0,
        max_hist=2,
    ) == (3, 2, 3)


def test_action_canonicalization_uses_no_role_delimiters_and_fills_unknown():
    flat = np.array([10, 11, 12], dtype=np.int16)
    offsets = np.array([0, 2, 2], dtype=np.int64)
    lengths = np.array([2, 0, 1], dtype=np.int32)

    combined, combined_offsets, combined_lengths = PairedJEPADataset._canonicalize_actions(
        flat,
        offsets,
        lengths,
        unknown_token=99,
    )

    np.testing.assert_array_equal(combined_lengths, np.array([2, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(combined_offsets, np.array([0, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(combined, np.array([10, 11, 99, 12], dtype=np.int16))


def test_legal_action_texts_use_only_acting_player_state():
    state_text = """
<bos>
<conditions>
noweather
<you> cantera <end_you>
<opponent> <end_opponent>
<end_conditions>
<begin_moves>
<move> blizzard ice special <end_move>
<move> recover normal status <end_move>
<end_moves>
<bench>
<poke1>
alakazam 1.00 psychic
<end_poke1>
<poke2>
chansey 0.00 normal fnt
<end_poke2>
<end_bench>
<eos>
"""

    legal, chosen_idx = wm_data._legal_action_texts_from_state(state_text, "move recover")

    assert legal == ["move blizzard", "move recover", "switch alakazam"]
    assert chosen_idx == 1

    force_state = state_text.replace("<you> cantera <end_you>", "<you> forceswitch <end_you>")
    legal, chosen_idx = wm_data._legal_action_texts_from_state(force_state, "move recover")

    assert legal == ["switch alakazam", "move recover"]
    assert chosen_idx == 1


def test_decision_value_and_action_value_heads_preserve_rollout_shapes():
    encoder = JEPADecisionStateEncoder(
        latent_dim=4,
        decision_dim=6,
        hidden_dim=8,
        n_layers=2,
        dropout=0.0,
    )
    value_head = JEPAValueHead(decision_dim=6, hidden_dim=8, n_layers=2, dropout=0.0)
    action_projector = JEPAActionProjector(
        action_latent_dim=3,
        decision_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )
    q_head = JEPAActionValueHead(decision_dim=6, hidden_dim=8, n_layers=2, dropout=0.0)

    self_state = torch.randn(2, 3, 4)
    opp_mu = torch.randn(2, 3, 4)
    opp_logvar = torch.randn(2, 3, 4)
    legal_actions = torch.randn(2, 3, 5, 3)

    decision_state = encoder(self_state, opp_mu, opp_logvar)
    value_logits = value_head(decision_state)
    action_state = action_projector(legal_actions)
    q_logits = q_head(decision_state, action_state)

    assert decision_state.shape == (2, 3, 6)
    assert value_logits.shape == (2, 3)
    assert action_state.shape == (2, 3, 5, 6)
    assert q_logits.shape == (2, 3, 5)


def test_compute_paired_losses_targets_predicted_opponent_actions():
    state = torch.zeros((1, 2))
    action = torch.zeros((1, 2))
    outputs = {
        "enc_p1_T": state,
        "enc_p2_T": state,
        "enc_p1_T1": state,
        "enc_p2_T1": state,
        "pred_p2_T_mu": state,
        "pred_p2_T_logvar": state,
        "pred_p1_T_mu": state,
        "pred_p1_T_logvar": state,
        "pred_p2_T": state,
        "pred_p1_T": state,
        "p1_action": action,
        "p2_action": action,
        "actual_p2_action_from_p1_perspective": torch.tensor([[1.0, 0.0]]),
        "actual_p1_action_from_p2_perspective": torch.tensor([[0.0, 2.0]]),
        "pred_p2_action_mu": torch.tensor([[1.0, 0.0]]),
        "pred_p2_action_logvar": torch.tensor([[0.0, 0.0]]),
        "pred_p1_action_mu": torch.tensor([[0.0, 2.0]]),
        "pred_p1_action_logvar": torch.tensor([[0.0, 0.0]]),
        "pred_p2_action": torch.tensor([[1.0, 0.0]]),
        "pred_p1_action": torch.tensor([[0.0, 2.0]]),
        "pred_p1_T1_mu": state,
        "pred_p1_T1_logvar": state,
        "pred_p2_T1_mu": state,
        "pred_p2_T1_logvar": state,
        "pred_p1_T1": state,
        "pred_p2_T1": state,
    }

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_sigreg=0.0,
        lambda_opponent_state=0.0,
        lambda_action=1.0,
        lambda_next_state=0.0,
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["action_loss_p1_to_p2"] == pytest.approx(0.0)
    assert metrics["action_loss_p2_to_p1"] == pytest.approx(0.0)


def test_compute_paired_losses_uses_gaussian_nll_for_prediction_heads():
    state_target = torch.tensor([[1.0, -1.0]])
    state_mu = torch.tensor([[0.0, 1.0]])
    state_logvar = torch.full_like(state_target, 1.3862944)  # log(4)
    action_target = torch.tensor([[2.0, -2.0]])
    action_mu = torch.tensor([[1.0, 0.0]])
    action_logvar = torch.full_like(action_target, -0.6931472)  # log(0.5)
    next_target = torch.tensor([[0.5, -0.5]])
    next_mu = torch.tensor([[0.0, 0.0]])
    next_logvar = torch.zeros_like(next_target)

    outputs = {
        "enc_p1_T": state_target,
        "enc_p2_T": state_target,
        "enc_p1_T1": next_target,
        "enc_p2_T1": next_target,
        "pred_p2_T_mu": state_mu,
        "pred_p2_T_logvar": state_logvar,
        "pred_p1_T_mu": state_mu,
        "pred_p1_T_logvar": state_logvar,
        "pred_p2_T": state_mu,
        "pred_p1_T": state_mu,
        "p1_action": torch.zeros_like(action_target),
        "p2_action": torch.zeros_like(action_target),
        "actual_p2_action_from_p1_perspective": action_target,
        "actual_p1_action_from_p2_perspective": action_target,
        "pred_p2_action_mu": action_mu,
        "pred_p2_action_logvar": action_logvar,
        "pred_p1_action_mu": action_mu,
        "pred_p1_action_logvar": action_logvar,
        "pred_p2_action": action_mu,
        "pred_p1_action": action_mu,
        "pred_p1_T1_mu": next_mu,
        "pred_p1_T1_logvar": next_logvar,
        "pred_p2_T1_mu": next_mu,
        "pred_p2_T1_logvar": next_logvar,
        "pred_p1_T1": next_mu,
        "pred_p2_T1": next_mu,
    }

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_sigreg=0.0,
        lambda_opponent_state=1.0,
        lambda_action=1.0,
        lambda_next_state=1.0,
        lambda_rank=0.0,
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    expected_state = (
        0.5 * (state_logvar + (state_target - state_mu).square() * torch.exp(-state_logvar))
    ).mean().item()
    expected_action = (
        0.5 * (action_logvar + (action_target - action_mu).square() * torch.exp(-action_logvar))
    ).mean().item()
    expected_next = (
        0.5 * (next_logvar + (next_target - next_mu).square() * torch.exp(-next_logvar))
    ).mean().item()

    assert metrics["opponent_state_loss"] == pytest.approx(expected_state)
    assert metrics["action_loss"] == pytest.approx(expected_action)
    assert metrics["next_state_loss"] == pytest.approx(expected_next)
    assert loss.item() == pytest.approx(expected_state + expected_action + expected_next)


def test_paired_jepa_forward_teacher_forces_actual_opponent_state_and_actions_for_next_state():
    class FakeBeliefPredictor:
        def __call__(self, history_context, current_state):
            state_mu = current_state + 10.0
            state_logvar = torch.zeros_like(current_state)
            action_mu = torch.full_like(current_state, 99.0)
            action_logvar = torch.zeros_like(current_state)
            return state_mu, state_logvar, action_mu, action_logvar

    class RecordingNextStatePredictor:
        def __init__(self):
            self.calls = []

        def __call__(self, current_state, own_action, opponent_state, opponent_action):
            self.calls.append((
                current_state.clone(),
                own_action.clone(),
                opponent_state.clone(),
                opponent_action.clone(),
            ))
            return torch.zeros_like(current_state), torch.zeros_like(current_state)

    class FakeDecisionStateEncoder:
        def __call__(self, self_state, opponent_state_mu, opponent_state_logvar):
            return self_state + opponent_state_mu * 0.0 + opponent_state_logvar * 0.0

    class ZeroValueHead:
        def __call__(self, decision_state):
            return torch.zeros(decision_state.shape[:-1], device=decision_state.device)

    class FakeActionProjector:
        def __call__(self, action_latent):
            return action_latent

    class ZeroActionValueHead:
        def __call__(self, decision_state, action_state):
            return torch.zeros(action_state.shape[:-1], device=action_state.device)

    class ZeroRankHead:
        def __call__(self, self_state, opponent_state):
            return torch.zeros(self_state.shape[:-1], device=self_state.device)

    class FakeModel:
        training = False
        _singleton_candidate_tokens = staticmethod(PairedJEPAModel._singleton_candidate_tokens)
        _singleton_candidate_mask = staticmethod(PairedJEPAModel._singleton_candidate_mask)
        _zero_chosen_indices = staticmethod(PairedJEPAModel._zero_chosen_indices)

        def __init__(self):
            self.opponent_belief_predictor = FakeBeliefPredictor()
            self.next_state_predictor = RecordingNextStatePredictor()
            self.rank_head = ZeroRankHead()
            self.decision_state_encoder = FakeDecisionStateEncoder()
            self.value_head = ZeroValueHead()
            self.action_projector = FakeActionProjector()
            self.action_value_head = ZeroActionValueHead()

        def encode_current_state(self, state_tokens, state_valid):
            return state_tokens.float()

        def encode_history_context(
            self,
            state_tokens,
            state_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        ):
            return torch.zeros_like(state_tokens.float())

        def encode_action_tokens(self, action_tokens):
            return action_tokens.float()

        def encode_action_candidates(self, action_tokens, action_mask=None):
            return action_tokens.float()

        def reparameterize(self, mu, logvar, sample):
            return mu

    fake = FakeModel()
    valid = torch.ones((1,), dtype=torch.bool)
    p1_state_T = torch.tensor([[1.0, 2.0]])
    p2_state_T = torch.tensor([[3.0, 4.0]])
    p1_state_T1 = torch.tensor([[5.0, 6.0]])
    p2_state_T1 = torch.tensor([[7.0, 8.0]])
    p1_action = torch.tensor([[11.0, 12.0]])
    p2_action = torch.tensor([[21.0, 22.0]])
    actual_p2_action = torch.tensor([[31.0, 32.0]])
    actual_p1_action = torch.tensor([[41.0, 42.0]])

    outputs = PairedJEPAModel.forward(
        fake,
        p1_state_T, valid,
        p1_state_T1, valid,
        p1_state_T, valid,
        p1_state_T, valid,
        p1_state_T1, valid,
        p1_state_T1, valid,
        p2_state_T, valid,
        p2_state_T1, valid,
        p2_state_T, valid,
        p2_state_T, valid,
        p2_state_T1, valid,
        p2_state_T1, valid,
        p1_action,
        p2_action,
        actual_p2_action,
        actual_p1_action,
        p1_next_legal_action_tokens=p1_action.unsqueeze(1),
        p1_next_legal_action_mask=torch.ones((1, 1), dtype=torch.bool),
        p2_next_legal_action_tokens=p2_action.unsqueeze(1),
        p2_next_legal_action_mask=torch.ones((1, 1), dtype=torch.bool),
        compute_td_bootstrap=True,
    )

    assert len(fake.next_state_predictor.calls) == 2
    p1_next_call, p2_next_call = fake.next_state_predictor.calls
    torch.testing.assert_close(p1_next_call[1], p1_action)
    torch.testing.assert_close(p1_next_call[2], p2_state_T)
    torch.testing.assert_close(p1_next_call[3], actual_p2_action)
    torch.testing.assert_close(p2_next_call[1], p2_action)
    torch.testing.assert_close(p2_next_call[2], p1_state_T)
    torch.testing.assert_close(p2_next_call[3], actual_p1_action)
    assert "p1_next_value_logit" in outputs
    assert "p2_next_value_logit" in outputs
    assert outputs["p1_next_q_logits"].shape == (1, 1)
    assert outputs["p2_next_q_logits"].shape == (1, 1)


def test_compute_paired_losses_uses_value_q_and_policy_heads():
    state = torch.zeros((2, 2))
    action = torch.zeros((2, 2))
    q_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    legal_mask = torch.ones((2, 2), dtype=torch.bool)
    chosen_idx = torch.tensor([1, 0])
    outputs = {
        "enc_p1_T": state,
        "enc_p2_T": state,
        "enc_p1_T1": state,
        "enc_p2_T1": state,
        "ctx_p1_T": state,
        "ctx_p2_T": state,
        "pred_p2_T_mu": state,
        "pred_p2_T_logvar": state,
        "pred_p1_T_mu": state,
        "pred_p1_T_logvar": state,
        "pred_p2_T": state,
        "pred_p1_T": state,
        "p1_action": action,
        "p2_action": action,
        "actual_p2_action_from_p1_perspective": action,
        "actual_p1_action_from_p2_perspective": action,
        "pred_p2_action_mu": action,
        "pred_p2_action_logvar": action,
        "pred_p1_action_mu": action,
        "pred_p1_action_logvar": action,
        "pred_p2_action": action,
        "pred_p1_action": action,
        "pred_p1_T1_mu": state,
        "pred_p1_T1_logvar": state,
        "pred_p2_T1_mu": state,
        "pred_p2_T1_logvar": state,
        "pred_p1_T1": state,
        "pred_p2_T1": state,
        "p1_value_logit": torch.zeros(2),
        "p2_value_logit": torch.zeros(2),
        "p1_q_logits": q_logits,
        "p2_q_logits": q_logits,
        "p1_legal_action_mask": legal_mask,
        "p2_legal_action_mask": legal_mask,
        "p1_chosen_legal_action_idx": chosen_idx,
        "p2_chosen_legal_action_idx": chosen_idx,
        "p1_won": torch.tensor([True, True]),
        "p2_won": torch.tensor([False, False]),
        "rank_valid": torch.tensor([True, True]),
    }

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_sigreg=0.0,
        lambda_opponent_state=0.0,
        lambda_action=0.0,
        lambda_next_state=0.0,
        lambda_rank=1.0,
        lambda_value=1.0,
        lambda_q_value=1.0,
        lambda_policy=1.0,
        lambda_value_teacher=0.0,
        lambda_q_teacher=0.0,
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    expected_value = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2), torch.ones(2)
    ).item() * 0.5 + torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2), torch.zeros(2)
    ).item() * 0.5
    expected_q = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([2.0, 2.0]), torch.ones(2))
        + torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([2.0, 2.0]), torch.zeros(2))
    ).item()
    expected_policy = torch.nn.functional.cross_entropy(
        q_logits,
        chosen_idx,
    ).item()

    assert metrics["rank_loss"] == pytest.approx(0.0)
    assert metrics["value_loss"] == pytest.approx(expected_value)
    assert metrics["q_value_loss"] == pytest.approx(expected_q)
    assert metrics["policy_loss"] == pytest.approx(expected_policy)
    assert loss.item() == pytest.approx(expected_value + expected_q + expected_policy)


def test_compute_paired_losses_accepts_rollout_axis():
    state = torch.zeros((2, 3, 2))
    action = torch.zeros((2, 3, 2))
    outputs = {
        "enc_p1_T": state,
        "enc_p2_T": state,
        "enc_p1_T1": state,
        "enc_p2_T1": state,
        "ctx_p1_T": state,
        "ctx_p2_T": state,
        "pred_p2_T_mu": state,
        "pred_p2_T_logvar": state,
        "pred_p1_T_mu": state,
        "pred_p1_T_logvar": state,
        "pred_p2_T": state,
        "pred_p1_T": state,
        "p1_action": action,
        "p2_action": action,
        "actual_p2_action_from_p1_perspective": action,
        "actual_p1_action_from_p2_perspective": action,
        "pred_p2_action_mu": action,
        "pred_p2_action_logvar": action,
        "pred_p1_action_mu": action,
        "pred_p1_action_logvar": action,
        "pred_p2_action": action,
        "pred_p1_action": action,
        "pred_p1_T1_mu": state,
        "pred_p1_T1_logvar": state,
        "pred_p2_T1_mu": state,
        "pred_p2_T1_logvar": state,
        "pred_p1_T1": state,
        "pred_p2_T1": state,
        "p1_value_logit": torch.zeros((2, 3)),
        "p2_value_logit": torch.zeros((2, 3)),
        "p1_q_logits": torch.zeros((2, 3, 2)),
        "p2_q_logits": torch.zeros((2, 3, 2)),
        "p1_legal_action_mask": torch.ones((2, 3, 2), dtype=torch.bool),
        "p2_legal_action_mask": torch.ones((2, 3, 2), dtype=torch.bool),
        "p1_chosen_legal_action_idx": torch.zeros((2, 3), dtype=torch.long),
        "p2_chosen_legal_action_idx": torch.zeros((2, 3), dtype=torch.long),
        "p1_won": torch.ones((2, 3), dtype=torch.bool),
        "p2_won": torch.zeros((2, 3), dtype=torch.bool),
        "rank_valid": torch.ones((2, 3), dtype=torch.bool),
    }

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_sigreg=0.0,
        lambda_opponent_state=1.0,
        lambda_action=1.0,
        lambda_next_state=1.0,
        lambda_rank=1.0,
        lambda_value=1.0,
        lambda_q_value=1.0,
        lambda_policy=1.0,
        lambda_value_teacher=0.0,
        lambda_q_teacher=0.0,
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    assert metrics["opponent_state_loss"] == pytest.approx(0.0)
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["next_state_loss"] == pytest.approx(0.0)
    assert metrics["rank_valid"] == pytest.approx(1.0)
    assert metrics["rank_loss"] == pytest.approx(0.0)
    assert metrics["value_loss"] == pytest.approx(torch.nn.functional.softplus(torch.tensor(0.0)).item())
    assert torch.isfinite(loss)


def test_compute_paired_losses_ignores_invalid_outcomes_for_value_q_policy():
    state = torch.zeros((2, 2))
    action = torch.zeros((2, 2))
    outputs = {
        "enc_p1_T": state,
        "enc_p2_T": state,
        "enc_p1_T1": state,
        "enc_p2_T1": state,
        "ctx_p1_T": state,
        "ctx_p2_T": state,
        "pred_p2_T_mu": state,
        "pred_p2_T_logvar": state,
        "pred_p1_T_mu": state,
        "pred_p1_T_logvar": state,
        "pred_p2_T": state,
        "pred_p1_T": state,
        "p1_action": action,
        "p2_action": action,
        "actual_p2_action_from_p1_perspective": action,
        "actual_p1_action_from_p2_perspective": action,
        "pred_p2_action_mu": action,
        "pred_p2_action_logvar": action,
        "pred_p1_action_mu": action,
        "pred_p1_action_logvar": action,
        "pred_p2_action": action,
        "pred_p1_action": action,
        "pred_p1_T1_mu": state,
        "pred_p1_T1_logvar": state,
        "pred_p2_T1_mu": state,
        "pred_p2_T1_logvar": state,
        "pred_p1_T1": state,
        "pred_p2_T1": state,
        "p1_value_logit": torch.tensor([1.0, -10.0]),
        "p2_value_logit": torch.tensor([0.0, 10.0]),
        "p1_q_logits": torch.tensor([[1.0, 0.0], [-10.0, 10.0]]),
        "p2_q_logits": torch.tensor([[0.0, 1.0], [10.0, -10.0]]),
        "p1_legal_action_mask": torch.ones((2, 2), dtype=torch.bool),
        "p2_legal_action_mask": torch.ones((2, 2), dtype=torch.bool),
        "p1_chosen_legal_action_idx": torch.zeros(2, dtype=torch.long),
        "p2_chosen_legal_action_idx": torch.zeros(2, dtype=torch.long),
        "p1_won": torch.tensor([True, False]),
        "p2_won": torch.tensor([False, False]),
        "rank_valid": torch.tensor([True, False]),
    }

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_sigreg=0.0,
        lambda_opponent_state=0.0,
        lambda_action=0.0,
        lambda_next_state=0.0,
        lambda_rank=1.0,
        lambda_value=1.0,
        lambda_q_value=1.0,
        lambda_policy=1.0,
        lambda_value_teacher=0.0,
        lambda_q_teacher=0.0,
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    expected_value = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([1.0]), torch.tensor([1.0]))
        + torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([0.0]))
    ).item()
    expected_q = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([1.0]), torch.tensor([1.0]))
        + torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([0.0]))
    ).item()
    expected_policy = 0.5 * (
        torch.nn.functional.cross_entropy(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        + torch.nn.functional.cross_entropy(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))
    ).item()
    assert metrics["rank_loss"] == pytest.approx(0.0)
    assert metrics["rank_valid"] == pytest.approx(0.5)
    assert metrics["value_loss"] == pytest.approx(expected_value)
    assert metrics["q_value_loss"] == pytest.approx(expected_q)
    assert metrics["policy_loss"] == pytest.approx(expected_policy)
    assert loss.item() == pytest.approx(expected_value + expected_q + expected_policy)
