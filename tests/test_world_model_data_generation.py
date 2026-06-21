import os
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.generate_world_model_data as wm_data
from metamon.jepa.model import compute_paired_losses

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
    np.testing.assert_array_equal(sample["p1_action"][2], np.array([12], dtype=np.int16))
    batch = collate_paired_fn([sample], pad_id=0)
    assert batch["p1_state_T"].shape[:2] == (1, 3)
    assert batch["p1_action"].shape[:2] == (1, 3)
    assert batch["p1_won"].shape == (1, 3)


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


def test_compute_paired_losses_supports_gaussian_beliefs_and_rank():
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
        "rank_p1_teacher": torch.tensor([1.0, 0.0]),
        "rank_p2_teacher": torch.tensor([0.0, 1.0]),
        "rank_p1_belief": torch.tensor([1.0, 0.0]),
        "rank_p2_belief": torch.tensor([0.0, 1.0]),
        "rank_p1_next_belief": torch.tensor([1.0, 0.0]),
        "rank_p2_next_belief": torch.tensor([0.0, 1.0]),
        "p1_won": torch.tensor([True, False]),
    }

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_sigreg=0.0,
        lambda_opponent_state=1.0,
        lambda_action=1.0,
        lambda_next_state=1.0,
        lambda_rank=1.0,
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    assert metrics["opponent_state_loss"] == pytest.approx(0.0)
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["next_state_loss"] == pytest.approx(0.0)
    assert metrics["rank_loss"] == pytest.approx(torch.nn.functional.softplus(torch.tensor(-1.0)).item())
    assert loss.item() == pytest.approx(metrics["rank_loss"])


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
        "rank_p1_teacher": torch.ones((2, 3)),
        "rank_p2_teacher": torch.zeros((2, 3)),
        "rank_p1_belief": torch.ones((2, 3)),
        "rank_p2_belief": torch.zeros((2, 3)),
        "rank_p1_next_belief": torch.ones((2, 3)),
        "rank_p2_next_belief": torch.zeros((2, 3)),
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
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    assert metrics["opponent_state_loss"] == pytest.approx(0.0)
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["next_state_loss"] == pytest.approx(0.0)
    assert metrics["rank_valid"] == pytest.approx(1.0)
    assert torch.isfinite(loss)


def test_compute_paired_losses_ignores_rank_invalid_ties():
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
        "rank_p1_teacher": torch.tensor([1.0, -10.0]),
        "rank_p2_teacher": torch.tensor([0.0, 10.0]),
        "rank_p1_belief": torch.tensor([1.0, -10.0]),
        "rank_p2_belief": torch.tensor([0.0, 10.0]),
        "rank_p1_next_belief": torch.tensor([1.0, -10.0]),
        "rank_p2_next_belief": torch.tensor([0.0, 10.0]),
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
        sigreg_num_slices=1,
        sigreg_num_points=2,
    )

    expected = torch.nn.functional.softplus(torch.tensor(-1.0)).item()
    assert metrics["rank_loss"] == pytest.approx(expected)
    assert metrics["rank_valid"] == pytest.approx(0.5)
    assert loss.item() == pytest.approx(expected)
