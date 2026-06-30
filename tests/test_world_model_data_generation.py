import os
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.generate_world_model_data as wm_data
from metamon.jepa.model import (
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
        path="p2.txt",
    )
    rows = _paired_transition_rows(p1, p2)
    assert [(r.p1_action_idx, r.p2_action_idx) for r in rows] == [(0, 0), (2, 3)]
    windows = _contiguous_rollout_windows(rows, rollout_len=1)

    acc = PairedShardAccumulator(format_names={0: "gen1ou"})
    acc.append(PairedBattle("battle-1", p1, p2, rows, windows))
    stats = acc.write(str(tmp_path), shard_idx=0)
    assert stats["transitions"] == 2
    assert stats["rollout_samples"] == 2
    assert stats["rollout_len"] == 1

    data = np.load(tmp_path / "paired_shard_0000.npz")
    np.testing.assert_array_equal(data["p1_target_state_idx"], np.array([[1], [3]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_next_state_idx"], np.array([[2], [4]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_action_idx"], np.array([[0], [2]], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_target_state_idx"], np.array([[1], [4]], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_next_state_idx"], np.array([[2], [5]], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_action_idx"], np.array([[0], [3]], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_battle_start"], np.array([0, 5], dtype=np.int64))
    np.testing.assert_array_equal(data["p2_battle_start"], np.array([0, 6], dtype=np.int64))
    np.testing.assert_array_equal(data["p1_battle_action_start"], np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(data["p2_battle_action_start"], np.array([0, 4], dtype=np.int64))
    for removed_key in (
        "p1_won",
        "p2_won",
        "rank_valid",
        "p1_legal_actions",
        "p2_legal_actions",
        "p1_chosen_legal_action_idx",
        "p2_chosen_legal_action_idx",
    ):
        assert removed_key not in data

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
    assert samples[0]["raw_battle_key"] == "battle-1"
    assert samples[0]["battle_id"] == 0
    np.testing.assert_array_equal(samples[0]["turn_number"], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(samples[1]["turn_number"], np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(samples[1]["subturn_idx"], np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(samples[1]["p1_target_state_idx_meta"], np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(samples[1]["p2_target_state_idx_meta"], np.array([4], dtype=np.int32))
    np.testing.assert_array_equal(samples[0]["p1_history_T"][0][0], np.array([90], dtype=np.int16))
    np.testing.assert_array_equal(samples[0]["p1_target_state_T"][0], np.array([101], dtype=np.int16))
    second = samples[1]
    assert len(second["p1_player_hist_T"]) == 1
    assert len(second["p1_player_hist_T"][0]) == 2
    assert len(second["p2_player_hist_T"][0]) == 3
    np.testing.assert_array_equal(
        second["p1_player_hist_T"][0][-1],
        np.array([11], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_player_hist_T"][0][-1],
        np.array([97], dtype=np.int16),
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
    assert batch["p1_history_T"].ndim == 4
    assert batch["p1_history_T"].shape[:2] == (2, 1)
    assert batch["p1_target_state_T"].shape[:2] == (2, 1)
    assert batch["p1_next_state_T1"].shape[:2] == (2, 1)
    assert batch["p1_action"].shape[:2] == (2, 1)
    assert batch["raw_battle_key"] == ["battle-1", "battle-1"]
    assert torch.equal(batch["turn_number"], torch.tensor([[1], [3]], dtype=torch.int32))
    assert torch.equal(batch["p1_target_state_idx_meta"], torch.tensor([[1], [3]], dtype=torch.int32))
    assert "p1_legal_actions" not in batch
    assert "p1_is_terminal" not in batch
    assert "rank_valid" not in batch

    capped_dataset = PairedJEPADataset(
        [str(tmp_path / "paired_shard_0000.npz")],
        struct_ids,
        shuffle_shards=False,
        max_history_blocks=2,
    )
    capped_second = list(capped_dataset)[1]
    capped_p1_history_T = capped_second["p1_history_T"][0]
    assert len(capped_p1_history_T) == 3
    np.testing.assert_array_equal(
        capped_p1_history_T[0],
        np.array([90], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_history_T[1],
        np.array([101], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_p1_history_T[2],
        np.array([102], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_target_state_T"][0],
        np.array([103], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_next_state_T1"][0],
        np.array([104], dtype=np.int16),
    )
    assert len(capped_second["p1_player_hist_T"][0]) == 2
    np.testing.assert_array_equal(
        capped_second["p1_player_hist_T"][0][-1],
        np.array([11], dtype=np.int16),
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
        path="p2.txt",
    )
    rows = _paired_transition_rows(p1, p2)
    windows = _contiguous_rollout_windows(rows, rollout_len=3)
    assert len(windows) == 1

    acc = PairedShardAccumulator(format_names={0: "gen1ou"}, rollout_len=3)
    acc.append(PairedBattle("battle-1", p1, p2, rows, windows))
    stats = acc.write(str(tmp_path), shard_idx=0)
    assert stats["rollout_samples"] == 1
    assert stats["transitions"] == 3

    data = np.load(tmp_path / "paired_shard_0000.npz")
    assert int(data["rollout_len"]) == 3
    np.testing.assert_array_equal(data["p1_target_state_idx"], np.array([[1, 2, 3]], dtype=np.int32))
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
    assert len(sample["p1_history_T"]) == 3
    assert len(sample["p1_target_state_T"]) == 3
    assert len(sample["p1_next_state_T1"]) == 3
    assert len(sample["p1_action"]) == 3
    assert "p1_won" not in sample
    assert "p1_is_terminal" not in sample
    np.testing.assert_array_equal(sample["p1_action"][2], np.array([12], dtype=np.int16))
    batch = collate_paired_fn([sample], pad_id=0)
    assert batch["p1_history_T"].shape[:2] == (1, 3)
    assert batch["p1_target_state_T"].shape[:2] == (1, 3)
    assert batch["p1_next_state_T1"].shape[:2] == (1, 3)
    assert batch["p1_action"].shape[:2] == (1, 3)
    assert "p1_won" not in batch
    assert "p1_is_terminal" not in batch


def test_paired_jepa_dataset_history_window_can_be_capped():
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        target_state_idx=4,
        action_base=0,
        max_hist=0,
    ) == (0, 0, 3)
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        target_state_idx=5,
        action_base=0,
        max_hist=0,
    ) == (0, 0, 4)
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        target_state_idx=4,
        action_base=0,
        max_hist=2,
    ) == (2, 1, 3)
    assert PairedJEPADataset._resolve_window(
        battle_start=0,
        target_state_idx=5,
        action_base=0,
        max_hist=2,
    ) == (3, 2, 4)


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


def test_parsed_pov_counts_chosen_and_opponent_moves(tmp_path, monkeypatch):
    class FakeTokenizer:
        def __init__(self):
            self.ids = {}

        def __getitem__(self, key):
            if key not in self.ids:
                self.ids[key] = len(self.ids) + 1
            return self.ids[key]

    monkeypatch.setattr(wm_data, "_TOKENIZER", FakeTokenizer())
    monkeypatch.setattr(wm_data, "_UNKNOWN_ID", 999)

    replay = """
<bos>
<turn> 1 <end_turn>
<eos>
<boa>
<chosen_move>move bodyslam<end_chosen_move>
<opponent_chosen_move>recover<end_opponent_chosen_move>
<eoa>
<bos>
<turn> 2 <end_turn>
<eos>
<terminal>won<end_terminal>
"""
    replay_path = tmp_path / "battle_WIN.txt"
    replay_path.write_text(replay)

    pov = wm_data._parse_single_battle_file_detailed(str(replay_path))

    assert pov is not None
    assert pov.move_counts == {"bodyslam": 1, "recover": 1}


def test_move_histogram_rows_divide_two_view_counts():
    rows = wm_data._move_histogram_rows({"bodyslam": 4, "recover": 1})

    assert rows[0]["move"] == "bodyslam"
    assert rows[0]["count"] == pytest.approx(2.0)
    assert rows[0]["raw_double_count"] == 4
    assert rows[1]["move"] == "recover"
    assert rows[1]["count"] == pytest.approx(0.5)
    assert wm_data._format_histogram_count(float(rows[1]["count"])) == "0.5"


def test_move_histogram_output_writes_markdown_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        wm_data,
        "_write_move_histogram_plot",
        lambda rows, png_path, title: False,
    )

    outputs = wm_data._write_move_histogram_outputs(
        str(tmp_path),
        {"bodyslam": 4, "recover": 2},
        title="Chosen Move Histogram (test)",
    )

    assert outputs["plot_path"] is None
    assert outputs["total_move_actions"] == pytest.approx(3.0)
    assert (tmp_path / "move_histogram.md").exists()
    assert (tmp_path / "move_histogram.csv").exists()
    assert "| 1 | bodyslam | 2 |" in (tmp_path / "move_histogram.md").read_text()
    assert "bodyslam,2," in (tmp_path / "move_histogram.csv").read_text()


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


def _zero_loss_outputs(state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "enc_p1_T": state,
        "enc_p2_T": state,
        "enc_p1_T1": state,
        "enc_p2_T1": state,
        "pred_p1_self_T_mu": state,
        "pred_p1_self_T_logvar": torch.zeros_like(state),
        "pred_p2_self_T_mu": state,
        "pred_p2_self_T_logvar": torch.zeros_like(state),
        "pred_p2_T_mu": state,
        "pred_p2_T_logvar": torch.zeros_like(state),
        "pred_p1_T_mu": state,
        "pred_p1_T_logvar": torch.zeros_like(state),
        "pred_p2_T": state,
        "pred_p1_T": state,
        "p1_action": action,
        "p2_action": action,
        "actual_p2_action_from_p1_perspective": action,
        "actual_p1_action_from_p2_perspective": action,
        "pred_p2_action_mu": action,
        "pred_p2_action_logvar": torch.zeros_like(action),
        "pred_p1_action_mu": action,
        "pred_p1_action_logvar": torch.zeros_like(action),
        "pred_p2_action": action,
        "pred_p1_action": action,
        "pred_p1_T1_mu": state,
        "pred_p1_T1_logvar": torch.zeros_like(state),
        "pred_p2_T1_mu": state,
        "pred_p2_T1_logvar": torch.zeros_like(state),
        "pred_p1_T1": state,
        "pred_p2_T1": state,
    }


def test_compute_paired_losses_targets_predicted_opponent_actions():
    state = torch.zeros((1, 2))
    action = torch.zeros((1, 2))
    outputs = _zero_loss_outputs(state, action)
    outputs["actual_p2_action_from_p1_perspective"] = torch.tensor([[1.0, 0.0]])
    outputs["actual_p1_action_from_p2_perspective"] = torch.tensor([[0.0, 2.0]])
    outputs["pred_p2_action_mu"] = torch.tensor([[1.0, 0.0]])
    outputs["pred_p1_action_mu"] = torch.tensor([[0.0, 2.0]])

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_self_state=0.0,
        lambda_opponent_state=0.0,
        lambda_action=1.0,
        lambda_next_state=0.0,
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
        "pred_p1_self_T_mu": state_mu,
        "pred_p1_self_T_logvar": state_logvar,
        "pred_p2_self_T_mu": state_mu,
        "pred_p2_self_T_logvar": state_logvar,
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
        lambda_self_state=1.0,
        lambda_opponent_state=1.0,
        lambda_action=1.0,
        lambda_next_state=1.0,
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

    assert metrics["self_state_loss"] == pytest.approx(expected_state)
    assert metrics["opponent_state_loss"] == pytest.approx(expected_state)
    assert metrics["action_loss"] == pytest.approx(expected_action)
    assert metrics["next_state_loss"] == pytest.approx(expected_next)
    assert loss.item() == pytest.approx(2 * expected_state + expected_action + expected_next)


def test_paired_jepa_forward_uses_predicted_beliefs_for_next_state():
    class SequenceGaussian:
        def __init__(self, values):
            self.values = list(values)

        def __call__(self, *args):
            value = self.values.pop(0)
            mu = torch.full((1, 2), value)
            return mu, torch.zeros_like(mu)

    class SequenceActionPolicy:
        def __init__(self):
            self.values = [50.0, 60.0]

        def __call__(self, self_state, opponent_state):
            value = self.values.pop(0)
            mu = torch.full_like(self_state, value)
            return mu, torch.zeros_like(mu)

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

    class FakeModel:
        training = False

        def __init__(self):
            self.self_belief_encoder = SequenceGaussian([10.0, 20.0])
            self.opp_belief_predictor = SequenceGaussian([30.0, 40.0])
            self.opp_action_policy_predictor = SequenceActionPolicy()
            self.next_state_predictor = RecordingNextStatePredictor()

        def encode_history_context(
            self,
            state_tokens,
            state_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        ):
            return (state_tokens.float(),)

        def encode_token_tokens(self, action_tokens):
            return action_tokens.float()

        def encode_action_tokens(self, action_tokens):
            return action_tokens.float()

        def reparameterize(self, mu, logvar, sample):
            return mu

    fake = FakeModel()
    history = torch.tensor([[[99.0, 99.0]]])
    history_valid = torch.ones((1, 1), dtype=torch.bool)
    p1_target = torch.tensor([[1.0, 2.0]])
    p2_target = torch.tensor([[3.0, 4.0]])
    p1_state_T1 = torch.tensor([[5.0, 6.0]])
    p2_state_T1 = torch.tensor([[7.0, 8.0]])
    p1_action = torch.tensor([[11.0, 12.0]])
    p2_action = torch.tensor([[21.0, 22.0]])
    actual_p2_action = torch.tensor([[31.0, 32.0]])
    actual_p1_action = torch.tensor([[41.0, 42.0]])

    outputs = PairedJEPAModel.forward(
        fake,
        p1_history_T=history,
        p1_history_T_valid=history_valid,
        p1_player_hist_T=history,
        p1_player_hist_T_valid=history_valid,
        p1_opponent_hist_T=history,
        p1_opponent_hist_T_valid=history_valid,
        p1_target_state_T=p1_target,
        p1_next_state_T1=p1_state_T1,
        p1_action_tokens=p1_action,
        actual_p2_action_from_p1_perspective_tokens=actual_p2_action,
        p2_history_T=history,
        p2_history_T_valid=history_valid,
        p2_player_hist_T=history,
        p2_player_hist_T_valid=history_valid,
        p2_opponent_hist_T=history,
        p2_opponent_hist_T_valid=history_valid,
        p2_target_state_T=p2_target,
        p2_next_state_T1=p2_state_T1,
        p2_action_tokens=p2_action,
        actual_p1_action_from_p2_perspective_tokens=actual_p1_action,
        sample_beliefs=False,
    )

    assert len(fake.next_state_predictor.calls) == 2
    p1_next_call, p2_next_call = fake.next_state_predictor.calls
    torch.testing.assert_close(p1_next_call[0], torch.full((1, 2), 10.0))
    torch.testing.assert_close(p1_next_call[1], p1_action)
    torch.testing.assert_close(p1_next_call[2], torch.full((1, 2), 30.0))
    torch.testing.assert_close(p1_next_call[3], torch.full((1, 2), 50.0))
    torch.testing.assert_close(p2_next_call[0], torch.full((1, 2), 20.0))
    torch.testing.assert_close(p2_next_call[1], p2_action)
    torch.testing.assert_close(p2_next_call[2], torch.full((1, 2), 40.0))
    torch.testing.assert_close(p2_next_call[3], torch.full((1, 2), 60.0))
    assert outputs["enc_p1_T"].equal(p1_target)
    assert outputs["actual_p2_action_from_p1_perspective"].equal(actual_p2_action)


def test_compute_paired_losses_accepts_rollout_axis():
    state = torch.zeros((2, 3, 2))
    action = torch.zeros((2, 3, 2))
    outputs = _zero_loss_outputs(state, action)

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_self_state=1.0,
        lambda_opponent_state=1.0,
        lambda_action=1.0,
        lambda_next_state=1.0,
        lambda_q_value=999.0,
        lambda_policy=999.0,
        gamma=0.1,
    )

    assert metrics["self_state_loss"] == pytest.approx(0.0)
    assert metrics["opponent_state_loss"] == pytest.approx(0.0)
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["next_state_loss"] == pytest.approx(0.0)
    for legacy_key in ("rank_loss", "rank_valid", "value_loss", "q_value_loss", "policy_loss"):
        assert legacy_key not in metrics
    assert torch.isfinite(loss)
