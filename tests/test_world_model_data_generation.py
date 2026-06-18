import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import orjson
import pytest
import torch

import scripts.generate_world_model_data as wm_data
from metamon.jepa.dataset import JEPADataset
from metamon.jepa.model import compute_paired_losses
from metamon.sl.train import WorldModelDataset
from metamon.sl.train import collate_fn as sl_collate_fn
from scripts.generate_world_model_data import (
    PairedBattle,
    PairedShardAccumulator,
    ShardAccumulator,
    TokenizedPOV,
    _paired_transition_rows,
    raw_battle_key,
    split_groups,
    group_txt_files,
    tokenize_battle,
)
from metamon.jepa.train_paired import PairedJEPADataset


def test_shard_accumulator_writes_transition_table_v2_uncompressed_npz(tmp_path):
    """Verify v2 ShardAccumulator writes separate player/opponent action arrays."""
    acc = ShardAccumulator(fmt="gen1ou", fmt_id=3)

    # Battle 0: 3 states (header + 2 real states), 2 transitions
    battle0_states = [
        np.array([10, 11], dtype=np.int16),   # header
        np.array([12], dtype=np.int16),       # state 0
        np.array([13, 14, 15], dtype=np.int16),  # state 1
    ]
    battle0_pa = [np.array([1], dtype=np.int16), np.array([2], dtype=np.int16)]
    battle0_oa = [np.array([3], dtype=np.int16), np.array([4], dtype=np.int16)]

    # Battle 1: 2 states (header + 1 real state), 1 transition
    battle1_states = [
        np.array([20], dtype=np.int16),   # header
        np.array([21, 22], dtype=np.int16),  # state 0
    ]
    battle1_pa = [np.array([5], dtype=np.int16)]
    battle1_oa = [np.array([6], dtype=np.int16)]

    acc.append(battle0_states, battle0_pa, battle0_oa, won=True)
    acc.append(battle1_states, battle1_pa, battle1_oa, won=False)

    stats = acc.write(str(tmp_path), shard_idx=0)
    shard_path = tmp_path / "seq_shard_0000.npz"

    assert stats["battles"] == 2
    assert stats["states"] == 5
    assert stats["transitions"] == 3

    with zipfile.ZipFile(shard_path) as zf:
        assert {info.compress_type for info in zf.infolist()} == {zipfile.ZIP_STORED}

    data = np.load(shard_path)
    expected_keys = {
        "states", "state_lengths", "state_offsets",
        "player_actions", "player_action_offsets", "player_action_lengths",
        "opponent_actions", "opponent_action_offsets", "opponent_action_lengths",
        "prev_state_idx", "next_state_idx",
        "battle_id", "turn_idx", "format_id",
        "won", "raw_battle_key",
        "battle_start", "battle_action_start",
        "format_name", "format_id_value",
    }
    assert set(data.files) == expected_keys

    # States: header(2) + state0(1) + state1(3) + header(1) + state0(2) = 9 tokens
    np.testing.assert_array_equal(
        data["states"], np.array([10, 11, 12, 13, 14, 15, 20, 21, 22], dtype=np.int16))
    np.testing.assert_array_equal(data["state_lengths"], np.array([2, 1, 3, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(data["state_offsets"], np.array([0, 2, 3, 6, 7], dtype=np.int64))

    # Transitions: prev skips header; battle0(states 0→1, 1→2), battle1(header→state0)
    # battle0: header@0, state0@1, state1@2 → prev=[1,2], next=[2,3]
    # battle1: header@3, state0@4            → prev=[4],   next=[5]
    np.testing.assert_array_equal(data["prev_state_idx"], np.array([1, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(data["next_state_idx"], np.array([2, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(data["battle_id"], np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(data["battle_start"], np.array([0, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(data["battle_action_start"], np.array([0, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(data["won"], np.array([True, False]))
    assert str(data["format_name"]) == "gen1ou"
    assert int(data["format_id_value"]) == 3


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

    acc = PairedShardAccumulator(fmt="gen1ou", fmt_id=0)
    acc.append(PairedBattle("battle-1", p1, p2, rows))
    stats = acc.write(str(tmp_path), shard_idx=0)
    assert stats["transitions"] == 2

    data = np.load(tmp_path / "paired_shard_0000.npz")
    np.testing.assert_array_equal(data["p1_state_idx"], np.array([1, 3], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_next_state_idx"], np.array([2, 4], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_action_idx"], np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_state_idx"], np.array([1, 4], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_next_state_idx"], np.array([2, 5], dtype=np.int32))
    np.testing.assert_array_equal(data["p2_action_idx"], np.array([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(data["p1_battle_start"], np.array([0, 5], dtype=np.int64))
    np.testing.assert_array_equal(data["p2_battle_start"], np.array([0, 6], dtype=np.int64))
    np.testing.assert_array_equal(data["p1_battle_action_start"], np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(data["p2_battle_action_start"], np.array([0, 4], dtype=np.int64))

    struct_ids = {
        "chosen_move": 5,
        "end_chosen_move": 6,
        "opponent_chosen_move": 7,
        "end_opponent_chosen_move": 8,
    }
    dataset = PairedJEPADataset(
        [str(tmp_path / "paired_shard_0000.npz")],
        struct_ids,
        shuffle_shards=False,
    )
    samples = list(dataset)
    assert len(samples) == 2
    second = samples[1]
    assert len(second["p1_player_hist_T"]) == 2
    assert len(second["p1_player_hist_T1"]) == 3
    assert len(second["p2_player_hist_T"]) == 3
    assert len(second["p2_player_hist_T1"]) == 4
    np.testing.assert_array_equal(
        second["p1_player_hist_T"][-1],
        np.array([5, 11, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p1_player_hist_T1"][-1],
        np.array([5, 12, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_player_hist_T"][-1],
        np.array([5, 97, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_player_hist_T1"][-1],
        np.array([5, 22, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p1_action"],
        np.array([5, 12, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_action"],
        np.array([5, 22, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["p2_action_from_p1_perspective"],
        np.array([7, 22, 8], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second["actual_p1_action_from_p2_perspective"],
        np.array([7, 12, 8], dtype=np.int16),
    )

    capped_dataset = PairedJEPADataset(
        [str(tmp_path / "paired_shard_0000.npz")],
        struct_ids,
        shuffle_shards=False,
        max_history_blocks=2,
    )
    capped_second = list(capped_dataset)[1]
    assert len(capped_second["p1_state_T"]) == 3
    assert len(capped_second["p1_state_T1"]) == 3
    np.testing.assert_array_equal(
        capped_second["p1_state_T"][0],
        np.array([90], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_state_T"][1],
        np.array([102], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_state_T"][2],
        np.array([103], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_state_T1"][0],
        np.array([90], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_state_T1"][1],
        np.array([103], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_state_T1"][2],
        np.array([104], dtype=np.int16),
    )
    assert len(capped_second["p1_player_hist_T"]) == 1
    assert len(capped_second["p1_player_hist_T1"]) == 1
    np.testing.assert_array_equal(
        capped_second["p1_player_hist_T"][0],
        np.array([5, 11, 6], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        capped_second["p1_player_hist_T1"][0],
        np.array([5, 12, 6], dtype=np.int16),
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


def test_compute_paired_losses_targets_predicted_opponent_actions():
    state = torch.zeros((1, 2))
    outputs = {
        "enc_p1_T": state,
        "enc_p2_T": state,
        "enc_p1_T1": state,
        "enc_p2_T1": state,
        "pred_p2_T": state,
        "pred_p1_T": state,
        "p1_action": state,
        "p2_action": state,
        "p2_action_from_p1_perspective": torch.tensor([[1.0, 0.0]]),
        "actual_p1_action_from_p2_perspective": torch.tensor([[0.0, 2.0]]),
        "pred_p2_action": torch.tensor([[1.0, 0.0]]),
        "pred_p1_action": torch.tensor([[0.0, 2.0]]),
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


def test_shard_accumulator_can_shuffle_transition_rows_without_misalignment(tmp_path):
    acc = ShardAccumulator(fmt="gen1ou", fmt_id=0)
    acc.append(
        [
            np.array([10], dtype=np.int16),
            np.array([11], dtype=np.int16),
            np.array([12], dtype=np.int16),
            np.array([13], dtype=np.int16),
        ],
        [np.array([1], dtype=np.int16),
         np.array([2], dtype=np.int16),
         np.array([3], dtype=np.int16)],
        [np.array([4], dtype=np.int16),
         np.array([5], dtype=np.int16),
         np.array([6], dtype=np.int16)],
        won=True,
        raw_battle_key="battle_a",
    )

    acc.write(str(tmp_path), shard_idx=0, rng=np.random.default_rng(1))
    data = np.load(tmp_path / "seq_shard_0000.npz")

    rows = set(
        zip(
            data["prev_state_idx"].tolist(),
            data["next_state_idx"].tolist(),
            data["battle_id"].tolist(),
            data["turn_idx"].tolist(),
        )
    )
    assert rows == {
        (1, 2, 0, 0),
        (2, 3, 0, 1),
        (3, 4, 0, 2),
    }
    np.testing.assert_array_equal(data["raw_battle_key"], np.array(["battle_a"]))


# Legacy tests removed (v1 format, depends on removed UniversalState/WorldModelObservationSpace):
#   test_tokenize_battle_wraps_states_with_bos_eos_by_default
#   test_sl_collate_strips_stored_bos_eos_before_prompt_builder


def test_jepa_dataset_v2_yields_battle_blocks(tmp_path):
    """JEPA v2 dataset yields block-level state/action histories."""
    bos, eos, boa, eoa = 1, 2, 3, 4
    cm, ecm, ocm, eocm = 5, 6, 7, 8

    acc = ShardAccumulator(fmt="gen1ou", fmt_id=0)
    # Battle: header + 2 states + 1 transition
    acc.append(
        [
            np.array([90, 91], dtype=np.int16),                # header
            np.array([bos, 10, eos], dtype=np.int16),          # state 0
            np.array([bos, 20, 21, eos], dtype=np.int16),      # state 1
        ],
        [np.array([50], dtype=np.int16)],    # player action content
        [np.array([60], dtype=np.int16)],    # opponent action content
        won=True,
    )
    acc.write(str(tmp_path), shard_idx=0)

    structural_ids = {
        "boa": boa, "eoa": eoa,
        "chosen_move": cm, "end_chosen_move": ecm,
        "opponent_chosen_move": ocm, "end_opponent_chosen_move": eocm,
    }

    dataset = JEPADataset(
        [str(tmp_path / "seq_shard_0000.npz")],
        structural_ids,
        shuffle_shards=False,
    )
    (
        state_blocks_N,
        state_blocks_N1,
        pa_hist_N,
        oa_hist_N,
        pa_hist_N1,
        oa_hist_N1,
        pa_tokens,
        oa_tokens,
    ) = next(iter(dataset))

    assert len(state_blocks_N) == 2
    np.testing.assert_array_equal(state_blocks_N[0], np.array([90, 91], dtype=np.int16))
    np.testing.assert_array_equal(state_blocks_N[1], np.array([bos, 10, eos], dtype=np.int16))
    assert pa_hist_N == []
    assert oa_hist_N == []

    assert len(state_blocks_N1) == 3
    np.testing.assert_array_equal(state_blocks_N1[0], np.array([90, 91], dtype=np.int16))
    np.testing.assert_array_equal(state_blocks_N1[1], np.array([bos, 10, eos], dtype=np.int16))
    np.testing.assert_array_equal(state_blocks_N1[2], np.array([bos, 20, 21, eos], dtype=np.int16))
    assert len(pa_hist_N1) == 1
    assert len(oa_hist_N1) == 1
    np.testing.assert_array_equal(pa_hist_N1[0], np.array([cm, 50, ecm], dtype=np.int16))
    np.testing.assert_array_equal(oa_hist_N1[0], np.array([ocm, 60, eocm], dtype=np.int16))

    # Action tokens with delimiters
    np.testing.assert_array_equal(pa_tokens, np.array([cm, 50, ecm], dtype=np.int16))
    np.testing.assert_array_equal(oa_tokens, np.array([ocm, 60, eocm], dtype=np.int16))


def test_split_aware_dataset_discovery(tmp_path):
    for split in ("train", "val"):
        split_dir = tmp_path / "gen1ou" / split
        split_dir.mkdir(parents=True)
        acc = ShardAccumulator(fmt="gen1ou", fmt_id=0)
        acc.append(
            [
                np.array([1, 10, 2], dtype=np.int16),
                np.array([1, 20, 2], dtype=np.int16),
            ],
            [np.array([4], dtype=np.int16)],   # player actions
            [np.array([5], dtype=np.int16)],   # opponent actions
            won=True,
            raw_battle_key=f"battle_{split}",
        )
        acc.write(str(split_dir), shard_idx=0)

    # Build dummy structural IDs for the test
    struct_ids = {
        "boa": 1, "eoa": 2,
        "chosen_move": 3, "end_chosen_move": 4,
        "opponent_chosen_move": 5, "end_opponent_chosen_move": 6,
    }

    jepa_train = JEPADataset.from_formats(str(tmp_path), ["gen1ou"], split="train", structural_token_ids=struct_ids)
    jepa_val = JEPADataset.from_formats(str(tmp_path), ["gen1ou"], split="val", structural_token_ids=struct_ids)
    sl_train = WorldModelDataset.from_formats(str(tmp_path), ["gen1ou"], split="train")
    sl_val = WorldModelDataset.from_formats(str(tmp_path), ["gen1ou"], split="val")

    assert all("/train/" in path for path in jepa_train.shard_paths)
    assert all("/val/" in path for path in jepa_val.shard_paths)
    assert all("/train/" in path for path in sl_train.shard_paths)
    assert all("/val/" in path for path in sl_val.shard_paths)

    with pytest.raises(FileNotFoundError):
        JEPADataset.from_formats(str(tmp_path), ["gen2ou"], split="train",
                                 structural_token_ids=struct_ids)


def test_tokenize_cached_parsed_replay_can_populate_transition_shard(tmp_path):
    cache_dir = Path(os.environ.get("METAMON_CACHE_DIR", "/workspace/poke-datasets"))
    parsed_file = (
        cache_dir
        / "parsed-replays"
        / "gen1ou"
        / "smogtours-gen1ou-749168_Unrated_encore90411_vs_mindplate96156_02-23-2024_WIN.json"
    )
    tokenizer_path = cache_dir / "tokenizers" / "WorldModelObservationSpace-v1.json"
    if not parsed_file.is_file() or not tokenizer_path.is_file():
        pytest.skip("cached parsed replay/tokenizer fixture is unavailable")

    local_replay = tmp_path / parsed_file.name
    shutil.copy2(parsed_file, local_replay)

    result = tokenize_battle((str(local_replay), str(tokenizer_path)))
    assert result is not None

    token_ids_list, actions_arr, won, max_state_len = result
    assert len(token_ids_list) >= 2
    assert len(actions_arr) == len(token_ids_list) - 1
    assert all(tokens.dtype == np.int16 for tokens in token_ids_list)
    assert max_state_len == max(len(tokens) for tokens in token_ids_list)
    from metamon.tokenizer import PokemonTokenizer

    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens_from_disk(str(tokenizer_path))
    bos_id = tokenizer["<bos>"]
    eos_id = tokenizer["<eos>"]
    assert all(int(tokens[0]) == bos_id for tokens in token_ids_list)
    assert all(int(tokens[-1]) == eos_id for tokens in token_ids_list)
    assert isinstance(won, bool)

    acc = ShardAccumulator(fmt="gen1ou", fmt_id=0)
    acc.append(token_ids_list, actions_arr, won)
    stats = acc.write(str(tmp_path), shard_idx=0)
    data = np.load(tmp_path / "seq_shard_0000.npz")

    assert stats["states"] == len(token_ids_list)
    assert stats["transitions"] == len(actions_arr)
    assert len(data["prev_state_idx"]) == len(actions_arr)
    assert int(data["prev_state_idx"][0]) == 0
    assert int(data["next_state_idx"][0]) == 1
    assert int(data["prev_state_idx"][-1]) == len(token_ids_list) - 2
    assert int(data["next_state_idx"][-1]) == len(token_ids_list) - 1
    assert set(np.unique(data["format_id"]).tolist()) == {0}
