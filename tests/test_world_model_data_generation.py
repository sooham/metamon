import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import orjson
import pytest

import scripts.generate_world_model_data as wm_data
from metamon.jepa.train import JEPADataset
from metamon.sl.train import WorldModelDataset
from metamon.sl.train import collate_fn as sl_collate_fn
from scripts.generate_world_model_data import (
    LengthStats,
    ShardAccumulator,
    group_json_files,
    raw_battle_key,
    split_groups,
    tokenize_battle,
)


def test_length_stats_tracks_min_max_and_weighted_average():
    stats = LengthStats()

    stats.update_many([5, 2, 8])
    stats.update_many([3])

    assert stats.count == 4
    assert stats.total == 18
    assert stats.min_len == 2
    assert stats.max_len == 8
    assert stats.avg == pytest.approx(4.5)
    assert stats.as_metadata() == {
        "state_len_count": 4,
        "state_len_min": 2,
        "state_len_max": 8,
        "state_len_avg": pytest.approx(4.5),
    }

    val_stats = LengthStats()
    val_stats.update_many([20, 1])
    stats.merge(val_stats)

    assert stats.count == 6
    assert stats.min_len == 1
    assert stats.max_len == 20


def test_shard_accumulator_writes_transition_table_uncompressed_npz(tmp_path):
    acc = ShardAccumulator(fmt="gen1ou", fmt_id=3)

    battle0_states = [
        np.array([10, 11], dtype=np.int16),
        np.array([12], dtype=np.int16),
        np.array([13, 14, 15], dtype=np.int16),
    ]
    battle1_states = [
        np.array([20], dtype=np.int16),
        np.array([21, 22], dtype=np.int16),
    ]

    acc.append(battle0_states, np.array([1, 2], dtype=np.int16), won=True)
    acc.append(battle1_states, np.array([-1], dtype=np.int16), won=False)

    stats = acc.write(str(tmp_path), shard_idx=0)
    shard_path = tmp_path / "seq_shard_0000.npz"

    assert stats["battles"] == 2
    assert stats["states"] == 5
    assert stats["transitions"] == 3
    assert stats["avg_len"] == pytest.approx(1.8)
    assert stats["min_len"] == 1
    assert stats["max_len"] == 3

    with zipfile.ZipFile(shard_path) as zf:
        # ZIP_STORED == 0; np.savez_compressed would use ZIP_DEFLATED.
        assert {info.compress_type for info in zf.infolist()} == {zipfile.ZIP_STORED}

    data = np.load(shard_path)
    assert set(data.files) == {
        "states",
        "state_lengths",
        "state_offsets",
        "prev_state_idx",
        "next_state_idx",
        "actions",
        "battle_id",
        "turn_idx",
        "format_id",
        "won",
        "raw_battle_key",
        "battle_start",
        "format_name",
        "format_id_value",
    }

    np.testing.assert_array_equal(data["states"], np.array([10, 11, 12, 13, 14, 15, 20, 21, 22], dtype=np.int16))
    np.testing.assert_array_equal(data["state_lengths"], np.array([2, 1, 3, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(data["state_offsets"], np.array([0, 2, 3, 6, 7], dtype=np.int64))
    np.testing.assert_array_equal(data["prev_state_idx"], np.array([0, 1, 3], dtype=np.int32))
    np.testing.assert_array_equal(data["next_state_idx"], np.array([1, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(data["actions"], np.array([1, 2, -1], dtype=np.int16))
    np.testing.assert_array_equal(data["battle_id"], np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(data["turn_idx"], np.array([0, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(data["format_id"], np.array([3, 3, 3], dtype=np.int16))
    np.testing.assert_array_equal(data["battle_start"], np.array([0, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(data["won"], np.array([True, False]))
    np.testing.assert_array_equal(data["raw_battle_key"], np.array(["", ""]))
    assert str(data["format_name"]) == "gen1ou"
    assert int(data["format_id_value"]) == 3


def test_raw_battle_grouping_keeps_win_loss_together():
    files = [
        "/tmp/gen1ou/battle_a_WIN.json",
        "/tmp/gen1ou/battle_a_LOSS.json",
        "/tmp/gen1ou/battle_b_WIN.json",
        "/tmp/gen1ou/standalone.json",
    ]

    assert raw_battle_key(files[0]) == "battle_a"
    assert raw_battle_key(files[1]) == "battle_a"
    assert raw_battle_key(files[3]) == "standalone"

    groups = group_json_files(files)
    rng = np.random.default_rng(0)
    train_keys, val_keys, train_files, val_files = split_groups(groups, 0.5, rng)

    assert set(train_keys).isdisjoint(val_keys)
    for split_files in (train_files, val_files):
        keys = {raw_battle_key(path) for path in split_files}
        if "battle_a" in keys:
            assert "/tmp/gen1ou/battle_a_WIN.json" in split_files
            assert "/tmp/gen1ou/battle_a_LOSS.json" in split_files


def test_shard_accumulator_can_shuffle_transition_rows_without_misalignment(tmp_path):
    acc = ShardAccumulator(fmt="gen1ou", fmt_id=0)
    acc.append(
        [
            np.array([10], dtype=np.int16),
            np.array([11], dtype=np.int16),
            np.array([12], dtype=np.int16),
            np.array([13], dtype=np.int16),
        ],
        np.array([1, 2, 3], dtype=np.int16),
        won=True,
        raw_battle_key="battle_a",
    )

    acc.write(str(tmp_path), shard_idx=0, rng=np.random.default_rng(1))
    data = np.load(tmp_path / "seq_shard_0000.npz")

    rows = set(
        zip(
            data["prev_state_idx"].tolist(),
            data["next_state_idx"].tolist(),
            data["actions"].tolist(),
            data["battle_id"].tolist(),
            data["turn_idx"].tolist(),
        )
    )
    assert rows == {
        (0, 1, 1, 0, 0),
        (1, 2, 2, 0, 1),
        (2, 3, 3, 0, 2),
    }
    np.testing.assert_array_equal(data["raw_battle_key"], np.array(["battle_a"]))


def test_tokenize_battle_wraps_states_with_bos_eos_by_default(tmp_path, monkeypatch):
    tokenizer_path = tmp_path / "tok.json"
    tokenizer_path.write_bytes(
        orjson.dumps({"<bos>": 1, "<eos>": 2, "foo": 3, "bar": 4})
    )
    parsed_path = tmp_path / "battle.json"
    parsed_path.write_bytes(
        orjson.dumps(
            {
                "states": [{"battle_won": False}, {"battle_won": True}],
                "actions": [5],
            }
        )
    )

    class DummyUniversalState:
        battle_won = False

        @classmethod
        def from_dict(cls, state):
            out = cls()
            out.battle_won = bool(state.get("battle_won", False))
            return out

    class DummyObservationSpace:
        def reset(self):
            pass

        def state_to_obs(self, state):
            return {"text": np.array("foo bar", dtype=np.str_)}

    monkeypatch.setattr(wm_data, "UniversalState", DummyUniversalState)
    monkeypatch.setattr(wm_data, "WorldModelObservationSpace", DummyObservationSpace)
    monkeypatch.setattr(wm_data, "_TOKENIZER", None)

    result = tokenize_battle((str(parsed_path), str(tokenizer_path)))
    assert result is not None
    token_ids_list, actions_arr, won, max_state_len = result

    assert won is True
    np.testing.assert_array_equal(actions_arr, np.array([5], dtype=np.int16))
    assert max_state_len == 4
    for tokens in token_ids_list:
        np.testing.assert_array_equal(tokens, np.array([1, 3, 4, 2], dtype=np.int16))

    raw_result = tokenize_battle((str(parsed_path), str(tokenizer_path), False))
    assert raw_result is not None
    raw_token_ids_list, _raw_actions, _raw_won, raw_max_state_len = raw_result
    assert raw_max_state_len == 2
    for tokens in raw_token_ids_list:
        np.testing.assert_array_equal(tokens, np.array([3, 4], dtype=np.int16))


def test_sl_collate_strips_stored_bos_eos_before_prompt_builder():
    bos_id = 1
    eos_id = 2
    pad_id = 99
    batch = [
        (
            4,
            5,
            3,
            np.array([bos_id, 10, 11, eos_id], dtype=np.int16),
            np.array([bos_id, 20, 21, 22, eos_id], dtype=np.int16),
        )
    ]

    state_t, state_next, actions, state_t_lengths, state_next_lengths = sl_collate_fn(
        batch,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_state_len=4,
    )

    np.testing.assert_array_equal(state_t.numpy(), np.array([[10, 11, pad_id, pad_id]]))
    np.testing.assert_array_equal(state_next.numpy(), np.array([[20, 21, 22, pad_id]]))
    np.testing.assert_array_equal(actions.numpy(), np.array([3]))
    np.testing.assert_array_equal(state_t_lengths.numpy(), np.array([2]))
    np.testing.assert_array_equal(state_next_lengths.numpy(), np.array([3]))


def test_jepa_dataset_yields_stored_boundaries_without_appending(tmp_path):
    acc = ShardAccumulator(fmt="gen1ou", fmt_id=0)
    acc.append(
        [
            np.array([1, 10, 2], dtype=np.int16),
            np.array([1, 20, 21, 2], dtype=np.int16),
        ],
        np.array([4], dtype=np.int16),
        won=True,
    )
    acc.write(str(tmp_path), shard_idx=0)

    dataset = JEPADataset([str(tmp_path / "seq_shard_0000.npz")], shuffle_shards=False)
    prev_tokens, next_tokens, action = next(iter(dataset))

    np.testing.assert_array_equal(prev_tokens, np.array([1, 10, 2], dtype=np.int16))
    np.testing.assert_array_equal(next_tokens, np.array([1, 20, 21, 2], dtype=np.int16))
    assert action == 4


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
            np.array([4], dtype=np.int16),
            won=True,
            raw_battle_key=f"battle_{split}",
        )
        acc.write(str(split_dir), shard_idx=0)

    jepa_train = JEPADataset.from_formats(str(tmp_path), ["gen1ou"], split="train")
    jepa_val = JEPADataset.from_formats(str(tmp_path), ["gen1ou"], split="val")
    sl_train = WorldModelDataset.from_formats(str(tmp_path), ["gen1ou"], split="train")
    sl_val = WorldModelDataset.from_formats(str(tmp_path), ["gen1ou"], split="val")

    assert all("/train/" in path for path in jepa_train.shard_paths)
    assert all("/val/" in path for path in jepa_val.shard_paths)
    assert all("/train/" in path for path in sl_train.shard_paths)
    assert all("/val/" in path for path in sl_val.shard_paths)

    with pytest.raises(FileNotFoundError):
        JEPADataset.from_formats(str(tmp_path), ["gen2ou"], split="train")


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
