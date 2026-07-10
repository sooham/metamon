import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from metamon.simple_world_model.action_vocab import ActionVocabulary
from metamon.simple_world_model.cache_latents import atomic_savez
from metamon.simple_world_model.data import (
    BalancedFormatBatchSampler,
    CompactFormatBatchSampler,
    LatentTransitionDataset,
    VStateDataset,
    assert_matching_cache,
    collate_latent,
    format_id_to_name,
)
from metamon.simple_world_model.model import (
    CausalLatentTransformer,
    SimpleWorldModel,
    StateVAE,
    c_losses,
    interleave_latent_history,
)
import metamon.simple_world_model.train as simple_world_model_train
from metamon.simple_world_model.train import build_arg_parser, train
from metamon.tokenizer import PokemonTokenizer


def _tiny_model(action_vocab_size: int = 8) -> SimpleWorldModel:
    return SimpleWorldModel(
        vocab_size=48,
        pad_id=0,
        action_vocab_size=action_vocab_size,
        latent_dim=8,
        v_cfg={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "max_seq_len": 16,
            "max_state_tokens": 8,
            "gradient_checkpointing": False,
        },
        action_encoder_cfg={"action_dim": 8},
        m_cfg={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "num_mixtures": 2,
            "max_context_transitions": 2,
            "gradient_checkpointing": False,
        },
        controller_cfg={"hidden_dim": 16, "dropout": 0.0},
    )


def test_wandb_stage_run_records_config_and_metrics(monkeypatch):
    class FakeRun:
        def __init__(self):
            self.defined_metrics = []
            self.logged = []
            self.finished = False

        def define_metric(self, name, **kwargs):
            self.defined_metrics.append((name, kwargs))

        def log(self, payload, *, step):
            self.logged.append((dict(payload), step))

        def finish(self):
            self.finished = True

    class FakeWandb:
        def __init__(self):
            self.run = FakeRun()
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self.run

    fake_wandb = FakeWandb()
    monkeypatch.setattr(simple_world_model_train, "_wandb", fake_wandb)
    args = build_arg_parser().parse_args([
        "--stage", "m", "--data_root", "data", "--formats", "gen1ou", "gen9ou",
        "--tokenizer_path", "tokenizer.json", "--wandb_project", "test-project",
        "--wandb_name", "m-smoke", "--wandb_log_interval", "17",
    ])

    logger = simple_world_model_train._start_wandb(
        args,
        stage="m",
        model=_tiny_model(),
        model_config={"latent_dim": 8},
        source_hash="dataset-hash",
        cache_hash="cache-hash",
        device=torch.device("cpu"),
    )

    assert logger is not None
    assert fake_wandb.init_kwargs["project"] == "test-project"
    assert fake_wandb.init_kwargs["job_type"] == "m"
    assert fake_wandb.init_kwargs["name"] == "m-smoke"
    assert fake_wandb.init_kwargs["tags"] == ["simple-world-model", "m", "gen1ou", "gen9ou"]
    config = fake_wandb.init_kwargs["config"]
    assert config["dataset_manifest_hash"] == "dataset-hash"
    assert config["latent_cache_manifest_hash"] == "cache-hash"
    assert config["model_config"] == {"latent_dim": 8}
    assert ("train/*", {"step_metric": "global_step"}) in fake_wandb.run.defined_metrics

    logger.log({"global_step": 17, "train/loss": 1.25}, step=17)
    logger.finish()
    assert fake_wandb.run.logged == [({"global_step": 17, "train/loss": 1.25}, 17)]
    assert fake_wandb.run.finished


def test_decoder_padding_does_not_change_valid_prefix_logits():
    torch.manual_seed(3)
    vae = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
    ).eval()
    z = torch.randn(1, 8)
    prefix = vae.decode(z, torch.tensor([[True, True, True]]))
    right_padded = vae.decode(z, torch.tensor([[True, True, True, False, False, False]]))
    torch.testing.assert_close(prefix, right_padded[:, :3], atol=1e-6, rtol=1e-6)


def test_encoder_header_state_padding_is_batch_invariant():
    torch.manual_seed(5)
    vae = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
    ).eval()
    single = vae(
        torch.tensor([[1, 2]]), torch.tensor([[3, 4]]),
        header_valid_mask=torch.tensor([[True, True]]),
        state_valid_mask=torch.tensor([[True, True]]), deterministic=True,
    )
    batched = vae(
        torch.tensor([[1, 2, 0, 0, 0], [5, 6, 7, 8, 9]]),
        torch.tensor([[3, 4, 0, 0], [10, 11, 12, 13]]),
        header_valid_mask=torch.tensor([[True, True, False, False, False], [True, True, True, True, True]]),
        state_valid_mask=torch.tensor([[True, True, False, False], [True, True, True, True]]), deterministic=True,
    )
    torch.testing.assert_close(single["mu"], batched["mu"][:1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(single["logits"], batched["logits"][:1, :2], atol=1e-6, rtol=1e-6)


def test_causal_transformer_has_no_future_leakage():
    torch.manual_seed(4)
    transformer = CausalLatentTransformer(
        latent_dim=4, action_dim=4, d_model=8, n_heads=2, n_layers=2,
        d_ff=16, max_context_transitions=2,
    ).eval()
    team = torch.randn(1, 4)
    states = torch.randn(1, 3, 4)
    own = torch.randn(1, 2, 4)
    opponent = torch.randn(1, 2, 4)
    baseline = transformer(team, states, own, opponent)
    future_changed = transformer(
        team,
        torch.cat([states[:, :2], states[:, 2:] + 100.0], dim=1),
        torch.cat([own[:, :1], own[:, 1:] - 100.0], dim=1),
        opponent,
    )
    # team, z0, own0, opp0, z1 are all before the changed own1/opp1/z2.
    torch.testing.assert_close(baseline["hidden"][:, :5], future_changed["hidden"][:, :5], atol=1e-6, rtol=1e-6)


def test_interleaving_is_team_state_own_opponent_state():
    team = torch.tensor([[100.0]])
    states = torch.tensor([[[1.0], [2.0], [3.0]]])
    own = torch.tensor([[[10.0], [11.0]]])
    opponent = torch.tensor([[[20.0], [21.0]]])
    tokens, valid, type_ids = interleave_latent_history(team, states, own, opponent)
    assert tokens[0, valid[0], 0].tolist() == [100.0, 1.0, 10.0, 20.0, 2.0, 11.0, 21.0, 3.0]
    assert type_ids[0, valid[0]].tolist() == [0, 1, 2, 3, 1, 2, 3, 1]


def test_action_vocabulary_round_trip_and_format_masks():
    vocab = ActionVocabulary.build([
        ("move: surf", "gen1ou"),
        ("switch: starmie", "gen1ou"),
        ("move thunderbolt", "gen9ou"),
    ])
    surf = vocab.encode("move surf")
    assert vocab.decode(surf) == "move surf"
    assert vocab.encode("move: surf") == surf
    assert vocab.format_mask("gen1ou")[surf]
    assert not vocab.format_mask("gen1ou")[vocab.encode("move thunderbolt")]
    restored = ActionVocabulary.from_state(vocab.to_state())
    assert restored.to_state() == vocab.to_state()


def test_format_id_map_accepts_current_and_legacy_metadata_spellings():
    assert format_id_to_name({"gen1ou": 0, "gen9ou": 1}) == {0: "gen1ou", 1: "gen9ou"}
    assert format_id_to_name({"0": "gen1ou", "1": "gen9ou"}) == {0: "gen1ou", 1: "gen9ou"}


def test_none_remains_in_m_but_is_excluded_from_controller_bc():
    vocab = ActionVocabulary.build([("move tackle", "gen1ou")])
    model = _tiny_model(len(vocab))
    out = model.forward_m(
        team_z=torch.randn(1, 8), state_z=torch.randn(1, 1, 8),
        own_history_action_ids=torch.empty(1, 0, dtype=torch.long),
        opponent_history_action_ids=torch.empty(1, 0, dtype=torch.long),
        current_own_action_ids=torch.tensor([vocab.none_id]),
        current_opponent_action_ids=torch.tensor([vocab.unknown_id]),
        state_valid_mask=torch.tensor([[True]]),
    )
    assert torch.isfinite(out["opponent_logits"]).all()
    controller = model.forward_c(
        z_t=out["z_t"], h_t=out["h"],
        legal_action_ids=torch.tensor([[vocab.none_id, vocab.encode("move tackle")]]),
        legal_action_mask=torch.tensor([[True, True]]),
    )
    eligible = torch.tensor([vocab.is_controller_id(i) for i in range(len(vocab))])
    loss, metrics, selected = c_losses(
        controller["controller_logits"],
        legal_action_ids=torch.tensor([[vocab.none_id, vocab.encode("move tackle")]]),
        legal_action_mask=torch.tensor([[True, True]]),
        chosen_legal_action_idx=torch.tensor([0]), controller_eligible=eligible,
    )
    assert loss.item() == 0.0
    assert metrics["controller_count"] == 0.0
    assert not bool(selected.item())


def test_balanced_sampler_emits_equal_generation_mass():
    sampler = BalancedFormatBatchSampler(
        ["gen1ou", "gen1ou", "gen1ou", "gen9ou"], [3, 2, 1, 4],
        batch_size=4, balanced=True, shuffle=False,
    )
    for batch in sampler:
        formats = ["gen1ou", "gen1ou", "gen1ou", "gen9ou"]
        assert sum(formats[index] == "gen1ou" for index in batch) == 2
        assert sum(formats[index] == "gen9ou" for index in batch) == 2


def _write_tiny_paired_dataset(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens({
        "team": 1, "state": 2, "a": 3, "b": 4, "move": 5, "tackle": 6,
        "switch": 7, "bench": 8, "growl": 9,
    })
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save_tokens_to_disk(str(tokenizer_path))

    team = tokenizer["team"]
    state = tokenizer["state"]
    move = tokenizer["move"]
    tackle = tokenizer["tackle"]
    switch = tokenizer["switch"]
    bench = tokenizer["bench"]
    growl = tokenizer["growl"]

    for split in ("train", "val"):
        directory = tmp_path / split
        directory.mkdir(parents=True)
        np.savez(
            directory / "paired_shard_0000.npz",
            p1_states=np.array([team, state, tokenizer["a"], state, tokenizer["b"]], dtype=np.int16),
            p1_state_offsets=np.array([0, 1, 3], dtype=np.int64),
            p1_state_lengths=np.array([1, 2, 2], dtype=np.int32),
            p2_states=np.array([team, state, tokenizer["b"], state, tokenizer["a"]], dtype=np.int16),
            p2_state_offsets=np.array([0, 1, 3], dtype=np.int64),
            p2_state_lengths=np.array([1, 2, 2], dtype=np.int32),
            p1_actions=np.array([move, tackle], dtype=np.int16),
            p1_action_offsets=np.array([0], dtype=np.int64), p1_action_lengths=np.array([2], dtype=np.int32),
            p1_opponent_actions=np.array([switch, bench], dtype=np.int16),
            p1_opponent_action_offsets=np.array([0], dtype=np.int64), p1_opponent_action_lengths=np.array([2], dtype=np.int32),
            p2_actions=np.array([switch, bench], dtype=np.int16),
            p2_action_offsets=np.array([0], dtype=np.int64), p2_action_lengths=np.array([2], dtype=np.int32),
            p2_opponent_actions=np.array([move, tackle], dtype=np.int16),
            p2_opponent_action_offsets=np.array([0], dtype=np.int64), p2_opponent_action_lengths=np.array([2], dtype=np.int32),
            p1_target_state_idx=np.array([[1]], dtype=np.int32), p1_next_state_idx=np.array([[2]], dtype=np.int32),
            p1_action_idx=np.array([[0]], dtype=np.int32),
            p2_target_state_idx=np.array([[1]], dtype=np.int32), p2_next_state_idx=np.array([[2]], dtype=np.int32),
            p2_action_idx=np.array([[0]], dtype=np.int32),
            p1_next_terminal_class=np.array([[1]], dtype=np.int16), battle_id=np.array([0], dtype=np.int32),
            format_id=np.array([[0]], dtype=np.int16),
            p1_battle_start=np.array([0, 3], dtype=np.int64), p2_battle_start=np.array([0, 3], dtype=np.int64),
            p1_battle_action_start=np.array([0, 1], dtype=np.int64), p2_battle_action_start=np.array([0, 1], dtype=np.int64),
            p1_legal_actions=np.array([[[move, tackle], [switch, bench]]], dtype=np.int16),
            p1_legal_action_mask=np.array([[True, True]]), p1_chosen_legal_action_idx=np.array([0], dtype=np.int16),
            p2_legal_actions=np.array([[[switch, bench], [move, growl]]], dtype=np.int16),
            p2_legal_action_mask=np.array([[True, True]]), p2_chosen_legal_action_idx=np.array([0], dtype=np.int16),
        )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "schema_version": "test", "formats": ["gen1ou"], "format_id_map": {"0": "gen1ou"},
    }))
    config = {
        "model": {
            "latent_dim": 8,
            "v": {"d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "max_seq_len": 16,
                  "max_state_tokens": 8, "gradient_checkpointing": False},
            "action_encoder": {"action_dim": 8},
            "m": {"d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32, "num_mixtures": 2,
                  "gradient_checkpointing": False},
            "controller": {"hidden_dim": 16},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return tokenizer_path, config_path


def test_compact_v_loader_samples_without_materializing_every_state_ref(tmp_path):
    tokenizer_path, _ = _write_tiny_paired_dataset(tmp_path)
    tokenizer = PokemonTokenizer().load_tokens_from_disk(str(tokenizer_path))
    dataset = VStateDataset(
        [str(tmp_path / "train" / "paired_shard_0000.npz")], data_root=tmp_path,
        max_state_tokens=8, formats=["gen1ou"],
    )
    assert len(dataset) == 4  # two non-header states from each POV
    assert not hasattr(dataset, "refs")
    sampler = CompactFormatBatchSampler(dataset, batch_size=2, balanced=False, shuffle=False)
    refs = next(iter(sampler))
    rows = [dataset[ref] for ref in refs]
    assert {row["fmt"] for row in rows} == {"gen1ou"}
    assert all(len(row["state"]) == 2 for row in rows)
    # Validation draws at most one sample from the raw battle, not p1+p2
    # duplicates, so its battle split remains genuinely disjoint.
    assert len(dataset.fixed_subset(4)) == 1


def test_cache_atomicity(monkeypatch, tmp_path):
    path = tmp_path / "sidecar.npz"

    def explode(*args, **kwargs):
        raise RuntimeError("disk failed")

    monkeypatch.setattr(np, "savez", explode)
    with pytest.raises(RuntimeError, match="disk failed"):
        atomic_savez(path, values=np.array([1]))
    assert not path.exists()
    assert not list(tmp_path.glob(".sidecar.*.npz"))


def test_v_cache_m_c_smoke_and_perspective_inversion(tmp_path):
    tokenizer_path, config_path = _write_tiny_paired_dataset(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    v_checkpoint = checkpoint_dir / "v.pt"
    cache_root = tmp_path / "cache"
    parser = build_arg_parser()

    train(parser.parse_args([
        "--stage", "v", "--data_root", str(tmp_path), "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path), "--config", str(config_path), "--save_dir", str(checkpoint_dir),
        "--checkpoint", str(v_checkpoint), "--batch_size", "1", "--max_updates", "1", "--val_interval", "1",
        "--val_samples", "4", "--print_interval", "0", "--no-compile", "--no-wandb",
    ]))
    assert v_checkpoint.exists()
    v_latest = checkpoint_dir / "v_latest.pt"
    assert torch.load(v_latest, map_location="cpu")["global_step"] == 1

    train(parser.parse_args([
        "--stage", "v", "--data_root", str(tmp_path), "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path), "--config", str(config_path), "--save_dir", str(checkpoint_dir),
        "--checkpoint", str(v_checkpoint), "--resume_checkpoint", str(v_latest),
        "--batch_size", "1", "--additional_updates", "1", "--val_interval", "1",
        "--val_samples", "4", "--print_interval", "0", "--no-compile", "--no-wandb",
    ]))
    assert torch.load(v_latest, map_location="cpu")["global_step"] == 2

    train(parser.parse_args([
        "--stage", "cache", "--data_root", str(tmp_path), "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path), "--v_checkpoint", str(v_checkpoint),
        "--latent_cache_root", str(cache_root), "--batch_token_budget", "64",
    ]))
    assert (cache_root / "manifest.json").exists()
    dataset = LatentTransitionDataset(cache_root, split="train", max_context_transitions=32, format_id_map={0: "gen1ou"})
    assert not hasattr(dataset, "refs")
    sampled_refs = next(iter(CompactFormatBatchSampler(dataset, batch_size=2, balanced=False, shuffle=False)))
    assert {ref.side for ref in sampled_refs} == {"p1", "p2"}
    p1 = dataset.sample_with_perspective(0, "p1")
    p2 = dataset.sample_with_perspective(0, "p2")
    assert p1["outcome"] == 0
    assert p2["outcome"] == 1
    assert p1["current_own_action_id"] == p2["current_opponent_action_id"]

    m_checkpoint = checkpoint_dir / "m.pt"
    train(parser.parse_args([
        "--stage", "m", "--data_root", str(tmp_path), "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path), "--config", str(config_path), "--save_dir", str(checkpoint_dir),
        "--checkpoint", str(m_checkpoint), "--v_checkpoint", str(v_checkpoint), "--latent_cache_root", str(cache_root),
        "--batch_size", "1", "--max_updates", "1", "--val_interval", "1", "--val_samples", "2",
        "--print_interval", "0", "--no-compile", "--no-wandb",
    ]))
    c_checkpoint = checkpoint_dir / "c.pt"
    train(parser.parse_args([
        "--stage", "c", "--data_root", str(tmp_path), "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path), "--config", str(config_path), "--save_dir", str(checkpoint_dir),
        "--checkpoint", str(c_checkpoint), "--v_checkpoint", str(v_checkpoint), "--m_checkpoint", str(m_checkpoint),
        "--latent_cache_root", str(cache_root), "--batch_size", "1", "--max_updates", "1", "--val_interval", "1",
        "--val_samples", "2", "--print_interval", "0", "--no-compile", "--no-wandb",
    ]))
    assert c_checkpoint.exists()
    checkpoint = torch.load(c_checkpoint, map_location="cpu")
    assert checkpoint["stage"] == "c"
    assert checkpoint["action_vocabulary"]
    assert checkpoint["dataset_manifest_hash"]
    assert checkpoint["latent_cache_manifest_hash"]

    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    vocabulary = ActionVocabulary.from_state(manifest["action_vocabulary"])
    # ``move growl`` is only a legal candidate in the tiny fixture, proving
    # cache-stage vocabulary construction includes legal actions, not clicks.
    assert vocabulary.decode(vocabulary.encode("move growl")) == "move growl"
    manifest["v_checkpoint_hash"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    tokenizer = PokemonTokenizer().load_tokens_from_disk(str(tokenizer_path))
    with pytest.raises(ValueError, match="manifest mismatch"):
        assert_matching_cache(
            cache_root, data_root=tmp_path, tokenizer=tokenizer, v_checkpoint_path=v_checkpoint, latent_dim=8,
        )
