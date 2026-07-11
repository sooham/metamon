import json
import random
from pathlib import Path
from types import SimpleNamespace

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
    vae_losses,
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


def test_v_battle_sampling_alpha_tempers_long_battle_weight(tmp_path):
    directory = tmp_path / "train"
    directory.mkdir(parents=True)
    # Battle 0 has one non-header state per POV; battle 1 has four. All
    # blocks have one token because this test exercises only reference draws.
    np.savez(
        directory / "paired_shard_0000.npz",
        p1_states=np.arange(7, dtype=np.int16),
        p1_state_offsets=np.arange(7, dtype=np.int64),
        p1_state_lengths=np.ones(7, dtype=np.int32),
        p2_states=np.arange(7, dtype=np.int16),
        p2_state_offsets=np.arange(7, dtype=np.int64),
        p2_state_lengths=np.ones(7, dtype=np.int32),
        p1_battle_start=np.array([0, 2, 7], dtype=np.int64),
        p2_battle_start=np.array([0, 2, 7], dtype=np.int64),
        format_name=np.array("gen1ou"),
    )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "schema_version": "test", "formats": ["gen1ou"],
    }))
    dataset = VStateDataset(
        [str(directory / "paired_shard_0000.npz")], data_root=tmp_path,
        formats=["gen1ou"],
    )

    def long_to_short_ratio(alpha: float) -> float:
        rng = random.Random(17)
        counts = [0, 0]
        for _ in range(20_000):
            population = dataset.draw_battle_population("gen1ou", rng, alpha=alpha)
            ref = dataset.draw_ref_from_battle_population(population, rng, side="p1")
            counts[ref.battle_id] += 1
        return counts[1] / counts[0]

    assert long_to_short_ratio(0.0) == pytest.approx(1.0, rel=0.04)
    assert long_to_short_ratio(0.5) == pytest.approx(2.0, rel=0.04)
    assert long_to_short_ratio(1.0) == pytest.approx(4.0, rel=0.04)


def test_v_restart_recipe_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--stage", "v", "--data_root", "data", "--formats", "gen1ou",
        "--tokenizer_path", "tokenizer.json",
    ])
    assert simple_world_model_train.DEFAULT_UPDATES["v"] == 200_000
    assert args.batch_size == 128
    assert args.lr == pytest.approx(3e-5)
    assert args.min_lr == pytest.approx(3e-6)
    assert args.lr_warmup_updates == 2_000
    assert args.lr_schedule_updates == 200_000
    assert args.weight_decay == pytest.approx(0.01)
    assert args.grad_clip == pytest.approx(1.0)
    assert args.grad_clip_fraction_window == 1_000
    assert args.val_interval == 5_000
    assert args.val_mc_samples == 4
    assert args.train_eval_samples == 2_000
    assert args.train_metric_window == 100
    assert args.early_stop_patience == 10
    assert args.kl_warmup_updates == 20_000
    assert args.free_bits == pytest.approx(0.02)
    assert args.encoder_token_mask_prob == pytest.approx(0.0)
    assert args.mean_recon_weight == pytest.approx(0.0)
    config_path = Path(simple_world_model_train.__file__).parent / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["latent_dim"] == 512
    assert config["model"]["v"]["dropout"] == pytest.approx(0.0)
    assert config["loss"]["beta_kl"] == pytest.approx(0.01)


def test_v_warmup_cosine_schedule():
    def lr(update):
        return simple_world_model_train._warmup_cosine_lr(
            update, peak_lr=3e-5, min_lr=3e-6,
            warmup_updates=2_000, schedule_updates=200_000,
        )

    assert lr(1) == pytest.approx(1.5e-8)
    assert lr(2_000) == pytest.approx(3e-5)
    assert lr(101_000) == pytest.approx(1.65e-5)
    assert lr(200_000) == pytest.approx(3e-6)
    assert lr(210_000) == pytest.approx(3e-6)


def test_v_encoder_mask_excludes_structural_and_padding_tokens():
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.tensor([[True, True, True, True, False]])
    structural = torch.zeros(10, dtype=torch.bool)
    structural[[1, 3]] = True
    masked, selected, eligible = simple_world_model_train._mask_non_structural_tokens(
        tokens, valid, structural_token_lookup=structural,
        mask_token_id=9, probability=1.0,
    )
    assert masked.tolist() == [[1, 9, 3, 9, 5]]
    assert selected == 2
    assert eligible == 2


def test_v_mean_reconstruction_blend_preserves_objective_scale():
    torch.manual_seed(19)
    model = _tiny_model().train()
    batch = {
        "header_tokens": torch.tensor([[1, 2], [2, 3]]),
        "header_mask": torch.tensor([[True, True], [True, True]]),
        "state_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "state_mask": torch.tensor([[True, True, True], [True, True, True]]),
    }
    args = SimpleNamespace(
        encoder_token_mask_prob=0.0, mean_recon_weight=0.25,
        free_bits=0.02, kl_capacity=0.0, kl_capacity_weight=0.0,
    )
    loss, metrics, outputs = simple_world_model_train._run_v_batch(
        model, batch, beta=0.01, args=args,
    )
    mean_outputs = {
        "logits": model.v.decode(outputs["mu"], outputs["state_valid_mask"]),
        "mu": outputs["mu"],
        "logvar": outputs["logvar"],
        "state_valid_mask": outputs["state_valid_mask"],
    }
    mean_loss, mean_metrics = vae_losses(
        mean_outputs, batch["state_tokens"], beta_kl=0.01, free_bits=0.02,
    )
    sampled_loss, sampled_metrics = vae_losses(
        outputs, batch["state_tokens"], beta_kl=0.01, free_bits=0.02,
    )
    expected = 0.75 * sampled_loss + 0.25 * mean_loss

    torch.testing.assert_close(loss, expected)
    assert metrics["loss"] == pytest.approx(float(expected.detach()))
    assert metrics["objective_recon_ce"] == pytest.approx(
        0.75 * sampled_metrics["recon_ce"] + 0.25 * mean_metrics["recon_ce"]
    )
    assert metrics["mean_recon_ce"] == pytest.approx(mean_metrics["recon_ce"])
    assert metrics["mean_recon_weight"] == pytest.approx(0.25)


def test_v_validation_reports_reproducible_mc_reconstruction_metrics(monkeypatch):
    torch.manual_seed(23)
    model = _tiny_model()
    batch = {
        "header_tokens": torch.tensor([[1, 2], [2, 3]]),
        "header_mask": torch.tensor([[True, True], [True, True]]),
        "state_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "state_mask": torch.tensor([[True, True, True], [True, True, True]]),
        "formats": ["gen1ou", "gen1ou"],
    }
    args = SimpleNamespace(
        beta_kl=0.01, free_bits=0.02, kl_capacity=0.0,
        kl_capacity_weight=0.0, val_mc_samples=4, seed=11,
    )
    monkeypatch.setattr(simple_world_model_train, "_v_format_metrics", lambda outputs, row: {})
    monkeypatch.setattr(
        simple_world_model_train, "_v_token_category_metrics",
        lambda outputs, row, tokenizer: {},
    )

    first = simple_world_model_train._validate_v(
        model, [batch], torch.device("cpu"), args, PokemonTokenizer(),
    )
    second = simple_world_model_train._validate_v(
        model, [batch], torch.device("cpu"), args, PokemonTokenizer(),
    )

    assert first["selection_score"] == pytest.approx(first["recon_ce"])
    assert first["recon_ce_mc"] > 0.0
    assert first["recon_ce_mc_std"] >= 0.0
    assert 0.0 <= first["recon_token_acc_mc"] <= 1.0
    assert first["recon_token_acc_mc_std"] >= 0.0
    assert first["recon_ce_mc"] == pytest.approx(second["recon_ce_mc"])
    assert first["recon_ce_mc_std"] == pytest.approx(second["recon_ce_mc_std"])
    assert first["recon_token_acc_mc"] == pytest.approx(second["recon_token_acc_mc"])
    assert first["recon_token_acc_mc_std"] == pytest.approx(second["recon_token_acc_mc_std"])
    assert first["aggregate_mean_rms"] >= 0.0
    assert first["aggregate_std_min"] > 0.0
    assert first["aggregate_std_max"] >= first["aggregate_std_min"]
    assert first["aggregate_cov_offdiag_rms"] >= 0.0
    assert first["aggregate_gaussian_kl_per_dim"] >= 0.0
    assert first["aggregate_gaussian_kl_per_dim"] == pytest.approx(
        second["aggregate_gaussian_kl_per_dim"]
    )
    assert model.training


def test_v_train_eval_metrics_preserve_validation_selection_and_report_gaps():
    result = simple_world_model_train._merge_v_train_eval_metrics(
        {
            "loss": 2.0, "recon_ce": 1.7, "recon_token_acc": 0.6,
            "recon_ce_mc": 1.9, "recon_token_acc_mc": 0.55,
            "selection_score": 1.7,
        },
        {
            "loss": 1.5, "recon_ce": 1.2, "recon_token_acc": 0.8,
            "recon_ce_mc": 1.4, "recon_token_acc_mc": 0.75,
            "selection_score": 1.2,
        },
    )
    assert result["selection_score"] == pytest.approx(1.7)
    assert "train_eval_selection_score" not in result
    assert result["generalization_gap_loss"] == pytest.approx(0.5)
    assert result["generalization_gap_recon_ce"] == pytest.approx(0.5)
    assert result["generalization_gap_recon_ce_mc"] == pytest.approx(0.5)
    assert result["generalization_gap_recon_token_acc"] == pytest.approx(0.2)
    assert result["generalization_gap_recon_token_acc_mc"] == pytest.approx(0.2)


def test_training_loop_logs_clipping_and_early_stops():
    class Loader:
        dataset = object()

        def __iter__(self):
            yield {}

    class Sampler:
        def set_epoch(self, epoch):
            self.epoch = epoch

    class Logger:
        def __init__(self):
            self.logged = []

        def log(self, payload, *, step):
            self.logged.append((dict(payload), step))

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    args = SimpleNamespace(
        max_updates=10, max_steps=0, additional_updates=0,
        grad_clip_fraction_window=4, grad_accum_steps=1, grad_clip=0.01,
        compile=False, batch_size=1, num_workers=0,
        wandb_log_interval=1, print_interval=0, val_interval=1,
        train_metric_window=2,
    )
    scores = iter([1.0, 2.0, 3.0])
    saves = []
    logger = Logger()

    def batch_loss(batch, validation):
        return model.weight.sum(), {"loss": float(model.weight.detach().sum())}

    simple_world_model_train._training_loop(
        stage="v", model=model, train_loader=Loader(), train_sampler=Sampler(),
        validation=lambda: {"selection_score": next(scores)},
        batch_loss=batch_loss, optimizer=optimizer, args=args,
        save_callback=lambda update, metrics, improved: saves.append((update, improved)),
        device=torch.device("cpu"), wandb_logger=logger,
        learning_rate_schedule=lambda update: 0.1 * update,
        early_stop_patience=2,
    )
    assert saves == [(1, True), (2, False), (3, False)]
    train_logs = [payload for payload, _ in logger.logged if "train/grad_clip_fraction" in payload]
    assert [payload["train/grad_clip_fraction"] for payload in train_logs] == [1.0, 1.0, 1.0]
    assert train_logs[0]["train_smooth/loss"] == pytest.approx(train_logs[0]["train/loss"])
    assert train_logs[1]["train_smooth/loss"] == pytest.approx(
        0.5 * (train_logs[0]["train/loss"] + train_logs[1]["train/loss"])
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.3)


def test_v_battle_sampling_alpha_defaults_and_bounds():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--stage", "v", "--data_root", "data", "--formats", "gen1ou",
        "--tokenizer_path", "tokenizer.json",
    ])
    assert args.v_battle_sampling_alpha == 0.5
    with pytest.raises(ValueError, match="between 0 and 1"):
        CompactFormatBatchSampler(
            type("Dataset", (), {"total_by_format": {"gen1ou": 1}})(),
            batch_size=1, battle_sampling_alpha=1.1,
        )


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
    saved_v = torch.load(v_checkpoint, map_location="cpu")
    assert saved_v["training_config"]["stage"] == "v"
    assert saved_v["training_config"]["beta_kl"] == pytest.approx(0.01)
    assert saved_v["training_config"]["mean_recon_weight"] == pytest.approx(0.0)
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
