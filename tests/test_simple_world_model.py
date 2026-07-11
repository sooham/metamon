import json
import math
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
    MODEL_VERSION,
    VStateDataset,
    assert_matching_cache,
    collate_latent,
    format_id_to_name,
)
from metamon.simple_world_model.model import (
    CausalLatentTransformer,
    SimpleWorldModel,
    StateVAE,
    aggregate_posterior_sigreg,
    c_losses,
    interleave_latent_history,
    vae_losses,
)
import metamon.simple_world_model.train as simple_world_model_train
from metamon.simple_world_model.train import build_arg_parser, train
from metamon.tokenizer import PokemonTokenizer


def _tiny_model(
    action_vocab_size: int = 8,
    *,
    decoder_conditioning: str = "additive",
    decoder_header_conditioning: str = "none",
    fixed_posterior_std: float | None = None,
    vocab_size: int = 48,
    pad_id: int = 0,
) -> SimpleWorldModel:
    return SimpleWorldModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
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
            "decoder_conditioning": decoder_conditioning,
            "decoder_header_conditioning": decoder_header_conditioning,
            "fixed_posterior_std": fixed_posterior_std,
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


def test_adaln_decoder_warm_start_is_exact_and_conditioners_receive_gradients():
    torch.manual_seed(31)
    legacy = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
        decoder_conditioning="additive",
    )
    conditioned = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
        decoder_conditioning="adaln",
    )
    missing, unexpected = conditioned.load_state_dict(legacy.state_dict(), strict=False)
    assert unexpected == []
    assert set(missing) == {
        f"decoder_blocks.{layer}.adaln.{parameter}"
        for layer in range(2)
        for parameter in ("weight", "bias")
    }

    z = torch.randn(3, 8)
    valid = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
        [True, True, False, False, False],
    ])
    legacy_logits = legacy.decode(z, valid)
    conditioned_logits = conditioned.decode(z, valid)
    torch.testing.assert_close(conditioned_logits, legacy_logits, atol=0.0, rtol=0.0)

    targets = torch.randint(1, 32, valid.shape)
    torch.nn.functional.cross_entropy(conditioned_logits[valid], targets[valid]).backward()
    for block in conditioned.decoder_blocks:
        weight_grad = block.adaln.weight.grad
        bias_grad = block.adaln.bias.grad
        assert weight_grad is not None and torch.isfinite(weight_grad).all()
        assert bias_grad is not None and torch.isfinite(bias_grad).all()
        assert float(weight_grad.abs().sum()) > 0.0
        assert float(bias_grad.abs().sum()) > 0.0


def test_adaln_decoder_padding_does_not_change_valid_prefix_logits():
    torch.manual_seed(32)
    vae = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
        decoder_conditioning="adaln",
    ).eval()
    # Move off the identity initialization so this covers active modulation.
    with torch.no_grad():
        for block in vae.decoder_blocks:
            block.adaln.weight.normal_(std=0.02)
    z = torch.randn(1, 8)
    prefix = vae.decode(z, torch.tensor([[True, True, True]]))
    right_padded = vae.decode(z, torch.tensor([[True, True, True, False, False, False]]))
    torch.testing.assert_close(prefix, right_padded[:, :3], atol=1e-6, rtol=1e-6)


def test_state_vae_rejects_unknown_decoder_conditioning():
    with pytest.raises(ValueError, match="decoder_conditioning"):
        StateVAE(
            vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
            n_layers=1, d_ff=32, max_seq_len=16, max_state_tokens=8,
            decoder_conditioning="memory",
        )


def test_header_cross_attention_zero_gate_is_exact_and_valid_header_changes_logits():
    torch.manual_seed(33)
    legacy = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
    ).eval()
    conditioned = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
        decoder_header_conditioning="cross_attention",
    ).eval()
    missing, unexpected = conditioned.load_state_dict(legacy.state_dict(), strict=False)
    assert unexpected == []
    assert set(missing) == {
        key for key in conditioned.state_dict() if key.startswith("decoder_header_")
    }

    z = torch.randn(2, 8)
    state_mask = torch.tensor([
        [True, True, True, False],
        [True, True, True, True],
    ])
    header = torch.tensor([[1, 2, 0], [3, 4, 5]])
    header_mask = header.ne(0)
    legacy_logits = legacy.decode(z, state_mask)
    conditioned_logits = conditioned.decode(
        z,
        state_mask,
        header_tokens=header,
        header_valid_mask=header_mask,
    )
    torch.testing.assert_close(conditioned_logits, legacy_logits, atol=0.0, rtol=0.0)

    assert conditioned.decoder_header_gate is not None
    with torch.no_grad():
        conditioned.decoder_header_gate.fill_(1.0)
    active_logits = conditioned.decode(
        z,
        state_mask,
        header_tokens=header,
        header_valid_mask=header_mask,
    )
    changed_header = header.clone()
    changed_header[0, 1] = 6
    changed_logits = conditioned.decode(
        z,
        state_mask,
        header_tokens=changed_header,
        header_valid_mask=header_mask,
    )
    assert not torch.equal(changed_logits[0], active_logits[0])
    torch.testing.assert_close(changed_logits[1], active_logits[1], atol=0.0, rtol=0.0)


def test_header_cross_attention_padding_storage_and_batch_are_invariant():
    torch.manual_seed(34)
    vae = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
        decoder_header_conditioning="cross_attention",
    ).eval()
    assert vae.decoder_header_gate is not None
    with torch.no_grad():
        vae.decoder_header_gate.fill_(1.0)

    z = torch.randn(1, 8)
    single = vae.decode(
        z,
        torch.tensor([[True, True, True]]),
        header_tokens=torch.tensor([[1, 2]]),
        header_valid_mask=torch.tensor([[True, True]]),
    )
    batch_z = torch.cat([z, torch.randn_like(z)], dim=0)
    state_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
    ])
    header_mask = torch.tensor([
        [True, True, False, False, False],
        [True, True, True, True, True],
    ])
    padded_header = torch.tensor([
        [1, 2, 7, 8, 9],
        [3, 4, 5, 6, 10],
    ])
    batched = vae.decode(
        batch_z,
        state_mask,
        header_tokens=padded_header,
        header_valid_mask=header_mask,
    )
    torch.testing.assert_close(single, batched[:1, :3], atol=1e-6, rtol=1e-6)

    different_padding = padded_header.clone()
    different_padding[0, 2:] = torch.tensor([11, 12, 13])
    changed_storage = vae.decode(
        batch_z,
        state_mask,
        header_tokens=different_padding,
        header_valid_mask=header_mask,
    )
    torch.testing.assert_close(batched[0], changed_storage[0], atol=0.0, rtol=0.0)


def test_header_conditioned_forward_has_no_target_state_decoder_skip(monkeypatch):
    torch.manual_seed(35)
    vae = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=1, d_ff=32, max_seq_len=16, max_state_tokens=8,
        decoder_header_conditioning="cross_attention",
    ).eval()
    assert vae.decoder_header_gate is not None
    with torch.no_grad():
        vae.decoder_header_gate.fill_(1.0)
    fixed_mu = torch.randn(1, 8)

    def fixed_encode(header_tokens, state_tokens=None, **kwargs):
        del state_tokens, kwargs
        mu = fixed_mu.expand(header_tokens.shape[0], -1)
        return mu, torch.zeros_like(mu)

    monkeypatch.setattr(vae, "encode", fixed_encode)
    header = torch.tensor([[1, 2]])
    header_mask = torch.tensor([[True, True]])
    state_mask = torch.tensor([[True, True, True]])
    first = vae(
        header,
        torch.tensor([[3, 4, 5]]),
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
        deterministic=True,
    )
    second = vae(
        header,
        torch.tensor([[6, 7, 8]]),
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
        deterministic=True,
    )
    torch.testing.assert_close(first["logits"], second["logits"], atol=0.0, rtol=0.0)


def test_state_vae_rejects_unknown_decoder_header_conditioning():
    with pytest.raises(ValueError, match="decoder_header_conditioning"):
        StateVAE(
            vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
            n_layers=1, d_ff=32, max_seq_len=16, max_state_tokens=8,
            decoder_header_conditioning="joint_encoder",
        )


def test_fixed_posterior_std_preserves_mu_and_sets_exact_sampling_scale(monkeypatch):
    torch.manual_seed(33)
    learned = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=1, d_ff=32, max_seq_len=16, max_state_tokens=8,
    ).eval()
    fixed = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=1, d_ff=32, max_seq_len=16, max_state_tokens=8,
        fixed_posterior_std=0.25,
    ).eval()
    # fixed_posterior_std adds no tensors, so legacy state loading stays strict.
    fixed.load_state_dict(learned.state_dict(), strict=True)
    header = torch.tensor([[1, 2, 0], [3, 4, 5]])
    state = torch.tensor([[6, 7, 0], [8, 9, 10]])
    header_mask = header.ne(0)
    state_mask = state.ne(0)
    learned_mu, _ = learned.encode(
        header, state,
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
    )
    fixed_mu, fixed_logvar = fixed.encode(
        header, state,
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
    )
    torch.testing.assert_close(fixed_mu, learned_mu, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        fixed_logvar,
        torch.full_like(fixed_logvar, 2.0 * math.log(0.25)),
        atol=0.0,
        rtol=0.0,
    )
    learned_team_mu, _ = learned.encode(
        header, header_valid_mask=header_mask,
    )
    fixed_team_mu, fixed_team_logvar = fixed.encode(
        header, header_valid_mask=header_mask,
    )
    torch.testing.assert_close(fixed_team_mu, learned_team_mu, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        fixed_team_logvar,
        torch.full_like(fixed_team_logvar, 2.0 * math.log(0.25)),
        atol=0.0,
        rtol=0.0,
    )

    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))
    sampled = fixed.sample(fixed_mu, fixed_logvar)
    torch.testing.assert_close(
        sampled - fixed_mu,
        torch.full_like(fixed_mu, 0.25),
        atol=1e-7,
        rtol=0.0,
    )


@pytest.mark.parametrize("fixed_std", [0.0, -0.1, float("inf"), float("nan")])
def test_state_vae_rejects_invalid_fixed_posterior_std(fixed_std):
    with pytest.raises(ValueError, match="fixed_posterior_std"):
        StateVAE(
            vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
            n_layers=1, d_ff=32, max_seq_len=16, max_state_tokens=8,
            fixed_posterior_std=fixed_std,
        )


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
    assert args.lambda_mu_sigreg == pytest.approx(0.0)
    assert args.mu_sigreg_warmup_updates == 2_000
    assert args.lambda_sampled_sigreg == pytest.approx(0.0)
    assert args.sampled_sigreg_warmup_updates == 2_000
    assert args.lambda_aggregate_sigreg == pytest.approx(0.0)
    assert args.aggregate_sigreg_warmup_updates == 2_000
    assert args.posterior_std_target is None
    assert args.posterior_std_weight == pytest.approx(0.0)
    assert args.posterior_std_warmup_updates == 2_000
    assert args.sigreg_num_slices == 128
    assert args.sigreg_num_points == 17
    assert args.sigreg_domain == pytest.approx(3.0)
    config_path = Path(simple_world_model_train.__file__).parent / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["latent_dim"] == 512
    assert config["model"]["v"]["dropout"] == pytest.approx(0.0)
    assert config["loss"]["beta_kl"] == pytest.approx(0.01)


def test_v_additive_checkpoint_safely_warm_starts_adaln(tmp_path):
    tokenizer = PokemonTokenizer().load_tokens({"foo": 1, "bar": 2})
    source_model = _tiny_model(
        decoder_conditioning="additive",
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    target_model = _tiny_model(
        decoder_conditioning="adaln",
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    v_base = {
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "max_seq_len": 16,
        "max_state_tokens": 8,
        "gradient_checkpointing": False,
    }
    # Absence of decoder_conditioning is the legacy additive checkpoint form.
    source_config = {"latent_dim": 8, "v": dict(v_base)}
    target_config = {
        "latent_dim": 8,
        "v": {**v_base, "decoder_conditioning": "adaln"},
    }
    checkpoint_path = tmp_path / "legacy_v.pt"
    torch.save({
        "model_version": MODEL_VERSION,
        "stage": "v",
        "model_state_dict": source_model.state_dict(),
        "model_config": source_config,
        "tokenizer_state": tokenizer.to_state(),
        "dataset_manifest_hash": "dataset-hash",
        "vocab_size": len(tokenizer),
        "pad_id": tokenizer.pad_token_id,
    }, checkpoint_path)
    args = SimpleNamespace(warm_start_checkpoint=str(checkpoint_path))

    loaded = simple_world_model_train._warm_start_v(
        args,
        model=target_model,
        tokenizer=tokenizer,
        source_hash="dataset-hash",
        model_config=target_config,
        device=torch.device("cpu"),
    )
    assert loaded is not None
    z = torch.randn(2, 8)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    torch.testing.assert_close(
        target_model.v.decode(z, valid),
        source_model.v.decode(z, valid),
        atol=0.0,
        rtol=0.0,
    )

    incompatible_target = _tiny_model(
        decoder_conditioning="adaln",
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    with pytest.raises(ValueError, match="dataset_manifest_hash"):
        simple_world_model_train._warm_start_v(
            args,
            model=incompatible_target,
            tokenizer=tokenizer,
            source_hash="different-dataset",
            model_config=target_config,
            device=torch.device("cpu"),
        )


def test_v_checkpoint_safely_warm_starts_raw_header_cross_attention(tmp_path):
    tokenizer = PokemonTokenizer().load_tokens({"foo": 1, "bar": 2})
    source_model = _tiny_model(
        decoder_conditioning="adaln",
        fixed_posterior_std=0.25,
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    target_model = _tiny_model(
        decoder_conditioning="adaln",
        decoder_header_conditioning="cross_attention",
        fixed_posterior_std=0.25,
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    v_base = {
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "max_seq_len": 16,
        "max_state_tokens": 8,
        "gradient_checkpointing": False,
        "decoder_conditioning": "adaln",
        "fixed_posterior_std": 0.25,
    }
    source_config = {"latent_dim": 8, "v": dict(v_base)}
    target_config = {
        "latent_dim": 8,
        "v": {**v_base, "decoder_header_conditioning": "cross_attention"},
    }
    checkpoint_path = tmp_path / "no_header_decoder_v.pt"
    torch.save({
        "model_version": MODEL_VERSION,
        "stage": "v",
        "model_state_dict": source_model.state_dict(),
        "model_config": source_config,
        "tokenizer_state": tokenizer.to_state(),
        "dataset_manifest_hash": "dataset-hash",
        "vocab_size": len(tokenizer),
        "pad_id": tokenizer.pad_token_id,
    }, checkpoint_path)
    args = SimpleNamespace(warm_start_checkpoint=str(checkpoint_path))
    loaded = simple_world_model_train._warm_start_v(
        args,
        model=target_model,
        tokenizer=tokenizer,
        source_hash="dataset-hash",
        model_config=target_config,
        device=torch.device("cpu"),
    )
    assert loaded is not None
    assert target_model.v.decoder_header_gate is not None
    assert torch.count_nonzero(target_model.v.decoder_header_gate) == 0

    header = torch.tensor([[1, 2], [2, 1]])
    state = torch.tensor([[1, 2, 0], [2, 1, 2]])
    header_mask = torch.tensor([[True, True], [True, True]])
    state_mask = torch.tensor([[True, True, False], [True, True, True]])
    source_outputs = source_model.encode_state(
        header,
        state,
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
        deterministic=True,
    )
    target_outputs = target_model.encode_state(
        header,
        state,
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
        deterministic=True,
    )
    torch.testing.assert_close(target_outputs["mu"], source_outputs["mu"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        target_outputs["logits"], source_outputs["logits"], atol=0.0, rtol=0.0,
    )


def test_v_warm_start_allows_fixed_posterior_std_and_preserves_deterministic_logits(
    tmp_path,
):
    tokenizer = PokemonTokenizer().load_tokens({"foo": 1, "bar": 2})
    source_model = _tiny_model(
        decoder_conditioning="additive",
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    target_model = _tiny_model(
        decoder_conditioning="additive",
        fixed_posterior_std=0.25,
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
    )
    v_base = {
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "d_ff": 32,
        "dropout": 0.0,
        "max_seq_len": 16,
        "max_state_tokens": 8,
        "gradient_checkpointing": False,
    }
    source_config = {"latent_dim": 8, "v": dict(v_base)}
    target_config = {
        "latent_dim": 8,
        "v": {**v_base, "fixed_posterior_std": 0.25},
    }
    checkpoint = {
        "model_version": MODEL_VERSION,
        "stage": "v",
        "model_state_dict": source_model.state_dict(),
        "model_config": source_config,
        "tokenizer_state": tokenizer.to_state(),
        "dataset_manifest_hash": "dataset-hash",
        "vocab_size": len(tokenizer),
        "pad_id": tokenizer.pad_token_id,
    }
    checkpoint_path = tmp_path / "learned_std_v.pt"
    torch.save(checkpoint, checkpoint_path)
    args = SimpleNamespace(warm_start_checkpoint=str(checkpoint_path))
    simple_world_model_train._warm_start_v(
        args,
        model=target_model,
        tokenizer=tokenizer,
        source_hash="dataset-hash",
        model_config=target_config,
        device=torch.device("cpu"),
    )

    header = torch.tensor([[1, 2], [2, 1]])
    state = torch.tensor([[1, 2, 0], [2, 1, 2]])
    header_mask = header.ne(0)
    state_mask = state.ne(0)
    source_outputs = source_model.encode_state(
        header,
        state,
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
        deterministic=True,
    )
    target_outputs = target_model.encode_state(
        header,
        state,
        header_valid_mask=header_mask,
        state_valid_mask=state_mask,
        deterministic=True,
    )
    torch.testing.assert_close(
        target_outputs["mu"], source_outputs["mu"], atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        target_outputs["logits"], source_outputs["logits"], atol=0.0, rtol=0.0,
    )
    torch.testing.assert_close(
        target_outputs["logvar"],
        torch.full_like(target_outputs["logvar"], 2.0 * math.log(0.25)),
        atol=0.0,
        rtol=0.0,
    )

    incompatible_checkpoint = dict(checkpoint)
    incompatible_checkpoint["model_config"] = {
        "latent_dim": 8,
        "v": {**v_base, "fixed_posterior_std": 0.5},
    }
    incompatible_path = tmp_path / "different_fixed_std_v.pt"
    torch.save(incompatible_checkpoint, incompatible_path)
    incompatible_args = SimpleNamespace(warm_start_checkpoint=str(incompatible_path))
    with pytest.raises(ValueError, match="fixed_posterior_std"):
        simple_world_model_train._warm_start_v(
            incompatible_args,
            model=_tiny_model(
                fixed_posterior_std=0.25,
                vocab_size=len(tokenizer),
                pad_id=tokenizer.pad_token_id,
            ),
            tokenizer=tokenizer,
            source_hash="dataset-hash",
            model_config=target_config,
            device=torch.device("cpu"),
        )


def test_v_resume_and_warm_start_are_mutually_exclusive():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--stage", "v", "--data_root", "data", "--formats", "gen1ou",
            "--tokenizer_path", "tokenizer.json",
            "--resume_checkpoint", "latest.pt",
            "--warm_start_checkpoint", "legacy.pt",
        ])


def test_v_team_aggregate_sigreg_cli_values_are_explicit():
    args = build_arg_parser().parse_args([
        "--stage", "v", "--data_root", "data", "--formats", "gen1ou",
        "--tokenizer_path", "tokenizer.json",
        "--lambda_team_aggregate_sigreg", "0.002",
        "--team_aggregate_sigreg_warmup_updates", "500",
        "--team_sigreg_batch_size", "32",
    ])
    assert args.lambda_team_aggregate_sigreg == pytest.approx(0.002)
    assert args.team_aggregate_sigreg_warmup_updates == 500
    assert args.team_sigreg_batch_size == 32


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


def test_v_pure_mean_reconstruction_decodes_once(monkeypatch):
    torch.manual_seed(27)
    model = _tiny_model().train()
    batch = {
        "header_tokens": torch.tensor([[1, 2], [2, 3]]),
        "header_mask": torch.tensor([[True, True], [True, True]]),
        "state_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "state_mask": torch.tensor([[True, True, True], [True, True, True]]),
    }
    args = SimpleNamespace(
        encoder_token_mask_prob=0.0, mean_recon_weight=1.0,
        free_bits=0.0, kl_capacity=0.0, kl_capacity_weight=0.0,
        lambda_sampled_sigreg=0.0,
    )
    original_decode = model.v.decode
    calls = 0

    def counted_decode(*decode_args, **decode_kwargs):
        nonlocal calls
        calls += 1
        return original_decode(*decode_args, **decode_kwargs)

    monkeypatch.setattr(model.v, "decode", counted_decode)
    loss, metrics, outputs = simple_world_model_train._run_v_batch(
        model, batch, beta=0.0, args=args,
    )
    expected, expected_metrics = vae_losses(
        outputs, batch["state_tokens"], beta_kl=0.0, free_bits=0.0,
    )
    assert calls == 1
    torch.testing.assert_close(outputs["z"], outputs["mu"])
    torch.testing.assert_close(loss, expected)
    assert metrics["objective_recon_ce"] == pytest.approx(
        expected_metrics["recon_ce"]
    )
    assert metrics["mean_recon_weight"] == 1.0


def test_v_mu_sigreg_regularizes_deployed_code_and_respects_warmup(monkeypatch):
    torch.manual_seed(29)
    model = _tiny_model().train()
    batch = {
        "header_tokens": torch.tensor([[1, 2], [2, 3]]),
        "header_mask": torch.tensor([[True, True], [True, True]]),
        "state_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "state_mask": torch.tensor([[True, True, True], [True, True, True]]),
    }
    args = SimpleNamespace(
        encoder_token_mask_prob=0.0, mean_recon_weight=0.0,
        free_bits=0.02, kl_capacity=0.0, kl_capacity_weight=0.0,
        lambda_mu_sigreg=0.4, sigreg_num_slices=8,
        sigreg_num_points=5, sigreg_domain=2.0,
    )
    monkeypatch.setattr(
        simple_world_model_train, "sigreg",
        lambda mu, **_: mu.float().square().mean(),
    )

    torch.manual_seed(31)
    base_values = vars(args).copy()
    base_values["lambda_mu_sigreg"] = 0.0
    base_args = SimpleNamespace(**base_values)
    base_loss, _, _ = simple_world_model_train._run_v_batch(
        model, batch, beta=0.01, args=base_args,
    )
    torch.manual_seed(31)
    loss, metrics, outputs = simple_world_model_train._run_v_batch(
        model, batch, beta=0.01, args=args, mu_sigreg_scale=0.25,
    )

    expected_prior = outputs["mu"].float().square().mean()
    expected_weight = 0.4 * 0.25
    torch.testing.assert_close(loss, base_loss + expected_weight * expected_prior)
    assert metrics["mu_sigreg_loss"] == pytest.approx(float(expected_prior.detach()))
    assert metrics["mu_sigreg_weight"] == pytest.approx(expected_weight)
    assert metrics["mu_sigreg_weighted"] == pytest.approx(
        float((expected_weight * expected_prior).detach())
    )


def test_v_sampled_sigreg_regularizes_reparameterized_posterior(monkeypatch):
    torch.manual_seed(37)
    model = _tiny_model().train()
    batch = {
        "header_tokens": torch.tensor([[1, 2], [2, 3]]),
        "header_mask": torch.tensor([[True, True], [True, True]]),
        "state_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "state_mask": torch.tensor([[True, True, True], [True, True, True]]),
    }
    args = SimpleNamespace(
        encoder_token_mask_prob=0.0, mean_recon_weight=0.0,
        free_bits=0.02, kl_capacity=0.0, kl_capacity_weight=0.0,
        lambda_mu_sigreg=0.0, lambda_sampled_sigreg=0.2,
        sigreg_num_slices=8, sigreg_num_points=5, sigreg_domain=2.0,
    )
    monkeypatch.setattr(
        simple_world_model_train, "sigreg",
        lambda latent, **_: latent.float().square().mean(),
    )
    base_values = vars(args).copy()
    base_values["lambda_sampled_sigreg"] = 0.0

    torch.manual_seed(41)
    base_loss, _, _ = simple_world_model_train._run_v_batch(
        model, batch, beta=0.01, args=SimpleNamespace(**base_values),
    )
    torch.manual_seed(41)
    loss, metrics, outputs = simple_world_model_train._run_v_batch(
        model, batch, beta=0.01, args=args, sampled_sigreg_scale=0.5,
    )

    expected_prior = outputs["z"].float().square().mean()
    expected_weight = 0.1
    torch.testing.assert_close(loss, base_loss + expected_weight * expected_prior)
    assert metrics["sampled_sigreg_loss"] == pytest.approx(float(expected_prior.detach()))
    assert metrics["sampled_sigreg_weight"] == pytest.approx(expected_weight)
    assert metrics["sampled_sigreg_weighted"] == pytest.approx(
        float((expected_weight * expected_prior).detach())
    )


def test_team_aggregate_sigreg_is_separate_balanced_and_backpropagates(monkeypatch):
    torch.manual_seed(39)
    model = _tiny_model().train()
    batch = {
        "header_tokens": torch.tensor([
            [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
        ]),
        "header_mask": torch.ones(6, 2, dtype=torch.bool),
        "state_tokens": torch.tensor([
            [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24],
        ]),
        "state_mask": torch.ones(6, 2, dtype=torch.bool),
        "formats": ["gen1ou", "gen9ou", "gen1ou", "gen9ou", "gen1ou", "gen9ou"],
    }
    team_batch = {
        "header_tokens": torch.tensor([[25, 26], [27, 28], [29, 30], [31, 32]]),
        "header_mask": torch.ones(4, 2, dtype=torch.bool),
        "state_tokens": torch.ones(4, 1, dtype=torch.long),
        "state_mask": torch.ones(4, 1, dtype=torch.bool),
        "formats": ["gen1ou", "gen9ou", "gen1ou", "gen9ou"],
    }
    values = dict(
        encoder_token_mask_prob=0.0, mean_recon_weight=0.0,
        free_bits=0.0, kl_capacity=0.0, kl_capacity_weight=0.0,
        lambda_mu_sigreg=0.0, lambda_sampled_sigreg=0.0,
        lambda_aggregate_sigreg=0.2, lambda_team_aggregate_sigreg=0.4,
        team_sigreg_batch_size=4,
        sigreg_num_slices=8, sigreg_num_points=5, sigreg_domain=2.0,
        posterior_std_target=None, posterior_std_weight=0.0,
    )
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_aggregate(mu, logvar, **_):
        mu.retain_grad()
        captured.append((mu, logvar))
        return mu.float().square().mean()

    monkeypatch.setattr(
        simple_world_model_train, "aggregate_posterior_sigreg", fake_aggregate,
    )
    base_values = {
        **values,
        "lambda_aggregate_sigreg": 0.0,
        "lambda_team_aggregate_sigreg": 0.0,
    }
    torch.manual_seed(43)
    base_loss, _, _ = simple_world_model_train._run_v_batch(
        model, batch, beta=0.0, args=SimpleNamespace(**base_values),
    )
    torch.manual_seed(43)
    loss, metrics, _ = simple_world_model_train._run_v_batch(
        model,
        batch,
        beta=0.0,
        args=SimpleNamespace(**values),
        team_batch=team_batch,
        aggregate_sigreg_scale=0.5,
        team_aggregate_sigreg_scale=0.5,
    )

    assert len(captured) == 2
    state_mu, _ = captured[0]
    team_mu, _ = captured[1]
    assert state_mu.shape == (6, 8)
    assert team_mu.shape == (4, 8)
    model.eval()
    with torch.no_grad():
        expected_team_mu, _ = model.v.encode(
            team_batch["header_tokens"],
            header_valid_mask=team_batch["header_mask"],
        )
    model.train()
    torch.testing.assert_close(team_mu.detach(), expected_team_mu)
    expected_state = state_mu.float().square().mean()
    expected_team = team_mu.float().square().mean()
    torch.testing.assert_close(
        loss,
        base_loss + 0.1 * expected_state + 0.2 * expected_team,
    )
    assert metrics["team_aggregate_sigreg_loss"] == pytest.approx(
        float(expected_team.detach())
    )
    assert metrics["team_aggregate_sigreg_weight"] == pytest.approx(0.2)
    assert metrics["team_aggregate_sigreg_weighted"] == pytest.approx(
        float((0.2 * expected_team).detach())
    )
    assert metrics["team_sigreg_examples"] == 4.0
    loss.backward()
    assert state_mu.grad is not None and torch.isfinite(state_mu.grad).all()
    assert team_mu.grad is not None and torch.isfinite(team_mu.grad).all()
    assert bool(team_mu.grad.abs().gt(0).any())


def test_team_sigreg_row_subset_is_capped_and_format_balanced():
    formats = ["gen9ou"] * 5 + ["gen1ou"] * 4 + ["gen2ou"]
    rows = simple_world_model_train._balanced_format_row_indices(
        formats, 7, device=torch.device("cpu"),
    )
    selected = [formats[int(row)] for row in rows]
    counts = {fmt: selected.count(fmt) for fmt in set(selected)}
    assert len(selected) == 7
    assert set(selected) == {"gen1ou", "gen2ou", "gen9ou"}
    assert max(counts.values()) - min(counts.values()) <= 2


def test_header_only_training_encode_uses_checkpointable_backward():
    torch.manual_seed(45)
    vae = StateVAE(
        vocab_size=32, pad_id=0, latent_dim=8, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, max_seq_len=16, max_state_tokens=8,
        gradient_checkpointing=True,
    ).train()
    header = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mu, logvar = vae.encode(header, header_valid_mask=header.ne(0))
    loss = mu.square().mean() + 0.01 * logvar.square().mean()
    loss.backward()
    grad = vae.token_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert bool(grad.abs().gt(0).any())


def test_aggregate_posterior_sigreg_integrates_diagonal_gaussians_and_backpropagates():
    directions = torch.eye(4)
    mu = torch.zeros(8, 4, requires_grad=True)
    logvar = torch.zeros(8, 4, requires_grad=True)
    exact_prior = aggregate_posterior_sigreg(
        mu, logvar, num_slices=4, num_points=17, domain=3.0,
        directions=directions,
    )
    assert float(exact_prior.detach()) == pytest.approx(0.0, abs=1e-12)

    shifted_mu = torch.full((8, 4), 0.75, requires_grad=True)
    narrow_logvar = torch.full((8, 4), -2.0, requires_grad=True)
    mismatch = aggregate_posterior_sigreg(
        shifted_mu, narrow_logvar, num_slices=4, num_points=17, domain=3.0,
        directions=directions,
    )
    assert float(mismatch.detach()) > 0.0
    mismatch.backward()
    assert shifted_mu.grad is not None and torch.isfinite(shifted_mu.grad).all()
    assert narrow_logvar.grad is not None and torch.isfinite(narrow_logvar.grad).all()


def test_v_posterior_std_target_adds_bounded_variance_transfer_loss():
    torch.manual_seed(43)
    model = _tiny_model().train()
    batch = {
        "header_tokens": torch.tensor([[1, 2], [2, 3]]),
        "header_mask": torch.tensor([[True, True], [True, True]]),
        "state_tokens": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "state_mask": torch.tensor([[True, True, True], [True, True, True]]),
    }
    values = dict(
        encoder_token_mask_prob=0.0, mean_recon_weight=1.0,
        free_bits=0.0, kl_capacity=0.0, kl_capacity_weight=0.0,
        lambda_mu_sigreg=0.0, lambda_sampled_sigreg=0.0,
        lambda_aggregate_sigreg=0.0, posterior_std_target=0.1,
        posterior_std_weight=0.4,
    )
    base_values = {**values, "posterior_std_weight": 0.0}
    torch.manual_seed(47)
    base_loss, _, _ = simple_world_model_train._run_v_batch(
        model, batch, beta=0.0, args=SimpleNamespace(**base_values),
    )
    torch.manual_seed(47)
    loss, metrics, outputs = simple_world_model_train._run_v_batch(
        model, batch, beta=0.0, args=SimpleNamespace(**values), posterior_std_scale=0.25,
    )
    expected = (outputs["logvar"].float().mul(0.5).exp() - 0.1).square().mean()
    effective_weight = 0.1
    torch.testing.assert_close(loss, base_loss + effective_weight * expected)
    assert metrics["posterior_std_target"] == pytest.approx(0.1)
    assert metrics["posterior_std_target_loss"] == pytest.approx(float(expected.detach()))
    assert metrics["posterior_std_target_weight"] == pytest.approx(effective_weight)


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
    assert first["aggregate_mu_mean_rms"] >= 0.0
    assert first["aggregate_mu_std_min"] > 0.0
    assert first["aggregate_mu_std_max"] >= first["aggregate_mu_std_min"]
    assert first["aggregate_mu_cov_offdiag_rms"] >= 0.0
    assert first["aggregate_mu_gaussian_kl_per_dim"] >= 0.0
    assert first["aggregate_gaussian_kl_per_dim"] == pytest.approx(
        second["aggregate_gaussian_kl_per_dim"]
    )
    assert first["aggregate_mu_gaussian_kl_per_dim"] == pytest.approx(
        second["aggregate_mu_gaussian_kl_per_dim"]
    )
    assert first["aggregate_sigreg"] >= 0.0
    assert first["aggregate_mu_sigreg"] >= 0.0
    assert first["aggregate_sigreg"] == pytest.approx(second["aggregate_sigreg"])
    assert first["aggregate_mu_sigreg"] == pytest.approx(second["aggregate_mu_sigreg"])
    assert first["state_aggregate_mean_rms"] == pytest.approx(first["aggregate_mean_rms"])
    assert first["state_aggregate_mu_std_mean"] == pytest.approx(first["aggregate_mu_std_mean"])
    assert first["state_aggregate_sigreg"] == pytest.approx(first["aggregate_sigreg"])
    assert first["state_aggregate_mu_sigreg"] == pytest.approx(first["aggregate_mu_sigreg"])
    assert first["team_aggregate_std_min"] > 0.0
    assert first["team_aggregate_sigreg"] >= 0.0
    assert first["team_aggregate_mu_sigreg"] >= 0.0
    assert first["team_aggregate_sigreg"] == pytest.approx(second["team_aggregate_sigreg"])
    assert first["team_aggregate_mu_sigreg"] == pytest.approx(
        second["team_aggregate_mu_sigreg"]
    )
    assert model.training


def test_v_validation_keeps_state_and_team_distribution_audits_separate(monkeypatch):
    tokenizer = PokemonTokenizer().load_tokens({"foo": 1})
    vocab_size = len(tokenizer) + 1

    class FakeV(torch.nn.Module):
        def encode(self, header_tokens, *, header_valid_mask):
            del header_valid_mask
            batch = header_tokens.shape[0]
            return (
                torch.full((batch, 4), 0.75, device=header_tokens.device),
                torch.full((batch, 4), -2.0, device=header_tokens.device),
            )

        def decode(self, z, valid_token_mask):
            logits = torch.zeros(*valid_token_mask.shape, vocab_size, device=z.device)
            logits[..., 1] = 1.0
            return logits

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v = FakeV()

        def encode_state(
            self, header_tokens, state_tokens, *, header_valid_mask,
            state_valid_mask, deterministic,
        ):
            del header_tokens, header_valid_mask, deterministic
            mu = torch.zeros(state_tokens.shape[0], 4, device=state_tokens.device)
            logvar = torch.zeros_like(mu)
            return {
                "logits": self.v.decode(mu, state_valid_mask),
                "mu": mu,
                "logvar": logvar,
                "z": mu,
                "state_valid_mask": state_valid_mask,
            }

    batch = {
        "header_tokens": torch.tensor([[1, 1], [1, 1]]),
        "header_mask": torch.ones(2, 2, dtype=torch.bool),
        "state_tokens": torch.tensor([[1], [1]]),
        "state_mask": torch.ones(2, 1, dtype=torch.bool),
        "formats": ["gen1ou", "gen1ou"],
    }
    args = SimpleNamespace(
        beta_kl=0.01, free_bits=0.0, kl_capacity=0.0,
        kl_capacity_weight=0.0, val_mc_samples=2, seed=13,
        sigreg_num_slices=4, sigreg_num_points=9, sigreg_domain=3.0,
    )
    monkeypatch.setattr(simple_world_model_train, "_v_format_metrics", lambda outputs, row: {})
    monkeypatch.setattr(
        simple_world_model_train, "_v_token_category_metrics",
        lambda outputs, row, selected_tokenizer: {},
    )
    model = FakeModel()

    first = simple_world_model_train._validate_v(
        model, [batch], torch.device("cpu"), args, tokenizer,
    )
    second = simple_world_model_train._validate_v(
        model, [batch], torch.device("cpu"), args, tokenizer,
    )

    assert first["state_aggregate_gaussian_kl"] == pytest.approx(0.0, abs=1e-12)
    assert first["state_aggregate_sigreg"] == pytest.approx(0.0, abs=1e-12)
    assert first["aggregate_sigreg"] == pytest.approx(first["state_aggregate_sigreg"])
    assert first["team_aggregate_gaussian_kl"] > 0.0
    assert first["team_aggregate_sigreg"] > 0.0
    assert first["team_aggregate_sigreg"] != pytest.approx(first["state_aggregate_sigreg"])
    for name in (
        "state_aggregate_sigreg", "state_aggregate_mu_sigreg",
        "team_aggregate_sigreg", "team_aggregate_mu_sigreg",
        "state_aggregate_gaussian_kl", "team_aggregate_gaussian_kl",
    ):
        assert first[name] == pytest.approx(second[name])
    assert first["selection_score"] == pytest.approx(first["recon_ce"])
    assert model.training


def test_v_validation_token_weights_unequal_batches_and_reports_mu_moments():
    tokenizer = PokemonTokenizer().load_tokens({"foo": 1, "bar": 2})
    vocab_size = len(tokenizer) + 1

    class FakeV(torch.nn.Module):
        def encode(self, header_tokens, *, header_valid_mask):
            del header_valid_mask
            mu = header_tokens[:, :2].float()
            return mu, torch.zeros_like(mu)

        def decode(self, z, valid_token_mask):
            logits = torch.zeros(*valid_token_mask.shape, vocab_size, device=z.device)
            logits[..., 1] = 4.0
            return logits

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v = FakeV()

        def encode_state(
            self, header_tokens, state_tokens, *, header_valid_mask,
            state_valid_mask, deterministic,
        ):
            del header_valid_mask, deterministic
            mu = header_tokens[:, :2].float()
            logvar = torch.zeros_like(mu)
            return {
                "logits": self.v.decode(mu, state_valid_mask),
                "mu": mu,
                "logvar": logvar,
                "z": mu,
                "state_valid_mask": state_valid_mask,
            }

    batches = [
        {
            "header_tokens": torch.tensor([[1, 0]]),
            "header_mask": torch.tensor([[True, True]]),
            "state_tokens": torch.tensor([[1]]),
            "state_mask": torch.tensor([[True]]),
            "formats": ["gen1ou"],
        },
        {
            "header_tokens": torch.tensor([[3, 0], [5, 0]]),
            "header_mask": torch.tensor([[True, True], [True, True]]),
            "state_tokens": torch.tensor([[2, 2, 2], [2, 1, 1]]),
            "state_mask": torch.tensor([[True, True, True], [True, False, False]]),
            "formats": ["gen1ou", "gen1ou"],
        },
    ]
    args = SimpleNamespace(
        beta_kl=0.01, free_bits=0.02, kl_capacity=0.0,
        kl_capacity_weight=0.0, val_mc_samples=2, seed=7,
    )

    result = simple_world_model_train._validate_v(
        FakeModel(), batches, torch.device("cpu"), args, tokenizer,
    )

    logits = torch.zeros(2, vocab_size)
    logits[:, 1] = 4.0
    correct_ce = float(torch.nn.functional.cross_entropy(logits[:1], torch.tensor([1])))
    wrong_ce = float(torch.nn.functional.cross_entropy(logits[1:], torch.tensor([2])))
    expected_ce = (correct_ce + 4.0 * wrong_ce) / 5.0
    assert result["recon_ce"] == pytest.approx(expected_ce)
    assert result["recon_token_acc"] == pytest.approx(1.0 / 5.0)
    assert result["recon_ce_mc"] == pytest.approx(expected_ce)
    assert result["recon_token_acc_mc"] == pytest.approx(1.0 / 5.0)
    assert result["recon_ce_mc_std"] == pytest.approx(0.0)
    assert result["recon_token_acc_mc_std"] == pytest.approx(0.0)
    # Per-format and content-category metrics use the same exact token totals.
    assert result["v_gen1ou_token_ce"] == pytest.approx(expected_ce)
    assert result["v_gen1ou_token_acc"] == pytest.approx(1.0 / 5.0)
    assert result["v_species_move_token_ce"] == pytest.approx(expected_ce)
    assert result["v_species_move_token_acc"] == pytest.approx(1.0 / 5.0)

    # mu rows are [1, 0], [3, 0], [5, 0]: population mean [3, 0]
    # and population variance [8/3, 0].
    assert result["aggregate_mu_mean_rms"] == pytest.approx((4.5) ** 0.5)
    assert result["aggregate_mu_std_max"] == pytest.approx((8.0 / 3.0) ** 0.5)
    assert result["aggregate_mu_std_min"] == pytest.approx(1e-6)
    assert result["aggregate_mu_cov_offdiag_rms"] == pytest.approx(0.0)
    assert math.isinf(result["aggregate_mu_gaussian_kl_per_dim"])
    # Existing aggregate metrics still describe sampled q(z): unit posterior
    # variance is added to each dimension, unlike the new mu-only summary.
    assert result["aggregate_std_max"] == pytest.approx((11.0 / 3.0) ** 0.5)
    assert result["aggregate_std_min"] == pytest.approx(1.0)


def test_aggregate_gaussian_metrics_identity_is_zero_and_singular_is_infinite():
    identity = torch.eye(3, dtype=torch.float64)
    zero = torch.zeros(3, dtype=torch.float64)
    metrics = simple_world_model_train._aggregate_gaussian_metrics(
        zero, identity * 10.0, 10, prefix="probe",
    )
    assert metrics["probe_gaussian_kl"] == pytest.approx(0.0, abs=1e-12)

    singular = identity.clone()
    singular[-1, -1] = 0.0
    metrics = simple_world_model_train._aggregate_gaussian_metrics(
        zero, singular * 10.0, 10, prefix="probe",
    )
    assert math.isinf(metrics["probe_gaussian_kl"])


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
