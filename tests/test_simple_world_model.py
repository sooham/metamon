import pytest
import torch
import yaml

from scripts.generate_world_model_data import PairedBattle, PairedShardAccumulator, TokenizedPOV, _contiguous_rollout_windows, _paired_transition_rows
from metamon.simple_world_model.train import _prepare_batch, build_arg_parser, train
from metamon.simple_world_model.checkpointing import (
    resume_training_state,
    save_simple_world_model_checkpoint,
)
from metamon.simple_world_model.model import (
    SimpleWorldModel,
    _latent_diagnostics,
    compute_simple_world_model_losses,
)
from metamon.tokenizer import PokemonTokenizer


def _tiny_model() -> SimpleWorldModel:
    return SimpleWorldModel(
        vocab_size=32,
        pad_id=0,
        latent_dim=8,
        v_cfg={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "max_seq_len": 6,
            "gradient_checkpointing": False,
        },
        action_encoder_cfg={
            "action_dim": 8,
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "max_seq_len": 3,
            "gradient_checkpointing": False,
        },
        m_cfg={
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "num_mixtures": 3,
            "gradient_checkpointing": False,
        },
        controller_cfg={
            "hidden_dim": 16,
            "dropout": 0.0,
        },
    )


def _tokens() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.tensor([
        [1, 2, 3, 4, 0, 0],
        [1, 5, 6, 7, 8, 0],
    ], dtype=torch.long)
    next_state = torch.tensor([
        [1, 2, 9, 4, 0, 0],
        [1, 5, 6, 10, 8, 0],
    ], dtype=torch.long)
    action = torch.tensor([
        [11, 12, 0],
        [13, 14, 0],
    ], dtype=torch.long)
    return state, next_state, action


def test_vm_forward_and_loss_are_finite():
    model = _tiny_model()
    state, next_state, action = _tokens()
    outputs = model.forward_vm(state, next_state, action)

    assert outputs["state_logits"].shape == (2, 6, 33)
    assert outputs["z_mu"].shape == (2, 8)
    assert outputs["mixture_means"].shape == (2, 3, 8)
    assert outputs["terminal_logits"].shape == (2, 6)

    loss, metrics = compute_simple_world_model_losses(
        outputs,
        {
            "state_tokens": state,
            "terminal_class": torch.tensor([0, 1], dtype=torch.long),
        },
        pad_id=0,
        components="vm",
    )

    assert torch.isfinite(loss)
    assert metrics["mdn_nll"] == pytest.approx(metrics["mdn_nll"])
    assert 0.0 <= metrics["terminal_acc"] <= 1.0


def test_mdn_temperature_sampling_shape_is_stable():
    model = _tiny_model()
    state, next_state, action = _tokens()
    outputs = model.forward_vm(state, next_state, action)

    for tau in (0.5, 1.0, 1.5):
        sample = model.m.sample_next_z(
            outputs["mixture_logits"],
            outputs["mixture_means"],
            outputs["mixture_log_scales"],
            tau=tau,
        )
        assert sample.shape == (2, 8)
        assert torch.isfinite(sample).all()


def test_controller_masks_legal_actions_and_computes_loss():
    model = _tiny_model()
    state, _, _ = _tokens()
    legal = torch.tensor([
        [[11, 12, 0], [15, 16, 0], [0, 0, 0]],
        [[13, 14, 0], [17, 18, 0], [19, 20, 0]],
    ], dtype=torch.long)
    mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    chosen = torch.tensor([1, 2], dtype=torch.long)
    outputs = model.forward_controller(state, legal)

    loss, metrics = compute_simple_world_model_losses(
        outputs,
        {
            "legal_action_mask": mask,
            "chosen_legal_action_idx": chosen,
        },
        pad_id=0,
        components="c",
    )

    assert outputs["controller_logits"].shape == (2, 3)
    assert torch.isfinite(loss)
    assert metrics["legal_action_count"] == pytest.approx(2.5)


def test_latent_diagnostics_detect_collapsed_zero_latents():
    metrics = _latent_diagnostics("z", torch.zeros(8, 4))

    assert metrics["z_std_per_dim"] == 0.0
    assert metrics["z_pairwise_distance"] == 0.0
    assert metrics["z_active_dims"] == 0.0


def test_prepare_batch_concatenates_team_header_with_every_state():
    batch = {
        "p1_history_T": torch.tensor([[[[21, 22, 0], [99, 0, 0]]]], dtype=torch.int32),
        "p1_target_state_T": torch.tensor([[[31, 32, 0]]], dtype=torch.int32),
        "p1_next_state_T1": torch.tensor([[[41, 42, 43]]], dtype=torch.int32),
        "p1_action": torch.tensor([[[51, 0]]], dtype=torch.int32),
        "p1_next_terminal_class": torch.tensor([[0]], dtype=torch.long),
    }

    prepared = _prepare_batch(batch, pad_id=0)

    assert torch.equal(prepared["state_tokens"], torch.tensor([[21, 22, 31, 32]], dtype=torch.int32))
    assert torch.equal(prepared["next_state_tokens"], torch.tensor([[21, 22, 41, 42, 43]], dtype=torch.int32))


def test_strict_resume_restores_optimizer_and_rejects_config_changes(tmp_path):
    model_cfg = {
        "latent_dim": 8,
        "v": {"d_model": 16},
        "action_encoder": {"action_dim": 8},
        "m": {"d_model": 16},
        "controller": {"hidden_dim": 16},
    }
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens({"foo": 1})
    path = tmp_path / "ckpt.pt"

    save_simple_world_model_checkpoint(
        str(path),
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=7,
        model_config=model_cfg,
        vocab_size=32,
        pad_id=0,
        tokenizer=tokenizer,
        components="vm",
        max_history_blocks=1,
    )

    restored_model = _tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    ckpt = resume_training_state(
        model=restored_model,
        optimizer=restored_optimizer,
        checkpoint_path=str(path),
        device=torch.device("cpu"),
        model_config=model_cfg,
        vocab_size=32,
        pad_id=0,
    )
    assert ckpt["global_step"] == 7

    changed_cfg = dict(model_cfg)
    changed_cfg["latent_dim"] = 16
    with pytest.raises(ValueError, match="signature"):
        resume_training_state(
            model=restored_model,
            optimizer=restored_optimizer,
            checkpoint_path=str(path),
            device=torch.device("cpu"),
            model_config=changed_cfg,
            vocab_size=32,
            pad_id=0,
        )


def _write_tiny_training_fixture(tmp_path):
    tokenizer = PokemonTokenizer()
    tokenizer.load_tokens({f"tok{i}": i for i in range(1, 40)})
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save_tokens_to_disk(str(tokenizer_path))

    cfg = {
        "model": {
            "latent_dim": 8,
            "v": {
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
                "gradient_checkpointing": False,
            },
            "action_encoder": {
                "action_dim": 8,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
                "max_seq_len": 3,
                "gradient_checkpointing": False,
            },
            "m": {
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
                "num_mixtures": 2,
                "gradient_checkpointing": False,
            },
            "controller": {
                "hidden_dim": 16,
                "dropout": 0.0,
            },
        },
        "loss": {
            "lambda_recon": 1.0,
            "beta_kl": 0.01,
            "lambda_mdn": 1.0,
            "lambda_terminal": 0.25,
            "lambda_controller_bc": 1.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    p1 = TokenizedPOV(
        state_token_arrays=[
            torch.tensor([1, 2], dtype=torch.int16).numpy(),
            torch.tensor([3, 4, 5], dtype=torch.int16).numpy(),
            torch.tensor([6, 7, 8], dtype=torch.int16).numpy(),
        ],
        player_action_arrays=[torch.tensor([9, 10], dtype=torch.int16).numpy()],
        opponent_action_arrays=[torch.tensor([11, 12], dtype=torch.int16).numpy()],
        turn_numbers=[1, 2],
        path="p1.txt",
        state_terminal_classes=[0, 0, 1],
        legal_action_arrays=[
            [
                torch.tensor([9, 10], dtype=torch.int16).numpy(),
                torch.tensor([13, 14], dtype=torch.int16).numpy(),
            ]
        ],
        chosen_legal_action_idx=[0],
    )
    p2 = TokenizedPOV(
        state_token_arrays=[
            torch.tensor([1, 2], dtype=torch.int16).numpy(),
            torch.tensor([15, 16], dtype=torch.int16).numpy(),
            torch.tensor([17, 18], dtype=torch.int16).numpy(),
        ],
        player_action_arrays=[torch.tensor([11, 12], dtype=torch.int16).numpy()],
        opponent_action_arrays=[torch.tensor([9, 10], dtype=torch.int16).numpy()],
        turn_numbers=[1, 2],
        path="p2.txt",
    )
    rows = _paired_transition_rows(p1, p2)
    windows = _contiguous_rollout_windows(rows, rollout_len=1)
    for split in ("train", "val"):
        split_dir = tmp_path / "data" / split
        split_dir.mkdir(parents=True)
        acc = PairedShardAccumulator(format_names={0: "gen1ou"}, rollout_len=1)
        acc.append(PairedBattle(f"battle-{split}", p1, p2, rows, windows))
        acc.write(str(split_dir), shard_idx=0)
    (tmp_path / "data" / "sequence_stats.json").write_text(
        '{"state_block_len": {"max": 4}, "temporal_sequence_len": {"max": 4}}'
    )
    return tokenizer_path, config_path


def test_train_vm_and_c_smoke_on_tiny_shard(tmp_path):
    tokenizer_path, config_path = _write_tiny_training_fixture(tmp_path)
    save_dir = tmp_path / "checkpoints"
    checkpoint = save_dir / "best.pt"

    args = build_arg_parser().parse_args([
        "--data_root", str(tmp_path / "data"),
        "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path),
        "--config", str(config_path),
        "--save_dir", str(save_dir),
        "--checkpoint", str(checkpoint),
        "--components", "vm",
        "--batch_size", "1",
        "--epochs", "1",
        "--max_steps", "1",
        "--num_workers", "0",
        "--val_interval", "0",
        "--val_max_batches", "0",
        "--print_interval", "0",
        "--no-wandb",
    ])
    train(args)
    assert checkpoint.exists()

    c_args = build_arg_parser().parse_args([
        "--data_root", str(tmp_path / "data"),
        "--formats", "gen1ou",
        "--tokenizer_path", str(tokenizer_path),
        "--config", str(config_path),
        "--save_dir", str(save_dir),
        "--checkpoint", str(checkpoint),
        "--components", "c",
        "--batch_size", "1",
        "--epochs", "1",
        "--max_steps", "1",
        "--num_workers", "0",
        "--val_interval", "0",
        "--val_max_batches", "0",
        "--print_interval", "0",
        "--no-wandb",
    ])
    train(c_args)
