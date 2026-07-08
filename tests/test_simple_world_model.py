import pytest
import torch
import yaml

from scripts.generate_world_model_data import PairedBattle, PairedShardAccumulator, TokenizedPOV, _contiguous_rollout_windows, _paired_transition_rows
from metamon.simple_world_model.train import _count_processed_tokens, _prepare_batch, build_arg_parser, train
from metamon.simple_world_model.checkpointing import (
    resume_training_state,
    save_simple_world_model_checkpoint,
)
from metamon.simple_world_model.model import (
    NUM_TERMINAL_CLASSES,
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


def _vm_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Batch=2. Each row: the full interleaved seen history as one sequence
    # (team header block + any prior states/actions + the current state), with
    # the current state present at the END of the sequence.
    history = torch.tensor([
        [1, 2, 3, 4, 0, 0],
        [1, 5, 6, 7, 8, 0],
    ], dtype=torch.long)
    next_history = torch.tensor([
        [1, 2, 3, 4, 9, 0],
        [1, 5, 6, 7, 8, 10],
    ], dtype=torch.long)
    action = torch.tensor([
        [11, 12, 0],
        [13, 14, 0],
    ], dtype=torch.long)
    return history, next_history, action


def test_vm_forward_and_loss_are_finite():
    model = _tiny_model()
    history, next_history, action = _vm_inputs()
    outputs = model.forward_vm(history, next_history, action)

    # Decoder reconstructs the ENTIRE history sequence.
    assert outputs["state_logits"].shape == (2, history.shape[-1], 33)
    assert outputs["history_tokens"].shape == history.shape
    assert outputs["z_mu"].shape == (2, 8)
    assert outputs["mixture_means"].shape == (2, 3, 8)
    assert outputs["terminal_logits"].shape == (2, NUM_TERMINAL_CLASSES)

    loss, metrics = compute_simple_world_model_losses(
        outputs,
        {
            "history_tokens": history,
            "terminal_class": torch.tensor([0, 1], dtype=torch.long),
        },
        pad_id=0,
        components="vm",
    )

    assert torch.isfinite(loss)
    assert metrics["mdn_nll"] == pytest.approx(metrics["mdn_nll"])
    assert 0.0 <= metrics["terminal_acc"] <= 1.0


def test_vm_decoder_reconstructs_full_history_not_just_current_state():
    model = _tiny_model()
    history, next_history, action = _vm_inputs()
    outputs = model.forward_vm(history, next_history, action)
    # state_logits must span the full history length, proving the decoder
    # reconstructs the entire input sequence (not a single current-state block).
    assert outputs["state_logits"].shape[1] == history.shape[-1]
    loss, metrics = compute_simple_world_model_losses(
        outputs,
        {"history_tokens": history, "terminal_class": torch.tensor([0, 0], dtype=torch.long)},
        pad_id=0, components="vm",
    )
    # recon accuracy is computed over the full history (non-pad tokens).
    assert metrics["recon_token_acc"] >= 0.0
    assert torch.isfinite(loss)


def test_mdn_temperature_sampling_shape_is_stable():
    model = _tiny_model()
    history, next_history, action = _vm_inputs()
    outputs = model.forward_vm(history, next_history, action)

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
    history, next_history, action = _vm_inputs()
    legal = torch.tensor([
        [[11, 12, 0], [15, 16, 0], [0, 0, 0]],
        [[13, 14, 0], [17, 18, 0], [19, 20, 0]],
    ], dtype=torch.long)
    mask = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    chosen = torch.tensor([1, 2], dtype=torch.long)
    outputs = model.forward_controller(history, legal)

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


def _collated_batch() -> dict:
    """A minimal collated batch with one battle, rollout_len=1.

    Layout (one battle):
      p1_history_T blocks: [team_header (21,22), prior_state (99)], valid=[T,T]
      p1_player_hist_T / p1_opponent_hist_T blocks: one action each
      target_state_T (current state) = (31,32)
      next_state_T1 = (41,42,43)
      p1_action = current player action (51)
      actual_p2_action_from_p1_perspective = current opponent action (52,53)
    Interleaved history_T = header(21,22) state(99) p_act(61) o_act(62) state_T(31,32)
    next_history_T       = history_T + p_T(51) o_T(52,53) next(41,42,43)
    """
    return {
        "p1_history_T": torch.tensor([[[[21, 22, 0], [99, 0, 0]]]], dtype=torch.int32),
        "p1_history_T_valid": torch.tensor([[[True, True]]], dtype=torch.bool),
        "p1_player_hist_T": torch.tensor([[[[61, 0, 0]]]], dtype=torch.int32),
        "p1_player_hist_T_valid": torch.tensor([[[True]]], dtype=torch.bool),
        "p1_opponent_hist_T": torch.tensor([[[[62, 0, 0]]]], dtype=torch.int32),
        "p1_opponent_hist_T_valid": torch.tensor([[[True]]], dtype=torch.bool),
        "p1_target_state_T": torch.tensor([[[31, 32, 0]]], dtype=torch.int32),
        "p1_next_state_T1": torch.tensor([[[41, 42, 43]]], dtype=torch.int32),
        "p1_action": torch.tensor([[[51, 0]]], dtype=torch.int32),
        "actual_p2_action_from_p1_perspective": torch.tensor([[[52, 53, 0]]], dtype=torch.int32),
        "p1_next_terminal_class": torch.tensor([[0]], dtype=torch.long),
    }


def test_prepare_batch_builds_interleaved_seen_history():
    prepared = _prepare_batch(_collated_batch(), pad_id=0)
    # history_T = team_header(21,22) || prior_state(99) || p_act(61) || o_act(62) || current_state(31,32)
    expected_hist = torch.tensor([[21, 22, 99, 61, 62, 31, 32]], dtype=torch.long)
    assert torch.equal(prepared["history_tokens"], expected_hist)
    # next_history_T = history_T || p_T(51) || o_T(52,53) || next(41,42,43)
    expected_next = torch.tensor([[21, 22, 99, 61, 62, 31, 32, 51, 52, 53, 41, 42, 43]], dtype=torch.long)
    assert torch.equal(prepared["next_history_tokens"], expected_next)
    # state_tokens is the raw current-state block (padded to its own width)
    assert torch.equal(prepared["state_tokens"], torch.tensor([[31, 32, 0]], dtype=torch.long))
    # current player action is carried through
    assert torch.equal(prepared["action_tokens"], torch.tensor([[51, 0]], dtype=torch.long))


def test_count_processed_tokens_vm_counts_history_encode_decode_next_and_action():
    prepared = _prepare_batch(_collated_batch(), pad_id=0)
    # vm consumes: history_tokens (V encode) + history_tokens AGAIN (V decode,
    # reconstructing the FULL history) + next_history_tokens (V encode, MDN
    # target) + action_tokens (action enc).
    hist_nonpad = 7      # 21,22,99,61,62,31,32
    next_nonpad = 13     # full next_history
    action_nonpad = 1    # 51
    expected_vm = 2 * hist_nonpad + next_nonpad + action_nonpad
    assert _count_processed_tokens(prepared, "vm", pad_id=0) == expected_vm


def test_count_processed_tokens_ignores_legal_actions_in_vm_mode():
    base = _collated_batch()
    prepared = _prepare_batch(base, pad_id=0)
    with_legal = dict(base)
    # Adding legal-action fields must not change the vm count (they are unused
    # in vm and only counted for c/all).
    with_legal["p1_legal_actions"] = torch.tensor([[[[61, 62], [0, 0]]]], dtype=torch.int32)
    with_legal["p1_legal_action_mask"] = torch.tensor([[[True, False]]], dtype=torch.bool)
    with_legal["p1_chosen_legal_action_idx"] = torch.tensor([[0]], dtype=torch.long)
    prepared_with = _prepare_batch(with_legal, pad_id=0)
    assert _count_processed_tokens(prepared_with, "vm", pad_id=0) == _count_processed_tokens(prepared, "vm", pad_id=0)
    # c counts history + legal actions only: 7 nonpad history + 2 nonpad legal
    assert _count_processed_tokens(prepared_with, "c", pad_id=0) == 9


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
