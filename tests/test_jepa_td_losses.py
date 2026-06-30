"""Tests for the active paired JEPA loss contract."""

import pytest
import torch

from metamon.jepa import model as jepa_model
from metamon.jepa.model import compute_paired_losses


def _outputs(
    batch_size: int = 2,
    rollout_len: int = 3,
    latent_dim: int = 5,
    action_dim: int = 4,
) -> dict[str, torch.Tensor]:
    state = torch.zeros(batch_size, rollout_len, latent_dim)
    action = torch.zeros(batch_size, rollout_len, action_dim)
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


def test_loss_ignores_legacy_td_kwargs_and_metrics():
    outputs = _outputs()

    loss, metrics = compute_paired_losses(
        outputs,
        gamma=0.25,
        lambda_rank=100.0,
        lambda_value=100.0,
        lambda_q_value=100.0,
        lambda_policy=100.0,
        lambda_value_teacher=100.0,
        lambda_q_teacher=100.0,
        lambda_sigreg=100.0,
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(0.0)
    assert metrics["sigreg_state_loss"] == pytest.approx(0.0)
    for removed_key in (
        "rank_loss",
        "rank_valid",
        "value_loss",
        "q_value_loss",
        "policy_loss",
        "p1_terminal_fraction",
        "p1_value_td_fraction",
        "p1_q_td_fraction",
        "sigreg_loss",
    ):
        assert removed_key not in metrics


def test_state_sigreg_contributes_for_all_pov_state_groups(monkeypatch):
    outputs = _outputs(batch_size=1, rollout_len=2, latent_dim=3, action_dim=2)
    outputs["enc_p1_history_states"] = torch.zeros((1, 2, 3, 3))
    outputs["enc_p1_history_states_valid"] = torch.tensor(
        [[[True, True, False], [True, False, False]]]
    )
    outputs["enc_p2_history_states"] = torch.zeros((1, 2, 3, 3))
    outputs["enc_p2_history_states_valid"] = torch.tensor(
        [[[True, False, False], [False, False, False]]]
    )
    call_sizes = []

    def fake_sigreg(embeddings, **_kwargs):
        call_sizes.append(int(embeddings.shape[0]))
        return embeddings.sum() * 0 + embeddings.new_tensor(float(embeddings.shape[0]))

    monkeypatch.setattr(jepa_model, "sigreg", fake_sigreg)

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_self_state=0.0,
        lambda_opponent_state=0.0,
        lambda_action=0.0,
        lambda_next_state=0.0,
        lambda_sigreg_state=0.5,
        sigreg_num_slices=4,
        sigreg_num_points=5,
        sigreg_domain=2.0,
    )

    assert call_sizes == [3, 2, 2, 1, 2, 2]
    assert metrics["sigreg_state_p1_history"] == pytest.approx(3.0)
    assert metrics["sigreg_state_p1_current"] == pytest.approx(2.0)
    assert metrics["sigreg_state_p1_next"] == pytest.approx(2.0)
    assert metrics["sigreg_state_p2_history"] == pytest.approx(1.0)
    assert metrics["sigreg_state_p2_current"] == pytest.approx(2.0)
    assert metrics["sigreg_state_p2_next"] == pytest.approx(2.0)
    assert metrics["sigreg_state_loss"] == pytest.approx(2.0)
    assert metrics["sigreg_state_weighted"] == pytest.approx(1.0)
    assert loss.item() == pytest.approx(1.0)


def test_self_state_loss_contributes_to_total():
    outputs = _outputs(batch_size=1, rollout_len=1, latent_dim=2, action_dim=2)
    outputs["enc_p1_T"] = torch.tensor([[[1.0, 0.0]]])
    outputs["enc_p2_T"] = torch.tensor([[[0.0, 2.0]]])
    outputs["pred_p1_self_T_mu"] = torch.zeros_like(outputs["enc_p1_T"])
    outputs["pred_p2_self_T_mu"] = torch.zeros_like(outputs["enc_p2_T"])

    loss, metrics = compute_paired_losses(
        outputs,
        lambda_self_state=1.0,
        lambda_opponent_state=0.0,
        lambda_action=0.0,
        lambda_next_state=0.0,
    )

    expected_p1 = 0.5 * torch.tensor([[[1.0, 0.0]]]).square().mean()
    expected_p2 = 0.5 * torch.tensor([[[0.0, 2.0]]]).square().mean()
    expected = 0.5 * (expected_p1 + expected_p2)
    assert metrics["self_state_loss"] == pytest.approx(expected.item())
    assert loss.item() == pytest.approx(expected.item())


def test_rollout_axis_supported():
    outputs = _outputs(batch_size=2, rollout_len=4)

    loss, metrics = compute_paired_losses(outputs)

    assert torch.isfinite(loss)
    assert metrics["self_state_loss"] == pytest.approx(0.0)
    assert metrics["opponent_state_loss"] == pytest.approx(0.0)
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["next_state_loss"] == pytest.approx(0.0)
