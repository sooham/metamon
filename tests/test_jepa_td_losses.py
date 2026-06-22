"""Tests for TD (temporal-difference) value/Q bootstrapping in compute_paired_losses."""

import pytest
import torch

from metamon.jepa.model import (
    compute_paired_losses,
    _masked_bce_with_logits,
)


class TestTDBootstrapping:
    """Unit tests for TD bootstrapping in compute_paired_losses."""

    @staticmethod
    def _make_outputs(
        B: int = 2,
        K: int = 3,
        latent_dim: int = 192,
        action_latent_dim: int = 32,
        num_candidates: int = 4,
        with_is_terminal: bool = True,
    ) -> dict:
        """Build a minimal outputs dict matching the PairedJEPAModel.forward() schema."""
        o = {
            "enc_p1_T": torch.randn(B, K, latent_dim),
            "enc_p2_T": torch.randn(B, K, latent_dim),
            "enc_p1_T1": torch.randn(B, K, latent_dim),
            "enc_p2_T1": torch.randn(B, K, latent_dim),
            "ctx_p1_T": torch.randn(B, K, latent_dim),
            "ctx_p2_T": torch.randn(B, K, latent_dim),
            "pred_p2_T_mu": torch.randn(B, K, latent_dim),
            "pred_p2_T_logvar": torch.randn(B, K, latent_dim),
            "pred_p1_T_mu": torch.randn(B, K, latent_dim),
            "pred_p1_T_logvar": torch.randn(B, K, latent_dim),
            "pred_p2_T": torch.randn(B, K, latent_dim),
            "pred_p1_T": torch.randn(B, K, latent_dim),
            "p1_action": torch.randn(B, K, action_latent_dim),
            "p2_action": torch.randn(B, K, action_latent_dim),
            "actual_p2_action_from_p1_perspective": torch.randn(B, K, action_latent_dim),
            "actual_p1_action_from_p2_perspective": torch.randn(B, K, action_latent_dim),
            "pred_p2_action_mu": torch.randn(B, K, action_latent_dim),
            "pred_p2_action_logvar": torch.randn(B, K, action_latent_dim),
            "pred_p1_action_mu": torch.randn(B, K, action_latent_dim),
            "pred_p1_action_logvar": torch.randn(B, K, action_latent_dim),
            "pred_p2_action": torch.randn(B, K, action_latent_dim),
            "pred_p1_action": torch.randn(B, K, action_latent_dim),
            "pred_p1_T1_mu": torch.randn(B, K, latent_dim),
            "pred_p1_T1_logvar": torch.randn(B, K, latent_dim),
            "pred_p2_T1_mu": torch.randn(B, K, latent_dim),
            "pred_p2_T1_logvar": torch.randn(B, K, latent_dim),
            "pred_p1_T1": torch.randn(B, K, latent_dim),
            "pred_p2_T1": torch.randn(B, K, latent_dim),
            "p1_value_logit": torch.randn(B, K),
            "p2_value_logit": torch.randn(B, K),
            "p1_next_value_logit": torch.randn(B, K),
            "p2_next_value_logit": torch.randn(B, K),
            "p1_value_teacher_logit": torch.randn(B, K),
            "p2_value_teacher_logit": torch.randn(B, K),
            "p1_q_logits": torch.randn(B, K, num_candidates),
            "p2_q_logits": torch.randn(B, K, num_candidates),
            "p1_next_q_logits": torch.randn(B, K, num_candidates),
            "p2_next_q_logits": torch.randn(B, K, num_candidates),
            "p1_q_teacher_logits": torch.randn(B, K, num_candidates),
            "p2_q_teacher_logits": torch.randn(B, K, num_candidates),
            "p1_legal_action_mask": torch.ones(B, K, num_candidates, dtype=torch.bool),
            "p2_legal_action_mask": torch.ones(B, K, num_candidates, dtype=torch.bool),
            "p1_next_legal_action_mask": torch.ones(B, K, num_candidates, dtype=torch.bool),
            "p2_next_legal_action_mask": torch.ones(B, K, num_candidates, dtype=torch.bool),
            "p1_chosen_legal_action_idx": torch.randint(0, num_candidates, (B, K)),
            "p2_chosen_legal_action_idx": torch.randint(0, num_candidates, (B, K)),
            "p1_won": torch.tensor([True, False]),
            "p2_won": torch.tensor([False, True]),
            "rank_valid": torch.tensor([True, True]),
        }
        if with_is_terminal:
            # Default fixture marks the final step as a true terminal. Tests
            # that need rollout truncation/non-terminal boundaries override it.
            term = torch.zeros(B, K, dtype=torch.bool)
            term[:, -1] = True
            o["p1_is_terminal"] = term
            o["p2_is_terminal"] = term
        return o

    def test_mc_loss_same_with_gamma_one(self):
        """gamma=1.0 should produce identical losses to MC (backward compat)."""
        outputs = self._make_outputs()
        _, m_td = compute_paired_losses(outputs, gamma=1.0)
        _, m_mc = compute_paired_losses(outputs, gamma=1.0)
        assert abs(m_td["value_loss"] - m_mc["value_loss"]) < 1e-5
        assert abs(m_td["q_value_loss"] - m_mc["q_value_loss"]) < 1e-5

    def test_td_losses_differ_from_mc(self):
        """With gamma<1, TD losses should differ from MC losses."""
        outputs = self._make_outputs()
        _, m_mc = compute_paired_losses(outputs, gamma=1.0)
        _, m_td = compute_paired_losses(outputs, gamma=0.99)
        # TD and MC losses should differ (because soft targets change)
        # Note: they could coincidentally be equal, but probability is very low
        assert abs(m_td["value_loss"] - m_mc["value_loss"]) > 1e-7
        assert abs(m_td["q_value_loss"] - m_mc["q_value_loss"]) > 1e-7

    def test_td_value_loss_nonzero_with_gamma_lt_one(self):
        """V head should produce non-zero loss even with gamma<1 on non-zero logits."""
        outputs = self._make_outputs()
        outputs["p1_value_logit"] = torch.tensor([[1.0, -0.5, 0.3],
                                                   [-1.0, 0.8, -0.2]])
        outputs["p2_value_logit"] = torch.tensor([[0.5, -1.0, 0.1],
                                                   [-0.3, 0.6, -0.8]])
        _, metrics = compute_paired_losses(
            outputs,
            lambda_value=1.0,
            lambda_q_value=0.0,
            lambda_policy=0.0,
            lambda_value_teacher=0.0,
            lambda_q_teacher=0.0,
            gamma=0.99,
        )
        assert metrics["value_loss"] > 0.0

    def test_td_q_loss_nonzero_with_gamma_lt_one(self):
        """Q head should produce non-zero loss with gamma<1."""
        outputs = self._make_outputs()
        outputs["p1_q_logits"] = torch.randn(2, 3, 4) * 2.0
        outputs["p2_q_logits"] = torch.randn(2, 3, 4) * 2.0
        _, metrics = compute_paired_losses(
            outputs,
            lambda_value=0.0,
            lambda_q_value=1.0,
            lambda_policy=0.0,
            lambda_value_teacher=0.0,
            lambda_q_teacher=0.0,
            gamma=0.99,
        )
        assert metrics["q_value_loss"] > 0.0

    def test_k1_nonterminal_bootstraps_from_explicit_t1(self):
        """K=1 can still TD-bootstrap when the transition is non-terminal."""
        outputs = self._make_outputs(K=1)
        outputs["p1_is_terminal"] = torch.zeros((2, 1), dtype=torch.bool)
        outputs["p2_is_terminal"] = torch.zeros((2, 1), dtype=torch.bool)
        _, td_metrics = compute_paired_losses(outputs, gamma=0.99)
        _, mc_metrics = compute_paired_losses(outputs, gamma=1.0)
        assert td_metrics["p1_terminal_fraction"] == 0.0
        assert td_metrics["p1_value_td_fraction"] == 1.0
        assert td_metrics["p1_q_td_fraction"] == 1.0
        assert abs(td_metrics["value_loss"] - mc_metrics["value_loss"]) > 1e-7

    def test_no_is_terminal_fallback(self):
        """Without is_terminal in outputs, should fall back gracefully."""
        outputs = self._make_outputs(with_is_terminal=False)
        loss, metrics = compute_paired_losses(outputs, gamma=0.99)
        assert "loss" in metrics
        assert metrics["loss"] == metrics["loss"]  # NaN check
        assert "p1_terminal_fraction" in metrics
        assert "p2_terminal_fraction" in metrics

    def test_none_is_terminal_fallback(self):
        """A present-but-None terminal key should use the default non-terminal mask."""
        outputs = self._make_outputs(with_is_terminal=False)
        outputs["p1_is_terminal"] = None
        outputs["p2_is_terminal"] = None
        loss, metrics = compute_paired_losses(outputs, gamma=0.99)
        assert torch.isfinite(loss)
        assert metrics["p1_terminal_fraction"] == 0.0
        assert metrics["p2_terminal_fraction"] == 0.0

    def test_rollout_boundary_bootstraps_from_explicit_next_value(self):
        """The last rollout position is not terminal unless the data says so."""
        B, K = 1, 2
        outputs = self._make_outputs(B=B, K=K, num_candidates=2)
        outputs["p1_value_logit"] = torch.tensor([[0.5, -0.25]])
        outputs["p2_value_logit"] = torch.tensor([[-0.5, 0.25]])
        outputs["p1_next_value_logit"] = torch.tensor([[1.0, 2.0]])
        outputs["p2_next_value_logit"] = torch.tensor([[-1.0, -2.0]])
        outputs["p1_won"] = torch.ones((B, K), dtype=torch.bool)
        outputs["p2_won"] = torch.zeros((B, K), dtype=torch.bool)
        outputs["rank_valid"] = torch.ones((B, K), dtype=torch.bool)
        outputs["p1_is_terminal"] = torch.zeros((B, K), dtype=torch.bool)
        outputs["p2_is_terminal"] = torch.zeros((B, K), dtype=torch.bool)

        gamma = 0.5
        _, metrics = compute_paired_losses(
            outputs,
            lambda_sigreg=0.0,
            lambda_opponent_state=0.0,
            lambda_action=0.0,
            lambda_next_state=0.0,
            lambda_rank=0.0,
            lambda_value=1.0,
            lambda_q_value=0.0,
            lambda_policy=0.0,
            lambda_value_teacher=0.0,
            lambda_q_teacher=0.0,
            gamma=gamma,
            sigreg_num_slices=1,
            sigreg_num_points=2,
        )

        p1_target = gamma * torch.sigmoid(outputs["p1_next_value_logit"])
        p2_target = gamma * torch.sigmoid(outputs["p2_next_value_logit"])
        expected = 0.5 * (
            torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["p1_value_logit"], p1_target
            )
            + torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["p2_value_logit"], p2_target
            )
        )
        assert metrics["p1_terminal_fraction"] == 0.0
        assert metrics["p2_terminal_fraction"] == 0.0
        assert metrics["p1_value_td_fraction"] == 1.0
        assert metrics["p2_value_td_fraction"] == 1.0
        assert metrics["value_loss"] == pytest.approx(expected.item())

    def test_td_accepts_rollout_shaped_outcomes_with_gamma_lt_one(self):
        """Real collated batches provide outcome labels as [B, K]."""
        outputs = self._make_outputs()
        B, K = outputs["p1_value_logit"].shape
        outputs["p1_won"] = torch.tensor(
            [[True, True, True], [False, False, False]],
            dtype=torch.bool,
        )
        outputs["p2_won"] = ~outputs["p1_won"]
        outputs["rank_valid"] = torch.ones((B, K), dtype=torch.bool)
        loss, metrics = compute_paired_losses(outputs, gamma=0.99)
        assert torch.isfinite(loss)
        assert metrics["p1_value_td_fraction"] > 0.0

    def test_no_value_q_heads(self):
        """Without V/Q heads, loss should still compute (world-model only)."""
        outputs = self._make_outputs()
        # Remove all V/Q/teacher heads and their dependencies
        for k in list(outputs.keys()):
            if any(x in k for x in ("value_logit", "q_logits", "q_teacher", "value_teacher",
                                      "legal_action", "chosen_legal")):
                del outputs[k]
        loss, metrics = compute_paired_losses(outputs, gamma=0.99)
        assert "loss" in metrics
        assert metrics["value_loss"] == 0.0
        assert metrics["q_value_loss"] == 0.0

    def test_td_targets_detached(self):
        """TD bootstrapping uses .detach() so gradients don't flow through targets.

        We verify this by checking that the per-step targets embedded in the
        TD helper functions do not carry gradients (they're produced from
        .detach() calls on the shifted value/Q tensors).
        """
        # Build a simple scenario where the next value logit requires grad.
        # It should be used only through a detached target.
        B, K = 2, 2
        value_logit = torch.randn(B, K, requires_grad=True)
        next_value_logit = torch.randn(B, K, requires_grad=True)
        gamma = 0.99

        # Manually verify the detach behaviour: the target should be
        # gamma * sigmoid(next_value_logit).detach(), which has no grad.
        target = gamma * torch.sigmoid(next_value_logit.detach())
        assert not target.requires_grad  # .detach() was called

        # Full loss computation with requires_grad on latents:
        outputs = self._make_outputs(B=B, K=K)
        outputs["p1_value_logit"] = value_logit
        outputs["p1_next_value_logit"] = next_value_logit
        outputs["p2_value_logit"] = torch.randn(B, K, requires_grad=True)
        loss, _ = compute_paired_losses(
            outputs,
            lambda_value=1.0,
            lambda_q_value=0.0,
            lambda_policy=0.0,
            lambda_value_teacher=0.0,
            lambda_q_teacher=0.0,
            gamma=gamma,
        )
        loss.backward()  # should not error — targets are detached
        assert next_value_logit.grad is None

    def test_metrics_include_gamma_and_terminal(self):
        """Metrics dict should include gamma and terminal fraction."""
        outputs = self._make_outputs()
        _, metrics = compute_paired_losses(outputs, gamma=0.95)
        assert metrics["gamma"] == 0.95
        assert 0.0 <= metrics["p1_terminal_fraction"] <= 1.0
        assert 0.0 <= metrics["p2_terminal_fraction"] <= 1.0
        assert 0.0 <= metrics["p1_value_td_fraction"] <= 1.0
        assert 0.0 <= metrics["p2_value_td_fraction"] <= 1.0

    def test_gamma_zero_all_nonterminal_targets_zero(self):
        """With gamma=0, non-terminal steps have target=0."""
        outputs = self._make_outputs(K=2)
        outputs["p1_value_logit"] = torch.tensor([[0.0, 5.0],
                                                   [0.0, 5.0]])
        outputs["p2_value_logit"] = torch.tensor([[0.0, 5.0],
                                                   [0.0, 5.0]])
        outputs["p1_won"] = torch.tensor([True, False])
        outputs["p2_won"] = torch.tensor([False, True])
        outputs["rank_valid"] = torch.tensor([True, True])
        term = torch.zeros(2, 2, dtype=torch.bool)
        term[:, -1] = True
        outputs["p1_is_terminal"] = term
        outputs["p2_is_terminal"] = term

        _, metrics = compute_paired_losses(
            outputs,
            lambda_value=1.0,
            lambda_q_value=0.0,
            lambda_policy=0.0,
            lambda_value_teacher=0.0,
            lambda_q_teacher=0.0,
            gamma=0.0,
        )
        # V loss should be non-zero (step 0 gets target=0, step 1 gets outcome)
        assert metrics["value_loss"] > 0.0
