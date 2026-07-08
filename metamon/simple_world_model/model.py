"""Simple p1-only world model with V, M, and C components.

The model follows the World Models split:

* V: a transformer beta-VAE over the FULL seen history for the POV player —
  the interleaved sequence ``[team_header, state_0, p_action_0, o_action_0,
  state_1, ..., p_action_{T-1}, o_action_{T-1}, state_T]`` (everything
  observed up to and including the current state) — compressed into a latent
  ``z_T``. The decoder reconstructs the ENTIRE history sequence from ``z_T``
  (the bottleneck is the beta-VAE latent), so ``z_T`` must carry the full
  observed context. The history window is capped by ``--max_history_blocks``
  (default 64) at the dataset level, matching the JEPA trainer's windowing.
* M: an MLP mixture-density transition model ``p(z_{T+1} | z_T, a_T)``. Since
  the full history context now lives inside ``z_T`` via V, M only needs the
  latest latent + the chosen action — no transformer needed.
* C: small MLP controller over z_T and M's no-action context.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from metamon.jepa.model import AttentionPool, MLP, TransformerBlock

TERMINAL_CLASSES = (
    "ongoing",
    "won",
    "lost",
    "forfeit_won",
    "forfeit_lost",
    "tie",
)
NUM_TERMINAL_CLASSES = len(TERMINAL_CLASSES)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx] = 0.0


def _flatten_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    leading = tuple(tokens.shape[:-1])
    return tokens.reshape(-1, tokens.shape[-1]), leading


def _restore_leading(x: torch.Tensor, leading: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*leading, *x.shape[1:])


class StateVAE(nn.Module):
    """Transformer beta-VAE over the full seen history sequence.

    ``encode`` sees the interleaved ``[team_header, states, player/opponent
    actions, ..., current_state_T]`` token sequence (capped to
    ``--max_history_blocks`` prior states by the dataset) and compresses it
    into a latent ``z_T`` via attention pooling. ``decode`` reconstructs the
    ENTIRE history sequence from ``z_T`` (the bottleneck is the beta-VAE
    latent), forcing ``z_T`` to carry the full observed context.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        latent_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        ffn_activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=False, ffn_activation=ffn_activation, use_rope=True,
            )
            for _ in range(n_layers)
        ])
        self.encoder_ln = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        self.mu_head = MLP(d_model, max(2 * d_model, latent_dim), latent_dim)
        self.logvar_head = MLP(d_model, max(2 * d_model, latent_dim), latent_dim)

        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.z_to_decoder = nn.Linear(latent_dim, d_model)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=False, ffn_activation=ffn_activation, use_rope=True,
            )
            for _ in range(n_layers)
        ])
        self.decoder_ln = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size + 1)
        self.apply(_init_weights)

    def _check_len(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"State sequence length {seq_len} exceeds V max_seq_len={self.max_seq_len}"
            )

    def _encode_flat(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the encoder body over the full history sequence and pool."""
        self._check_len(token_ids.shape[-1])
        valid_mask = token_ids != self.pad_id
        x = self.token_embedding(token_ids)
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=valid_mask)
        x = self.encoder_ln(x)
        return self.pool(x, valid_mask)

    def encode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the full history into ``(mu, logvar)`` for ``z_T``."""
        flat, leading = _flatten_tokens(token_ids)
        if self.gradient_checkpointing and self.training:
            pooled = torch.utils.checkpoint.checkpoint(
                self._encode_flat, flat, use_reentrant=False,
            )
        else:
            pooled = self._encode_flat(flat)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-12.0, 8.0)
        return _restore_leading(mu, leading), _restore_leading(logvar, leading)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _decode_flat(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        self._check_len(seq_len)
        positions = torch.arange(seq_len, device=z.device)
        pos = self.position_embedding(positions).unsqueeze(0).expand(z.shape[0], -1, -1)
        x = pos + self.z_to_decoder(z).unsqueeze(1)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_ln(x)
        return self.output_head(x)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        leading = tuple(z.shape[:-1])
        flat_z = z.reshape(-1, z.shape[-1])
        logits = self._decode_flat(flat_z, seq_len)
        return _restore_leading(logits, leading)

    def forward(
        self,
        token_ids: torch.Tensor,
        reconstruct_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(token_ids)
        z = self.reparameterize(mu, logvar)
        seq_len = reconstruct_len if reconstruct_len is not None else token_ids.shape[-1]
        logits = self.decode(z, seq_len)
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}


class ActionEncoder(nn.Module):
    """Small transformer encoder for short action token sequences."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        action_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 32,
        ffn_activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=False, ffn_activation=ffn_activation, use_rope=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        self.proj = MLP(d_model, max(2 * d_model, action_dim), action_dim)
        self.apply(_init_weights)

    def _encode_flat(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[-1] > self.max_seq_len:
            raise ValueError(
                f"Action sequence length {token_ids.shape[-1]} exceeds max_seq_len={self.max_seq_len}"
            )
        valid_mask = token_ids != self.pad_id
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, key_padding_mask=valid_mask)
        x = self.ln_final(x)
        return self.proj(self.pool(x, valid_mask))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat, leading = _flatten_tokens(token_ids)
        if self.gradient_checkpointing and self.training:
            emb = torch.utils.checkpoint.checkpoint(
                self._encode_flat, flat, use_reentrant=False,
            )
        else:
            emb = self._encode_flat(flat)
        return emb.reshape(*leading, self.action_dim)


class MemoryMDN(nn.Module):
    """MLP mixture-density transition model for p(z_{t+1} | z_t, a_t).

    Since V now encodes the FULL seen history into ``z_T``, the transition
    model no longer needs a transformer to carry temporal context — everything
    is already in ``z_T``. A plain MLP from ``(z_T, action_emb)`` to the MDN
    parameters suffices.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        action_dim: int = 256,
        d_model: int = 1024,
        n_heads: int = 8,          # unused (MLP); kept for config compatibility
        n_layers: int = 4,
        d_ff: int = 4096,          # unused (MLP); kept for config compatibility
        dropout: float = 0.0,
        num_mixtures: int = 5,
        ffn_activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_mixtures = num_mixtures
        self.gradient_checkpointing = gradient_checkpointing
        act = {"gelu": nn.GELU, "relu": nn.ReLU}.get(ffn_activation, nn.GELU)
        in_dim = latent_dim + action_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, d_model), act()]
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(d_model, d_model))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(act())
        layers.append(nn.Linear(d_model, d_model))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.no_action_embedding = nn.Parameter(torch.zeros(action_dim))
        self.mixture_logits = nn.Linear(d_model, num_mixtures)
        self.mixture_means = nn.Linear(d_model, num_mixtures * latent_dim)
        self.mixture_log_scales = nn.Linear(d_model, num_mixtures * latent_dim)
        self.terminal_head = nn.Linear(d_model, NUM_TERMINAL_CLASSES)
        self.apply(_init_weights)

    def context(self, z: torch.Tensor, action_emb: torch.Tensor | None = None) -> torch.Tensor:
        """MLP over ``(z, action_emb)`` -> no-action context h (for controller)."""
        leading = tuple(z.shape[:-1])
        z_flat = z.reshape(-1, z.shape[-1])
        if action_emb is None:
            a_flat = self.no_action_embedding.to(dtype=z_flat.dtype, device=z_flat.device).expand(z_flat.shape[0], -1)
        else:
            a_flat = action_emb.reshape(-1, action_emb.shape[-1])
        h = self.ln_final(self.trunk(torch.cat([z_flat, a_flat], dim=-1)))
        return h.reshape(*leading, self.d_model)

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.context(z, action_emb)
        leading = tuple(h.shape[:-1])
        h_flat = h.reshape(-1, h.shape[-1])
        logits = self.mixture_logits(h_flat).reshape(*leading, self.num_mixtures)
        means = self.mixture_means(h_flat).reshape(*leading, self.num_mixtures, self.latent_dim)
        log_scales = self.mixture_log_scales(h_flat).reshape(*leading, self.num_mixtures, self.latent_dim)
        terminal_logits = self.terminal_head(h_flat).reshape(*leading, NUM_TERMINAL_CLASSES)
        return {
            "h": h,
            "mixture_logits": logits,
            "mixture_means": means,
            "mixture_log_scales": log_scales.clamp(-7.0, 3.0),
            "terminal_logits": terminal_logits,
        }

    @torch.no_grad()
    def sample_next_z(
        self,
        mixture_logits: torch.Tensor,
        mixture_means: torch.Tensor,
        mixture_log_scales: torch.Tensor,
        tau: float = 1.0,
    ) -> torch.Tensor:
        tau = max(float(tau), 1e-6)
        dist = torch.distributions.Categorical(logits=mixture_logits / tau)
        component = dist.sample()
        gather_idx = component.unsqueeze(-1).unsqueeze(-1).expand(*component.shape, 1, self.latent_dim)
        mean = mixture_means.gather(-2, gather_idx).squeeze(-2)
        scale = torch.exp(mixture_log_scales.gather(-2, gather_idx).squeeze(-2)) * tau
        return mean + torch.randn_like(mean) * scale


class Controller(nn.Module):
    """Two-layer MLP controller that scores encoded legal actions."""

    def __init__(
        self,
        *,
        latent_dim: int = 1024,
        h_dim: int = 1024,
        action_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + h_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor, h: torch.Tensor, legal_action_embs: torch.Tensor) -> torch.Tensor:
        query = self.net(torch.cat([z, h], dim=-1))
        return torch.einsum("...d,...ld->...l", query, legal_action_embs)


class SimpleWorldModel(nn.Module):
    """Container for V, M, C, and the shared action encoder."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        latent_dim: int = 1024,
        v_cfg: dict[str, Any] | None = None,
        action_encoder_cfg: dict[str, Any] | None = None,
        m_cfg: dict[str, Any] | None = None,
        controller_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        v_cfg = dict(v_cfg or {})
        action_encoder_cfg = dict(action_encoder_cfg or {})
        m_cfg = dict(m_cfg or {})
        controller_cfg = dict(controller_cfg or {})

        action_dim = int(action_encoder_cfg.get("action_dim", 256))
        self.latent_dim = latent_dim
        self.pad_id = pad_id
        self.v = StateVAE(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            **v_cfg,
        )
        self.action_encoder = ActionEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            **action_encoder_cfg,
        )
        self.m = MemoryMDN(
            latent_dim=latent_dim,
            action_dim=action_dim,
            **m_cfg,
        )
        self.c = Controller(
            latent_dim=latent_dim,
            h_dim=self.m.d_model,
            action_dim=action_dim,
            **controller_cfg,
        )

    def forward_vm(
        self,
        history_tokens: torch.Tensor,
        next_history_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """V/M training step.

        ``history_tokens`` is the FULL interleaved seen history up to and
        including the current state ``state_T`` (team header + prior states +
        prior player/opponent actions + current state), capped to
        ``--max_history_blocks`` by the dataset. V encodes it into ``z_T``
        (with grad) and the decoder reconstructs the ENTIRE history sequence
        from ``z_T`` (the bottleneck is the beta-VAE latent).

        ``next_history_tokens`` is the same history extended with the current
        player action, opponent action, and next state ``state_{T+1}``; V
        encodes it (no grad) into the MDN target ``z_{T+1}``, re-sampled each
        step (Ha & Schmidhuber 2018, App. A.2).

        ``action_tokens`` is the current POV action, encoded by the action
        encoder and fed to the MLP MDN ``M``.
        """
        mu, logvar = self.v.encode(history_tokens)
        z = StateVAE.reparameterize(mu, logvar)
        # Decode/reconstruct the ENTIRE history fed to the encoder.
        state_logits = self.v.decode(z, history_tokens.shape[-1])
        with torch.no_grad():
            next_mu, next_logvar = self.v.encode(next_history_tokens)
            next_std = torch.exp(0.5 * next_logvar)
            next_z = next_mu + torch.randn_like(next_std) * next_std
        action_emb = self.action_encoder(action_tokens)
        m_out = self.m(z, action_emb)
        return {
            "state_logits": state_logits,
            "history_tokens": history_tokens,
            "z": z,
            "z_mu": mu,
            "z_logvar": logvar,
            "next_z_mu": next_mu,
            "next_z_logvar": next_logvar,
            "next_z": next_z,
            "action_emb": action_emb,
            **m_out,
        }

    def forward_controller(
        self,
        history_tokens: torch.Tensor,
        legal_action_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mu, _ = self.v.encode(history_tokens)
        h = self.m.context(mu, action_emb=None)
        legal_embs = self.action_encoder(legal_action_tokens)
        logits = self.c(mu, h, legal_embs)
        return {
            "controller_logits": logits,
            "controller_z_mu": mu,
            "controller_h": h,
            "legal_action_embs": legal_embs,
        }


def mdn_nll(
    target: torch.Tensor,
    mixture_logits: torch.Tensor,
    mixture_means: torch.Tensor,
    mixture_log_scales: torch.Tensor,
) -> torch.Tensor:
    target = target.unsqueeze(-2)
    inv_scale = torch.exp(-mixture_log_scales)
    standardized = (target - mixture_means) * inv_scale
    log_prob = (
        -0.5 * standardized.square()
        - mixture_log_scales
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    log_mix = F.log_softmax(mixture_logits, dim=-1)
    return -torch.logsumexp(log_mix + log_prob, dim=-1).mean()


def _masked_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = targets != pad_id
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    preds = logits.argmax(dim=-1)
    return (preds[mask] == targets[mask]).float().mean()


def _latent_diagnostics(name: str, z: torch.Tensor, active_std_threshold: float = 1e-3) -> dict[str, float]:
    with torch.no_grad():
        flat = z.detach().float().reshape(-1, z.shape[-1])
        if flat.numel() == 0:
            return {
                f"{name}_norm": 0.0,
                f"{name}_std_per_dim": 0.0,
                f"{name}_pairwise_distance": 0.0,
                f"{name}_abs_mean": 0.0,
                f"{name}_active_dims": 0.0,
            }
        std = flat.std(dim=0, unbiased=False)
        n = min(flat.shape[0], 64)
        if n >= 2:
            pairwise = torch.pdist(flat[:n], p=2).mean()
        else:
            pairwise = flat.new_tensor(0.0)
        return {
            f"{name}_norm": flat.norm(dim=-1).mean().item(),
            f"{name}_std_per_dim": std.mean().item(),
            f"{name}_pairwise_distance": pairwise.item(),
            f"{name}_abs_mean": flat.mean(dim=0).abs().mean().item(),
            f"{name}_active_dims": float((std > active_std_threshold).sum().item()),
        }


def _terminal_class_counts(target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        flat = target.detach().reshape(-1)
        return {
            f"terminal_count_{name}": float((flat == idx).sum().item())
            for idx, name in enumerate(TERMINAL_CLASSES)
        }


def compute_simple_world_model_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    pad_id: int,
    components: str = "vm",
    lambda_recon: float = 1.0,
    beta_kl: float = 0.01,
    lambda_mdn: float = 1.0,
    lambda_terminal: float = 0.25,
    lambda_controller_bc: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    total = torch.zeros((), device=next(iter(outputs.values())).device)

    if components in {"vm", "all"}:
        # Reconstruction target is the ENTIRE history sequence fed to the
        # encoder (team header + states + actions + current state), so z_T must
        # carry the full observed context. Prefer the history tensor returned
        # by the model (matches the decoded length exactly).
        recon_target = (
            outputs["history_tokens"].long()
            if "history_tokens" in outputs
            else batch["history_tokens"].long()
        )
        recon_ce = F.cross_entropy(
            outputs["state_logits"].reshape(-1, outputs["state_logits"].shape[-1]),
            recon_target.reshape(-1),
            ignore_index=pad_id,
        )
        kl_per_sample = -0.5 * (
            1.0 + outputs["z_logvar"] - outputs["z_mu"].square() - outputs["z_logvar"].exp()
        ).sum(dim=-1)
        kl = kl_per_sample.mean()
        nll = mdn_nll(
            outputs["next_z"].detach(),
            outputs["mixture_logits"],
            outputs["mixture_means"],
            outputs["mixture_log_scales"],
        )
        terminal_ce = F.cross_entropy(
            outputs["terminal_logits"].reshape(-1, NUM_TERMINAL_CLASSES),
            batch["terminal_class"].long().reshape(-1),
        )
        vm_loss = (
            lambda_recon * recon_ce
            + beta_kl * kl
            + lambda_mdn * nll
            + lambda_terminal * terminal_ce
        )
        total = total + vm_loss
        terminal_pred = outputs["terminal_logits"].argmax(dim=-1)
        terminal_target = batch["terminal_class"].long()
        terminal_acc = (terminal_pred == terminal_target).float().mean()
        metrics.update({
            "loss_vm": vm_loss.item(),
            "recon_ce": recon_ce.item(),
            "recon_token_acc": _masked_token_accuracy(outputs["state_logits"], recon_target, pad_id).item(),
            "kl": kl.item(),
            "kl_per_dim": (kl / outputs["z_mu"].shape[-1]).item(),
            "kl_weighted": (beta_kl * kl).item(),
            "mdn_nll": nll.item(),
            "terminal_ce": terminal_ce.item(),
            "terminal_acc": terminal_acc.item(),
            "vae_logvar_mean": outputs["z_logvar"].detach().float().mean().item(),
            "vae_logvar_min": outputs["z_logvar"].detach().float().min().item(),
            "vae_logvar_max": outputs["z_logvar"].detach().float().max().item(),
            # Per-sample mean (not a batch-raw count) so averaging across
            # validation/epoch steps is meaningful.
            "nonpad_state_tokens": float((recon_target != pad_id).float().sum(dim=-1).mean().item()),
        })
        metrics.update(_latent_diagnostics("z_mu", outputs["z_mu"]))
        metrics.update(_latent_diagnostics("next_z_mu", outputs["next_z_mu"]))
        with torch.no_grad():
            probs = F.softmax(outputs["mixture_logits"].detach().float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            metrics.update({
                "mdn_mixture_entropy": entropy.item(),
                "mdn_max_mixture_prob": probs.max(dim=-1).values.mean().item(),
                "mdn_scale_mean": outputs["mixture_log_scales"].detach().float().exp().mean().item(),
                "mdn_scale_min": outputs["mixture_log_scales"].detach().float().exp().min().item(),
                "mdn_scale_max": outputs["mixture_log_scales"].detach().float().exp().max().item(),
            })
        metrics.update(_terminal_class_counts(terminal_target))

    if components in {"c", "all"}:
        controller_logits = outputs["controller_logits"]
        legal_mask = batch["legal_action_mask"]
        chosen_idx = batch["chosen_legal_action_idx"].long()
        masked_logits = controller_logits.masked_fill(~legal_mask, torch.finfo(controller_logits.dtype).min)
        controller_ce = F.cross_entropy(masked_logits, chosen_idx)
        total = total + lambda_controller_bc * controller_ce
        controller_pred = masked_logits.argmax(dim=-1)
        controller_acc = (controller_pred == chosen_idx).float().mean()
        metrics.update({
            "loss_controller": controller_ce.item(),
            "loss_controller_weighted": (lambda_controller_bc * controller_ce).item(),
            "controller_acc": controller_acc.item(),
            "legal_action_count": legal_mask.float().sum(dim=-1).mean().item(),
        })

    metrics["loss"] = total.item()
    return total, metrics
