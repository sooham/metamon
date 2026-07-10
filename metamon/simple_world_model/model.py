"""V/M/C world model.

This module intentionally does not retain the old full-token-history VAE
contract.  V only encodes a permanent team header plus one visible state; M is
where temporal memory lives, in a causal sequence of cached state latents and
canonical action IDs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from metamon.jepa.model import AttentionPool, MLP, TransformerBlock


# These source-protocol labels are retained for paired-shard compatibility.
TERMINAL_CLASSES = (
    "ongoing",
    "won",
    "lost",
    "forfeit_won",
    "forfeit_lost",
    "tie",
)
NUM_TERMINAL_CLASSES = len(TERMINAL_CLASSES)
OUTCOME_CLASSES = ("win", "loss", "tie")
NUM_OUTCOME_CLASSES = len(OUTCOME_CLASSES)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


def _flatten_leading(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    return x.reshape(-1, x.shape[-1]), tuple(x.shape[:-1])


def _restore_leading(x: torch.Tensor, leading: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*leading, *x.shape[1:])


def _as_valid_mask(tokens: torch.Tensor, mask: torch.Tensor | None, pad_id: int) -> torch.Tensor:
    if mask is None:
        return tokens.ne(pad_id)
    if mask.shape != tokens.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match tokens {tuple(tokens.shape)}")
    return mask.to(device=tokens.device, dtype=torch.bool)


class StateVAE(nn.Module):
    """Bidirectional VAE for one visible state conditioned on a team header.

    The decoder API deliberately requires ``valid_token_mask``.  Passing this
    mask to every decoder block prevents a right-padding token from becoming a
    key that changes a valid prefix's logits.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        max_state_tokens: int | None = None,
        theta: float = 10000.0,  # Kept in the public config for compatibility.
        ffn_activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        del theta
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.max_state_tokens = int(max_state_tokens or max_seq_len)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.token_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.segment_embedding = nn.Embedding(2, d_model)  # header / visible state
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout, max_seq_len,
                    causal=False, ffn_activation=ffn_activation, use_rope=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.encoder_ln = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        self.mu_head = MLP(d_model, max(d_model * 2, latent_dim), latent_dim)
        self.logvar_head = MLP(d_model, max(d_model * 2, latent_dim), latent_dim)

        self.position_embedding = nn.Embedding(self.max_state_tokens, d_model)
        self.z_to_decoder = nn.Linear(latent_dim, d_model)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout, self.max_state_tokens,
                    causal=False, ffn_activation=ffn_activation, use_rope=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder_ln = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size + 1)
        self.apply(_init_weights)

    def _check_encoder_len(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"header + state sequence length {seq_len} exceeds V max_seq_len={self.max_seq_len}"
            )

    def _check_state_len(self, seq_len: int) -> None:
        if seq_len > self.max_state_tokens:
            raise ValueError(
                f"state reconstruction length {seq_len} exceeds V max_state_tokens={self.max_state_tokens}"
            )

    def _encode_flat(
        self,
        header_tokens: torch.Tensor,
        state_tokens: torch.Tensor | None,
        header_mask: torch.Tensor,
        state_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # ``header_tokens`` and ``state_tokens`` arrive independently right
        # padded by the collator.  Concatenating their rectangular tensors
        # would put header padding *between* the two real segments for shorter
        # samples.  That changes RoPE positions according to the other rows in
        # the batch, making posterior latents batch-composition dependent.
        # Compact each row into ``[valid header, valid state, right padding]``
        # before the transformer so every real state starts after its own
        # header, regardless of neighbouring sequence lengths.
        width = header_tokens.shape[-1] + (0 if state_tokens is None else state_tokens.shape[-1])
        tokens = torch.full(
            (header_tokens.shape[0], width + 1), self.pad_id,
            dtype=header_tokens.dtype, device=header_tokens.device,
        )
        segment_ids = torch.zeros_like(tokens)
        header_lengths = header_mask.long().sum(dim=-1)
        total_lengths = header_lengths.clone()
        sentinel = torch.full_like(header_tokens, width)
        header_positions = header_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
        header_positions = torch.where(header_mask, header_positions, sentinel)
        tokens.scatter_(1, header_positions, header_tokens)

        if state_tokens is not None:
            assert state_mask is not None
            state_lengths = state_mask.long().sum(dim=-1)
            total_lengths = total_lengths + state_lengths
            state_positions = header_lengths[:, None] + state_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
            state_sentinel = torch.full_like(state_positions, width)
            state_positions = torch.where(state_mask, state_positions, state_sentinel)
            tokens.scatter_(1, state_positions, state_tokens)
            segment_ids.scatter_(1, state_positions, torch.ones_like(state_tokens))

        tokens = tokens[:, :width]
        segment_ids = segment_ids[:, :width]
        valid = torch.arange(width, device=tokens.device)[None, :] < total_lengths[:, None]
        self._check_encoder_len(tokens.shape[-1])
        x = self.token_embedding(tokens) + self.segment_embedding(segment_ids)
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=valid)
        x = self.encoder_ln(x)
        return self.pool(x, valid)

    def encode(
        self,
        header_tokens: torch.Tensor,
        state_tokens: torch.Tensor | None = None,
        *,
        header_valid_mask: torch.Tensor | None = None,
        state_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior parameters for ``header`` or ``header + state``.

        Supplying no state is used only for a cached team-header latent.  V
        training and online state encoding always pass the visible state.
        """
        if header_tokens.ndim < 2:
            raise ValueError("header_tokens must have a sequence dimension")
        if state_tokens is not None and header_tokens.shape[:-1] != state_tokens.shape[:-1]:
            raise ValueError("header and state leading dimensions must match")
        header_mask = _as_valid_mask(header_tokens, header_valid_mask, self.pad_id)
        state_mask = (
            _as_valid_mask(state_tokens, state_valid_mask, self.pad_id)
            if state_tokens is not None
            else None
        )
        flat_header, leading = _flatten_leading(header_tokens)
        flat_header_mask, _ = _flatten_leading(header_mask)
        flat_state = flat_state_mask = None
        if state_tokens is not None:
            flat_state, state_leading = _flatten_leading(state_tokens)
            flat_state_mask, _ = _flatten_leading(state_mask)  # type: ignore[arg-type]
            if state_leading != leading:
                raise ValueError("header and state leading dimensions must match")
        if self.gradient_checkpointing and self.training:
            # checkpoint accepts None poorly on older PyTorch versions; normal
            # state encoding always has a state, header-only cache encoding does
            # not need gradients.
            if flat_state is not None:
                pooled = torch.utils.checkpoint.checkpoint(
                    self._encode_flat,
                    flat_header,
                    flat_state,
                    flat_header_mask,
                    flat_state_mask,
                    use_reentrant=False,
                )
            else:
                pooled = self._encode_flat(flat_header, None, flat_header_mask, None)
        else:
            pooled = self._encode_flat(flat_header, flat_state, flat_header_mask, flat_state_mask)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-12.0, 8.0)
        return _restore_leading(mu, leading), _restore_leading(logvar, leading)

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # Compatibility spelling for code that previously called reparameterize.
    reparameterize = sample

    def _decode_flat(self, z: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        if valid_token_mask.ndim != 2 or valid_token_mask.shape[0] != z.shape[0]:
            raise ValueError("decoder valid_token_mask must be [batch, state_tokens]")
        seq_len = int(valid_token_mask.shape[-1])
        self._check_state_len(seq_len)
        positions = torch.arange(seq_len, device=z.device)
        x = self.position_embedding(positions).unsqueeze(0) + self.z_to_decoder(z).unsqueeze(1)
        # Critical: keep padded positions out of keys in *every* attention
        # block, not only a final pooling operation.
        for block in self.decoder_blocks:
            x = block(x, key_padding_mask=valid_token_mask)
        x = self.decoder_ln(x)
        return self.output_head(x)

    def decode(self, z: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        """Decode a state; a per-sample validity mask is mandatory.

        The explicit mask avoids a subtle padding leak: a valid query in a
        short state must receive identical logits when the batch adds more
        right padding for a longer neighbouring state.
        """
        if valid_token_mask is None:
            raise TypeError("StateVAE.decode requires valid_token_mask")
        if tuple(z.shape[:-1]) != tuple(valid_token_mask.shape[:-1]):
            raise ValueError("z and valid_token_mask leading dimensions must match")
        leading = tuple(z.shape[:-1])
        flat_z = z.reshape(-1, z.shape[-1])
        flat_mask = valid_token_mask.reshape(-1, valid_token_mask.shape[-1]).bool()
        logits = self._decode_flat(flat_z, flat_mask)
        return _restore_leading(logits, leading)

    def forward(
        self,
        header_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        *,
        header_valid_mask: torch.Tensor | None = None,
        state_valid_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_mask = _as_valid_mask(state_tokens, state_valid_mask, self.pad_id)
        mu, logvar = self.encode(
            header_tokens,
            state_tokens,
            header_valid_mask=header_valid_mask,
            state_valid_mask=state_mask,
        )
        z = self.sample(mu, logvar, deterministic=deterministic)
        return {
            "logits": self.decode(z, state_mask),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "state_valid_mask": state_mask,
        }


def interleave_latent_history(
    team_z: torch.Tensor,
    state_tokens: torch.Tensor,
    own_action_tokens: torch.Tensor,
    opponent_action_tokens: torch.Tensor,
    state_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interleave latent blocks as ``team, z0, own0, opp0, z1, ...``.

    ``state_tokens`` contains states through the current state.  Its right
    padding is represented by ``state_valid_mask``; action entries beyond the
    last valid state are never exposed to the causal Transformer.
    """
    if team_z.ndim != 2 or state_tokens.ndim != 3:
        raise ValueError("team_z must be [B,D] and state_tokens must be [B,S,D]")
    batch, max_states, dim = state_tokens.shape
    if team_z.shape != (batch, dim):
        raise ValueError("team_z and state_tokens latent dimensions must match")
    if own_action_tokens.shape[:2] != (batch, max(max_states - 1, 0)):
        raise ValueError("own action sequence must contain one entry per state transition")
    if opponent_action_tokens.shape != own_action_tokens.shape:
        raise ValueError("own and opponent action sequence shapes must match")
    if state_valid_mask is None:
        state_valid_mask = torch.ones(batch, max_states, dtype=torch.bool, device=state_tokens.device)
    else:
        state_valid_mask = state_valid_mask.bool().to(state_tokens.device)
    if not bool(state_valid_mask[:, 0].all()):
        raise ValueError("each latent history must contain at least state_0")

    max_len = 3 * max_states - 1
    out = state_tokens.new_zeros(batch, max_len, dim)
    valid = torch.zeros(batch, max_len, dtype=torch.bool, device=state_tokens.device)
    type_ids = torch.zeros(batch, max_len, dtype=torch.long, device=state_tokens.device)
    # A small loop over batch items is intentional: it makes the actual causal
    # layout obvious and avoids accidentally treating padded transitions as
    # observed history. Context is at most 32 transitions by default.
    for row in range(batch):
        n_states = int(state_valid_mask[row].sum().item())
        cursor = 0
        out[row, cursor] = team_z[row]
        valid[row, cursor] = True
        type_ids[row, cursor] = 0
        cursor += 1
        for state_idx in range(n_states):
            out[row, cursor] = state_tokens[row, state_idx]
            valid[row, cursor] = True
            type_ids[row, cursor] = 1
            cursor += 1
            if state_idx < n_states - 1:
                out[row, cursor] = own_action_tokens[row, state_idx]
                valid[row, cursor] = True
                type_ids[row, cursor] = 2
                cursor += 1
                out[row, cursor] = opponent_action_tokens[row, state_idx]
                valid[row, cursor] = True
                type_ids[row, cursor] = 3
                cursor += 1
    return out, valid, type_ids


class CausalLatentTransformer(nn.Module):
    """Causal temporal model over compact state/action latent blocks."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.0,
        max_context_transitions: int = 32,
        ffn_activation: str = "gelu",
        gradient_checkpointing: bool = False,
        **_: Any,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.max_context_transitions = int(max_context_transitions)
        self.max_seq_len = 3 * self.max_context_transitions + 2
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.type_embedding = nn.Embedding(4, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout, self.max_seq_len,
                    causal=True, ffn_activation=ffn_activation, use_rope=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.apply(_init_weights)

    def _forward_embedded(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_seq_len:
            raise ValueError(
                f"latent history length {x.shape[1]} exceeds M max_seq_len={self.max_seq_len}; "
                "reduce max_context_transitions"
            )
        for block in self.blocks:
            x = block(x, key_padding_mask=valid_mask)
        return self.ln_final(x)

    def forward(
        self,
        team_z: torch.Tensor,
        state_z: torch.Tensor,
        own_action_embeddings: torch.Tensor,
        opponent_action_embeddings: torch.Tensor,
        state_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent_blocks = self.latent_proj(torch.cat([team_z.unsqueeze(1), state_z], dim=1))
        # Build separate projected tensors then use the pure interleaver.  The
        # first latent block is team_z; remaining blocks are current/prior z.
        projected_team = latent_blocks[:, 0]
        projected_states = latent_blocks[:, 1:]
        projected_own = self.action_proj(own_action_embeddings)
        projected_opp = self.action_proj(opponent_action_embeddings)
        tokens, valid_mask, type_ids = interleave_latent_history(
            projected_team,
            projected_states,
            projected_own,
            projected_opp,
            state_valid_mask,
        )
        x = tokens + self.type_embedding(type_ids)
        if self.gradient_checkpointing and self.training:
            hidden = torch.utils.checkpoint.checkpoint(self._forward_embedded, x, valid_mask, use_reentrant=False)
        else:
            hidden = self._forward_embedded(x, valid_mask)
        last_idx = valid_mask.long().sum(dim=-1).sub(1)
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
        return {"h": h, "hidden": hidden, "valid_mask": valid_mask, "type_ids": type_ids}


class TransitionMDNHead(nn.Module):
    """Diagonal Gaussian mixture for ``p(z[t+1] | h[t], own, opponent)``."""

    def __init__(self, *, h_dim: int, action_dim: int, latent_dim: int, num_mixtures: int = 5):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_mixtures = int(num_mixtures)
        hidden = max(h_dim, action_dim * 2)
        self.trunk = nn.Sequential(
            nn.Linear(h_dim + action_dim * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.mixture_logits = nn.Linear(hidden, num_mixtures)
        self.mixture_means = nn.Linear(hidden, num_mixtures * latent_dim)
        self.mixture_log_scales = nn.Linear(hidden, num_mixtures * latent_dim)
        self.apply(_init_weights)

    def forward(self, h: torch.Tensor, own_action: torch.Tensor, opponent_action: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.trunk(torch.cat([h, own_action, opponent_action], dim=-1))
        batch = h.shape[0]
        return {
            "mixture_logits": self.mixture_logits(x),
            "mixture_means": self.mixture_means(x).view(batch, self.num_mixtures, self.latent_dim),
            "mixture_log_scales": self.mixture_log_scales(x)
            .view(batch, self.num_mixtures, self.latent_dim)
            .clamp(-7.0, 3.0),
        }

    @torch.no_grad()
    def sample(self, params: dict[str, torch.Tensor], *, deterministic: bool = False) -> torch.Tensor:
        logits = params["mixture_logits"]
        means = params["mixture_means"]
        log_scales = params["mixture_log_scales"]
        component = logits.argmax(dim=-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        gather = component[:, None, None].expand(-1, 1, self.latent_dim)
        mean = means.gather(1, gather).squeeze(1)
        if deterministic:
            return mean
        scale = log_scales.gather(1, gather).squeeze(1).exp()
        return mean + torch.randn_like(mean) * scale


class LegalActionController(nn.Module):
    """Behavior-cloning prior over the currently legal own action IDs."""

    def __init__(self, *, latent_dim: int, h_dim: int, action_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(latent_dim + h_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(_init_weights)

    def forward(self, z_t: torch.Tensor, h_t: torch.Tensor, legal_action_embeddings: torch.Tensor) -> torch.Tensor:
        q = self.query(torch.cat([z_t, h_t], dim=-1))
        return torch.einsum("bd,bld->bl", q, legal_action_embeddings)


class SimpleWorldModel(nn.Module):
    """Container for a current-state VAE, causal M, and legal-action C."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        action_vocab_size: int = 3,
        latent_dim: int = 128,
        v_cfg: dict[str, Any] | None = None,
        action_encoder_cfg: dict[str, Any] | None = None,
        m_cfg: dict[str, Any] | None = None,
        controller_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        v_cfg = dict(v_cfg or {})
        # action_encoder_cfg is retained as a config name so older config files
        # can be upgraded in place.  It now configures categorical embeddings.
        action_cfg = dict(action_encoder_cfg or {})
        m_cfg = dict(m_cfg or {})
        controller_cfg = dict(controller_cfg or {})
        self.latent_dim = int(latent_dim)
        self.pad_id = int(pad_id)
        self.action_dim = int(action_cfg.pop("action_dim", 128))
        self.v = StateVAE(vocab_size=vocab_size, pad_id=pad_id, latent_dim=latent_dim, **v_cfg)
        self.action_embedding = nn.Embedding(action_vocab_size, self.action_dim, padding_idx=0)
        self.m = CausalLatentTransformer(
            latent_dim=latent_dim,
            action_dim=self.action_dim,
            **m_cfg,
        )
        h_dim = self.m.d_model
        self.opponent_head = nn.Sequential(
            nn.Linear(h_dim + self.action_dim, h_dim), nn.GELU(), nn.Linear(h_dim, action_vocab_size)
        )
        self.transition_head = TransitionMDNHead(
            h_dim=h_dim,
            action_dim=self.action_dim,
            latent_dim=latent_dim,
            num_mixtures=int(m_cfg.get("num_mixtures", 5)),
        )
        self.done_head = nn.Linear(h_dim, 1)
        self.value_head = nn.Linear(h_dim, NUM_OUTCOME_CLASSES)
        self.c = LegalActionController(
            latent_dim=latent_dim,
            h_dim=h_dim,
            action_dim=self.action_dim,
            **controller_cfg,
        )
        self.apply(_init_weights)

    def encode_state(
        self,
        header_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        *,
        header_valid_mask: torch.Tensor | None = None,
        state_valid_mask: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        return self.v(
            header_tokens,
            state_tokens,
            header_valid_mask=header_valid_mask,
            state_valid_mask=state_valid_mask,
            deterministic=deterministic,
        )

    def _actions(self, ids: torch.Tensor) -> torch.Tensor:
        return self.action_embedding(ids.long())

    def encode_history(
        self,
        team_z: torch.Tensor,
        state_z: torch.Tensor,
        own_action_ids: torch.Tensor,
        opponent_action_ids: torch.Tensor,
        state_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.m(
            team_z,
            state_z,
            self._actions(own_action_ids),
            self._actions(opponent_action_ids),
            state_valid_mask,
        )

    def forward_m(
        self,
        *,
        team_z: torch.Tensor,
        state_z: torch.Tensor,
        own_history_action_ids: torch.Tensor,
        opponent_history_action_ids: torch.Tensor,
        current_own_action_ids: torch.Tensor,
        current_opponent_action_ids: torch.Tensor,
        state_valid_mask: torch.Tensor | None = None,
        action_logit_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        history = self.encode_history(
            team_z, state_z, own_history_action_ids, opponent_history_action_ids, state_valid_mask
        )
        h = history["h"]
        own = self._actions(current_own_action_ids)
        opponent = self._actions(current_opponent_action_ids)
        opponent_logits = self.opponent_head(torch.cat([h, own], dim=-1))
        if action_logit_mask is not None:
            opponent_logits = opponent_logits.masked_fill(~action_logit_mask.bool(), torch.finfo(opponent_logits.dtype).min)
        transition = self.transition_head(h, own, opponent)
        return {
            **history,
            **transition,
            "opponent_logits": opponent_logits,
            "done_logits": self.done_head(h).squeeze(-1),
            "value_logits": self.value_head(h),
            "z_t": state_z[
                torch.arange(state_z.shape[0], device=state_z.device),
                (state_valid_mask if state_valid_mask is not None else torch.ones(state_z.shape[:2], device=state_z.device, dtype=torch.bool))
                .long()
                .sum(dim=-1)
                .sub(1),
            ],
        }

    def forward_c(
        self,
        *,
        z_t: torch.Tensor,
        h_t: torch.Tensor,
        legal_action_ids: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.c(z_t, h_t, self._actions(legal_action_ids))
        if legal_action_mask is not None:
            logits = logits.masked_fill(~legal_action_mask.bool(), torch.finfo(logits.dtype).min)
        return {"controller_logits": logits}


def mdn_nll(
    target: torch.Tensor,
    mixture_logits: torch.Tensor,
    mixture_means: torch.Tensor,
    mixture_log_scales: torch.Tensor,
) -> torch.Tensor:
    target = target.unsqueeze(1)
    standardized = (target - mixture_means) * torch.exp(-mixture_log_scales)
    log_prob = (
        -0.5 * standardized.square() - mixture_log_scales - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    return -torch.logsumexp(F.log_softmax(mixture_logits, dim=-1) + log_prob, dim=-1).mean()


def _masked_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if not bool(valid_mask.any()):
        return logits.new_zeros(())
    return logits.argmax(dim=-1)[valid_mask].eq(targets[valid_mask]).float().mean()


def vae_losses(
    outputs: dict[str, torch.Tensor],
    state_targets: torch.Tensor,
    *,
    beta_kl: float,
    free_bits: float = 0.0,
    capacity: float = 0.0,
    capacity_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = outputs["state_valid_mask"].bool()
    ce = F.cross_entropy(outputs["logits"].reshape(-1, outputs["logits"].shape[-1]), state_targets.reshape(-1), reduction="none")
    recon_ce = ce[valid.reshape(-1)].mean() if bool(valid.any()) else ce.mean() * 0.0
    kl_per_dim = -0.5 * (1.0 + outputs["logvar"] - outputs["mu"].square() - outputs["logvar"].exp())
    raw_kl = kl_per_dim.sum(dim=-1).mean()
    free_kl = kl_per_dim.clamp_min(float(free_bits)).sum(dim=-1).mean()
    # Capacity is expressed per latent dimension.  It can be ramped by the
    # trainer; free bits remains active even when no capacity target is used.
    capacity_target = float(capacity) * outputs["mu"].shape[-1]
    capacity_term = (raw_kl - capacity_target).abs() if capacity_weight else raw_kl.new_zeros(())
    loss = recon_ce + float(beta_kl) * free_kl + float(capacity_weight) * capacity_term
    return loss, {
        "loss": float(loss.detach()),
        "recon_ce": float(recon_ce.detach()),
        "recon_token_acc": float(_masked_token_accuracy(outputs["logits"], state_targets, valid).detach()),
        "kl": float(raw_kl.detach()),
        "kl_per_dim": float((raw_kl / outputs["mu"].shape[-1]).detach()),
        "free_kl": float(free_kl.detach()),
        "beta_kl": float(beta_kl),
        "weighted_kl": float((float(beta_kl) * free_kl).detach()),
        "capacity_term": float(capacity_term.detach()),
        "z_norm": float(outputs["mu"].detach().float().norm(dim=-1).mean()),
    }


def _binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Mann–Whitney AUROC without a sklearn dependency."""
    scores = scores.detach().float().reshape(-1)
    targets = targets.detach().bool().reshape(-1)
    positives = int(targets.sum())
    negatives = int((~targets).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float)
    return float(((ranks[targets].sum() - positives * (positives + 1) / 2) / (positives * negatives)).cpu())


def _binary_pr_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Area under the precision/recall curve with a deterministic threshold sweep."""
    scores = scores.detach().float().reshape(-1)
    targets = targets.detach().bool().reshape(-1)
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    ranked = targets[order].float()
    tp = ranked.cumsum(0)
    precision = tp / torch.arange(1, len(ranked) + 1, device=scores.device)
    recall = tp / positives
    recall = torch.cat([recall.new_zeros(1), recall])
    precision = torch.cat([precision[:1], precision])
    return float(((recall[1:] - recall[:-1]) * precision[1:]).sum().cpu())


def m_losses(
    outputs: dict[str, torch.Tensor],
    *,
    next_z: torch.Tensor,
    opponent_action_ids: torch.Tensor,
    done_targets: torch.Tensor,
    outcome_targets: torch.Tensor,
    lambda_opponent: float = 1.0,
    lambda_mdn: float = 1.0,
    lambda_done: float = 0.25,
    lambda_value: float = 0.25,
    done_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    opp_ce = F.cross_entropy(outputs["opponent_logits"], opponent_action_ids.long())
    nll = mdn_nll(next_z, outputs["mixture_logits"], outputs["mixture_means"], outputs["mixture_log_scales"])
    pos_weight = outputs["done_logits"].new_tensor(float(done_pos_weight))
    done_bce = F.binary_cross_entropy_with_logits(outputs["done_logits"], done_targets.float(), pos_weight=pos_weight)
    value_ce = F.cross_entropy(outputs["value_logits"], outcome_targets.long())
    loss = lambda_opponent * opp_ce + lambda_mdn * nll + lambda_done * done_bce + lambda_value * value_ce
    with torch.no_grad():
        opp_rank = outputs["opponent_logits"].topk(min(5, outputs["opponent_logits"].shape[-1]), dim=-1).indices
        target = opponent_action_ids[:, None]
        probs = F.softmax(outputs["value_logits"].float(), dim=-1)
        one_hot = F.one_hot(outcome_targets.long(), NUM_OUTCOME_CLASSES).float()
        done_prob = torch.sigmoid(outputs["done_logits"])
    return loss, {
        "loss": float(loss.detach()),
        "opponent_ce": float(opp_ce.detach()),
        "opponent_top1": float(opp_rank[:, :1].eq(target).any(dim=-1).float().mean()),
        "opponent_top5": float(opp_rank.eq(target).any(dim=-1).float().mean()),
        "mdn_nll": float(nll.detach()),
        "done_bce": float(done_bce.detach()),
        "done_auroc": _binary_auc(done_prob, done_targets),
        "done_pr_auc": _binary_pr_auc(done_prob, done_targets),
        "value_ce": float(value_ce.detach()),
        "value_brier": float((probs - one_hot).square().sum(dim=-1).mean()),
    }


def c_losses(
    logits: torch.Tensor,
    *,
    legal_action_ids: torch.Tensor,
    legal_action_mask: torch.Tensor,
    chosen_legal_action_idx: torch.Tensor,
    controller_eligible: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Legal-action CE, excluding `none` / unknown replay targets from C."""
    chosen_ids = legal_action_ids.gather(1, chosen_legal_action_idx[:, None]).squeeze(1)
    valid_choice = legal_action_mask.gather(1, chosen_legal_action_idx[:, None]).squeeze(1)
    eligible = controller_eligible[chosen_ids] & valid_choice
    if not bool(eligible.any()):
        zero = logits.sum() * 0.0
        return zero, {"loss": 0.0, "controller_count": 0.0, "controller_acc": 0.0, "controller_top3": 0.0}, eligible
    selected_logits = logits[eligible]
    selected_target = chosen_legal_action_idx[eligible]
    loss = F.cross_entropy(selected_logits, selected_target)
    topk = selected_logits.topk(min(3, selected_logits.shape[-1]), dim=-1).indices
    return loss, {
        "loss": float(loss.detach()),
        "controller_count": float(eligible.sum()),
        "controller_acc": float(selected_logits.argmax(dim=-1).eq(selected_target).float().mean()),
        "controller_top3": float(topk.eq(selected_target[:, None]).any(dim=-1).float().mean()),
        "legal_action_count": float(legal_action_mask[eligible].sum(dim=-1).float().mean()),
    }, eligible


# Kept as a clear failure rather than silently running a stale objective.
def compute_simple_world_model_losses(*_: Any, **__: Any) -> tuple[torch.Tensor, dict[str, float]]:
    raise RuntimeError(
        "The old combined simple-world-model objective was removed. "
        "Use vae_losses(), m_losses(), or c_losses() for --stage v/m/c."
    )
