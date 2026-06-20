"""LeJEPA (Latent-Euclidean JEPA) model for paired-POV world-model learning.

Architecture overview (v4 — paired current-state belief + next-state rollout)
-----------------------------------------------------------------------------

    POV state/header blocks through T ─► JEPAEncoder per block ─┐
    Historical player actions ────────► JEPAActionEncoder ──────┤
    Historical opponent actions ──────► JEPAActionEncoder ──────┤
                                                                ▼
    [team, state₀, p_action₀, o_action₀, state₁, ... state_{T-1}]
                                                                │
                                                                ▼
                                            JEPATemporalEncoder ──► history ctx_T

    Current visible state block T ─► JEPAEncoder ─► z_T
    Next visible state block T+1 ──► JEPAEncoder ─► z_{T+1} target

    history ctx_T + z_T ─► JEPAOpponentBeliefPredictor shared backbone
                         ├─ state head  ─► pred_opp_state_mu/logvar
                         └─ action head ─► pred_opp_action_mu/logvar

    Current player action ───► JEPAActionEncoder ─► own_action
    Current opponent action ─► JEPAActionEncoder ─► opponent_action target

    z_T + own_action + sampled opponent state/action beliefs
        ─► JEPANextStatePredictor ─► pred_z_{T+1}_mu/logvar

    ┌────────────────────────────────────────────────────────────────────┐
    │ LOSSES:                                                             │
    │   MSE(pred_opp_state_mu, z_opp_T)      — opponent state mean        │
    │   MSE(pred_opp_action_mu, action_opp)  — opponent action mean       │
    │   MSE(pred_z_{T+1}_mu, z_{T+1})        — next current-state latent  │
    │   Bradley-Terry rank loss from battle outcome                       │
    │   SIGReg on current, next-target, and history-context state latents │
    │   Optional SIGReg on true action encoder outputs                    │
    └────────────────────────────────────────────────────────────────────┘

Modules:

1. **JEPAEncoder φ** — bidirectional transformer over one team-header or state
   block. Attention pools over non-pad tokens → state/header block embedding.

2. **JEPAActionEncoder ψ** — smaller bidirectional transformer over action text
   (e.g. "<chosen_move>blizzard<end_chosen_move>").  Shares the token embedding
   matrix with JEPAEncoder.  Attention pool → MLP → action_latent_dim.

3. **JEPATemporalEncoder τ** — transformer over interleaved historical block
   embeddings.  The current state block is dropped from this history before
   temporal encoding, so the output is a prior context rather than a
   next-state target.

4. **JEPAOpponentBeliefPredictor β** — shared MLP backbone over
   (history_context, current_state_z), with separate Gaussian heads for the
   opponent current-state latent and opponent next-action latent.

5. **JEPANextStatePredictor μ** — diagonal-Gaussian MLP:
   (current_state_z, own_action, sampled/predicted opponent state,
   sampled/predicted opponent action) → next current-state latent.

Losses
------

*Next-state MSE* — MSE between the target next current-state block latent and
the next-state predictor mean::

    L_next = || z_{T+1} - pred_z_{T+1}_mu ||²

*Opponent state MSE* — MSE on predicted mean vs target opponent latent::

    L_os = || z_opp_T - pred_opp_state_mu ||²

*Action MSE* — MSE on predicted mean vs target action latent::

    L_oa = || actual_opp_action - pred_opp_action_mu ||²

*SIGReg* — on current-state encoder outputs, next-state target encoder
outputs, and history-context outputs.  Action SIGReg is configurable and is off
by default; predicted Gaussian samples are not regularized.

*Rank loss* — Bradley-Terry/logistic loss from the p1 outcome label, evaluated
on true current-state pairs, current belief pairs, and next-state belief pairs.

Total::

    L = λ_os L_os + λ_oa L_oa + λ_next L_next + λ_rank L_rank
        + λ_sigreg_state L_sigreg_state + λ_sigreg_action L_sigreg_action
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Context length constants ──────────────────────────────────────────────
# Maximum token count for an individual header/state block when no config
# overrides the encoder. Full battle length is handled by the temporal encoder
# over block embeddings, not token-level attention over a flat prefix.
CONTEXT_LENGTH: int = 2048

# Latent dimension for the main encoder (size of the deterministic embedding e).
LATENT_DIM: int = 192

# Latent dimension for action embeddings (output of JEPAActionEncoder).
ACTION_LATENT_DIM: int = 32

# ── SIGReg defaults ──────────────────────────────────────────────────────
# Number of random projection directions for sketching (resampled each step).
SIGREG_NUM_SLICES: int = 128
# Number of trapezoidal quadrature points for Epps-Pulley integration.
SIGREG_NUM_POINTS: int = 17
# Integration domain for the characteristic function: [0, domain].
SIGREG_DOMAIN: float = 3.0

_SIGREG_GRID_CACHE: dict[tuple[str, int, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


# ═════════════════════════════════════════════════════════════════════════
# Building blocks
# ═════════════════════════════════════════════════════════════════════════


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al. 2021) applied per-head to Q and K."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        S = x.shape[-2]
        dtype = x.dtype
        if not hasattr(self, "_cos_bf16"):
            self._cos_bf16 = {}
            self._sin_bf16 = {}
        if dtype not in self._cos_bf16:
            self._cos_bf16[dtype] = self.cos_cached.to(dtype=dtype)
            self._sin_bf16[dtype] = self.sin_cached.to(dtype=dtype)
        cos = self._cos_bf16[dtype][offset : offset + S]
        sin = self._sin_bf16[dtype][offset : offset + S]
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE.

    Supports both causal (autoregressive) and bidirectional (encoder)
    modes via the ``causal`` flag.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        causal: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_head, max_seq_len=max_seq_len) if use_rope else None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA bool masks use True for positions that participate in attention.
            # Shape broadcasts across query positions and attention heads.
            attn_mask = key_padding_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with attention + FFN (SwiGLU or GELU)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        causal: bool = True,
        ffn_activation: str = "gelu",
        use_rope: bool = True,
    ):
        super().__init__()
        self.ffn_activation = ffn_activation
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(
            d_model, n_heads, dropout, max_seq_len, causal=causal, use_rope=use_rope,
        )
        self.ln2 = nn.LayerNorm(d_model)
        if ffn_activation == "swiglu":
            self.ffn_w1 = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_w2 = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_out = nn.Linear(d_ff, d_model, bias=False)
        elif ffn_activation == "gelu":
            self.ffn = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_out = nn.Linear(d_ff, d_model, bias=False)
        else:
            raise ValueError(f"Unknown ffn_activation: {ffn_activation}")
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.dropout(self._ffn(self.ln2(x)))
        return x

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_activation == "swiglu":
            gate = F.silu(self.ffn_w1(x))
            up = self.ffn_w2(x)
            return self.ffn_out(gate * up)
        else:
            return self.ffn_out(F.gelu(self.ffn(x)))



# ═════════════════════════════════════════════════════════════════════════
# Shared modules
# ═════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Two-layer MLP with LayerNorm + GELU, used as projector / pred_proj."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Learned masked attention pooling over token states."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)  # (B, S)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# ═════════════════════════════════════════════════════════════════════════
# JEPAEncoder — bidirectional transformer over one header/state block
# ═════════════════════════════════════════════════════════════════════════

class JEPAEncoder(nn.Module):
    """Transformer encoder that maps one state/header token block to an embedding.

    Returns:
      - ``e``: (B, latent_dim) — deterministic embedding, used as input /
               target for the JEPA predictor and regularised via SIGReg.

    Architecture
    ------------
    ``token_ids`` → shared token embedding →
    N × transformer blocks (bidirectional) →
    attention pool over non-pad positions →
    MLP projector → e.

    Parameters
    ----------
    vocab_size : int
        State vocabulary size.
    pad_id : int
        Token ID for padding (embedding vector is zeroed).
    latent_dim : int
        Dimensionality of the output embedding.
    gradient_checkpointing : bool
        If True, wrap each transformer block with
        ``torch.utils.checkpoint.checkpoint`` to trade compute for memory.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, theta, ffn_activation :
        Standard transformer hyperparameters.
    proj_hidden_dim : int or None
        Hidden dimension for the projector MLP.  Defaults to 4× d_model.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        latent_dim: int = LATENT_DIM,
        gradient_checkpointing: bool = False,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        ffn_activation: str = "gelu",
        proj_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(
            vocab_size + 1, d_model, padding_idx=pad_id
        )  # +1 for unused index 0 (token IDs are 1-based)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=False,  # bidirectional for encoding
                ffn_activation=ffn_activation,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)

        # MLP projector: pooled representation → deterministic embedding.
        proj_hidden = proj_hidden_dim or (4 * d_model)
        self.proj_e = MLP(d_model, proj_hidden, latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx] = 0.0

    def forward(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode a token sequence.

        Args:
            token_ids: (B, S) int — state/header block token IDs.

        Returns:
            e: (B, latent_dim) — deterministic embedding.
        """
        valid_mask = token_ids != self.pad_id  # (B, S), True for real tokens

        # Token embeddings
        x = self.token_embedding(token_ids)  # (B, S, d_model)

        # Transformer blocks (bidirectional).
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, valid_mask, use_reentrant=False
                )
            else:
                x = block(x, key_padding_mask=valid_mask)
        x = self.ln_final(x)  # (B, S, d_model)

        # Learned attention pool over non-pad positions.
        pooled = self.pool(x, valid_mask)  # (B, d_model)

        return self.proj_e(pooled)  # (B, latent_dim)


# ═════════════════════════════════════════════════════════════════════════
# JEPAActionEncoder — smaller bidirectional transformer → action embedding
# ═════════════════════════════════════════════════════════════════════════

class JEPAActionEncoder(nn.Module):
    """Small bidirectional transformer that encodes action text to a fixed-size
    action embedding.

    Action text includes structural delimiters:
      - Player:   ``<chosen_move> blizzard <end_chosen_move>``
      - Opponent: ``<opponent_chosen_move> lovelykiss <end_opponent_chosen_move>``

    Shares the token embedding matrix with ``JEPAEncoder`` via an input
    projection from the main encoder's ``d_model`` to a (typically smaller)
    internal dimension.

    Uses **learned positional embeddings** (not RoPE) since action sequences
    are very short (2–4 tokens).

    Architecture
    ------------
    ``token_ids`` → shared token embedding → Linear(d_model_enc → d_model) →
    + learned positional embedding →
    N × transformer blocks (bidirectional, no RoPE) →
    attention pool over non-pad positions →
    MLP projector → action_latent_dim.

    Parameters
    ----------
    token_embedding : nn.Embedding
        Shared token embedding from the main JEPAEncoder.
    pad_id : int
        Token ID for padding.
    action_latent_dim : int
        Dimensionality of the output action embedding.
    gradient_checkpointing : bool
        If True, wrap each transformer block with checkpoint.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, theta, ffn_activation :
        Transformer hyperparameters for this smaller encoder.
    encoder_d_model : int
        Dimensionality of the shared token embedding (from JEPAEncoder).
    """

    def __init__(
        self,
        token_embedding: nn.Embedding,
        pad_id: int,
        action_latent_dim: int = ACTION_LATENT_DIM,
        encoder_d_model: int = 512,
        gradient_checkpointing: bool = False,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        theta: float = 10000.0,
        ffn_activation: str = "gelu",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.action_latent_dim = action_latent_dim
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing

        # Share the token embedding and project down to our internal dim.
        self.token_embedding = token_embedding
        self.input_proj = nn.Linear(encoder_d_model, d_model, bias=False)

        # Learned positional embedding (not RoPE — actions are too short to benefit).
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=False,  # bidirectional
                ffn_activation=ffn_activation,
                use_rope=False,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)

        # MLP projector: pooled → action_latent_dim.
        proj_hidden = 4 * d_model
        self.proj = MLP(d_model, proj_hidden, action_latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx] = 0.0

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode action text.

        Args:
            token_ids: (B, S) int — action token IDs with delimiters,
                       padded with ``pad_id``.

        Returns:
            emb: (B, action_latent_dim) — action embedding.
        """
        valid_mask = token_ids != self.pad_id  # (B, S)
        S = token_ids.shape[1]

        # Shared token embedding → project down to our d_model.
        x = self.token_embedding(token_ids)        # (B, S, encoder_d_model)
        x = self.input_proj(x)                      # (B, S, d_model)

        # Add learned positional embedding.
        x = x + self.pos_embedding[:, :S, :]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, valid_mask, use_reentrant=False
                )
            else:
                x = block(x, key_padding_mask=valid_mask)
        x = self.ln_final(x)

        pooled = self.pool(x, valid_mask)  # (B, d_model)
        return self.proj(pooled)            # (B, action_latent_dim)


# ═════════════════════════════════════════════════════════════════════════
# JEPATemporalEncoder — interleaved historical block embeddings → context
# ═════════════════════════════════════════════════════════════════════════

class JEPATemporalEncoder(nn.Module):
    """Transformer over state/header and side-specific action block embeddings."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_latent_dim: int = ACTION_LATENT_DIM,
        gradient_checkpointing: bool = False,
        n_heads: int = 6,
        n_layers: int = 4,
        d_ff: int = 768,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        ffn_activation: str = "gelu",
        proj_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gradient_checkpointing = gradient_checkpointing

        self.action_proj = nn.Linear(action_latent_dim, latent_dim, bias=False)
        self.type_embedding = nn.Embedding(3, latent_dim)  # state, player action, opponent action
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                latent_dim, n_heads, d_ff, dropout, max_seq_len,
                causal=False,
                ffn_activation=ffn_activation,
                use_rope=False,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(latent_dim)
        self.pool = AttentionPool(latent_dim)
        proj_hidden = proj_hidden_dim or (4 * latent_dim)
        self.proj_e = MLP(latent_dim, proj_hidden, latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        state_embs: torch.Tensor,
        state_valid: torch.Tensor,
        player_action_embs: torch.Tensor,
        player_action_valid: torch.Tensor,
        opponent_action_embs: torch.Tensor,
        opponent_action_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an interleaved block history.

        Args:
            state_embs: (B, S, latent_dim), including team header at index 0.
            state_valid: (B, S)
            player_action_embs: (B, A, action_latent_dim)
            player_action_valid: (B, A)
            opponent_action_embs: (B, A, action_latent_dim)
            opponent_action_valid: (B, A)
        """
        B, S, D = state_embs.shape
        A = player_action_embs.shape[1]
        max_seq = max(1 + 3 * max(S - 1, A), 1)
        if max_seq > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Temporal sequence length {max_seq} exceeds max_seq_len "
                f"{self.pos_embedding.shape[1]}"
            )

        x = state_embs.new_zeros((B, max_seq, D))
        valid = torch.zeros((B, max_seq), device=state_embs.device, dtype=torch.bool)
        type_ids = torch.zeros((B, max_seq), device=state_embs.device, dtype=torch.long)

        for i in range(S):
            # state_embs layout is [team_header, state_0, state_1, ...].
            # Temporal layout is:
            #   header, state_0, p_action_0, o_action_0, state_1, ...
            pos = 0 if i == 0 else 1 + 3 * (i - 1)
            x[:, pos, :] = state_embs[:, i, :]
            valid[:, pos] = state_valid[:, i]
            type_ids[:, pos] = 0

            action_idx = i - 1
            if i >= 1 and action_idx < A:
                pa_pos = pos + 1
                oa_pos = pos + 2
                x[:, pa_pos, :] = self.action_proj(player_action_embs[:, action_idx, :])
                x[:, oa_pos, :] = self.action_proj(opponent_action_embs[:, action_idx, :])
                valid[:, pa_pos] = player_action_valid[:, action_idx]
                valid[:, oa_pos] = opponent_action_valid[:, action_idx]
                type_ids[:, pa_pos] = 1
                type_ids[:, oa_pos] = 2

        x = x + self.type_embedding(type_ids) + self.pos_embedding[:, :max_seq, :]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, valid, use_reentrant=False
                )
            else:
                x = block(x, key_padding_mask=valid)
        x = self.ln_final(x)
        pooled = self.pool(x, valid)
        return self.proj_e(pooled)


class JEPAOpponentBeliefPredictor(nn.Module):
    """Shared-backbone predictor for opponent state AND opponent action beliefs.

    Takes (history_context, current_state) through a shared MLP backbone,
    then forks into two heads:

      - State head:  diagonal Gaussian over opponent current-state latent
      - Action head: diagonal Gaussian over opponent next-action latent

    Both means are trained via MSE against their respective targets; the
    variances are trained only via downstream gradients through reparameterized
    samples (and, for the action head, through the next-state predictor).

    Merging replaces the separate ``JEPAOpponentStatePredictor`` and
    ``JEPAPairedActionPredictor`` with a single module that learns a shared
    representation of the opponent from the observable history + current board.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_latent_dim: int = ACTION_LATENT_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 4,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (4 * latent_dim)

        # ── Shared backbone ──
        in_dim = 2 * latent_dim  # [history_context, current_state]
        backbone_layers: list[nn.Module] = []
        for i in range(n_layers - 1):
            backbone_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)

        # ── State head ──
        self.state_head = nn.Linear(hidden_dim, 2 * latent_dim)

        # ── Action head ──
        self.action_head = nn.Linear(hidden_dim, 2 * action_latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        history_context: torch.Tensor,
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (state_mu, state_logvar, action_mu, action_logvar)."""
        shared = self.backbone(torch.cat([history_context, current_state], dim=-1))
        state_mu, state_logvar = self.state_head(shared).chunk(2, dim=-1)
        action_mu, action_logvar = self.action_head(shared).chunk(2, dim=-1)
        return state_mu, state_logvar, action_mu, action_logvar


class JEPANextStatePredictor(nn.Module):
    """Diagonal-Gaussian predictor for the next self-POV state latent.

    Takes current self-state, own action, predicted opponent state, and
    predicted opponent action — all of which feed through a small MLP that
    outputs mu and logvar for the next-state latent.

    The world is stochastic (damage rolls, status effects, speed ties, …),
    so a deterministic predictor would be misspecified.  The mean is trained
    via MSE against the target encoder latent; the variance is trained only
    via downstream gradients through reparameterized samples.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_latent_dim: int = ACTION_LATENT_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 4,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (4 * latent_dim)
        layers: list[nn.Module] = []
        in_dim = 2 * latent_dim + 2 * action_latent_dim
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ])
                in_dim = hidden_dim
            else:
                layers.append(nn.Linear(in_dim, 2 * latent_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        current_state: torch.Tensor,
        own_action: torch.Tensor,
        opponent_state: torch.Tensor,
        opponent_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(torch.cat([
            current_state,
            own_action,
            opponent_state,
            opponent_action,
        ], dim=-1))
        mu, logvar = x.chunk(2, dim=-1)
        return mu, logvar


class JEPAPairwiseRankHead(nn.Module):
    """Predict relative advantage for ``self`` given a paired opponent latent."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 3,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (4 * latent_dim)
        layers: list[nn.Module] = []
        in_dim = 4 * latent_dim
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ])
                in_dim = hidden_dim
            else:
                layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, self_state: torch.Tensor, opponent_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self_state,
            opponent_state,
            self_state - opponent_state,
            self_state * opponent_state,
        ], dim=-1)
        return self.net(x).squeeze(-1)


class PairedJEPAModel(nn.Module):
    """JEPA trained from both synchronized player perspectives of one battle.

    For each transition it encodes both POV histories through state T, predicts
    each hidden opponent POV from the visible POV, predicts the opponent's next
    action, and then predicts each POV's next state latent.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        latent_dim: int = LATENT_DIM,
        action_latent_dim: int = ACTION_LATENT_DIM,
        encoder_cfg: Optional[dict] = None,
        temporal_encoder_cfg: Optional[dict] = None,
        action_encoder_cfg: Optional[dict] = None,
        opponent_state_predictor_cfg: Optional[dict] = None,
        action_predictor_cfg: Optional[dict] = None,
        next_state_predictor_cfg: Optional[dict] = None,
        rank_head_cfg: Optional[dict] = None,
        opponent_belief_predictor_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.latent_dim = latent_dim
        self.action_latent_dim = action_latent_dim

        enc_cfg = encoder_cfg or {}
        temp_cfg = temporal_encoder_cfg or {}
        act_enc_cfg = action_encoder_cfg or {}

        self.encoder = JEPAEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            **enc_cfg,
        )
        self.action_encoder = JEPAActionEncoder(
            token_embedding=self.encoder.token_embedding,
            pad_id=pad_id,
            action_latent_dim=action_latent_dim,
            encoder_d_model=enc_cfg.get("d_model", 512),
            **act_enc_cfg,
        )
        self.temporal_encoder = JEPATemporalEncoder(
            latent_dim=latent_dim,
            action_latent_dim=action_latent_dim,
            **temp_cfg,
        )
        # ── Merged opponent belief predictor (state + action) ──
        # Replaces the old separate opponent_state_predictor and action_predictor.
        # Config: prefer the new "opponent_belief_predictor" key; fall back to
        # "opponent_state_predictor" for backward compat (ignoring action_predictor
        # cfg since the merged backbone subsumes it).
        belief_cfg = opponent_belief_predictor_cfg or opponent_state_predictor_cfg or {}
        self.opponent_belief_predictor = JEPAOpponentBeliefPredictor(
            latent_dim=latent_dim,
            action_latent_dim=action_latent_dim,
            **belief_cfg,
        )
        self.next_state_predictor = JEPANextStatePredictor(
            latent_dim=latent_dim,
            action_latent_dim=action_latent_dim,
            **(next_state_predictor_cfg or {}),
        )
        self.rank_head = JEPAPairwiseRankHead(
            latent_dim=latent_dim,
            **(rank_head_cfg or {}),
        )

    def _encode_state_blocks(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        max_chunk_tokens: int = 65536,
    ) -> torch.Tensor:
        """Encode state blocks in micro-batches to bound peak FFN memory.

        Args:
            max_chunk_tokens: Maximum tokens per encoder call.  Lower = less
                peak memory, higher = fewer Python-loop iterations.
                65536 tokens → ~256 MiB FFN intermediate (bf16, d_ff=2048).
        """
        B, S, T = tokens.shape
        out = self.encoder.token_embedding.weight.new_zeros((B, S, self.latent_dim))
        if S == 0 or not valid.any():
            return out
        flat_tokens = tokens.reshape(B * S, T)
        flat_valid = valid.reshape(B * S)
        valid_idx = flat_valid.nonzero(as_tuple=True)[0]
        # Vectorized token count per valid block, then cumulative sum for chunking.
        n_tokens = (flat_tokens[valid_idx] != self.pad_id).sum(dim=1)  # (V,)
        cum_tokens = torch.cumsum(n_tokens, dim=0)
        total_tokens = cum_tokens[-1].item()
        if total_tokens <= max_chunk_tokens:
            encoded = self.encoder(flat_tokens[valid_idx])
            out.reshape(B * S, self.latent_dim)[valid_idx] = encoded
            return out
        # Find split points where cum_tokens crosses multiples of max_chunk_tokens.
        boundaries = torch.searchsorted(
            cum_tokens,
            torch.arange(max_chunk_tokens, total_tokens, max_chunk_tokens,
                         device=cum_tokens.device),
        )
        chunk_bounds = torch.cat([
            torch.tensor([0], device=cum_tokens.device),
            boundaries,
            torch.tensor([len(valid_idx)], device=cum_tokens.device),
        ])
        for i in range(len(chunk_bounds) - 1):
            chunk_idx = valid_idx[chunk_bounds[i]:chunk_bounds[i + 1]]
            encoded = self.encoder(flat_tokens[chunk_idx])
            out.reshape(B * S, self.latent_dim)[chunk_idx] = encoded
        return out

    def _encode_action_blocks(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        max_chunk_blocks: int = 512,
    ) -> torch.Tensor:
        B, S, T = tokens.shape
        out = self.encoder.token_embedding.weight.new_zeros((B, S, self.action_latent_dim))
        if S == 0 or not valid.any():
            return out
        flat_tokens = tokens.reshape(B * S, T)
        flat_valid = valid.reshape(B * S)
        valid_idx = flat_valid.nonzero(as_tuple=True)[0]
        for start in range(0, len(valid_idx), max_chunk_blocks):
            chunk_idx = valid_idx[start:start + max_chunk_blocks]
            encoded = self.action_encoder(flat_tokens[chunk_idx])
            out.reshape(B * S, self.action_latent_dim)[chunk_idx] = encoded
        return out

    def encode_history(
        self,
        state_tokens: torch.Tensor,
        state_valid: torch.Tensor,
        player_hist_tokens: torch.Tensor,
        player_hist_valid: torch.Tensor,
        opponent_hist_tokens: torch.Tensor,
        opponent_hist_valid: torch.Tensor,
    ) -> torch.Tensor:
        state_embs = self._encode_state_blocks(state_tokens, state_valid)
        player_embs = self._encode_action_blocks(player_hist_tokens, player_hist_valid)
        opponent_embs = self._encode_action_blocks(opponent_hist_tokens, opponent_hist_valid)
        return self.temporal_encoder(
            state_embs,
            state_valid,
            player_embs,
            player_hist_valid,
            opponent_embs,
            opponent_hist_valid,
        )

    @staticmethod
    def _last_valid_indices(valid: torch.Tensor) -> torch.Tensor:
        counts = valid.long().sum(dim=1)
        return (counts - 1).clamp_min(0)

    @staticmethod
    def _drop_current_state_from_history(valid: torch.Tensor) -> torch.Tensor:
        """Return a validity mask for prior/history states, excluding current.

        The team header is kept even for the first decision state.  Action
        histories are left untouched; they represent already-observed actions.
        """
        hist_valid = valid.clone()
        counts = valid.long().sum(dim=1)
        rows = torch.arange(valid.shape[0], device=valid.device)
        idx = (counts - 1).clamp_min(0)
        drop = counts > 1
        hist_valid[rows[drop], idx[drop]] = False
        return hist_valid

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        logvar = logvar.clamp(min=-8.0, max=6.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode_current_state(
        self,
        state_tokens: torch.Tensor,
        state_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode only the last valid state block, not the whole POV history."""
        B = state_tokens.shape[0]
        idx = self._last_valid_indices(state_valid)
        rows = torch.arange(B, device=state_tokens.device)
        return self.encoder(state_tokens[rows, idx, :])

    def encode_history_context(
        self,
        state_tokens: torch.Tensor,
        state_valid: torch.Tensor,
        player_hist_tokens: torch.Tensor,
        player_hist_valid: torch.Tensor,
        opponent_hist_tokens: torch.Tensor,
        opponent_hist_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode prior POV context separately from the current state block."""
        return self.encode_history(
            state_tokens,
            self._drop_current_state_from_history(state_valid),
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        )

    def forward(
        self,
        p1_state_T: torch.Tensor,
        p1_state_T_valid: torch.Tensor,
        p1_state_T1: torch.Tensor,
        p1_state_T1_valid: torch.Tensor,
        p1_player_hist_T: torch.Tensor,
        p1_player_hist_T_valid: torch.Tensor,
        p1_opponent_hist_T: torch.Tensor,
        p1_opponent_hist_T_valid: torch.Tensor,
        p1_player_hist_T1: torch.Tensor,
        p1_player_hist_T1_valid: torch.Tensor,
        p1_opponent_hist_T1: torch.Tensor,
        p1_opponent_hist_T1_valid: torch.Tensor,
        p2_state_T: torch.Tensor,
        p2_state_T_valid: torch.Tensor,
        p2_state_T1: torch.Tensor,
        p2_state_T1_valid: torch.Tensor,
        p2_player_hist_T: torch.Tensor,
        p2_player_hist_T_valid: torch.Tensor,
        p2_opponent_hist_T: torch.Tensor,
        p2_opponent_hist_T_valid: torch.Tensor,
        p2_player_hist_T1: torch.Tensor,
        p2_player_hist_T1_valid: torch.Tensor,
        p2_opponent_hist_T1: torch.Tensor,
        p2_opponent_hist_T1_valid: torch.Tensor,
        p1_action_tokens: torch.Tensor,
        p2_action_tokens: torch.Tensor,
        actual_p2_action_from_p1_perspective_tokens: torch.Tensor,
        actual_p1_action_from_p2_perspective_tokens: torch.Tensor,
        sample_beliefs: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        sample = self.training if sample_beliefs is None else sample_beliefs

        z_p1_T = self.encode_current_state(p1_state_T, p1_state_T_valid)
        z_p2_T = self.encode_current_state(p2_state_T, p2_state_T_valid)
        z_p1_T1 = self.encode_current_state(p1_state_T1, p1_state_T1_valid)
        z_p2_T1 = self.encode_current_state(p2_state_T1, p2_state_T1_valid)

        ctx_p1_T = self.encode_history_context(
            p1_state_T, p1_state_T_valid,
            p1_player_hist_T, p1_player_hist_T_valid,
            p1_opponent_hist_T, p1_opponent_hist_T_valid,
        )
        ctx_p2_T = self.encode_history_context(
            p2_state_T, p2_state_T_valid,
            p2_player_hist_T, p2_player_hist_T_valid,
            p2_opponent_hist_T, p2_opponent_hist_T_valid,
        )

        # ── Predict opponent state AND action from shared backbone ──
        (pred_p2_T_mu, pred_p2_T_logvar,
         pred_p2_action_mu, pred_p2_action_logvar) = self.opponent_belief_predictor(ctx_p1_T, z_p1_T)
        (pred_p1_T_mu, pred_p1_T_logvar,
         pred_p1_action_mu, pred_p1_action_logvar) = self.opponent_belief_predictor(ctx_p2_T, z_p2_T)
        pred_p2_T = self.reparameterize(pred_p2_T_mu, pred_p2_T_logvar, sample)
        pred_p1_T = self.reparameterize(pred_p1_T_mu, pred_p1_T_logvar, sample)
        pred_p2_action = self.reparameterize(pred_p2_action_mu, pred_p2_action_logvar, sample)
        pred_p1_action = self.reparameterize(pred_p1_action_mu, pred_p1_action_logvar, sample)

        p1_action = self.action_encoder(p1_action_tokens)
        p2_action = self.action_encoder(p2_action_tokens)
        actual_p2_action_from_p1_perspective = self.action_encoder(
            actual_p2_action_from_p1_perspective_tokens
        )
        actual_p1_action_from_p2_perspective = self.action_encoder(
            actual_p1_action_from_p2_perspective_tokens
        )

        # ── Predict next visible state (stochastic) ──
        pred_p1_T1_mu, pred_p1_T1_logvar = self.next_state_predictor(
            z_p1_T, p1_action, pred_p2_T, pred_p2_action
        )
        pred_p2_T1_mu, pred_p2_T1_logvar = self.next_state_predictor(
            z_p2_T, p2_action, pred_p1_T, pred_p1_action
        )
        pred_p1_T1 = self.reparameterize(pred_p1_T1_mu, pred_p1_T1_logvar, sample)
        pred_p2_T1 = self.reparameterize(pred_p2_T1_mu, pred_p2_T1_logvar, sample)

        rank_p1_teacher = self.rank_head(z_p1_T, z_p2_T)
        rank_p2_teacher = self.rank_head(z_p2_T, z_p1_T)
        rank_p1_belief = self.rank_head(z_p1_T, pred_p2_T)
        rank_p2_belief = self.rank_head(z_p2_T, pred_p1_T)
        rank_p1_next_belief = self.rank_head(pred_p1_T1, pred_p2_T)
        rank_p2_next_belief = self.rank_head(pred_p2_T1, pred_p1_T)

        return {
            "enc_p1_T": z_p1_T,
            "enc_p2_T": z_p2_T,
            "enc_p1_T1": z_p1_T1,
            "enc_p2_T1": z_p2_T1,
            "ctx_p1_T": ctx_p1_T,
            "ctx_p2_T": ctx_p2_T,
            "pred_p2_T_mu": pred_p2_T_mu,
            "pred_p2_T_logvar": pred_p2_T_logvar,
            "pred_p1_T_mu": pred_p1_T_mu,
            "pred_p1_T_logvar": pred_p1_T_logvar,
            "pred_p2_T": pred_p2_T,
            "pred_p1_T": pred_p1_T,
            "p1_action": p1_action,
            "p2_action": p2_action,
            "actual_p2_action_from_p1_perspective": actual_p2_action_from_p1_perspective,
            "actual_p1_action_from_p2_perspective": actual_p1_action_from_p2_perspective,
            "pred_p2_action_mu": pred_p2_action_mu,
            "pred_p2_action_logvar": pred_p2_action_logvar,
            "pred_p1_action_mu": pred_p1_action_mu,
            "pred_p1_action_logvar": pred_p1_action_logvar,
            "pred_p2_action": pred_p2_action,
            "pred_p1_action": pred_p1_action,
            "pred_p1_T1_mu": pred_p1_T1_mu,
            "pred_p1_T1_logvar": pred_p1_T1_logvar,
            "pred_p2_T1_mu": pred_p2_T1_mu,
            "pred_p2_T1_logvar": pred_p2_T1_logvar,
            "pred_p1_T1": pred_p1_T1,
            "pred_p2_T1": pred_p2_T1,
            "rank_p1_teacher": rank_p1_teacher,
            "rank_p2_teacher": rank_p2_teacher,
            "rank_p1_belief": rank_p1_belief,
            "rank_p2_belief": rank_p2_belief,
            "rank_p1_next_belief": rank_p1_next_belief,
            "rank_p2_next_belief": rank_p2_next_belief,
        }

    def save_checkpoint(self, path: str, **extra) -> None:
        ckpt = {"model_state_dict": self.state_dict(), **extra}
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None) -> dict:
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["model_state_dict"])
        return ckpt


# ═════════════════════════════════════════════════════════════════════════
# SIGReg — Sketched Isotropic Gaussian Regularization (Epps-Pulley test)
# ═════════════════════════════════════════════════════════════════════════

def _sigreg_grid(
    device: torch.device,
    num_points: int,
    domain: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached fp32 integration points, target CF, and weights."""
    key = (str(device), num_points, float(domain))
    cached = _SIGREG_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    t = torch.linspace(0, domain, num_points, device=device, dtype=torch.float32)
    dt = domain / (num_points - 1)
    weights = torch.full((num_points,), 2 * dt, device=device, dtype=torch.float32)
    weights[0] = dt
    weights[-1] = dt

    phi = torch.exp(-0.5 * t ** 2)
    weights = weights * phi

    cached = (t, phi, weights)
    _SIGREG_GRID_CACHE[key] = cached
    return cached


def sigreg(
    embeddings: torch.Tensor,
    num_slices: int = SIGREG_NUM_SLICES,
    num_points: int = SIGREG_NUM_POINTS,
    domain: float = SIGREG_DOMAIN,
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization via the Epps-Pulley test.

    Projects embeddings along random directions and compares the empirical
    characteristic function of each 1-D projection to that of N(0, 1).

    Per LeJEPA (Balestriero & LeCun 2025), SIGReg provides:
      - Bounded gradients and curvature (unlike moment-based tests)
      - Full identifiability of the Gaussian (unlike VICReg-style moment matching)
      - Linear O(N) complexity in batch size
      - DDP-friendly via simple averaging

    Args:
        embeddings: (B, D) — batch of D-dimensional embeddings.
        num_slices: number of random projection directions (resampled each call).
        num_points: number of quadrature points for integration.
        domain:     integration domain [0, domain] for the CF.

    Returns:
        Scalar SIGReg loss (averaged over slices, scaled by batch size).
    """
    # Characteristic-function math is sensitive enough that bf16 trig/exp adds
    # avoidable noise. Keep gradients, but run SIGReg itself in fp32.
    embeddings = embeddings.float()
    B, D = embeddings.shape
    device = embeddings.device

    # Sample random projection directions (unit norm).
    A = torch.randn(D, num_slices, device=device, dtype=torch.float32)
    A = A / A.norm(p=2, dim=0, keepdim=True)

    # Project embeddings onto random directions.
    proj = embeddings @ A  # (B, num_slices)

    # Integration points, Gaussian target CF, and trapezoidal weights.
    t, phi, weights = _sigreg_grid(device, num_points, domain)

    # Empirical CF via separate cos / sin — avoids complex64 precision loss.
    x_t = proj.unsqueeze(-1) * t             # (B, num_slices, T)
    err = (x_t.cos().mean(dim=0) - phi).square() + x_t.sin().mean(dim=0).square()
    # err: (num_slices, T)

    # Trapezoidal integration over t, average over slices, scale by batch.
    statistic = (err @ weights) * B          # (num_slices,)
    return statistic.mean()


def gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    min_logvar: float = -8.0,
    max_logvar: float = 6.0,
) -> torch.Tensor:
    """Diagonal Gaussian negative log likelihood, averaged over batch/dim."""
    logvar = logvar.clamp(min=min_logvar, max=max_logvar)
    return 0.5 * (logvar + (target - mu).square() * torch.exp(-logvar)).mean()


def pairwise_rank_loss(
    p1_score: torch.Tensor,
    p2_score: torch.Tensor,
    p1_won: torch.Tensor,
    rank_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bradley-Terry/logistic ranking loss from the p1 outcome label."""
    sign = p1_won.to(dtype=p1_score.dtype).mul(2.0).sub(1.0)
    margin = p1_score - p2_score
    loss = F.softplus(-sign * margin)
    if rank_valid is None:
        return loss.mean()
    valid = rank_valid.to(device=loss.device, dtype=torch.bool)
    if not bool(valid.any()):
        return loss.new_tensor(0.0)
    return loss[valid].mean()


def compute_paired_losses(
    outputs: dict[str, torch.Tensor],
    lambda_sigreg: float = 0.1,
    lambda_sigreg_state: float | None = None,
    lambda_sigreg_action: float | None = None,
    lambda_opponent_state: float = 1.0,
    lambda_action: float = 1.0,
    lambda_next_state: float = 1.0,
    lambda_rank: float = 1.0,
    sigreg_num_slices: int = SIGREG_NUM_SLICES,
    sigreg_num_points: int = SIGREG_NUM_POINTS,
    sigreg_domain: float = SIGREG_DOMAIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute paired-POV JEPA losses.

    Loss terms:
      - opponent_state_loss: MSE between predicted mu and target latent
      - action_loss: MSE between predicted mu and target action latent
      - next_state_loss: MSE between predicted next-state mu and target latent
        (the next-state predictor is stochastic — mu+logvar — because the
        world has inherent randomness: damage rolls, status, speed ties, …)
      - rank_loss: pairwise ranking from the winner label
      - SIGReg on state encoder outputs (current, next, context).
        Predicted next-state latents are NOT regularised — they are Gaussian
        samples (mu + ε·σ) by construction.

    If lambda_sigreg_state/lambda_sigreg_action are None, both fall back to
    lambda_sigreg for backward compatibility.
    """
    if lambda_sigreg_state is None:
        lambda_sigreg_state = lambda_sigreg
    if lambda_sigreg_action is None:
        lambda_sigreg_action = lambda_sigreg
    enc_p1_T = outputs["enc_p1_T"]
    enc_p2_T = outputs["enc_p2_T"]
    enc_p1_T1 = outputs["enc_p1_T1"]
    enc_p2_T1 = outputs["enc_p2_T1"]

    opponent_state_loss_p1_to_p2 = F.mse_loss(outputs["pred_p2_T_mu"], enc_p2_T)
    opponent_state_loss_p2_to_p1 = F.mse_loss(outputs["pred_p1_T_mu"], enc_p1_T)
    opponent_state_loss = 0.5 * (opponent_state_loss_p1_to_p2 + opponent_state_loss_p2_to_p1)

    action_loss_p1_to_p2 = F.mse_loss(
        outputs["pred_p2_action_mu"],
        outputs["actual_p2_action_from_p1_perspective"],
    )
    action_loss_p2_to_p1 = F.mse_loss(
        outputs["pred_p1_action_mu"],
        outputs["actual_p1_action_from_p2_perspective"],
    )
    action_loss = 0.5 * (action_loss_p1_to_p2 + action_loss_p2_to_p1)

    next_state_loss = 0.5 * (
        F.mse_loss(outputs["pred_p1_T1_mu"], enc_p1_T1)
        + F.mse_loss(outputs["pred_p2_T1_mu"], enc_p2_T1)
    )
    next_state_loss_p1 = F.mse_loss(outputs["pred_p1_T1_mu"], enc_p1_T1)
    next_state_loss_p2 = F.mse_loss(outputs["pred_p2_T1_mu"], enc_p2_T1)

    if "p1_won" in outputs:
        p1_won = outputs["p1_won"].to(device=enc_p1_T.device)
        if "rank_valid" in outputs:
            rank_valid = outputs["rank_valid"].to(device=enc_p1_T.device)
        elif "p2_won" in outputs:
            rank_valid = p1_won.bool() ^ outputs["p2_won"].to(device=enc_p1_T.device).bool()
        else:
            rank_valid = torch.ones_like(p1_won, dtype=torch.bool, device=enc_p1_T.device)
        rank_loss_teacher = pairwise_rank_loss(
            outputs["rank_p1_teacher"],
            outputs["rank_p2_teacher"],
            p1_won,
            rank_valid,
        )
        rank_loss_belief = pairwise_rank_loss(
            outputs["rank_p1_belief"],
            outputs["rank_p2_belief"],
            p1_won,
            rank_valid,
        )
        rank_loss_next = pairwise_rank_loss(
            outputs["rank_p1_next_belief"],
            outputs["rank_p2_next_belief"],
            p1_won,
            rank_valid,
        )
        rank_loss = (rank_loss_teacher + rank_loss_belief + rank_loss_next) / 3
    else:
        rank_valid = torch.zeros(enc_p1_T.shape[0], dtype=torch.bool, device=enc_p1_T.device)
        rank_loss_teacher = enc_p1_T.new_tensor(0.0)
        rank_loss_belief = enc_p1_T.new_tensor(0.0)
        rank_loss_next = enc_p1_T.new_tensor(0.0)
        rank_loss = enc_p1_T.new_tensor(0.0)

    # SIGReg on current-state latents: enc_p1_T and enc_p2_T are the JEPAEncoder
    # outputs (deterministic state embeddings, latent_dim=192) for each player at
    # the current time step T.  This is the "ground truth" latent of the visible board
    # state — a single μ vector, NOT the predicted Gaussian distribution.
    sigreg_current = (
        sigreg(enc_p1_T, sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(enc_p2_T, sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    sigreg_next_true = (
        sigreg(enc_p1_T1, sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(enc_p2_T1, sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    sigreg_context = (
        sigreg(outputs.get("ctx_p1_T", enc_p1_T), sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs.get("ctx_p2_T", enc_p2_T), sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 2
    sigreg_action_true = (
        sigreg(outputs["p1_action"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["p2_action"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["actual_p2_action_from_p1_perspective"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
        + sigreg(outputs["actual_p1_action_from_p2_perspective"], sigreg_num_slices, sigreg_num_points, sigreg_domain)
    ) / 4
    sigreg_state_loss = (
        sigreg_current + sigreg_next_true + sigreg_context
    ) / 3
    sigreg_action_loss = sigreg_action_true
    sigreg_loss = (
        sigreg_current
        + sigreg_next_true
        + sigreg_context
        + sigreg_action_true
    ) / 4

    pred_loss = (
        lambda_opponent_state * opponent_state_loss
        + lambda_action * action_loss
        + lambda_next_state * next_state_loss
        + lambda_rank * rank_loss
    )
    total_loss = (
        pred_loss
        + lambda_sigreg_state * sigreg_state_loss
        + lambda_sigreg_action * sigreg_action_loss
    )

    metrics = {
        "loss": total_loss.item(),
        "opponent_state_loss": opponent_state_loss.item(),
        "opponent_state_loss_p1_to_p2": opponent_state_loss_p1_to_p2.item(),
        "opponent_state_loss_p2_to_p1": opponent_state_loss_p2_to_p1.item(),
        "action_loss": action_loss.item(),
        "action_loss_p1_to_p2": action_loss_p1_to_p2.item(),
        "action_loss_p2_to_p1": action_loss_p2_to_p1.item(),
        "next_state_loss": next_state_loss.item(),
        "next_state_loss_p1": next_state_loss_p1.item(),
        "next_state_loss_p2": next_state_loss_p2.item(),
        "rank_loss": rank_loss.item(),
        "rank_loss_teacher": rank_loss_teacher.item(),
        "rank_loss_belief": rank_loss_belief.item(),
        "rank_loss_next": rank_loss_next.item(),
        "rank_valid": float(rank_valid.float().mean().item()),
        "pred_loss": pred_loss.item(),
        "sigreg_current": sigreg_current.item(),
        "sigreg_next_true": sigreg_next_true.item(),
        "sigreg_context": sigreg_context.item(),
        "sigreg_action_true": sigreg_action_true.item(),
        "sigreg_action_own": (sigreg(outputs["p1_action"], sigreg_num_slices, sigreg_num_points, sigreg_domain) + sigreg(outputs["p2_action"], sigreg_num_slices, sigreg_num_points, sigreg_domain)).item() / 2,
        "sigreg_action_opponent": (sigreg(outputs["actual_p2_action_from_p1_perspective"], sigreg_num_slices, sigreg_num_points, sigreg_domain) + sigreg(outputs["actual_p1_action_from_p2_perspective"], sigreg_num_slices, sigreg_num_points, sigreg_domain)).item() / 2,
        "sigreg_enc": ((sigreg_current + sigreg_next_true + sigreg_context) / 3).item(),
        "sigreg_state_loss": sigreg_state_loss.item(),
        "sigreg_action_loss": sigreg_action_loss.item(),
        "sigreg_loss": sigreg_loss.item(),
        "next_state_logvar_p1": outputs["pred_p1_T1_logvar"].mean().item(),
        "next_state_logvar_p2": outputs["pred_p2_T1_logvar"].mean().item(),
    }
    return total_loss, metrics
