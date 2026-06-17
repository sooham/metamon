"""LeJEPA (Latent-Euclidean JEPA) model for world-model state learning.

Architecture overview (v3 — block encoding + temporal action prediction)
-------------------------------------------------------------------------

    Header/state blocks₀..N ──► JEPAEncoder per block ─┐
    Historical player actions ─► JEPAActionEncoder ────┤
    Historical opponent actions ─► JEPAActionEncoder ──┤
                                                       ▼
    [team, state₀, p_action₀, o_action₀, state₁, ...]
                                                       │
                                                       ▼
                                             JEPATemporalEncoder ──► enc_N
                                                       │
                                                       ▼
                                      JEPAActionPredictor ──► pred_o
                                      (MLP, no conditioning)

    Current player action ─► JEPAActionEncoder ─► actual_p_emb
    Current opponent action ─► JEPAActionEncoder ─► actual_o_emb

    enc_N + actual_p_emb + pred_o ─► JEPAPredictor ─► pred_enc_{N+1}

    Blocks₀..N+1 through the same block+temporal encoders ─► enc_{N+1} target

    ┌──────────────────────────────────────────────────────────────┐
    │ LOSSES:                                                       │
    │   MSE(pred_o_emb, actual_o_emb)   — opponent action           │
    │   MSE(pred_enc_{N+1}, enc_{N+1})  — state prediction          │
    │   SIGReg(enc_N, enc_{N+1}, pred_enc_{N+1})  — state space     │
    │   SIGReg(actual_p_emb, actual_o_emb) — action encoder outputs │
    └──────────────────────────────────────────────────────────────┘

Modules:

1. **JEPAEncoder φ** — bidirectional transformer over one team-header or state
   block. Attention pools over non-pad tokens → state/header block embedding.

2. **JEPAActionEncoder ψ** — smaller bidirectional transformer over action text
   (e.g. "<chosen_move>blizzard<end_chosen_move>").  Shares the token embedding
   matrix with JEPAEncoder.  Attention pool → MLP → action_latent_dim.

3. **JEPATemporalEncoder τ** — transformer over interleaved block embeddings:
   [team, state_0, player_action_0, opponent_action_0, state_1, ...] → enc_N.

4. **JEPAActionPredictor α** — small MLP: enc_N → pred_o_emb.

5. **JEPAPredictor μ** — AdaLN-zero conditional transformer:
   (enc_N, actual_p_emb, pred_o_emb) → pred_enc_{N+1}.  Conditioning uses the
   chosen player action plus the predicted opponent action.

Losses
------

*State MSE* — MSE between target prefix embedding and predictor estimate::

    L_state = || enc_{N+1} - predictor(enc_N, actual_p_emb, pred_o_emb) ||²

*Action MSE* — MSE for opponent action prediction::

    L_oa = || actual_o_emb - pred_o_emb ||²

*SIGReg* — on encoder outputs (enc_N, enc_{N+1}, pred_enc_{N+1}) and
action encoder outputs (actual_p_emb, actual_o_emb).  NOT on
JEPAActionPredictor outputs.

Total::

    L = L_state + L_oa + λ · L_sigreg
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
# Building blocks (shared with metamon/sl/model.py structure)
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero modulation: apply per-dimension shift and scale."""
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.

    A conditioning vector ``c`` modulates the layer-norm statistics and
    gates the attention and FFN sublayers.  Zero-initialization of the
    modulation parameters means the block starts as an identity (the
    residual path passes the input through unchanged), letting the model
    gradually learn to use the conditioning signal.

    Based on the LeJEPA / DiT design (Balestriero & LeCun 2025, Peebles
    & Xie 2023).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        ffn_activation: str = "gelu",
    ):
        super().__init__()
        self.ffn_activation = ffn_activation
        # No elementwise_affine — AdaLN replaces the learned affine params.
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            d_model, n_heads, dropout, max_seq_len, causal=True,
            use_rope=False,  # predictor uses learned positional embeddings
        )
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

        # AdaLN-zero modulation: SiLU → Linear(dim → 6×dim).
        # Outputs: shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True)
        )
        # Zero-initialise the final layer so the block is an identity at init.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            shift_attn, scale_attn, gate_attn,
            shift_ffn, scale_ffn, gate_ffn,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        # Attention sublayer with AdaLN + gating.
        x = x + gate_attn * self.dropout(
            self.attn(modulate(self.norm1(x), shift_attn, scale_attn))
        )
        # FFN sublayer with AdaLN + gating — modulate before FFN.
        x = x + gate_ffn * self._ffn(
            modulate(self.norm2(x), shift_ffn, scale_ffn)
        )
        return x

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_activation == "swiglu":
            gate = F.silu(self.ffn_w1(x))
            up = self.ffn_w2(x)
            x = self.ffn_out(gate * up)
        else:
            x = self.ffn(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.ffn_out(x)
            x = self.dropout(x)
        return x


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
# JEPATemporalEncoder — interleaved block embeddings → prefix embedding
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


# ═════════════════════════════════════════════════════════════════════════
# JEPAActionPredictor — enc_N → predicted opponent action embedding
# ═════════════════════════════════════════════════════════════════════════

class JEPAActionPredictor(nn.Module):
    """Simple MLP that predicts the next opponent action embedding.

    Input:  enc_N  (B, latent_dim) — prefix embedding up to state N.
    Output: (B, action_latent_dim) — predicted opponent action embedding.

    No conditioning, no transformer — a straightforward MLP.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_latent_dim: int = ACTION_LATENT_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_latent_dim = action_latent_dim

        if hidden_dim is None:
            hidden_dim = 4 * latent_dim

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ])
                in_dim = hidden_dim
            else:
                layers.append(nn.Linear(in_dim, action_latent_dim))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, enc_N: torch.Tensor) -> torch.Tensor:
        """Predict opponent action embedding.

        Args:
            enc_N: (B, latent_dim) — prefix embedding up to state N.

        Returns:
            pred_o_emb: (B, action_latent_dim).
        """
        return self.net(enc_N)


# ═════════════════════════════════════════════════════════════════════════
# JEPAPredictor — (enc_N, player_action, opponent_action) → pred_enc_{N+1}
# ═════════════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """Action-conditioned predictor with AdaLN-zero.

    Maps the deterministic previous-prefix embedding, chosen player action,
    and predicted opponent action to an estimate of the next-prefix embedding::

        predicted_next = PRED(enc_N, player_action_emb, pred_o_emb)

    The player and opponent action embeddings are concatenated, projected into
    the predictor's ``d_model`` space, and used as the AdaLN-zero conditioning
    signal.

    Architecture (AdaLN-zero, based on LeJEPA / DiT)
    --------------------------------------------------
    1. Project ``enc_N`` from ``latent_dim`` to ``d_model``.
    2. Add learned positional embedding (single token).
    3. Project ``[player_action_emb, pred_o_emb]`` from
       2×action_latent_dim to ``d_model`` conditioning vector.
    4. Run through N ConditionalBlocks — each block's layer-norm statistics
       and sublayer gates are modulated by the action conditioning via an
       AdaLN-zero MLP.
    5. LayerNorm → pool → MLP projection back to ``latent_dim``.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input/output latent vectors.
    action_latent_dim : int
        Dimensionality of a single action embedding (player or opponent).
        The conditioning input is 2× this (player + opponent).
    gradient_checkpointing : bool
        If True, wrap each ConditionalBlock with checkpoint.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len :
        Transformer hyperparameters for the predictor.
    proj_hidden_dim : int or None
        Hidden dimension for the output projector MLP.  Defaults to 4× d_model.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_latent_dim: int = ACTION_LATENT_DIM,
        gradient_checkpointing: bool = False,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 64,
        proj_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gradient_checkpointing = gradient_checkpointing

        # Project state embedding into predictor space.
        self.state_proj = nn.Linear(latent_dim, d_model, bias=False)

        # Learned positional embedding for a single token.
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Action conditioning: project player + predicted opponent action embeds.
        self.action_cond_proj = nn.Linear(2 * action_latent_dim, d_model, bias=False)

        # AdaLN-zero conditional blocks.
        self.blocks = nn.ModuleList([
            ConditionalBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                ffn_activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        # MLP projector: pooled predictor output → embedding.
        pred_hidden = proj_hidden_dim or (4 * d_model)
        self.out_proj = MLP(d_model, pred_hidden, latent_dim)

        self.apply(self._init_weights)

        # Re-apply zero-init to AdaLN modulation layers — they were
        # overwritten by _init_weights (which normal-inits all Linears).
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        enc_N: torch.Tensor,
        player_action_emb: torch.Tensor,
        pred_o_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-prefix deterministic embedding.

        Args:
            enc_N:             (B, latent_dim) — prefix embedding up to state N.
            player_action_emb: (B, action_latent_dim) — chosen player action.
            pred_o_emb:        (B, action_latent_dim) — predicted opponent action.

        Returns:
            predicted_next: (B, latent_dim).
        """
        # State token: project + add positional embedding.
        x = self.state_proj(enc_N).unsqueeze(1)        # (B, 1, d_model)
        x = x + self.pos_embedding                       # (B, 1, d_model)

        # Action conditioning vector.
        action_cond = torch.cat([player_action_emb, pred_o_emb], dim=-1)
        c = self.action_cond_proj(action_cond)           # (B, d_model)

        # AdaLN-zero conditional blocks.
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, use_reentrant=False
                )
            else:
                x = block(x, c)
        x = self.ln_final(x)

        # Pool from the (only) token position.
        pooled = x[:, -1, :]  # (B, d_model)

        return self.out_proj(pooled)  # (B, latent_dim)


# ═════════════════════════════════════════════════════════════════════════
# Full JEPA model — encoder + action encoder + action predictor + predictor
# ═════════════════════════════════════════════════════════════════════════

class JEPAModel(nn.Module):
    """Top-level JEPA model for learning representations from battle histories.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    pad_id, bos_id, eos_id : int
        Special token IDs.
    latent_dim : int
        Dimensionality of the latent space (state embeddings).
    action_latent_dim : int
        Dimensionality of action embeddings.
    encoder_cfg, action_encoder_cfg, action_predictor_cfg, predictor_cfg : dict
        Sub-module configuration dicts.
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
        action_predictor_cfg: Optional[dict] = None,
        predictor_cfg: Optional[dict] = None,
        **kwargs,  # absorb legacy keys silently
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
        act_pred_cfg = action_predictor_cfg or {}
        pred_cfg = predictor_cfg or {}

        # Block encoder φ — encodes one team-header or state block.
        self.encoder = JEPAEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            **enc_cfg,
        )

        # Action encoder ψ — produces action embeddings from action text.
        self.action_encoder = JEPAActionEncoder(
            token_embedding=self.encoder.token_embedding,
            pad_id=pad_id,
            action_latent_dim=action_latent_dim,
            encoder_d_model=enc_cfg.get("d_model", 512),
            **act_enc_cfg,
        )

        # Temporal encoder τ — consumes interleaved state/action block embeddings.
        self.temporal_encoder = JEPATemporalEncoder(
            latent_dim=latent_dim,
            action_latent_dim=action_latent_dim,
            **temp_cfg,
        )

        # Action predictor α — predicts the opponent action embedding from enc_N.
        self.action_predictor = JEPAActionPredictor(
            latent_dim=latent_dim,
            action_latent_dim=action_latent_dim,
            **act_pred_cfg,
        )

        # State predictor μ — predicts enc_{N+1} from
        # (enc_N, player action, predicted opponent action).
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            action_latent_dim=action_latent_dim,
            **pred_cfg,
        )

    def forward(
        self,
        state_N_tokens: torch.Tensor,
        state_N_valid: torch.Tensor,
        state_N1_tokens: torch.Tensor,
        state_N1_valid: torch.Tensor,
        player_hist_N_tokens: torch.Tensor,
        player_hist_N_valid: torch.Tensor,
        opponent_hist_N_tokens: torch.Tensor,
        opponent_hist_N_valid: torch.Tensor,
        player_hist_N1_tokens: torch.Tensor,
        player_hist_N1_valid: torch.Tensor,
        opponent_hist_N1_tokens: torch.Tensor,
        opponent_hist_N1_valid: torch.Tensor,
        player_action_tokens: torch.Tensor,
        opponent_action_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            state_N_tokens:   (B, S_N, T) int — header/states through state N.
            state_N1_tokens:  (B, S_N1, T) int — header/states through state N+1.
            player_action_tokens:   (B, A_p) int — player action text tokens (with delimiters).
            opponent_action_tokens: (B, A_o) int — opponent action text tokens (with delimiters).

        Returns:
            Dict with keys:
                enc_N, enc_N1              — (B, latent_dim) encoder outputs.
                pred_enc_N1                — (B, latent_dim) predicted next state embedding.
                pred_o_emb                 — (B, action_latent_dim) predicted opponent action.
                actual_p_emb, actual_o_emb — (B, action_latent_dim) actual action embeddings.
        """
        def encode_state_blocks(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            B, S, T = tokens.shape
            out = self.encoder.token_embedding.weight.new_zeros((B, S, self.latent_dim))
            if S == 0 or not valid.any():
                return out
            flat_valid = valid.reshape(B * S)
            encoded = self.encoder(tokens.reshape(B * S, T)[flat_valid])
            out.reshape(B * S, self.latent_dim)[flat_valid] = encoded
            return out

        def encode_action_blocks(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            B, S, T = tokens.shape
            out = self.encoder.token_embedding.weight.new_zeros((B, S, self.action_latent_dim))
            if S == 0 or not valid.any():
                return out
            flat_valid = valid.reshape(B * S)
            encoded = self.action_encoder(tokens.reshape(B * S, T)[flat_valid])
            out.reshape(B * S, self.action_latent_dim)[flat_valid] = encoded
            return out

        # ── Encode prefix blocks ──
        state_N_embs = encode_state_blocks(state_N_tokens, state_N_valid)
        state_N1_embs = encode_state_blocks(state_N1_tokens, state_N1_valid)
        player_hist_N_embs = encode_action_blocks(player_hist_N_tokens, player_hist_N_valid)
        opponent_hist_N_embs = encode_action_blocks(opponent_hist_N_tokens, opponent_hist_N_valid)
        player_hist_N1_embs = encode_action_blocks(player_hist_N1_tokens, player_hist_N1_valid)
        opponent_hist_N1_embs = encode_action_blocks(opponent_hist_N1_tokens, opponent_hist_N1_valid)

        enc_N = self.temporal_encoder(
            state_N_embs, state_N_valid,
            player_hist_N_embs, player_hist_N_valid,
            opponent_hist_N_embs, opponent_hist_N_valid,
        )
        enc_N1 = self.temporal_encoder(
            state_N1_embs, state_N1_valid,
            player_hist_N1_embs, player_hist_N1_valid,
            opponent_hist_N1_embs, opponent_hist_N1_valid,
        )

        # ── Encode ground-truth action texts ──
        actual_p_emb = self.action_encoder(player_action_tokens)     # (B, action_latent_dim)
        actual_o_emb = self.action_encoder(opponent_action_tokens)   # (B, action_latent_dim)

        # ── Predict opponent action ──
        pred_o_emb = self.action_predictor(enc_N)       # (B, action_latent_dim)

        # ── Predict next state ──
        pred_enc_N1 = self.predictor(enc_N, actual_p_emb, pred_o_emb)  # (B, latent_dim)

        return {
            "enc_N": enc_N,
            "enc_N1": enc_N1,
            "pred_enc_N1": pred_enc_N1,
            "pred_o_emb": pred_o_emb,
            "actual_p_emb": actual_p_emb,
            "actual_o_emb": actual_o_emb,
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


# ═════════════════════════════════════════════════════════════════════════
# Loss computation
# ═════════════════════════════════════════════════════════════════════════

def compute_losses(
    outputs: dict[str, torch.Tensor],
    lambda_sigreg: float = 0.1,
    sigreg_num_slices: int = SIGREG_NUM_SLICES,
    sigreg_num_points: int = SIGREG_NUM_POINTS,
    sigreg_domain: float = SIGREG_DOMAIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the JEPA loss: state prediction + opponent-action prediction + SIGReg.

    No stop-gradient — SIGReg prevents representational collapse without
    asymmetric architecture tricks.

    Losses:
      - state_loss:   MSE(enc_{N+1}, pred_enc_{N+1})
      - opponent_action_loss: MSE(actual_o_emb, pred_o_emb)
      - sigreg_enc:    SIGReg on encoder outputs (enc_N, enc_{N+1}, pred_enc_{N+1})
      - sigreg_act:    SIGReg on action encoder outputs (actual_p_emb, actual_o_emb)

    SIGReg is NOT applied to JEPAActionPredictor outputs.

    Args:
        outputs: Dict from ``JEPAModel.forward()`` with keys
                 "enc_N", "enc_N1", "pred_enc_N1", "pred_o_emb",
                 "actual_p_emb", "actual_o_emb".
        lambda_sigreg: Weight for SIGReg.
        sigreg_num_slices, sigreg_num_points, sigreg_domain:
            SIGReg hyperparameters.

    Returns:
        total_loss: scalar tensor.
        metrics: dict with per-loss-component values.
    """
    enc_N = outputs["enc_N"]
    enc_N1 = outputs["enc_N1"]
    pred_enc_N1 = outputs["pred_enc_N1"]
    pred_o_emb = outputs["pred_o_emb"]
    actual_p_emb = outputs["actual_p_emb"]
    actual_o_emb = outputs["actual_o_emb"]

    # ── 1. State prediction loss ────────────────────────────────────
    state_loss = F.mse_loss(enc_N1, pred_enc_N1)

    # ── 2. Opponent action prediction loss ──────────────────────────
    oa_loss = F.mse_loss(actual_o_emb, pred_o_emb)

    # ── 3. SIGReg on encoder outputs (state space) ──────────────────
    sigreg_enc = sigreg(
        torch.cat([enc_N, enc_N1, pred_enc_N1], dim=0),
        sigreg_num_slices,
        sigreg_num_points,
        sigreg_domain,
    )

    # ── 4. SIGReg on action encoder outputs (action space) ──────────
    sigreg_act = sigreg(
        torch.cat([actual_p_emb, actual_o_emb], dim=0),
        sigreg_num_slices,
        sigreg_num_points,
        sigreg_domain,
    )

    # ── Total loss ──────────────────────────────────────────────────
    pred_loss = state_loss + oa_loss
    sigreg_loss = sigreg_enc + sigreg_act
    total_loss = pred_loss + lambda_sigreg * sigreg_loss

    metrics = {
        "loss": total_loss.item(),
        "state_loss": state_loss.item(),
        "player_action_loss": 0.0,
        "opponent_action_loss": oa_loss.item(),
        "pred_loss": pred_loss.item(),
        "sigreg_enc": sigreg_enc.item(),
        "sigreg_act": sigreg_act.item(),
        "sigreg_loss": sigreg_loss.item(),
    }

    return total_loss, metrics
