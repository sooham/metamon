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

    Current player action text ───► JEPAActionEncoder ─► own_action
    Current opponent action text ─► JEPAActionEncoder ─► opponent_action target

    z_T + own_action + actual opponent state/action
        ─► JEPANextStatePredictor ─► pred_z_{T+1}_mu/logvar

    pred_opp_state_mu/logvar + z_T ─► JEPADecisionStateEncoder
                                     ├─ JEPAValueHead ─► V_logit(s)
                                     └─ JEPAActionValueHead(legal a) ─► Q_logit(s,a)

    ┌────────────────────────────────────────────────────────────────────┐
    │ LOSSES:                                                             │
    │   Gaussian NLL(z_opp_T | pred_opp_state_mu/logvar) — opponent state │
    │   Gaussian NLL(action_opp | pred_opp_action_mu/logvar) — action     │
    │   Gaussian NLL(z_{T+1} | pred_z_{T+1}_mu/logvar) — next state       │
    │   BCE on V(s) and Q(s, chosen_action) vs POV terminal outcome       │
    │   Masked CE over the acting player's legal Q logits                 │
    │   Lower-weight V/Q teacher losses using actual paired opponent z    │
    │   SIGReg on current, next-target, and history-context state latents │
    │   Optional SIGReg on true action encoder outputs                    │
    └────────────────────────────────────────────────────────────────────┘

Modules:

1. **JEPAEncoder φ** — bidirectional transformer over one team-header or state
   block. Attention pools over non-pad tokens → state/header block embedding.

2. **JEPAActionEncoder ψ** — smaller bidirectional transformer over canonical
   action text content (e.g. "move blizzard" or "switch alakazam", with no
   player/opponent delimiter tokens).  Shares the token embedding matrix with
   JEPAEncoder.  Attention pool → MLP → action_latent_dim.

3. **JEPATemporalEncoder τ** — transformer over interleaved historical block
   embeddings.  The current state block is dropped from this history before
   temporal encoding, so the output is a prior context rather than a
   next-state target.

4. **JEPAOpponentBeliefPredictor β** — shared MLP backbone over
   (history_context, current_state_z), with separate Gaussian heads for the
   opponent current-state latent and opponent next-action latent.

5. **JEPANextStatePredictor μ** — diagonal-Gaussian MLP:
   (current_state_z, own_action, opponent_state, opponent_action) → next
   current-state latent. During paired supervised training, opponent state and
   opponent action are teacher-forced from the actual paired POV/action.

Losses
------

*Next-state NLL* — constant-free diagonal Gaussian negative log likelihood of
the target next current-state block latent under the next-state predictor
distribution::

    L_next = 0.5 * mean(logvar_next + ||z_{T+1} - mu_next||² / exp(logvar_next))

*Opponent state NLL* — constant-free diagonal Gaussian negative log likelihood
of the target opponent latent under the belief predictor distribution::

    L_os = 0.5 * mean(logvar_opp + ||z_opp_T - mu_opp||² / exp(logvar_opp))

*Action NLL* — constant-free diagonal Gaussian negative log likelihood of the
target opponent action latent under the belief predictor action distribution::

    L_oa = 0.5 * mean(logvar_action + ||actual_opp_action - mu_action||² / exp(logvar_action))

*SIGReg* — on current-state encoder outputs, next-state target encoder
outputs, and history-context outputs.  Action SIGReg is configurable and is off
by default; predicted Gaussian samples are not regularized.

*Value / action-value losses* — offline actor-critic style supervision from
the terminal outcome of each POV.  V(s) is action-free; Q(s,a) scores only the
acting player's legal candidates.  Opponent legal actions are never required.

Total::

    L = λ_os L_os + λ_oa L_oa + λ_next L_next
        + λ_v L_v + λ_q L_q + λ_policy L_policy + λ_teacher L_teacher
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

# Projected belief/action decision space used by V(s) and Q(s,a).
DECISION_DIM: int = 384

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

    def _transformer_forward(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run embedding + transformer stack → pooled latent.

        Separate from ``forward`` so the whole stack can be wrapped
        in a single gradient checkpoint.  That saves only *token_ids*
        (int64) for the backward recompute instead of per-block
        ``[B, T, d_model]`` activations, cutting peak memory ~100× when
        encoding many state blocks in one call.
        """
        valid_mask = token_ids != self.pad_id
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, key_padding_mask=valid_mask)
        x = self.ln_final(x)
        pooled = self.pool(x, valid_mask)
        return self.proj_e(pooled)

    def forward(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode a token sequence.

        Args:
            token_ids: (B, S) int — state/header block token IDs.

        Returns:
            e: (B, latent_dim) — deterministic embedding.
        """
        if self.gradient_checkpointing and self.training:
            # Checkpoint the ENTIRE transformer stack to bound autograd
            # context.  The saved context is only *token_ids* (int64),
            # not per-block ``[B, T, d_model]`` activations.
            return torch.utils.checkpoint.checkpoint(
                self._transformer_forward, token_ids, use_reentrant=False
            )
        return self._transformer_forward(token_ids)


# ═════════════════════════════════════════════════════════════════════════
# JEPAActionEncoder — smaller bidirectional transformer → action embedding
# ═════════════════════════════════════════════════════════════════════════

class JEPAActionEncoder(nn.Module):
    """Small bidirectional transformer that encodes action text to a fixed-size
    action embedding.

    Action text is canonicalized before encoding and does not include
    player/opponent role delimiters. The temporal encoder receives separate
    type embeddings for player vs opponent history blocks, and the next-state
    predictor receives own/opponent action embeddings in distinct input slots.

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

        if self.gradient_checkpointing and self.training:
            # Checkpoint the ENTIRE transformer stack to save only one copy
            # of [B, max_seq, d_model] instead of one per block (4× reduction).
            return torch.utils.checkpoint.checkpoint(
                self._transformer_forward, x, valid, use_reentrant=False
            )
        return self._transformer_forward(x, valid)

    def _transformer_forward(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
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

    Both heads are trained with diagonal Gaussian NLL against their respective
    targets. Their reparameterized samples also feed downstream prediction and
    ranking losses.

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

    Takes current self-state, own action, opponent state, and an opponent action
    embedding — all of which feed through a small MLP that outputs mu and logvar
    for the next-state latent. During paired supervised training, the opponent
    state/action inputs are teacher-forced from the actual paired POV state and
    the actual action taken between the current and next states.

    The world is stochastic (damage rolls, status effects, speed ties, …),
    so a deterministic predictor would be misspecified. The output distribution
    is trained with diagonal Gaussian NLL against the target encoder latent.
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


class JEPAGatedResidualBlock(nn.Module):
    """Small gated residual MLP block for decision-state refinement."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (2 * dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + torch.sigmoid(self.gate(h)) * self.ff(h)


class JEPADecisionStateEncoder(nn.Module):
    """Fuse self state with predicted opponent Gaussian belief for decisions."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        decision_dim: int = DECISION_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 4,
        dropout: float = 0.1,
        min_logvar: float = -8.0,
        max_logvar: float = 6.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (2 * decision_dim)
        self.decision_dim = decision_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        self.self_proj = nn.Sequential(
            nn.Linear(latent_dim, decision_dim),
            nn.LayerNorm(decision_dim),
            nn.GELU(),
        )
        self.opp_mu_proj = nn.Sequential(
            nn.Linear(latent_dim, decision_dim),
            nn.LayerNorm(decision_dim),
            nn.GELU(),
        )
        self.opp_logvar_proj = nn.Sequential(
            nn.Linear(latent_dim, decision_dim),
            nn.LayerNorm(decision_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(5 * decision_dim, decision_dim),
            nn.LayerNorm(decision_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            JEPAGatedResidualBlock(decision_dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(decision_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        self_state: torch.Tensor,
        opponent_state_mu: torch.Tensor,
        opponent_state_logvar: torch.Tensor,
    ) -> torch.Tensor:
        self_h = self.self_proj(self_state)
        mu_h = self.opp_mu_proj(opponent_state_mu)
        logvar_h = self.opp_logvar_proj(
            opponent_state_logvar.clamp(min=self.min_logvar, max=self.max_logvar)
        )
        x = self.fuse(torch.cat([
            self_h,
            mu_h,
            logvar_h,
            self_h - mu_h,
            self_h * mu_h,
        ], dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class JEPAValueHead(nn.Module):
    """Action-free value critic V(s), returned as a terminal-outcome logit."""

    def __init__(
        self,
        decision_dim: int = DECISION_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or decision_dim
        layers: list[nn.Module] = []
        in_dim = decision_dim
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
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

    def forward(self, decision_state: torch.Tensor) -> torch.Tensor:
        return self.net(decision_state).squeeze(-1)


class JEPAActionProjector(nn.Module):
    """Project action encoder latents into the decision/Q space."""

    def __init__(
        self,
        action_latent_dim: int = ACTION_LATENT_DIM,
        decision_dim: int = DECISION_DIM,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or decision_dim
        self.net = nn.Sequential(
            nn.Linear(action_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decision_dim),
            nn.LayerNorm(decision_dim),
            nn.GELU(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, action_latent: torch.Tensor) -> torch.Tensor:
        return self.net(action_latent)


class JEPAActionValueHead(nn.Module):
    """Q(s,a) scorer over current-player legal action candidates."""

    def __init__(
        self,
        decision_dim: int = DECISION_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (2 * decision_dim)
        self.bilinear = nn.Bilinear(decision_dim, decision_dim, 1, bias=False)
        layers: list[nn.Module] = []
        in_dim = 4 * decision_dim
        for i in range(n_layers):
            if i < n_layers - 1:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            else:
                layers.append(nn.Linear(in_dim, 1))
        self.joint = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Bilinear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, decision_state: torch.Tensor, action_state: torch.Tensor) -> torch.Tensor:
        if action_state.ndim == decision_state.ndim + 1:
            decision_state = decision_state.unsqueeze(-2).expand_as(action_state)
        x = torch.cat([
            decision_state,
            action_state,
            decision_state - action_state,
            decision_state * action_state,
        ], dim=-1)
        return (self.bilinear(decision_state, action_state) + self.joint(x)).squeeze(-1)


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
        decision_state_encoder_cfg: Optional[dict] = None,
        value_head_cfg: Optional[dict] = None,
        action_projector_cfg: Optional[dict] = None,
        action_value_head_cfg: Optional[dict] = None,
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
        decision_cfg = decision_state_encoder_cfg or {}
        self.decision_state_encoder = JEPADecisionStateEncoder(
            latent_dim=latent_dim,
            **decision_cfg,
        )
        decision_dim = decision_cfg.get("decision_dim", DECISION_DIM)
        self.value_head = JEPAValueHead(
            decision_dim=decision_dim,
            **(value_head_cfg or {}),
        )
        self.action_projector = JEPAActionProjector(
            action_latent_dim=action_latent_dim,
            decision_dim=decision_dim,
            **(action_projector_cfg or {}),
        )
        self.action_value_head = JEPAActionValueHead(
            decision_dim=decision_dim,
            **(action_value_head_cfg or {}),
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
        if state_tokens.ndim == 4:
            B, K, S, T = state_tokens.shape
            _, _, A, AT = player_hist_tokens.shape
            _, _, OA, OAT = opponent_hist_tokens.shape
            flat_state_tokens = state_tokens.reshape(B * K, S, T)
            flat_state_valid = state_valid.reshape(B * K, S)
            flat_player_tokens = player_hist_tokens.reshape(B * K, A, AT)
            flat_player_valid = player_hist_valid.reshape(B * K, A)
            flat_opponent_tokens = opponent_hist_tokens.reshape(B * K, OA, OAT)
            flat_opponent_valid = opponent_hist_valid.reshape(B * K, OA)
            encoded = self.encode_history(
                flat_state_tokens,
                flat_state_valid,
                flat_player_tokens,
                flat_player_valid,
                flat_opponent_tokens,
                flat_opponent_valid,
            )
            return encoded.reshape(B, K, self.latent_dim)

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
        if state_tokens.ndim == 4:
            B, K, S, T = state_tokens.shape
            flat_tokens = state_tokens.reshape(B * K, S, T)
            flat_valid = state_valid.reshape(B * K, S)
            encoded = self.encode_current_state(flat_tokens, flat_valid)
            return encoded.reshape(B, K, self.latent_dim)

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
        if state_valid.ndim == 3:
            B, K, S = state_valid.shape
            flat_valid = state_valid.reshape(B * K, S)
            hist_valid = self._drop_current_state_from_history(flat_valid).reshape(B, K, S)
        else:
            hist_valid = self._drop_current_state_from_history(state_valid)
        return self.encode_history(
            state_tokens,
            hist_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        )

    def encode_action_tokens(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Encode action text, preserving an optional rollout dimension."""
        if action_tokens.ndim == 3:
            B, K, T = action_tokens.shape
            encoded = self.action_encoder(action_tokens.reshape(B * K, T))
            return encoded.reshape(B, K, self.action_latent_dim)
        return self.action_encoder(action_tokens)

    def encode_action_candidates(
        self,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        max_chunk_blocks: int = 512,
    ) -> torch.Tensor:
        """Encode legal action candidate text with optional candidate masks."""
        if action_tokens.ndim not in {3, 4}:
            raise ValueError(
                f"action candidate tokens must be [B,C,T] or [B,K,C,T], got {action_tokens.shape}"
            )
        out_shape = (*action_tokens.shape[:-1], self.action_latent_dim)
        out = self.encoder.token_embedding.weight.new_zeros(out_shape)
        flat_tokens = action_tokens.reshape(-1, action_tokens.shape[-1])
        if action_mask is None:
            flat_valid = (flat_tokens != self.pad_id).any(dim=-1)
        else:
            flat_valid = action_mask.reshape(-1).to(device=action_tokens.device, dtype=torch.bool)
        if not bool(flat_valid.any()):
            return out
        valid_idx = flat_valid.nonzero(as_tuple=True)[0]
        flat_out = out.reshape(-1, self.action_latent_dim)
        for start in range(0, len(valid_idx), max_chunk_blocks):
            chunk_idx = valid_idx[start:start + max_chunk_blocks]
            flat_out[chunk_idx] = self.action_encoder(flat_tokens[chunk_idx])
        return out

    @staticmethod
    def _singleton_candidate_tokens(action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.ndim == 3:
            return action_tokens.unsqueeze(2)
        return action_tokens.unsqueeze(1)

    @staticmethod
    def _singleton_candidate_mask(action_tokens: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (*action_tokens.shape[:-1], 1),
            dtype=torch.bool,
            device=action_tokens.device,
        )

    @staticmethod
    def _zero_chosen_indices(action_tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            action_tokens.shape[:-1],
            dtype=torch.long,
            device=action_tokens.device,
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
        p1_legal_action_tokens: torch.Tensor | None = None,
        p1_legal_action_mask: torch.Tensor | None = None,
        p1_chosen_legal_action_idx: torch.Tensor | None = None,
        p2_legal_action_tokens: torch.Tensor | None = None,
        p2_legal_action_mask: torch.Tensor | None = None,
        p2_chosen_legal_action_idx: torch.Tensor | None = None,
        sample_beliefs: Optional[bool] = None,
        *,
        p1_next_legal_action_tokens: torch.Tensor | None = None,
        p1_next_legal_action_mask: torch.Tensor | None = None,
        p2_next_legal_action_tokens: torch.Tensor | None = None,
        p2_next_legal_action_mask: torch.Tensor | None = None,
        compute_td_bootstrap: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        sample = self.training if sample_beliefs is None else sample_beliefs
        if compute_td_bootstrap is None:
            compute_td_bootstrap = (
                p1_next_legal_action_tokens is not None
                or p2_next_legal_action_tokens is not None
            )

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

        p1_action = self.encode_action_tokens(p1_action_tokens)
        p2_action = self.encode_action_tokens(p2_action_tokens)
        actual_p2_action_from_p1_perspective = self.encode_action_tokens(
            actual_p2_action_from_p1_perspective_tokens
        )
        actual_p1_action_from_p2_perspective = self.encode_action_tokens(
            actual_p1_action_from_p2_perspective_tokens
        )

        # ── Predict next visible state (stochastic) ──
        pred_p1_T1_mu, pred_p1_T1_logvar = self.next_state_predictor(
            z_p1_T, p1_action, z_p2_T, actual_p2_action_from_p1_perspective
        )
        pred_p2_T1_mu, pred_p2_T1_logvar = self.next_state_predictor(
            z_p2_T, p2_action, z_p1_T, actual_p1_action_from_p2_perspective
        )
        pred_p1_T1 = self.reparameterize(pred_p1_T1_mu, pred_p1_T1_logvar, sample)
        pred_p2_T1 = self.reparameterize(pred_p2_T1_mu, pred_p2_T1_logvar, sample)

        if p1_legal_action_tokens is None:
            p1_legal_action_tokens = self._singleton_candidate_tokens(p1_action_tokens)
        if p1_legal_action_mask is None:
            p1_legal_action_mask = self._singleton_candidate_mask(p1_action_tokens)
        if p1_chosen_legal_action_idx is None:
            p1_chosen_legal_action_idx = self._zero_chosen_indices(p1_action_tokens)
        if p2_legal_action_tokens is None:
            p2_legal_action_tokens = self._singleton_candidate_tokens(p2_action_tokens)
        if p2_legal_action_mask is None:
            p2_legal_action_mask = self._singleton_candidate_mask(p2_action_tokens)
        if p2_chosen_legal_action_idx is None:
            p2_chosen_legal_action_idx = self._zero_chosen_indices(p2_action_tokens)

        p1_decision_state = self.decision_state_encoder(
            z_p1_T, pred_p2_T_mu, pred_p2_T_logvar
        )
        p2_decision_state = self.decision_state_encoder(
            z_p2_T, pred_p1_T_mu, pred_p1_T_logvar
        )
        zero_p2_logvar = torch.zeros_like(z_p2_T)
        zero_p1_logvar = torch.zeros_like(z_p1_T)
        p1_decision_state_teacher = self.decision_state_encoder(
            z_p1_T, z_p2_T, zero_p2_logvar
        )
        p2_decision_state_teacher = self.decision_state_encoder(
            z_p2_T, z_p1_T, zero_p1_logvar
        )

        p1_value_logit = self.value_head(p1_decision_state)
        p2_value_logit = self.value_head(p2_decision_state)
        p1_value_teacher_logit = self.value_head(p1_decision_state_teacher)
        p2_value_teacher_logit = self.value_head(p2_decision_state_teacher)

        p1_legal_action = self.encode_action_candidates(
            p1_legal_action_tokens,
            p1_legal_action_mask,
        )
        p2_legal_action = self.encode_action_candidates(
            p2_legal_action_tokens,
            p2_legal_action_mask,
        )
        p1_legal_action_h = self.action_projector(p1_legal_action)
        p2_legal_action_h = self.action_projector(p2_legal_action)
        p1_q_logits = self.action_value_head(p1_decision_state, p1_legal_action_h)
        p2_q_logits = self.action_value_head(p2_decision_state, p2_legal_action_h)
        p1_q_teacher_logits = self.action_value_head(
            p1_decision_state_teacher,
            p1_legal_action_h,
        )
        p2_q_teacher_logits = self.action_value_head(
            p2_decision_state_teacher,
            p2_legal_action_h,
        )

        ctx_p1_T1 = None
        ctx_p2_T1 = None
        pred_p2_bootstrap_mu = None
        pred_p2_bootstrap_logvar = None
        pred_p1_bootstrap_mu = None
        pred_p1_bootstrap_logvar = None
        p1_next_decision_state = None
        p2_next_decision_state = None
        p1_next_value_logit = None
        p2_next_value_logit = None
        p1_next_q_logits = None
        p2_next_q_logits = None
        if compute_td_bootstrap:
            with torch.no_grad():
                ctx_p1_T1 = self.encode_history_context(
                    p1_state_T1, p1_state_T1_valid,
                    p1_player_hist_T1, p1_player_hist_T1_valid,
                    p1_opponent_hist_T1, p1_opponent_hist_T1_valid,
                )
                ctx_p2_T1 = self.encode_history_context(
                    p2_state_T1, p2_state_T1_valid,
                    p2_player_hist_T1, p2_player_hist_T1_valid,
                    p2_opponent_hist_T1, p2_opponent_hist_T1_valid,
                )
                # Bootstrap decision states for the actual T+1 states. These
                # are semi-gradient TD targets, including at rollout-window
                # boundaries where there is no k+1 row inside the same item.
                (pred_p2_bootstrap_mu, pred_p2_bootstrap_logvar,
                 _pred_p2_action_T1_mu, _pred_p2_action_T1_logvar) = self.opponent_belief_predictor(ctx_p1_T1, z_p1_T1)
                (pred_p1_bootstrap_mu, pred_p1_bootstrap_logvar,
                 _pred_p1_action_T1_mu, _pred_p1_action_T1_logvar) = self.opponent_belief_predictor(ctx_p2_T1, z_p2_T1)
                p1_next_decision_state = self.decision_state_encoder(
                    z_p1_T1, pred_p2_bootstrap_mu, pred_p2_bootstrap_logvar
                )
                p2_next_decision_state = self.decision_state_encoder(
                    z_p2_T1, pred_p1_bootstrap_mu, pred_p1_bootstrap_logvar
                )
                p1_next_value_logit = self.value_head(p1_next_decision_state)
                p2_next_value_logit = self.value_head(p2_next_decision_state)
                if p1_next_legal_action_tokens is not None:
                    if p1_next_legal_action_mask is None:
                        p1_next_legal_action_mask = (
                            p1_next_legal_action_tokens != self.pad_id
                        ).any(dim=-1)
                    p1_next_legal_action = self.encode_action_candidates(
                        p1_next_legal_action_tokens,
                        p1_next_legal_action_mask,
                    )
                    p1_next_legal_action_h = self.action_projector(p1_next_legal_action)
                    p1_next_q_logits = self.action_value_head(
                        p1_next_decision_state,
                        p1_next_legal_action_h,
                    )
                if p2_next_legal_action_tokens is not None:
                    if p2_next_legal_action_mask is None:
                        p2_next_legal_action_mask = (
                            p2_next_legal_action_tokens != self.pad_id
                        ).any(dim=-1)
                    p2_next_legal_action = self.encode_action_candidates(
                        p2_next_legal_action_tokens,
                        p2_next_legal_action_mask,
                    )
                    p2_next_legal_action_h = self.action_projector(p2_next_legal_action)
                    p2_next_q_logits = self.action_value_head(
                        p2_next_decision_state,
                        p2_next_legal_action_h,
                    )

        outputs = {
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
            "p1_decision_state": p1_decision_state,
            "p2_decision_state": p2_decision_state,
            "p1_decision_state_teacher": p1_decision_state_teacher,
            "p2_decision_state_teacher": p2_decision_state_teacher,
            "p1_value_logit": p1_value_logit,
            "p2_value_logit": p2_value_logit,
            "p1_value_teacher_logit": p1_value_teacher_logit,
            "p2_value_teacher_logit": p2_value_teacher_logit,
            "p1_legal_action": p1_legal_action,
            "p2_legal_action": p2_legal_action,
            "p1_legal_action_mask": p1_legal_action_mask,
            "p2_legal_action_mask": p2_legal_action_mask,
            "p1_chosen_legal_action_idx": p1_chosen_legal_action_idx,
            "p2_chosen_legal_action_idx": p2_chosen_legal_action_idx,
            "p1_q_logits": p1_q_logits,
            "p2_q_logits": p2_q_logits,
            "p1_q_teacher_logits": p1_q_teacher_logits,
            "p2_q_teacher_logits": p2_q_teacher_logits,
        }
        if p1_next_value_logit is not None and p2_next_value_logit is not None:
            outputs.update({
                "ctx_p1_T1": ctx_p1_T1,
                "ctx_p2_T1": ctx_p2_T1,
                "pred_p2_T1_belief_mu": pred_p2_bootstrap_mu,
                "pred_p2_T1_belief_logvar": pred_p2_bootstrap_logvar,
                "pred_p1_T1_belief_mu": pred_p1_bootstrap_mu,
                "pred_p1_T1_belief_logvar": pred_p1_bootstrap_logvar,
                "p1_next_decision_state": p1_next_decision_state,
                "p2_next_decision_state": p2_next_decision_state,
                "p1_next_value_logit": p1_next_value_logit,
                "p2_next_value_logit": p2_next_value_logit,
            })
        if p1_next_q_logits is not None and p1_next_legal_action_mask is not None:
            outputs["p1_next_q_logits"] = p1_next_q_logits
            outputs["p1_next_legal_action_mask"] = p1_next_legal_action_mask
        if p2_next_q_logits is not None and p2_next_legal_action_mask is not None:
            outputs["p2_next_q_logits"] = p2_next_q_logits
            outputs["p2_next_legal_action_mask"] = p2_next_legal_action_mask
        return outputs

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
    embeddings = embeddings.float().reshape(-1, embeddings.shape[-1])
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
    """Constant-free diagonal Gaussian NLL, averaged over batch/dim."""
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


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _expand_to_reference(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = tensor.to(device=reference.device, dtype=dtype or reference.dtype)
    while tensor.ndim > reference.ndim and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    while tensor.ndim < reference.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor.expand_as(reference)


def _outcome_targets_and_valid(outputs: dict[str, torch.Tensor], reference: torch.Tensor) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = reference.device
    if "p1_won" not in outputs or "p2_won" not in outputs:
        shape = reference.shape[:-1] if reference.ndim > 0 else reference.shape
        valid = torch.zeros(shape, dtype=torch.bool, device=device)
        target = torch.zeros(shape, dtype=reference.dtype, device=device)
        return target, target, valid

    p1_won = outputs["p1_won"].to(device=device, dtype=reference.dtype)
    p2_won = outputs["p2_won"].to(device=device, dtype=reference.dtype)
    if "rank_valid" in outputs:
        valid = outputs["rank_valid"].to(device=device, dtype=torch.bool)
    else:
        valid = p1_won.bool() ^ p2_won.bool()
    return p1_won, p2_won, valid


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = _expand_to_reference(valid, logits, dtype=torch.bool)
    target = _expand_to_reference(target, logits, dtype=logits.dtype)
    if not bool(valid.any()):
        return _zero_like_loss(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        weight = _expand_to_reference(weight, logits, dtype=logits.dtype)
        loss = loss * weight
    return loss[valid].mean()


def _chosen_action_logits(
    logits: torch.Tensor,
    chosen_idx: torch.Tensor,
) -> torch.Tensor:
    idx = chosen_idx.to(device=logits.device, dtype=torch.long).clamp(
        min=0,
        max=max(logits.shape[-1] - 1, 0),
    )
    return logits.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def _masked_policy_cross_entropy(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    chosen_idx: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    chosen_idx = chosen_idx.to(device=logits.device, dtype=torch.long)
    valid = valid.to(device=logits.device, dtype=torch.bool)
    while valid.ndim < legal_mask.ndim - 1:
        valid = valid.unsqueeze(-1)
    chosen_in_range = (chosen_idx >= 0) & (chosen_idx < logits.shape[-1])
    safe_idx = chosen_idx.clamp(min=0, max=max(logits.shape[-1] - 1, 0))
    chosen_legal = legal_mask.gather(-1, safe_idx.unsqueeze(-1)).squeeze(-1)
    sample_valid = valid & legal_mask.any(dim=-1) & chosen_in_range & chosen_legal
    if not bool(sample_valid.any()):
        return _zero_like_loss(logits)

    masked_logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    flat_logits = masked_logits.reshape(-1, masked_logits.shape[-1])
    flat_target = safe_idx.reshape(-1)
    flat_valid = sample_valid.reshape(-1)
    loss = F.cross_entropy(flat_logits[flat_valid], flat_target[flat_valid], reduction="none")
    if weight is not None:
        weight = _expand_to_reference(weight, sample_valid, dtype=logits.dtype)
        flat_weight = weight.reshape(-1)
        loss = loss * flat_weight[flat_valid]
    return loss.mean()


def _advantage_weights(
    value_logit: torch.Tensor,
    outcome: torch.Tensor,
    valid: torch.Tensor,
    temperature: float | None,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor | None:
    if temperature is None or temperature <= 0:
        return None
    outcome = _expand_to_reference(outcome, value_logit, dtype=value_logit.dtype)
    valid = _expand_to_reference(valid, value_logit, dtype=torch.bool)
    value = torch.sigmoid(value_logit).detach()
    weight = torch.exp((outcome - value) / float(temperature))
    weight = weight.clamp(min=clamp_min, max=clamp_max)
    return torch.where(valid, weight, torch.ones_like(weight))


def compute_paired_losses(
    outputs: dict[str, torch.Tensor],
    lambda_sigreg: float = 0.1,
    lambda_sigreg_state: float | None = None,
    lambda_sigreg_action: float | None = None,
    lambda_opponent_state: float = 1.0,
    lambda_action: float = 1.0,
    lambda_next_state: float = 1.0,
    lambda_rank: float = 1.0,
    lambda_value: float = 1.0,
    lambda_q_value: float = 1.0,
    lambda_policy: float = 1.0,
    lambda_value_teacher: float = 0.25,
    lambda_q_teacher: float = 0.25,
    advantage_temperature: float | None = None,
    advantage_weight_min: float = 0.1,
    advantage_weight_max: float = 10.0,
    gamma: float = 1.0,
    sigreg_num_slices: int = SIGREG_NUM_SLICES,
    sigreg_num_points: int = SIGREG_NUM_POINTS,
    sigreg_domain: float = SIGREG_DOMAIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute paired-POV JEPA losses.

    Loss terms:
      - opponent_state_loss: diagonal Gaussian NLL of target opponent latent
      - action_loss: diagonal Gaussian NLL of target opponent action latent
      - next_state_loss: diagonal Gaussian NLL of target next-state latent
        (the next-state predictor is stochastic — mu+logvar — because the
        world has inherent randomness: damage rolls, status, speed ties, …)
      - value_loss: rollout-horizon TD(n) BCE on V(s) per POV.
        Non-terminal steps use the furthest valid in-window bootstrap state
        γⁿ·σ(V(T+n)).detach() as a soft target; true terminal steps use the
        discounted binary outcome.  γ=1 keeps the previous MC supervision path
        for backward compatibility.
      - q_value_loss: rollout-horizon TD(n) BCE on Q(s, chosen_action) per POV.
        Non-terminal steps use γⁿ·σ(maxₐ Q(T+n,a)).detach(); true terminal
        steps use the discounted binary outcome.
      - policy_loss: masked CE over only that POV's legal action candidates
      - SIGReg on state encoder outputs (current, next, context).
        Predicted next-state latents are NOT regularised — they are Gaussian
        samples (mu + ε·σ) by construction.

    If lambda_sigreg_state/lambda_sigreg_action are None, both fall back to
    lambda_sigreg for backward compatibility.

    gamma controls TD discount. For backward compatibility, 1.0 disables TD
    and uses the previous MC target for all steps. Typical TD values are
    0.95–0.99. With K-step rollout tensors, each position bootstraps from the
    furthest valid T+n state still present in that rollout, so the first
    position in a K-step non-terminal window uses TD(K). True terminal
    transitions stop the return early and use the discounted battle outcome.
    """
    if lambda_sigreg_state is None:
        lambda_sigreg_state = lambda_sigreg
    if lambda_sigreg_action is None:
        lambda_sigreg_action = lambda_sigreg
    enc_p1_T = outputs["enc_p1_T"]
    enc_p2_T = outputs["enc_p2_T"]
    enc_p1_T1 = outputs["enc_p1_T1"]
    enc_p2_T1 = outputs["enc_p2_T1"]

    opponent_state_loss_p1_to_p2 = gaussian_nll(
        enc_p2_T,
        outputs["pred_p2_T_mu"],
        outputs["pred_p2_T_logvar"],
    )
    opponent_state_loss_p2_to_p1 = gaussian_nll(
        enc_p1_T,
        outputs["pred_p1_T_mu"],
        outputs["pred_p1_T_logvar"],
    )
    opponent_state_loss = 0.5 * (opponent_state_loss_p1_to_p2 + opponent_state_loss_p2_to_p1)

    action_loss_p1_to_p2 = gaussian_nll(
        outputs["actual_p2_action_from_p1_perspective"],
        outputs["pred_p2_action_mu"],
        outputs["pred_p2_action_logvar"],
    )
    action_loss_p2_to_p1 = gaussian_nll(
        outputs["actual_p1_action_from_p2_perspective"],
        outputs["pred_p1_action_mu"],
        outputs["pred_p1_action_logvar"],
    )
    action_loss = 0.5 * (action_loss_p1_to_p2 + action_loss_p2_to_p1)

    next_state_loss = 0.5 * (
        gaussian_nll(enc_p1_T1, outputs["pred_p1_T1_mu"], outputs["pred_p1_T1_logvar"])
        + gaussian_nll(enc_p2_T1, outputs["pred_p2_T1_mu"], outputs["pred_p2_T1_logvar"])
    )
    next_state_loss_p1 = gaussian_nll(
        enc_p1_T1,
        outputs["pred_p1_T1_mu"],
        outputs["pred_p1_T1_logvar"],
    )
    next_state_loss_p2 = gaussian_nll(
        enc_p2_T1,
        outputs["pred_p2_T1_mu"],
        outputs["pred_p2_T1_logvar"],
    )

    p1_outcome, p2_outcome, outcome_valid = _outcome_targets_and_valid(outputs, enc_p1_T)
    rank_valid = outcome_valid
    rank_loss_teacher = enc_p1_T.new_tensor(0.0)
    rank_loss_belief = enc_p1_T.new_tensor(0.0)
    rank_loss_next = enc_p1_T.new_tensor(0.0)
    rank_loss = enc_p1_T.new_tensor(0.0)

    # ── TD bootstrapping helpers ────────────────────────────────────
    # True TD targets bootstrap from actual future states in the rollout. The
    # model emits V/Q logits for each T+1 state, so rollout-window boundaries
    # are not treated as environment terminals.
    use_td = abs(float(gamma) - 1.0) > 1e-8  # skip TD construction when γ≈1

    def _terminal_mask(name: str, reference: torch.Tensor) -> torch.Tensor:
        mask = outputs.get(name)
        if mask is None:
            return torch.zeros_like(reference, dtype=torch.bool)
        return _expand_to_reference(mask, reference, dtype=torch.bool)

    def _next_value_logits(
        current_logit: torch.Tensor,
        next_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_logit = outputs.get(next_key)
        if next_logit is not None:
            return (
                _expand_to_reference(next_logit, current_logit, dtype=current_logit.dtype),
                torch.ones_like(current_logit, dtype=torch.bool),
            )
        # Backward-compatible fallback for unit tests/legacy callers that only
        # provide current-step logits. The final rollout position is not
        # bootstrappable in this fallback and will use the outcome target.
        next_logit = torch.zeros_like(current_logit)
        valid = torch.zeros_like(current_logit, dtype=torch.bool)
        if current_logit.ndim >= 2 and current_logit.shape[-1] > 1:
            next_logit[..., :-1] = current_logit[..., 1:]
            valid[..., :-1] = True
        return next_logit, valid

    def _next_q_logits(
        current_logits: torch.Tensor,
        current_legal_mask: torch.Tensor,
        next_logits_key: str,
        next_mask_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_logits = outputs.get(next_logits_key)
        next_mask = outputs.get(next_mask_key)
        if next_logits is not None and next_mask is not None:
            next_logits = next_logits.to(device=current_logits.device, dtype=current_logits.dtype)
            next_mask = next_mask.to(device=current_logits.device, dtype=torch.bool)
            return next_logits, next_mask
        # Legacy fallback: bootstrap only interior rollout steps by shifting the
        # current-step candidate scores. The final position has no valid next
        # candidates and will use the outcome target.
        next_logits = torch.zeros_like(current_logits)
        next_mask = torch.zeros_like(current_legal_mask, dtype=torch.bool)
        if current_logits.ndim >= 3 and current_logits.shape[-2] > 1:
            next_logits[..., :-1, :] = current_logits[..., 1:, :]
            next_mask[..., :-1, :] = current_legal_mask.to(
                device=current_logits.device,
                dtype=torch.bool,
            )[..., 1:, :]
        return next_logits, next_mask

    def _discount_like(reference: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        gamma_tensor = reference.new_tensor(float(gamma))
        return torch.pow(gamma_tensor, exponent.to(device=reference.device, dtype=reference.dtype))

    def _td_target(
        value_logit: torch.Tensor,   # [B, K]
        next_value_logit: torch.Tensor,
        outcome: torch.Tensor,       # [B] or [B, K]
        is_terminal: torch.Tensor,   # [B, K]
        bootstrap_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rollout-horizon TD(n) soft targets for V logits.

        Non-terminal:  γ^n · σ(V_{T+n}).detach()
        Terminal:      γ^d · outcome, where d is the terminal transition offset
        """
        if not use_td:
            return outcome, torch.zeros_like(value_logit, dtype=torch.bool)
        next_value_logit = _expand_to_reference(
            next_value_logit,
            value_logit,
            dtype=value_logit.dtype,
        )
        outcome_expanded = _expand_to_reference(outcome, value_logit, dtype=value_logit.dtype)
        is_terminal = _expand_to_reference(is_terminal, value_logit, dtype=torch.bool)
        bootstrap_valid = _expand_to_reference(bootstrap_valid, value_logit, dtype=torch.bool)
        if value_logit.ndim < 2 or value_logit.shape[-1] <= 1:
            soft_target = float(gamma) * torch.sigmoid(next_value_logit).detach()
            target = torch.where(is_terminal | ~bootstrap_valid, outcome_expanded, soft_target)
            return target, bootstrap_valid & ~is_terminal

        rollout_len = value_logit.shape[-1]
        target = torch.empty_like(value_logit)
        td_used = torch.zeros_like(value_logit, dtype=torch.bool)
        for step in range(rollout_len):
            terminal_after = is_terminal[..., step:]
            valid_after = bootstrap_valid[..., step:]
            offsets = torch.arange(
                rollout_len - step,
                device=value_logit.device,
                dtype=torch.long,
            )

            terminal_offsets = torch.where(
                terminal_after,
                offsets,
                torch.full_like(offsets, rollout_len - step),
            ).min(dim=-1).values
            has_terminal = terminal_offsets < (rollout_len - step)

            bootstrap_offsets = torch.where(
                valid_after,
                offsets,
                torch.full_like(offsets, -1),
            ).max(dim=-1).values
            has_bootstrap = bootstrap_offsets >= 0
            bootstrap_idx = step + bootstrap_offsets.clamp_min(0)
            bootstrap_logit = next_value_logit.gather(
                -1,
                bootstrap_idx.unsqueeze(-1),
            ).squeeze(-1)

            outcome_step = outcome_expanded[..., step]
            terminal_target = _discount_like(value_logit, terminal_offsets) * outcome_step
            bootstrap_target = (
                _discount_like(value_logit, bootstrap_offsets + 1)
                * torch.sigmoid(bootstrap_logit).detach()
            )
            step_target = torch.where(has_terminal, terminal_target, bootstrap_target)
            step_target = torch.where(
                has_terminal | has_bootstrap,
                step_target,
                outcome_step,
            )
            target[..., step] = step_target
            td_used[..., step] = has_bootstrap & ~has_terminal

        return target, td_used

    def _td_q_target(
        chosen_q: torch.Tensor,            # [B, K]
        next_q_logits: torch.Tensor,       # [B, K, C]
        next_legal_mask: torch.Tensor,     # [B, K, C]
        outcome: torch.Tensor,             # [B] or [B, K]
        is_terminal: torch.Tensor,         # [B, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build rollout-horizon TD(n) soft targets for Q logits.

        Non-terminal:  γ^n · σ(max_candidates Q_{T+n}).detach()
        Terminal:      γ^d · outcome, where d is the terminal transition offset
        """
        if not use_td:
            return outcome, torch.zeros_like(chosen_q, dtype=torch.bool)
        device = chosen_q.device
        next_q_logits = next_q_logits.to(device=device, dtype=chosen_q.dtype)
        next_legal_mask = next_legal_mask.to(device=device, dtype=torch.bool)
        bootstrap_valid = next_legal_mask.any(dim=-1)
        masked = next_q_logits.masked_fill(
            ~next_legal_mask,
            torch.finfo(next_q_logits.dtype).min,
        )
        q_next_max = masked.max(dim=-1).values
        q_next_max = torch.where(bootstrap_valid, q_next_max, torch.zeros_like(q_next_max))
        outcome_expanded = _expand_to_reference(outcome, chosen_q, dtype=chosen_q.dtype)
        is_terminal = _expand_to_reference(is_terminal, chosen_q, dtype=torch.bool)
        bootstrap_valid = _expand_to_reference(bootstrap_valid, chosen_q, dtype=torch.bool)
        if chosen_q.ndim < 2 or chosen_q.shape[-1] <= 1:
            soft_target = float(gamma) * torch.sigmoid(q_next_max).detach()
            target = torch.where(is_terminal | ~bootstrap_valid, outcome_expanded, soft_target)
            return target, bootstrap_valid & ~is_terminal

        rollout_len = chosen_q.shape[-1]
        target = torch.empty_like(chosen_q)
        td_used = torch.zeros_like(chosen_q, dtype=torch.bool)
        for step in range(rollout_len):
            terminal_after = is_terminal[..., step:]
            valid_after = bootstrap_valid[..., step:]
            offsets = torch.arange(
                rollout_len - step,
                device=chosen_q.device,
                dtype=torch.long,
            )

            terminal_offsets = torch.where(
                terminal_after,
                offsets,
                torch.full_like(offsets, rollout_len - step),
            ).min(dim=-1).values
            has_terminal = terminal_offsets < (rollout_len - step)

            bootstrap_offsets = torch.where(
                valid_after,
                offsets,
                torch.full_like(offsets, -1),
            ).max(dim=-1).values
            has_bootstrap = bootstrap_offsets >= 0
            bootstrap_idx = step + bootstrap_offsets.clamp_min(0)
            bootstrap_q = q_next_max.gather(-1, bootstrap_idx.unsqueeze(-1)).squeeze(-1)

            outcome_step = outcome_expanded[..., step]
            terminal_target = _discount_like(chosen_q, terminal_offsets) * outcome_step
            bootstrap_target = (
                _discount_like(chosen_q, bootstrap_offsets + 1)
                * torch.sigmoid(bootstrap_q).detach()
            )
            step_target = torch.where(has_terminal, terminal_target, bootstrap_target)
            step_target = torch.where(
                has_terminal | has_bootstrap,
                step_target,
                outcome_step,
            )
            target[..., step] = step_target
            td_used[..., step] = has_bootstrap & ~has_terminal

        return target, td_used

    p1_is_terminal = _terminal_mask("p1_is_terminal", enc_p1_T[..., 0])
    p2_is_terminal = _terminal_mask("p2_is_terminal", enc_p2_T[..., 0])
    p1_value_bootstrap_valid = torch.zeros_like(p1_is_terminal, dtype=torch.bool)
    p2_value_bootstrap_valid = torch.zeros_like(p2_is_terminal, dtype=torch.bool)
    p1_q_bootstrap_valid = torch.zeros_like(p1_is_terminal, dtype=torch.bool)
    p2_q_bootstrap_valid = torch.zeros_like(p2_is_terminal, dtype=torch.bool)

    if "p1_value_logit" in outputs and "p2_value_logit" in outputs:
        p1_next_v, p1_value_bootstrap_valid = _next_value_logits(
            outputs["p1_value_logit"],
            "p1_next_value_logit",
        )
        p2_next_v, p2_value_bootstrap_valid = _next_value_logits(
            outputs["p2_value_logit"],
            "p2_next_value_logit",
        )
        p1_v_target, p1_value_bootstrap_valid = _td_target(
            outputs["p1_value_logit"], p1_next_v, p1_outcome,
            p1_is_terminal, p1_value_bootstrap_valid)
        p2_v_target, p2_value_bootstrap_valid = _td_target(
            outputs["p2_value_logit"], p2_next_v, p2_outcome,
            p2_is_terminal, p2_value_bootstrap_valid)
        value_loss_p1 = _masked_bce_with_logits(
            outputs["p1_value_logit"],
            p1_v_target,
            outcome_valid,
        )
        value_loss_p2 = _masked_bce_with_logits(
            outputs["p2_value_logit"],
            p2_v_target,
            outcome_valid,
        )
        value_loss = 0.5 * (value_loss_p1 + value_loss_p2)
    else:
        value_loss_p1 = value_loss_p2 = value_loss = enc_p1_T.new_tensor(0.0)

    if "p1_q_logits" in outputs and "p2_q_logits" in outputs:
        p1_chosen_q = _chosen_action_logits(
            outputs["p1_q_logits"],
            outputs["p1_chosen_legal_action_idx"],
        )
        p2_chosen_q = _chosen_action_logits(
            outputs["p2_q_logits"],
            outputs["p2_chosen_legal_action_idx"],
        )
        p1_adv_weight = _advantage_weights(
            outputs.get("p1_value_logit", p1_chosen_q),
            p1_outcome,
            outcome_valid,
            advantage_temperature,
            advantage_weight_min,
            advantage_weight_max,
        )
        p2_adv_weight = _advantage_weights(
            outputs.get("p2_value_logit", p2_chosen_q),
            p2_outcome,
            outcome_valid,
            advantage_temperature,
            advantage_weight_min,
            advantage_weight_max,
        )
        p1_next_q, p1_next_q_mask = _next_q_logits(
            outputs["p1_q_logits"],
            outputs["p1_legal_action_mask"],
            "p1_next_q_logits",
            "p1_next_legal_action_mask",
        )
        p2_next_q, p2_next_q_mask = _next_q_logits(
            outputs["p2_q_logits"],
            outputs["p2_legal_action_mask"],
            "p2_next_q_logits",
            "p2_next_legal_action_mask",
        )
        p1_q_target, p1_q_bootstrap_valid = _td_q_target(
            p1_chosen_q, p1_next_q, p1_next_q_mask,
            p1_outcome, p1_is_terminal)
        p2_q_target, p2_q_bootstrap_valid = _td_q_target(
            p2_chosen_q, p2_next_q, p2_next_q_mask,
            p2_outcome, p2_is_terminal)
        q_value_loss_p1 = _masked_bce_with_logits(
            p1_chosen_q,
            p1_q_target,
            outcome_valid,
            p1_adv_weight,
        )
        q_value_loss_p2 = _masked_bce_with_logits(
            p2_chosen_q,
            p2_q_target,
            outcome_valid,
            p2_adv_weight,
        )
        q_value_loss = 0.5 * (q_value_loss_p1 + q_value_loss_p2)
        policy_loss_p1 = _masked_policy_cross_entropy(
            outputs["p1_q_logits"],
            outputs["p1_legal_action_mask"],
            outputs["p1_chosen_legal_action_idx"],
            outcome_valid,
            p1_adv_weight,
        )
        policy_loss_p2 = _masked_policy_cross_entropy(
            outputs["p2_q_logits"],
            outputs["p2_legal_action_mask"],
            outputs["p2_chosen_legal_action_idx"],
            outcome_valid,
            p2_adv_weight,
        )
        policy_loss = 0.5 * (policy_loss_p1 + policy_loss_p2)
    else:
        p1_chosen_q = p2_chosen_q = enc_p1_T.new_zeros(enc_p1_T.shape[:-1])
        q_value_loss_p1 = q_value_loss_p2 = q_value_loss = enc_p1_T.new_tensor(0.0)
        policy_loss_p1 = policy_loss_p2 = policy_loss = enc_p1_T.new_tensor(0.0)
        p1_adv_weight = p2_adv_weight = None

    if "p1_value_teacher_logit" in outputs and "p2_value_teacher_logit" in outputs:
        value_teacher_loss_p1 = _masked_bce_with_logits(
            outputs["p1_value_teacher_logit"],
            p1_outcome,
            outcome_valid,
        )
        value_teacher_loss_p2 = _masked_bce_with_logits(
            outputs["p2_value_teacher_logit"],
            p2_outcome,
            outcome_valid,
        )
        value_teacher_loss = 0.5 * (value_teacher_loss_p1 + value_teacher_loss_p2)
    else:
        value_teacher_loss_p1 = value_teacher_loss_p2 = value_teacher_loss = enc_p1_T.new_tensor(0.0)

    if "p1_q_teacher_logits" in outputs and "p2_q_teacher_logits" in outputs:
        p1_teacher_q = _chosen_action_logits(
            outputs["p1_q_teacher_logits"],
            outputs["p1_chosen_legal_action_idx"],
        )
        p2_teacher_q = _chosen_action_logits(
            outputs["p2_q_teacher_logits"],
            outputs["p2_chosen_legal_action_idx"],
        )
        q_teacher_loss_p1 = _masked_bce_with_logits(
            p1_teacher_q,
            p1_outcome,
            outcome_valid,
            p1_adv_weight,
        )
        q_teacher_loss_p2 = _masked_bce_with_logits(
            p2_teacher_q,
            p2_outcome,
            outcome_valid,
            p2_adv_weight,
        )
        q_teacher_loss = 0.5 * (q_teacher_loss_p1 + q_teacher_loss_p2)
    else:
        q_teacher_loss_p1 = q_teacher_loss_p2 = q_teacher_loss = enc_p1_T.new_tensor(0.0)

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
        + lambda_value * value_loss
        + lambda_q_value * q_value_loss
        + lambda_policy * policy_loss
        + lambda_value_teacher * value_teacher_loss
        + lambda_q_teacher * q_teacher_loss
    )
    total_loss = (
        pred_loss
        + lambda_sigreg_state * sigreg_state_loss
        + lambda_sigreg_action * sigreg_action_loss
    )
    p1_value_td_used = p1_value_bootstrap_valid & ~p1_is_terminal
    p2_value_td_used = p2_value_bootstrap_valid & ~p2_is_terminal
    p1_q_td_used = p1_q_bootstrap_valid & ~p1_is_terminal
    p2_q_td_used = p2_q_bootstrap_valid & ~p2_is_terminal

    metrics = {
        "loss": total_loss.item(),
        "gamma": float(gamma),
        "p1_terminal_fraction": p1_is_terminal.float().mean().item(),
        "p2_terminal_fraction": p2_is_terminal.float().mean().item(),
        "p1_value_td_fraction": p1_value_td_used.float().mean().item(),
        "p2_value_td_fraction": p2_value_td_used.float().mean().item(),
        "p1_q_td_fraction": p1_q_td_used.float().mean().item(),
        "p2_q_td_fraction": p2_q_td_used.float().mean().item(),
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
        "value_loss": value_loss.item(),
        "value_loss_p1": value_loss_p1.item(),
        "value_loss_p2": value_loss_p2.item(),
        "q_value_loss": q_value_loss.item(),
        "q_value_loss_p1": q_value_loss_p1.item(),
        "q_value_loss_p2": q_value_loss_p2.item(),
        "policy_loss": policy_loss.item(),
        "policy_loss_p1": policy_loss_p1.item(),
        "policy_loss_p2": policy_loss_p2.item(),
        "value_teacher_loss": value_teacher_loss.item(),
        "value_teacher_loss_p1": value_teacher_loss_p1.item(),
        "value_teacher_loss_p2": value_teacher_loss_p2.item(),
        "q_teacher_loss": q_teacher_loss.item(),
        "q_teacher_loss_p1": q_teacher_loss_p1.item(),
        "q_teacher_loss_p2": q_teacher_loss_p2.item(),
        "p1_value_prob": (
            torch.sigmoid(outputs["p1_value_logit"]).detach().float()[outcome_valid].mean().item()
            if "p1_value_logit" in outputs and bool(outcome_valid.any())
            else 0.0
        ),
        "p2_value_prob": (
            torch.sigmoid(outputs["p2_value_logit"]).detach().float()[outcome_valid].mean().item()
            if "p2_value_logit" in outputs and bool(outcome_valid.any())
            else 0.0
        ),
        "p1_chosen_q": (
            p1_chosen_q.detach().float()[outcome_valid].mean().item()
            if bool(outcome_valid.any())
            else 0.0
        ),
        "p2_chosen_q": (
            p2_chosen_q.detach().float()[outcome_valid].mean().item()
            if bool(outcome_valid.any())
            else 0.0
        ),
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
