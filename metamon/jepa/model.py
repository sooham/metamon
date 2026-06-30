"""Variational JEPA model for paired-POV world-model learning.

Architecture overview (v4 — paired current-state belief + next-state rollout)
-----------------------------------------------------------------------------

    POV state/header/action blocks to predict state t ─► JEPAEncoder per block ─┐
                                                                                ▼
    history context ctx_T = [team, state_0, p_action_0, o_action_0, state_1, ... state_T-1, p_action_T-1 o_action_T-1]
    the history context is bounded by max history

    Current history ─► SelfStateBeliefEncoder ─► z_T belief sampled as current self state
    Current history ─► OpponentStateBeliefPredictor ─► z_opp_T belief sampled as current opponent state

    z_T, z_opp_T -> JEPAOpponentBeliefPolicy -> mean, variance of action for opponent


    z_T + z_opp_T + p_action_T + o_action_T
        ─► JEPANextStatePredictor ─► z_T+1_mu/logvar -> sample z_T+1

    ┌────────────────────────────────────────────────────────────────────┐
    │ LOSSES:                                                             │
    │   Gaussian NLL(z_T | pred_self_state_mu/logvar from self viewpoint) │
    │   Gaussian NLL(z_opp_T | pred_opp_state_mu/logvar from opponent viewpoint) — opponent state │
    │   Gaussian NLL(action_opp | pred_opp_action_mu/logvar) — action     │
    │   Gaussian NLL(z_T+1 | pred_z_T+1_mu/logvar from self viewpoint) — next state       │
    └────────────────────────────────────────────────────────────────────┘

Modules:

1. **JEPAEncoder φ** — bidirectional transformer over one team-header or state
   block. Attention pools over non-pad tokens → state/header block embedding.

2. **SelfStateBeliefEncoder, OpponentStateBeliefPredictor are instances of JEPAStateBeliefEncoder** — shared model module over
   (history_context), with separate gaussian output which we sample from

3. **JEPAOpponentBeliefPolicy** — given opponent state and current player state, get belief for action

4. **JEPANextStatePredictor** — transformer embedder:
   [current_state_z, own_action, opponent_state, predicted_opponent_action] → next
   current-state latent. During paired supervised training, the next-state
   predictor consumes sampled belief priors from the self-state, opponent-state,
   and opponent-action predictors.

Losses
------

*Self-state NLL* — constant-free diagonal Gaussian negative log likelihood of
the target current-state block latent under the self-state belief distribution::

    L_ss = 0.5 * mean(logvar_self + ||z_T - mu_self||² / exp(logvar_self))

*Next-state NLL* — constant-free diagonal Gaussian negative log likelihood of
the target next current-state block latent under the next-state predictor
distribution::

    L_next = 0.5 * mean(logvar_next + ||z_T+1 - mu_next||² / exp(logvar_next))

*Opponent state NLL* — constant-free diagonal Gaussian negative log likelihood
of the target opponent latent under the belief predictor distribution::

    L_os = 0.5 * mean(logvar_opp + ||z_opp_T - mu_opp||² / exp(logvar_opp))

*Action NLL* — constant-free diagonal Gaussian negative log likelihood of the
target opponent action latent under the belief predictor action distribution::

    L_oa = 0.5 * mean(logvar_action + ||actual_opp_action - mu_action||² / exp(logvar_action))

*SIGReg (not used right now)* — on current-state encoder outputs, next-state target encoder
outputs, and history-context outputs.  Action SIGReg is configurable and is off
by default; predicted Gaussian samples are not regularized.

Total::

    L = λ_ss L_ss + λ_os L_os + λ_oa L_oa + λ_next L_next
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
# only used as default if tokenizer and world model dataset do not contain max_seq_len
CONTEXT_LENGTH: int = 100

# Latent dimension for the main encoder (size of the deterministic embedding e).
LATENT_DIM: int = 100

# ── SIGReg defaults ──────────────────────────────────────────────────────
# Number of random projection directions for sketching (resampled each step).
SIGREG_NUM_SLICES: int = 128
# Number of trapezoidal quadrature points for Epps-Pulley integration.
SIGREG_NUM_POINTS: int = 17
# Integration domain for the characteristic function: [0, domain].
SIGREG_DOMAIN: float = 3.0

_SIGREG_GRID_CACHE: dict[tuple[str, int, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def format_tensor_debug(
    name: str,
    tensor: torch.Tensor,
    *,
    max_values: int = 16,
    pad_id: int | None = None,
    tokenizer: object | None = None,
) -> str:
    """Return a compact, deterministic debug summary for one tensor."""
    with torch.no_grad():
        x = tensor.detach()
        parts = [
            f"{name}: shape={tuple(x.shape)}",
            f"dtype={x.dtype}",
            f"device={x.device}",
        ]
        if tensor.requires_grad:
            parts.append("requires_grad=True")

        numel = x.numel()
        if numel == 0:
            return " | ".join(parts + ["numel=0"])

        if x.dtype == torch.bool:
            true_count = int(x.sum().item())
            parts.append(f"true={true_count}/{numel}")
        elif x.is_floating_point():
            xf = x.float()
            finite = torch.isfinite(xf)
            finite_count = int(finite.sum().item())
            parts.append(f"finite={finite_count}/{numel}")
            if finite_count:
                vals = xf[finite]
                parts.extend([
                    f"min={vals.min().item():.6g}",
                    f"max={vals.max().item():.6g}",
                    f"mean={vals.mean().item():.6g}",
                    f"std={vals.std(unbiased=False).item():.6g}",
                ])
        else:
            parts.extend([
                f"min={int(x.min().item())}",
                f"max={int(x.max().item())}",
            ])
            if pad_id is not None:
                parts.append(f"non_pad={int((x != pad_id).sum().item())}/{numel}")

        preview = x.reshape(-1)[:max_values].detach().cpu().tolist()
        parts.append(f"flat[:{max_values}]={preview}")

        if tokenizer is not None and not x.is_floating_point() and x.dtype != torch.bool and x.ndim >= 1:
            row = x.reshape(-1, x.shape[-1])[0].detach().cpu().tolist()
            if pad_id is not None:
                row = [int(v) for v in row if int(v) != pad_id]
            else:
                row = [int(v) for v in row]
            row = row[:max_values]
            try:
                tokens = tokenizer.detokenize(row)  # type: ignore[attr-defined]
                parts.append(f"tokens[:{max_values}]={tokens}")
            except Exception as exc:
                parts.append(f"tokens_error={type(exc).__name__}: {exc}")

        return " | ".join(parts)


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """Two-layer MLP with LayerNorm + GELU, used as projector / pred_proj.

    Input:
        x: ``(B, input_dim)`` — batch of feature vectors.

    Output:
        y: ``(B, output_dim)`` — projected vectors.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Learned masked attention pooling over a token / block sequence.

    Scores each position with a small MLP, masks out padding, then
    returns a weighted sum of the valid embeddings — a single vector
    per batch element.  This is a learnable alternative to mean-pooling
    or CLS-pooling.

    Input:
        x:          ``(B, S, d_model)`` — per-position embeddings (token
                    states from a transformer output, or block embeddings
                    from a temporal encoder).
        valid_mask: ``(B, S)`` — ``True`` for real positions, ``False``
                    for padding. sets these positions weights to -infinity before softmax.

    Output:
        pooled: ``(B, d_model)`` — one vector per batch item, the
                weighted combination of all valid positions.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
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
    """Transformer encoder that maps one header/state/action token block to an embedding.

    Returns:
      - ``e``: (B, latent_dim) — deterministic embedding, used as input in history

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
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        latent_dim: int = LATENT_DIM,
        gradient_checkpointing: bool = False,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        ffn_activation: str = "gelu",
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
        self.proj_e = MLP(d_model, max(2 * d_model, latent_dim), latent_dim)
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
        (int32) for the backward recompute instead of per-block
        ``[B, T, d_model]`` activations, cutting peak memory ~100× when
        encoding many state blocks in one call.
        """
        valid_mask = token_ids != self.pad_id
        x = self.token_embedding(token_ids)
        for block in self.blocks: # includes residual connections
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
            # context.  The saved context is only *token_ids* (int32),
            # not per-block ``[B, T, d_model]`` activations.
            return torch.utils.checkpoint.checkpoint(
                self._transformer_forward, token_ids, use_reentrant=False
            )
        return self._transformer_forward(token_ids)


# ═════════════════════════════════════════════════════════════════════════
# JEPAStateBeliefEncoder , produces the belief of the next state for the player
# ═════════════════════════════════════════════════════════════════════════

class JEPAStateBeliefEncoder(nn.Module):
    """
    Transformer over header/state/action history interleaved to produce belief over the next state
    Used for the current state from the known players perspective: (SelfStateBeliefEncoder)
    and for opponent state (OpponentStateBeliefPredictor)
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        gradient_checkpointing: bool = False,
        n_heads: int = 5,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 4096, # this is the default, real value will be determined from max_history and dataset
        ffn_activation: str = "gelu",
        proj_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([
            TransformerBlock(
                latent_dim, n_heads, d_ff, dropout, max_seq_len,
                causal=False,
                ffn_activation=ffn_activation,
                use_rope=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(latent_dim)
        #self.pool = attentionpool(latent_dim)  # todo: i don't think attention pooling will be useful here
        #proj_hidden = proj_hidden_dim or (4 * latent_dim)
        #self.proj_e = mlp(latent_dim, proj_hidden, latent_dim)
        # todo: will decoding this from the same head be an issue given that
        # the latent is much smaller than the encoding
        self.state_belief_head_mu = MLP(latent_dim, 2 * latent_dim, latent_dim)
        self.state_belief_head_logvar = MLP(latent_dim, 2 * latent_dim, latent_dim)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an interleaved block history into a Gaussian belief.

        Args:
            state_embs: ``[B, S, D]`` or ``[B, K, S, D]``. Index 0 is the
                team header, later indices are state blocks.
            state_valid: ``[B, S]`` or ``[B, K, S]``.
            player_action_embs: ``[B, A, D]`` or ``[B, K, A, D]``.
            player_action_valid: ``[B, A]`` or ``[B, K, A]``.
            opponent_action_embs: ``[B, A, D]`` or ``[B, K, A, D]``.
            opponent_action_valid: ``[B, A]`` or ``[B, K, A]``.
        """
        if state_embs.ndim == 4:
            b, k, s, d = state_embs.shape
            a = player_action_embs.shape[2]
            oa = opponent_action_embs.shape[2]
            mu, logvar = self.forward(
                state_embs.reshape(b * k, s, d),
                state_valid.reshape(b * k, s),
                player_action_embs.reshape(b * k, a, d),
                player_action_valid.reshape(b * k, a),
                opponent_action_embs.reshape(b * k, oa, d),
                opponent_action_valid.reshape(b * k, oa),
            )
            return mu.reshape(b, k, d), logvar.reshape(b, k, d)

        x, valid = self._interleave_history_blocks(
            state_embs,
            state_valid,
            player_action_embs,
            player_action_valid,
            opponent_action_embs,
            opponent_action_valid,
        )

        if self.gradient_checkpointing and self.training:
            # checkpoint the entire transformer stack to save only one copy
            # of [b, max_seq, d_model] instead of one per block (4× reduction).
            return torch.utils.checkpoint.checkpoint(
                self._transformer_forward, x, valid, use_reentrant=False
            )
        return self._transformer_forward(x, valid)

    @staticmethod
    def _interleave_history_blocks(
        state_embs: torch.Tensor,
        state_valid: torch.Tensor,
        player_action_embs: torch.Tensor,
        player_action_valid: torch.Tensor,
        opponent_action_embs: torch.Tensor,
        opponent_action_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave as ``team, state_i, player_action_i, opponent_action_i``.

        ``state_embs`` is expected to have already had the latest/current state
        masked out by ``encode_history_context``. Actions remain valid, so the
        context can end on the latest observed action pair rather than leaking
        the state produced by those actions.
        """
        b, s, d = state_embs.shape
        a = player_action_embs.shape[1]
        oa = opponent_action_embs.shape[1]
        max_steps = max(max(s - 1, 0), a, oa)
        max_seq = 1 + 3 * max_steps

        x = state_embs.new_zeros((b, max_seq, d))
        valid = torch.zeros((b, max_seq), device=state_embs.device, dtype=torch.bool)
        if s == 0:
            return x, valid

        x[:, 0, :] = state_embs[:, 0, :]
        valid[:, 0] = state_valid[:, 0]

        for step in range(max_steps):
            state_idx = step + 1
            state_pos = 1 + 3 * step
            player_pos = state_pos + 1
            opponent_pos = state_pos + 2

            if state_idx < s:
                step_state_valid = state_valid[:, state_idx]
                x[:, state_pos, :] = state_embs[:, state_idx, :]
                valid[:, state_pos] = step_state_valid
            else:
                step_state_valid = torch.zeros(b, device=state_embs.device, dtype=torch.bool)

            if step < a:
                x[:, player_pos, :] = player_action_embs[:, step, :]
                valid[:, player_pos] = player_action_valid[:, step] & step_state_valid
            if step < oa:
                x[:, opponent_pos, :] = opponent_action_embs[:, step, :]
                valid[:, opponent_pos] = opponent_action_valid[:, step] & step_state_valid

        return x, valid

    def _transformer_forward(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, key_padding_mask=valid)
        x = self.ln_final(x)
        idx = (valid.long().sum(dim=1) - 1).clamp_min(0)
        rows = torch.arange(x.shape[0], device=x.device)
        h = x[rows, idx, :]
        state_belief_mu = self.state_belief_head_mu(h)
        state_belief_logvar = self.state_belief_head_logvar(h)
        return state_belief_mu, state_belief_logvar


class JEPAOpponentBeliefPolicy(nn.Module):
    """Predict the opponent action latent distribution from paired state beliefs.

    Inputs:
        current_state: ``(..., latent_dim)`` self/current POV belief sample.
        opponent_state: ``(..., latent_dim)`` opponent POV belief sample.

    Returns:
        action_mu, action_logvar: each ``(..., action_latent_dim)``.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int | None = None,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (4 * latent_dim)
        in_dim = 4 * latent_dim
        layers: list[nn.Module] = []
        for _ in range(max(n_layers - 1, 0)):
            layers.extend([
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        current_state: torch.Tensor,
        opponent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([
            current_state,
            opponent_state,
            current_state - opponent_state,
            current_state * opponent_state,
        ], dim=-1)
        x = self.backbone(x)
        return self.mu_head(x), self.logvar_head(x)


class JEPANextStatePredictor(nn.Module):
    """Transformer embedder for the next self-POV state belief.

    The predictor treats the four latent inputs as a short sequence ordered as:
    ``current_state, opponent_state, own_action, opponent_action``. RoPE gives
    the four slots positional structure inside the transformer. The
    contextualized current-state token is projected with linear Gaussian heads
    to produce next-state ``mu`` and ``logvar``.

    The public argument order is kept compatible with existing call sites:
    ``(current_state, own_action, opponent_state, opponent_action)``.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int | None = None,
        n_heads: int = 5,
        n_layers: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        ffn_activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.max_seq_len = 4
        d_ff = d_ff or hidden_dim or (4 * latent_dim)
        if latent_dim % n_heads != 0:
            raise ValueError(
                f"latent_dim={latent_dim} must be divisible by n_heads={n_heads}"
            )
        if (latent_dim // n_heads) % 2 != 0:
            raise ValueError(
                "RoPE requires an even per-head dimension; got "
                f"latent_dim={latent_dim}, n_heads={n_heads}"
            )
        self.latent_dim = latent_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([
            TransformerBlock(
                latent_dim,
                n_heads,
                d_ff,
                dropout,
                self.max_seq_len,
                causal=False,
                ffn_activation=ffn_activation,
                use_rope=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(latent_dim)
        self.mu_head = MLP(latent_dim, latent_dim*2, latent_dim)
        self.logvar_head = MLP(latent_dim, latent_dim*2, latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return x[:, 0, :]

    def forward(
        self,
        current_state: torch.Tensor,
        own_action: torch.Tensor,
        opponent_state: torch.Tensor,
        opponent_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading_shape = current_state.shape[:-1]
        x = torch.stack(
            [current_state, opponent_state, own_action, opponent_action],
            dim=-2,
        )
        x = x.reshape(-1, 4, self.latent_dim)
        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._transformer_forward, x, use_reentrant=False
            )
        else:
            x = self._transformer_forward(x)
        x = x.reshape(*leading_shape, self.latent_dim)
        return self.mu_head(x), self.logvar_head(x)


def _valid_rope_heads(latent_dim: int, n_heads: int) -> bool:
    return n_heads > 0 and latent_dim % n_heads == 0 and (latent_dim // n_heads) % 2 == 0


def _choose_rope_heads(latent_dim: int, preferred: int, fallback: int = 5) -> int:
    for n_heads in (preferred, fallback):
        if _valid_rope_heads(latent_dim, int(n_heads)):
            return int(n_heads)
    for n_heads in range(min(latent_dim, max(preferred, fallback, 16)), 0, -1):
        if _valid_rope_heads(latent_dim, n_heads):
            return n_heads
    raise ValueError(f"No valid RoPE head count for latent_dim={latent_dim}")


def _latent_transformer_cfg(
    cfg: Optional[dict],
    latent_dim: int,
    *,
    default_heads: int = 5,
) -> dict:
    cleaned = dict(cfg or {})
    cleaned.pop("d_model", None)
    cleaned["n_heads"] = _choose_rope_heads(
        latent_dim,
        int(cleaned.get("n_heads", default_heads)),
        default_heads,
    )
    return cleaned


def _opponent_policy_cfg(cfg: Optional[dict]) -> dict:
    cleaned = dict(cfg or {})
    if "hidden_dim" not in cleaned and "d_ff" in cleaned:
        cleaned["hidden_dim"] = cleaned["d_ff"]
    for key in (
        "d_model",
        "d_ff",
        "n_heads",
        "max_seq_len",
        "ffn_activation",
        "gradient_checkpointing",
    ):
        cleaned.pop(key, None)
    return cleaned


def _next_state_predictor_cfg(
    cfg: Optional[dict],
    latent_dim: int,
    *,
    default_heads: int = 5,
) -> dict:
    cleaned = dict(cfg or {})
    cleaned.pop("d_model", None)
    cleaned["n_heads"] = _choose_rope_heads(
        latent_dim,
        int(cleaned.get("n_heads", default_heads)),
        default_heads,
    )
    return cleaned

# class JEPAGatedResidualBlock(nn.Module):
#     """Small gated residual MLP block for decision-state refinement."""
#  # TODO : revise

#     def __init__(
#         self,
#         dim: int,
#         hidden_dim: int | None = None,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         hidden_dim = hidden_dim or (2 * dim)
#         self.norm = nn.LayerNorm(dim)
#         self.ff = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#         )
#         self.gate = nn.Linear(dim, dim)
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.norm(x)
#         return x + torch.sigmoid(self.gate(h)) * self.ff(h)


# class JEPADecisionStateEncoder(nn.Module):
#     """Fuse self state with predicted opponent Gaussian belief for decisions."""
#  # TODO : revise

#     def __init__(
#         self,
#         latent_dim: int = LATENT_DIM,
#         decision_dim: int = DECISION_DIM,
#         hidden_dim: int | None = None,
#         n_layers: int = 4,
#         dropout: float = 0.1,
#         min_logvar: float = -8.0,
#         max_logvar: float = 6.0,
#     ):
#         super().__init__()
#         hidden_dim = hidden_dim or (2 * decision_dim)
#         self.decision_dim = decision_dim
#         self.min_logvar = min_logvar
#         self.max_logvar = max_logvar

#         self.self_proj = nn.Sequential(
#             nn.Linear(latent_dim, decision_dim, bias=False),
#             nn.LayerNorm(decision_dim),
#             nn.GELU(),
#         )
#         self.opp_mu_proj = nn.Sequential(
#             nn.Linear(latent_dim, decision_dim, bias=False),
#             nn.LayerNorm(decision_dim),
#             nn.GELU(),
#         )
#         self.opp_logvar_proj = nn.Sequential(
#             nn.Linear(latent_dim, decision_dim, bias=False),
#             nn.LayerNorm(decision_dim),
#             nn.GELU(),
#         )
#         self.fuse = nn.Sequential(
#             nn.Linear(5 * decision_dim, decision_dim, bias=False),
#             nn.LayerNorm(decision_dim),
#             nn.GELU(),
#         )
#         self.blocks = nn.ModuleList([
#             JEPAGatedResidualBlock(decision_dim, hidden_dim=hidden_dim, dropout=dropout)
#             for _ in range(n_layers)
#         ])
#         self.out_norm = nn.LayerNorm(decision_dim)
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(
#         self,
#         self_state: torch.Tensor,
#         opponent_state_mu: torch.Tensor,
#         opponent_state_logvar: torch.Tensor,
#     ) -> torch.Tensor:
#         self_h = self.self_proj(self_state)
#         mu_h = self.opp_mu_proj(opponent_state_mu)
#         logvar_h = self.opp_logvar_proj(
#             opponent_state_logvar.clamp(min=self.min_logvar, max=self.max_logvar)
#         )
#         x = self.fuse(torch.cat([
#             self_h,
#             mu_h,
#             logvar_h,
#             self_h - mu_h,
#             self_h * mu_h,
#         ], dim=-1))
#         for block in self.blocks:
#             x = block(x)
#         return self.out_norm(x)


# class JEPAValueHead(nn.Module):
#     """Action-free value critic V(s), returned as a terminal-outcome logit."""
#  # TODO : revise

#     def __init__(
#         self,
#         decision_dim: int = DECISION_DIM,
#         hidden_dim: int | None = None,
#         n_layers: int = 3,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         hidden_dim = hidden_dim or decision_dim
#         layers: list[nn.Module] = []
#         in_dim = decision_dim
#         for i in range(n_layers):
#             if i < n_layers - 1:
#                 layers.extend([
#                     nn.Linear(in_dim, hidden_dim, bias=False),
#                     nn.LayerNorm(hidden_dim),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                 ])
#                 in_dim = hidden_dim
#             else:
#                 layers.append(nn.Linear(in_dim, 1))
#         self.net = nn.Sequential(*layers)
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(self, decision_state: torch.Tensor) -> torch.Tensor:
#         return self.net(decision_state).squeeze(-1)


# class JEPAActionProjector(nn.Module):
#     """Project action encoder latents into the decision/Q space."""
#  # TODO : revise

#     def __init__(
#         self,
#         action_latent_dim: int = ACTION_LATENT_DIM,
#         decision_dim: int = DECISION_DIM,
#         hidden_dim: int | None = None,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         hidden_dim = hidden_dim or decision_dim
#         self.net = nn.Sequential(
#             nn.Linear(action_latent_dim, hidden_dim, bias=False),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, decision_dim, bias=False),
#             nn.LayerNorm(decision_dim),
#             nn.GELU(),
#         )
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(self, action_latent: torch.Tensor) -> torch.Tensor:
#         return self.net(action_latent)


# class JEPAActionValueHead(nn.Module):
#     """Q(s,a) scorer over current-player legal action candidates."""
#  # TODO : revise

#     def __init__(
#         self,
#         decision_dim: int = DECISION_DIM,
#         hidden_dim: int | None = None,
#         n_layers: int = 3,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         hidden_dim = hidden_dim or (2 * decision_dim)
#         self.bilinear = nn.Bilinear(decision_dim, decision_dim, 1, bias=False)
#         layers: list[nn.Module] = []
#         in_dim = 4 * decision_dim
#         for i in range(n_layers):
#             if i < n_layers - 1:
#                 layers.extend([
#                     nn.Linear(in_dim, hidden_dim, bias=False),
#                     nn.LayerNorm(hidden_dim),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                 ])
#                 in_dim = hidden_dim
#             else:
#                 layers.append(nn.Linear(in_dim, 1))
#         self.joint = nn.Sequential(*layers)
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Bilinear)):
#             nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if getattr(module, "bias", None) is not None:
#                 nn.init.zeros_(module.bias)

#     def forward(self, decision_state: torch.Tensor, action_state: torch.Tensor) -> torch.Tensor:
#         if action_state.ndim == decision_state.ndim + 1:
#             decision_state = decision_state.unsqueeze(-2).expand_as(action_state)
#         x = torch.cat([
#             decision_state,
#             action_state,
#             decision_state - action_state,
#             decision_state * action_state,
#         ], dim=-1)
#         return (self.bilinear(decision_state, action_state) + self.joint(x)).squeeze(-1)


class PairedJEPAModel(nn.Module):
    """JEPA trained from both synchronized player perspectives of one battle.

    Each transition receives explicit target-excluded histories.  For target
    state T, the history contains the team header plus prior state/action
    blocks, but not state T.  The model predicts self state T, opponent state
    T, opponent action T, and next self state T+1 with diagonal Gaussians.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        latent_dim: int = LATENT_DIM,
        encoder_cfg: Optional[dict] = None,
        self_belief_encoder_cfg: Optional[dict] = None,
        opponent_belief_predictor_cfg: Optional[dict] = None,
        opponent_policy_belief_cfg: Optional[dict] = None,
        next_state_predictor_cfg: Optional[dict] = None,
        encoder_chunk_tokens: int = 65536,
        belief_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.latent_dim = latent_dim
        self.action_latent_dim = latent_dim
        self.debug_tensors = False
        self.debug_tensor_max_steps = 1
        self.debug_tensor_max_values = 16
        self._debug_tensor_forward_count = 0
        self.encoder_chunk_tokens = int(encoder_chunk_tokens)
        self.belief_batch_size = int(belief_batch_size)

        enc_cfg = dict(encoder_cfg or {})
        self_belief_enc_cfg = _latent_transformer_cfg(self_belief_encoder_cfg, latent_dim)
        opp_belief_predict_cfg = _latent_transformer_cfg(opponent_belief_predictor_cfg, latent_dim)
        opp_policy_belief_cfg = _opponent_policy_cfg(opponent_policy_belief_cfg)
        next_state_predictor_cfg = _next_state_predictor_cfg(next_state_predictor_cfg, latent_dim)


        self.encoder = JEPAEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            **enc_cfg,
        )
        self.self_belief_encoder = JEPAStateBeliefEncoder(
            latent_dim=latent_dim,
            **self_belief_enc_cfg,
        )

        self.opp_belief_predictor = JEPAStateBeliefEncoder(
            latent_dim=latent_dim,
            **opp_belief_predict_cfg,
        )

        self.opp_action_policy_predictor = JEPAOpponentBeliefPolicy(
            latent_dim=latent_dim,
            **opp_policy_belief_cfg,
        )

        self.next_state_predictor = JEPANextStatePredictor(
            latent_dim=latent_dim,
            **next_state_predictor_cfg,
        )

    def set_debug_tensor_logging(
        self,
        enabled: bool,
        *,
        max_steps: int = 1,
        max_values: int = 16,
    ) -> None:
        """Enable concise stdout tensor summaries for the next forward calls."""
        self.debug_tensors = bool(enabled)
        self.debug_tensor_max_steps = max(0, int(max_steps))
        self.debug_tensor_max_values = max(1, int(max_values))
        self._debug_tensor_forward_count = 0

    def _debug_enabled_for_forward(self) -> bool:
        return self.debug_tensors and self._debug_tensor_forward_count < self.debug_tensor_max_steps

    def _debug_dump_tensors(
        self,
        title: str,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        if not self._debug_enabled_for_forward():
            return
        print(f"\n[JEPA tensor debug] {title}", flush=True)
        for name, tensor in tensors.items():
            print(
                "  " + format_tensor_debug(
                    name,
                    tensor,
                    max_values=self.debug_tensor_max_values,
                    pad_id=self.pad_id,
                ),
                flush=True,
            )

    def _encode_blocks(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        max_chunk_tokens: int | None = None,
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
        flat_out = out.reshape(B * S, self.latent_dim)

        token_budget = self.encoder_chunk_tokens if max_chunk_tokens is None else int(max_chunk_tokens)
        if token_budget <= 0:
            flat_out[valid_idx] = self.encoder(flat_tokens[valid_idx])
            return out

        blocks_per_chunk = max(1, int(token_budget) // max(int(T), 1))
        for start in range(0, int(valid_idx.numel()), blocks_per_chunk):
            chunk_idx = valid_idx[start : start + blocks_per_chunk]
            flat_out[chunk_idx] = self.encoder(flat_tokens[chunk_idx])
        return out

    @staticmethod
    def _forward_belief_encoder_chunked(
        owner: object,
        encoder: nn.Module,
        *ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a belief encoder with an optional cap on flattened batch size."""
        max_batch = int(getattr(owner, "belief_batch_size", 0) or 0)
        if max_batch <= 0 or len(ctx) != 6:
            return encoder(*ctx)

        (
            state_embs,
            state_valid,
            player_action_embs,
            player_action_valid,
            opponent_action_embs,
            opponent_action_valid,
        ) = ctx

        if state_embs.ndim == 4:
            b, k, s, d = state_embs.shape
            a = player_action_embs.shape[2]
            oa = opponent_action_embs.shape[2]
            mu, logvar = PairedJEPAModel._forward_belief_encoder_chunked(
                owner,
                encoder,
                state_embs.reshape(b * k, s, d),
                state_valid.reshape(b * k, s),
                player_action_embs.reshape(b * k, a, d),
                player_action_valid.reshape(b * k, a),
                opponent_action_embs.reshape(b * k, oa, d),
                opponent_action_valid.reshape(b * k, oa),
            )
            return mu.reshape(b, k, d), logvar.reshape(b, k, d)

        if state_embs.ndim != 3 or state_embs.shape[0] <= max_batch:
            return encoder(*ctx)

        mu_chunks: list[torch.Tensor] = []
        logvar_chunks: list[torch.Tensor] = []
        for start in range(0, int(state_embs.shape[0]), max_batch):
            sl = slice(start, start + max_batch)
            mu, logvar = encoder(
                state_embs[sl],
                state_valid[sl],
                player_action_embs[sl],
                player_action_valid[sl],
                opponent_action_embs[sl],
                opponent_action_valid[sl],
            )
            mu_chunks.append(mu)
            logvar_chunks.append(logvar)
        return torch.cat(mu_chunks, dim=0), torch.cat(logvar_chunks, dim=0)

    def encode_history(
        self,
        state_tokens: torch.Tensor,
        state_valid: torch.Tensor,
        player_hist_tokens: torch.Tensor,
        player_hist_valid: torch.Tensor,
        opponent_hist_tokens: torch.Tensor,
        opponent_hist_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_tokens.ndim == 4:
            B, K, S, T = state_tokens.shape # B = batch, K = turn, S = states, T = tokens
            _, _, A, AT = player_hist_tokens.shape # A = action
            _, _, OA, OAT = opponent_hist_tokens.shape # OA = opponent action
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
            (
                state_embs,
                hist_valid,
                player_embs,
                player_valid,
                opponent_embs,
                opponent_valid,
            ) = encoded
            return (
                state_embs.reshape(B, K, S, self.latent_dim),
                hist_valid.reshape(B, K, S),
                player_embs.reshape(B, K, A, self.latent_dim),
                player_valid.reshape(B, K, A),
                opponent_embs.reshape(B, K, OA, self.latent_dim),
                opponent_valid.reshape(B, K, OA),
            )

        state_embs = self._encode_blocks(state_tokens, state_valid)
        player_embs = self._encode_blocks(player_hist_tokens, player_hist_valid)
        opponent_embs = self._encode_blocks(opponent_hist_tokens, opponent_hist_valid)
        return (
            state_embs,
            state_valid,
            player_embs,
            player_hist_valid,
            opponent_embs,
            opponent_hist_valid,
        )

    @staticmethod
    def _last_valid_indices(valid: torch.Tensor) -> torch.Tensor:
        counts = valid.to(dtype=torch.bool).int().sum(dim=-1)
        return (counts - 1).clamp_min(0)

    @staticmethod
    def _drop_current_state_from_history(valid: torch.Tensor) -> torch.Tensor:
        """Return a validity mask for prior/history states, excluding current.

        The team header is kept even for the first decision state.  Action
        histories are left untouched; they represent already-observed actions.
        """
        if valid.ndim == 3:
            b, k, s = valid.shape
            flat = valid.reshape(b * k, s)
            return PairedJEPAModel._drop_current_state_from_history(flat).reshape(b, k, s)
        if valid.ndim != 2:
            raise ValueError(f"state valid mask must be [B,S] or [B,K,S], got {valid.shape}")

        hist_valid = valid.clone()
        counts = valid.to(dtype=torch.bool).int().sum(dim=1)
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
        """Encode the last valid block from a block history tensor."""
        if state_tokens.ndim == 4:
            b, k, s, t = state_tokens.shape
            encoded = self.encode_current_state(
                state_tokens.reshape(b * k, s, t),
                state_valid.reshape(b * k, s),
            )
            return encoded.reshape(b, k, self.latent_dim)
        if state_tokens.ndim != 3:
            raise ValueError(f"state_tokens must be [B,S,T] or [B,K,S,T], got {state_tokens.shape}")
        idx = self._last_valid_indices(state_valid)
        rows = torch.arange(state_tokens.shape[0], device=state_tokens.device)
        return self.encoder(state_tokens[rows, idx, :])

    def encode_history_context(
        self,
        state_tokens: torch.Tensor,
        state_valid: torch.Tensor,
        player_hist_tokens: torch.Tensor,
        player_hist_valid: torch.Tensor,
        opponent_hist_tokens: torch.Tensor,
        opponent_hist_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an already target-excluded POV history."""
        return self.encode_history(
            state_tokens,
            state_valid,
            player_hist_tokens,
            player_hist_valid,
            opponent_hist_tokens,
            opponent_hist_valid,
        )

    def encode_token_tokens(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Encode one token block, preserving an optional rollout dimension."""
        if action_tokens.ndim == 3:
            B, K, T = action_tokens.shape
            encoded = self.encoder(action_tokens.reshape(B * K, T))
            return encoded.reshape(B, K, self.latent_dim)
        if action_tokens.ndim != 2:
            raise ValueError(f"token block must be [B,T] or [B,K,T], got {action_tokens.shape}")
        return self.encoder(action_tokens)

    encode_action_tokens = encode_token_tokens

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
            dtype=torch.int32,
            device=action_tokens.device,
        )

    def forward(
        self,
        *,
        p1_history_T: torch.Tensor,
        p1_history_T_valid: torch.Tensor,
        p1_player_hist_T: torch.Tensor,
        p1_player_hist_T_valid: torch.Tensor,
        p1_opponent_hist_T: torch.Tensor,
        p1_opponent_hist_T_valid: torch.Tensor,
        p1_target_state_T: torch.Tensor,
        p1_next_state_T1: torch.Tensor,
        p1_action_tokens: torch.Tensor,
        actual_p2_action_from_p1_perspective_tokens: torch.Tensor,
        p2_history_T: torch.Tensor,
        p2_history_T_valid: torch.Tensor,
        p2_player_hist_T: torch.Tensor,
        p2_player_hist_T_valid: torch.Tensor,
        p2_opponent_hist_T: torch.Tensor,
        p2_opponent_hist_T_valid: torch.Tensor,
        p2_target_state_T: torch.Tensor,
        p2_next_state_T1: torch.Tensor,
        p2_action_tokens: torch.Tensor,
        actual_p1_action_from_p2_perspective_tokens: torch.Tensor,
        sample_beliefs: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        sample = self.training if sample_beliefs is None else sample_beliefs
        debug_this_forward = bool(getattr(self, "debug_tensors", False)) and (
            int(getattr(self, "_debug_tensor_forward_count", 0))
            < int(getattr(self, "debug_tensor_max_steps", 0))
        )
        if debug_this_forward:
            self._debug_dump_tensors(
                "PairedJEPAModel.forward raw token inputs",
                {
                    "p1_history_T": p1_history_T,
                    "p1_history_T_valid": p1_history_T_valid,
                    "p1_player_hist_T": p1_player_hist_T,
                    "p1_player_hist_T_valid": p1_player_hist_T_valid,
                    "p1_opponent_hist_T": p1_opponent_hist_T,
                    "p1_opponent_hist_T_valid": p1_opponent_hist_T_valid,
                    "p1_target_state_T": p1_target_state_T,
                    "p1_next_state_T1": p1_next_state_T1,
                    "p1_action_tokens": p1_action_tokens,
                    "actual_p2_action_from_p1_perspective_tokens": actual_p2_action_from_p1_perspective_tokens,
                    "p2_history_T": p2_history_T,
                    "p2_history_T_valid": p2_history_T_valid,
                    "p2_player_hist_T": p2_player_hist_T,
                    "p2_player_hist_T_valid": p2_player_hist_T_valid,
                    "p2_opponent_hist_T": p2_opponent_hist_T,
                    "p2_opponent_hist_T_valid": p2_opponent_hist_T_valid,
                    "p2_target_state_T": p2_target_state_T,
                    "p2_next_state_T1": p2_next_state_T1,
                    "p2_action_tokens": p2_action_tokens,
                    "actual_p1_action_from_p2_perspective_tokens": actual_p1_action_from_p2_perspective_tokens,
                },
            )

        ctx_p1_T = self.encode_history_context(
            p1_history_T, p1_history_T_valid,
            p1_player_hist_T, p1_player_hist_T_valid,
            p1_opponent_hist_T, p1_opponent_hist_T_valid,
        )
        ctx_p2_T = self.encode_history_context(
            p2_history_T, p2_history_T_valid,
            p2_player_hist_T, p2_player_hist_T_valid,
            p2_opponent_hist_T, p2_opponent_hist_T_valid,
        )
        if debug_this_forward:
            (
                p1_ctx_state_embs,
                p1_ctx_state_valid,
                p1_ctx_player_embs,
                p1_ctx_player_valid,
                p1_ctx_opponent_embs,
                p1_ctx_opponent_valid,
            ) = ctx_p1_T
            (
                p2_ctx_state_embs,
                p2_ctx_state_valid,
                p2_ctx_player_embs,
                p2_ctx_player_valid,
                p2_ctx_opponent_embs,
                p2_ctx_opponent_valid,
            ) = ctx_p2_T
            self._debug_dump_tensors(
                "JEPAStateBeliefEncoder inputs after encode_history_context",
                {
                    "p1_ctx_state_embs": p1_ctx_state_embs,
                    "p1_ctx_state_valid": p1_ctx_state_valid,
                    "p1_ctx_player_embs": p1_ctx_player_embs,
                    "p1_ctx_player_valid": p1_ctx_player_valid,
                    "p1_ctx_opponent_embs": p1_ctx_opponent_embs,
                    "p1_ctx_opponent_valid": p1_ctx_opponent_valid,
                    "p2_ctx_state_embs": p2_ctx_state_embs,
                    "p2_ctx_state_valid": p2_ctx_state_valid,
                    "p2_ctx_player_embs": p2_ctx_player_embs,
                    "p2_ctx_player_valid": p2_ctx_player_valid,
                    "p2_ctx_opponent_embs": p2_ctx_opponent_embs,
                    "p2_ctx_opponent_valid": p2_ctx_opponent_valid,
                },
            )
        z_p1_T = self.encode_token_tokens(p1_target_state_T)
        z_p2_T = self.encode_token_tokens(p2_target_state_T)
        z_p1_T1 = self.encode_token_tokens(p1_next_state_T1)
        z_p2_T1 = self.encode_token_tokens(p2_next_state_T1)
        if debug_this_forward:
            self._debug_dump_tensors(
                "JEPAEncoder target/action latent outputs",
                {
                    "z_p1_T": z_p1_T,
                    "z_p2_T": z_p2_T,
                    "z_p1_T1": z_p1_T1,
                    "z_p2_T1": z_p2_T1,
                },
            )

        pred_p1_self_T_mu, pred_p1_self_T_logvar = PairedJEPAModel._forward_belief_encoder_chunked(
            self, self.self_belief_encoder, *ctx_p1_T
        )
        pred_p2_self_T_mu, pred_p2_self_T_logvar = PairedJEPAModel._forward_belief_encoder_chunked(
            self, self.self_belief_encoder, *ctx_p2_T
        )
        pred_p2_T_mu, pred_p2_T_logvar = PairedJEPAModel._forward_belief_encoder_chunked(
            self, self.opp_belief_predictor, *ctx_p1_T
        )
        pred_p1_T_mu, pred_p1_T_logvar = PairedJEPAModel._forward_belief_encoder_chunked(
            self, self.opp_belief_predictor, *ctx_p2_T
        )
        if debug_this_forward:
            self._debug_dump_tensors(
                "JEPAStateBeliefEncoder Gaussian outputs",
                {
                    "pred_p1_self_T_mu": pred_p1_self_T_mu,
                    "pred_p1_self_T_logvar": pred_p1_self_T_logvar,
                    "pred_p2_self_T_mu": pred_p2_self_T_mu,
                    "pred_p2_self_T_logvar": pred_p2_self_T_logvar,
                    "pred_p2_T_mu": pred_p2_T_mu,
                    "pred_p2_T_logvar": pred_p2_T_logvar,
                    "pred_p1_T_mu": pred_p1_T_mu,
                    "pred_p1_T_logvar": pred_p1_T_logvar,
                },
            )

        pred_p1_self_T = self.reparameterize(pred_p1_self_T_mu, pred_p1_self_T_logvar, sample)
        pred_p2_self_T = self.reparameterize(pred_p2_self_T_mu, pred_p2_self_T_logvar, sample)
        pred_p2_T = self.reparameterize(pred_p2_T_mu, pred_p2_T_logvar, sample)
        pred_p1_T = self.reparameterize(pred_p1_T_mu, pred_p1_T_logvar, sample)
        if debug_this_forward:
            self._debug_dump_tensors(
                "JEPAOpponentBeliefPolicy inputs",
                {
                    "pred_p1_self_T": pred_p1_self_T,
                    "pred_p2_self_T": pred_p2_self_T,
                    "pred_p2_T": pred_p2_T,
                    "pred_p1_T": pred_p1_T,
                },
            )

        pred_p2_action_mu, pred_p2_action_logvar = self.opp_action_policy_predictor(
            pred_p1_self_T, pred_p2_T
        )
        pred_p1_action_mu, pred_p1_action_logvar = self.opp_action_policy_predictor(
            pred_p2_self_T, pred_p1_T
        )
        pred_p2_action = self.reparameterize(pred_p2_action_mu, pred_p2_action_logvar, sample)
        pred_p1_action = self.reparameterize(pred_p1_action_mu, pred_p1_action_logvar, sample)
        if debug_this_forward:
            self._debug_dump_tensors(
                "JEPAOpponentBeliefPolicy Gaussian outputs",
                {
                    "pred_p2_action_mu": pred_p2_action_mu,
                    "pred_p2_action_logvar": pred_p2_action_logvar,
                    "pred_p1_action_mu": pred_p1_action_mu,
                    "pred_p1_action_logvar": pred_p1_action_logvar,
                    "pred_p2_action": pred_p2_action,
                    "pred_p1_action": pred_p1_action,
                },
            )

        p1_action = self.encode_action_tokens(p1_action_tokens)
        p2_action = self.encode_action_tokens(p2_action_tokens)
        actual_p2_action_from_p1_perspective = self.encode_action_tokens(
            actual_p2_action_from_p1_perspective_tokens
        )
        actual_p1_action_from_p2_perspective = self.encode_action_tokens(
            actual_p1_action_from_p2_perspective_tokens
        )
        if debug_this_forward:
            self._debug_dump_tensors(
                "JEPANextStatePredictor inputs",
                {
                    "p1_next_current_state": pred_p1_self_T,
                    "p1_next_own_action": p1_action,
                    "p1_next_opponent_state": pred_p2_T,
                    "p1_next_opponent_action": pred_p2_action,
                    "p2_next_current_state": pred_p2_self_T,
                    "p2_next_own_action": p2_action,
                    "p2_next_opponent_state": pred_p1_T,
                    "p2_next_opponent_action": pred_p1_action,
                    "actual_p2_action_from_p1_perspective": actual_p2_action_from_p1_perspective,
                    "actual_p1_action_from_p2_perspective": actual_p1_action_from_p2_perspective,
                },
            )

        # ── Predict next visible state (stochastic) ──
        pred_p1_T1_mu, pred_p1_T1_logvar = self.next_state_predictor(
            pred_p1_self_T, p1_action, pred_p2_T, pred_p2_action
        )
        pred_p2_T1_mu, pred_p2_T1_logvar = self.next_state_predictor(
            pred_p2_self_T, p2_action, pred_p1_T, pred_p1_action
        )
        pred_p1_T1 = self.reparameterize(pred_p1_T1_mu, pred_p1_T1_logvar, sample)
        pred_p2_T1 = self.reparameterize(pred_p2_T1_mu, pred_p2_T1_logvar, sample)

        outputs = {
            "enc_p1_T": z_p1_T,
            "enc_p2_T": z_p2_T,
            "enc_p1_T1": z_p1_T1,
            "enc_p2_T1": z_p2_T1,
            "pred_p1_self_T_mu": pred_p1_self_T_mu,
            "pred_p1_self_T_logvar": pred_p1_self_T_logvar,
            "pred_p2_self_T_mu": pred_p2_self_T_mu,
            "pred_p2_self_T_logvar": pred_p2_self_T_logvar,
            "pred_p1_self_T": pred_p1_self_T,
            "pred_p2_self_T": pred_p2_self_T,
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
        }
        if debug_this_forward:
            self._debug_dump_tensors("PairedJEPAModel.forward outputs", outputs)
            self._debug_tensor_forward_count += 1
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


def compute_paired_losses(
    outputs: dict[str, torch.Tensor],
    lambda_self_state: float = 1.0,
    lambda_opponent_state: float = 1.0,
    lambda_action: float = 1.0,
    lambda_next_state: float = 1.0,
    **_: object,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the active NLL-only paired-POV JEPA losses."""
    enc_p1_T = outputs["enc_p1_T"]
    enc_p2_T = outputs["enc_p2_T"]
    enc_p1_T1 = outputs["enc_p1_T1"]
    enc_p2_T1 = outputs["enc_p2_T1"]

    self_state_loss_p1 = gaussian_nll(
        enc_p1_T,
        outputs["pred_p1_self_T_mu"],
        outputs["pred_p1_self_T_logvar"],
    )
    self_state_loss_p2 = gaussian_nll(
        enc_p2_T,
        outputs["pred_p2_self_T_mu"],
        outputs["pred_p2_self_T_logvar"],
    )
    self_state_loss = 0.5 * (self_state_loss_p1 + self_state_loss_p2)

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
    opponent_state_loss = 0.5 * (
        opponent_state_loss_p1_to_p2 + opponent_state_loss_p2_to_p1
    )

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
    next_state_loss = 0.5 * (next_state_loss_p1 + next_state_loss_p2)

    total_loss = (
        lambda_self_state * self_state_loss
        + lambda_opponent_state * opponent_state_loss
        + lambda_action * action_loss
        + lambda_next_state * next_state_loss
    )

    metrics = {
        "loss": total_loss.item(),
        "pred_loss": total_loss.item(),
        "self_state_loss": self_state_loss.item(),
        "self_state_loss_p1": self_state_loss_p1.item(),
        "self_state_loss_p2": self_state_loss_p2.item(),
        "opponent_state_loss": opponent_state_loss.item(),
        "opponent_state_loss_p1_to_p2": opponent_state_loss_p1_to_p2.item(),
        "opponent_state_loss_p2_to_p1": opponent_state_loss_p2_to_p1.item(),
        "action_loss": action_loss.item(),
        "action_loss_p1_to_p2": action_loss_p1_to_p2.item(),
        "action_loss_p2_to_p1": action_loss_p2_to_p1.item(),
        "next_state_loss": next_state_loss.item(),
        "next_state_loss_p1": next_state_loss_p1.item(),
        "next_state_loss_p2": next_state_loss_p2.item(),
        "self_state_logvar_p1": outputs["pred_p1_self_T_logvar"].mean().item(),
        "self_state_logvar_p2": outputs["pred_p2_self_T_logvar"].mean().item(),
        "opponent_state_logvar_p1_to_p2": outputs["pred_p2_T_logvar"].mean().item(),
        "opponent_state_logvar_p2_to_p1": outputs["pred_p1_T_logvar"].mean().item(),
        "action_logvar_p1_to_p2": outputs["pred_p2_action_logvar"].mean().item(),
        "action_logvar_p2_to_p1": outputs["pred_p1_action_logvar"].mean().item(),
        "next_state_logvar_p1": outputs["pred_p1_T1_logvar"].mean().item(),
        "next_state_logvar_p2": outputs["pred_p2_T1_logvar"].mean().item(),
    }
    return total_loss, metrics
