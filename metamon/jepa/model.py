"""LeJEPA (Latent-Euclidean JEPA) model for world-model state learning.

Architecture overview (LeJEPA — Balestriero & LeCun 2025)
--------------------------------------------------------

Encoder(state at time T-1)              Encoder(state at time T)
     |                                        |
     v                                        v
   e_{t-1} (deterministic)  --->  predictor(action) --->  e_{t} (deterministic)
                                                              |
                                                     SIGReg → push e toward N(0, I)

The encoder produces a single deterministic embedding *e* used by the JEPA
predictor.  SIGReg (Sketched Isotropic Gaussian Regularization) constrains
the embeddings to follow an isotropic Gaussian distribution, which is
provably optimal for minimizing downstream prediction risk.

No VAE decoder, no stop-gradient, no teacher-student, no KL divergence.
A single hyperparameter λ balances prediction loss vs. SIGReg.

Two modules:

1. **Encoder φ** — bidirectional transformer → mean pool over non-pad tokens
   → ``e`` (deterministic embedding used for JEPA prediction and SIGReg).

2. **Predictor μ** — A small causal transformer that maps ``(e_prev, action)``
   → ``predicted_next``.  Action is a compact index 0..13.

Losses
------

*JEPA loss* — MSE between target embedding and predictor estimate
(no stop-gradient — SIGReg prevents collapse without asymmetry)::

    L_jepa = || e_next - predictor(e_prev, action) ||²

*SIGReg loss* — Epps-Pulley characteristic function test sketched over
random projection directions, pushing the embedding distribution toward
N(0, I).  Applied to both e_prev and e_next.

Total::

    L = (1 - λ) · L_jepa + λ · (SIGReg(e_prev) + SIGReg(e_next)) / 2
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: calculate and find out (gen1ou is 199)
MAX_STATE_LENGTH: int = 200

# Latent dimension (size of the deterministic embedding e).
LATENT_DIM: int = 32

# Number of possible actions (-1..12 → 0..13 after +1 shift).
NUM_ACTIONS: int = 14

# ── SIGReg defaults ──────────────────────────────────────────────────────
# Number of random projection directions for sketching (resampled each step).
SIGREG_NUM_SLICES: int = 256
# Number of trapezoidal quadrature points for Epps-Pulley integration.
SIGREG_NUM_POINTS: int = 17
# Integration domain for the characteristic function: [-5, 5].
SIGREG_DOMAIN: float = 5.0


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
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_head, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
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
    ):
        super().__init__()
        self.ffn_activation = ffn_activation
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(
            d_model, n_heads, dropout, max_seq_len, causal=causal
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
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
# Encoder — bidirectional transformer → deterministic embedding e
# ═════════════════════════════════════════════════════════════════════════

class JEPAEncoder(nn.Module):
    """Transformer encoder that maps a state token sequence to a single
    deterministic embedding.

    Returns:
      - ``e``: (B, latent_dim) — deterministic embedding, used as input /
               target for the JEPA predictor and regularised via SIGReg.

    Architecture
    ------------
    ``token_ids`` → token embedding →
    N × transformer blocks (bidirectional) →
    mean pool over non-pad positions →
    linear projection → e.

    Parameters
    ----------
    vocab_size : int
        State vocabulary size.
    pad_id : int
        Token ID for padding (embedding vector is zeroed).
    latent_dim : int
        Dimensionality of the output embedding.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, theta, ffn_activation :
        Standard transformer hyperparameters.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        latent_dim: int = LATENT_DIM,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
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

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=False,  # bidirectional for encoding
                ffn_activation=ffn_activation,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # Single projection: pooled representation → deterministic embedding.
        self.proj_e = nn.Linear(d_model, latent_dim, bias=False)

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
            token_ids: (B, S) int — state token IDs (padded with ``pad_id``).

        Returns:
            e: (B, latent_dim) — deterministic embedding.
        """
        # Token embeddings
        x = self.token_embedding(token_ids)  # (B, S, d_model)

        # Transformer blocks (bidirectional)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)  # (B, S, d_model)

        # Mean pool over non-pad positions
        pad_mask = (token_ids != self.pad_id).unsqueeze(-1).float()  # (B, S, 1)
        x_sum = (x * pad_mask).sum(dim=1)          # (B, d_model)
        x_count = pad_mask.sum(dim=1).clamp(min=1) # (B, 1)
        pooled = x_sum / x_count                   # (B, d_model)

        return self.proj_e(pooled)  # (B, latent_dim)


# ═════════════════════════════════════════════════════════════════════════
# Predictor — action-conditioned transformer: e_prev + action → e_next
# ═════════════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """Action-conditioned predictor transformer.

    Maps the deterministic previous-state embedding and the action to an
    estimate of the next-state deterministic embedding::

        predicted_next = PRED(e_prev, action_compact_idx)

    Architecture
    ------------
    1. Project ``e_prev`` from ``latent_dim`` to ``d_model``.
    2. Look up action embedding from a dedicated 14-entry table.
    3. Stack as a 2-token causal sequence: ``[prev_proj, action_emb]``.
    4. Run through N causal transformer blocks.
    5. Pool from the action position (position -1).
    6. Linear projection back to ``latent_dim``.

    The action position has full causal context over the prev-state
    embedding, allowing the transformer to combine "what state are we in"
    with "what action was taken."

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input/output latent vectors.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len :
        Transformer hyperparameters for the predictor.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Project deterministic embedding into predictor space.
        self.prev_proj = nn.Linear(latent_dim, d_model, bias=False)

        # Dedicated action embedding: 14 entries (compact indices 0..13).
        self.action_embedding = nn.Embedding(NUM_ACTIONS, d_model)

        # Causal transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=True,
                ffn_activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, e_prev: torch.Tensor, action_compact_idx: torch.Tensor
    ) -> torch.Tensor:
        """Predict next-state deterministic embedding.

        Args:
            e_prev:             (B, latent_dim) — deterministic prev-state embedding.
            action_compact_idx: (B,) int 0..13  — compact action index.

        Returns:
            predicted_next: (B, latent_dim).
        """
        # Project prev-state embedding.
        prev_emb = self.prev_proj(e_prev).unsqueeze(1)  # (B, 1, d_model)

        # Look up action embedding.
        action_emb = self.action_embedding(action_compact_idx).unsqueeze(1)  # (B, 1, d_model)

        # 2-token causal sequence: [prev, action]
        x = torch.cat([prev_emb, action_emb], dim=1)  # (B, 2, d_model)

        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)

        # Pool from the action position (has causal context over prev_emb).
        pooled = x[:, -1, :]  # (B, d_model)

        return self.out_proj(pooled)  # (B, latent_dim)


# ═════════════════════════════════════════════════════════════════════════
# Full LeJEPA model — encoder + predictor
# ═════════════════════════════════════════════════════════════════════════

class JEPAModel(nn.Module):
    """Top-level LeJEPA model for learning representations from world-model states.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    pad_id, bos_id, eos_id : int
        Special token IDs (bos_id, eos_id kept for compatibility).
    latent_dim : int
        Dimensionality of the latent space.
    encoder_cfg, predictor_cfg : dict
        Sub-module configuration dicts.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        latent_dim: int = LATENT_DIM,
        encoder_cfg: Optional[dict] = None,
        predictor_cfg: Optional[dict] = None,
        **kwargs,  # absorb legacy keys (decoder_cfg) silently
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.latent_dim = latent_dim

        enc_cfg = encoder_cfg or {}
        pred_cfg = predictor_cfg or {}

        # Encoder φ — produces deterministic embedding e.
        self.encoder = JEPAEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            **enc_cfg,
        )

        # Predictor μ — maps (e_prev, action) → predicted e_next.
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            **pred_cfg,
        )

    def forward(
        self,
        prev_state_tokens: torch.Tensor,
        next_state_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            prev_state_tokens: (B, S_prev) int — previous state tokens (with <eos>).
            next_state_tokens: (B, S_next) int — next state tokens (with <eos>).
            actions:           (B,) int       — action indices (-1 .. 12).

        Returns:
            Dict with keys:
                e_prev, e_next     — (B, latent_dim) deterministic embeddings.
                predicted_next     — (B, latent_dim) predictor output.
        """
        # ── Encode previous state ──
        e_prev = self.encoder(prev_state_tokens)

        # ── Encode next state ──
        e_next = self.encoder(next_state_tokens)

        # ── Predict next from prev + action ──
        compact_idx = actions + 1  # -1..12 → 0..13
        predicted_next = self.predictor(e_prev, compact_idx)

        return {
            "e_prev": e_prev,
            "e_next": e_next,
            "predicted_next": predicted_next,
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
        domain:     integration domain [-domain, domain] for the CF.

    Returns:
        Scalar SIGReg loss (averaged over slices, scaled by batch size).
    """
    B, D = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype

    # Sample random projection directions (unit norm).
    A = torch.randn(D, num_slices, device=device, dtype=dtype)
    A = A / A.norm(p=2, dim=0, keepdim=True)

    # Project embeddings onto random directions: (B, num_slices).
    proj = embeddings @ A  # (B, num_slices)

    # Integration points for the characteristic function.
    t = torch.linspace(-domain, domain, num_points, device=device, dtype=dtype)

    # Target CF: φ(t) = exp(-t²/2) for N(0, 1).
    # Gaussian weighting window w(t) = exp(-t²/2) (from Epps-Pulley).
    target_cf = torch.exp(-0.5 * t ** 2)         # (T,)
    window = target_cf.clone()                    # w(t) = exp(-t²/2)

    # Empirical CF: (1/B) Σ exp(i · t · proj_n).
    # proj: (B, S) → unsqueeze → (B, S, 1) * t → (B, S, T).
    # exp(i·x) = cos(x) + i·sin(x); |·|² works on complex.
    proj_expanded = proj.unsqueeze(-1) * t  # (B, num_slices, T)
    ecf = torch.exp(1j * proj_expanded.to(torch.complex64)).mean(dim=0)  # (num_slices, T)

    # Weighted L² distance: |ecf - target_cf|² · w(t).
    err = (ecf - target_cf.to(torch.complex64)).abs().square()
    err = err.to(dtype) * window  # (num_slices, T)

    # Trapezoidal integration over t, average over slices, scale by batch size.
    integrated = torch.trapz(err, t, dim=1)  # (num_slices,)
    return integrated.mean() * B


# ═════════════════════════════════════════════════════════════════════════
# Loss computation
# ═════════════════════════════════════════════════════════════════════════

def compute_losses(
    outputs: dict[str, torch.Tensor],
    lambda_sigreg: float = 0.05,
    sigreg_num_slices: int = SIGREG_NUM_SLICES,
    sigreg_num_points: int = SIGREG_NUM_POINTS,
    sigreg_domain: float = SIGREG_DOMAIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the LeJEPA loss: JEPA prediction + SIGReg.

    No stop-gradient — SIGReg prevents representational collapse without
    asymmetric architecture tricks.

    Args:
        outputs: Dict from ``JEPAModel.forward()`` with keys
                 "e_prev", "e_next", "predicted_next".
        lambda_sigreg: Weight for SIGReg (0..1).  Default 0.05.
        sigreg_num_slices, sigreg_num_points, sigreg_domain:
            SIGReg hyperparameters.

    Returns:
        total_loss: scalar tensor.
        metrics: dict with per-loss-component values.
    """
    e_prev = outputs["e_prev"]
    e_next = outputs["e_next"]
    predicted_next = outputs["predicted_next"]

    # ── 1. JEPA loss: MSE between target and prediction ──────────────
    # No stop-gradient — both encoders get gradient signal.
    # SIGReg prevents collapse; asymmetry is unnecessary.
    jepa_loss = F.mse_loss(e_next, predicted_next)

    # ── 2. SIGReg on both embeddings ─────────────────────────────────
    sigreg_prev = sigreg(e_prev, sigreg_num_slices, sigreg_num_points, sigreg_domain)
    sigreg_next = sigreg(e_next, sigreg_num_slices, sigreg_num_points, sigreg_domain)
    sigreg_loss = (sigreg_prev + sigreg_next) / 2.0

    # ── Total loss ──────────────────────────────────────────────────
    total_loss = (1.0 - lambda_sigreg) * jepa_loss + lambda_sigreg * sigreg_loss

    metrics = {
        "loss": total_loss.item(),
        "jepa_loss": jepa_loss.item(),
        "sigreg_prev": sigreg_prev.item(),
        "sigreg_next": sigreg_next.item(),
        "sigreg_loss": sigreg_loss.item(),
    }

    return total_loss, metrics
