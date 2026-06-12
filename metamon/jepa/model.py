"""JEPA (Joint Embedding Predictive Architecture) model for raw replay learning.

Architecture overview
---------------------

Given a raw replay string R tokenized via BPE into a sequence of token IDs
``x = (x_0, ..., x_{T-1})``, three modules operate on it:

1. **Encoder φ** — ``ENC_phi(x) → v_replay``
   A transformer encoder (bidirectional attention) that reads the full
   (unmasked) replay and produces a latent vector ``v_replay``.  For the
   VAE regularisation, the encoder actually outputs ``(mu, logvar)`` and
   ``v_replay`` is sampled via reparameterisation::

       v_replay = mu + std * ε,   ε ~ N(0, I)

   This ``v_replay`` serves two purposes:
   - **JEPA target** — the predictor tries to estimate it from a masked view.
   - **VAE latent** — the decoder reconstructs the replay from it and the KL
     loss pushes ``(mu, std)`` toward an isotropic Gaussian N(0, I).

2. **Encoder ψ** — ``ENC_psi(x_masked) → v_masked``
   A second transformer encoder with **different weights** that processes
   a masked version of the replay (random spans replaced with ``<mask>``
   tokens).  Its output is a deterministic latent vector.

3. **Predictor μ** — ``PRED_mu(v_masked) → e_replay``
   A small MLP that maps the masked-view latent to an estimate ``e_replay``
   of the unmasked-view latent ``v_replay``.

4. **Decoder** — ``DEC(x, v_replay) → reconstructed x``
   An autoregressive transformer decoder that reconstructs the original
   replay tokens from the latent ``v_replay``.  Uses teacher forcing during
   training and autoregressive sampling at inference.

Losses
------

*JEPA (contrastive prediction) loss* — MSE between the target representation
and the predictor's estimate::

    L_jepa = || v_replay - PRED_mu(v_masked) ||²

*Reconstruction loss* — standard cross-entropy over the token vocabulary
(teacher-forced autoregressive decoding)::

    L_recon = -Σ_t log p(x_t | x_{<t}, v_replay)

*KL divergence* — regularises the latent distribution toward an isotropic
Gaussian prior (β-VAE style)::

    L_kl = D_KL( N(mu, σ²) || N(0, I) )

Total loss::

    L = L_jepa + β_recon * L_recon + β_kl * L_kl

File layout
-----------
- ``JEPAEncoder``       — transformer encoder (φ / ψ share the same class,
                          instantiated with separate weights).
- ``JEPAPredictor``     — MLP mapping latent → latent.
- ``JEPADecoder``       — autoregressive transformer decoder for VAE
                          reconstruction.
- ``JEPAModel``         — top-level container, forward pass, loss computation.
- ``compute_losses``    — standalone loss-aggregation function.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════
# Layout constants (derived from expected BPE token lengths for a full
# replay — these are rough upper bounds; actual sequences will be padded
# or truncated to these limits).
# ═════════════════════════════════════════════════════════════════════════

# Maximum number of BPE tokens in a single raw-replay string.
# Replays in early generations are shorter; Gen 9 replays can be ~15–30 kB
# of text, which at ~4 chars/token (BPE) yields ~4k–8k tokens.
# We allocate 8192 to be safe and will truncate if exceeded.
MAX_SEQ_LENGTH: int = 8192

# Latent dimension (size of v_replay / v_masked / e_replay).
LATENT_DIM: int = 256

# ═════════════════════════════════════════════════════════════════════════
# Building blocks (shared with metamon/sl/model.py structure)
# ═════════════════════════════════════════════════════════════════════════


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al. 2021) applied per-head to Q and K.

    Identical to the RoPE in ``metamon/sl/model.py``, kept here so the
    JEPA module is self-contained.
    """

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
# Encoder φ / ψ — bidirectional transformer → pooled latent vector
# ═════════════════════════════════════════════════════════════════════════

class JEPAEncoder(nn.Module):
    """Transformer encoder that maps a BPE token sequence to a latent vector.

    Two instances of this class are created with **separate weights**:
    - ``ENC_phi`` (target encoder) — processes the full, unmasked replay.
      Outputs ``(mu, logvar)`` for VAE regularisation; ``v_replay`` is
      sampled via reparameterisation.
    - ``ENC_psi`` (context encoder) — processes the masked replay.
      Outputs a deterministic ``v_masked`` (mean-pooled representation,
      no reparameterisation).

    Architecture
    ------------
    ``token_ids`` → token embedding + positional → N × transformer blocks
    (bidirectional attention) → mean pool over non-pad positions → linear
    projection(s) → latent vector(s).

    Parameters
    ----------
    vocab_size : int
        BPE vocabulary size (includes <mask>, <pad>, <bos>, <eos>, etc.).
    pad_id : int
        Token ID for padding.
    latent_dim : int
        Dimensionality of the output latent vector.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, theta,
    ffn_activation :
        Standard transformer hyperparameters.
    vae_mode : bool
        If True, outputs ``(mu, logvar)`` for VAE reparameterisation.
        If False, outputs a single deterministic vector.
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
        vae_mode: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.vae_mode = vae_mode

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

        # Projection head: pooled representation → latent
        # For VAE mode we need two projections (mu and logvar).
        if vae_mode:
            self.proj_mu = nn.Linear(d_model, latent_dim, bias=False)
            self.proj_logvar = nn.Linear(d_model, latent_dim, bias=False)
        else:
            self.proj = nn.Linear(d_model, latent_dim, bias=False)

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a token sequence into a latent vector.

        Args:
            token_ids: (B, S) int — BPE token IDs (padded with ``pad_id``).

        Returns:
            If ``vae_mode=False``: ``v`` — (B, latent_dim) deterministic vector.
            If ``vae_mode=True``:  ``(mu, logvar)`` — both (B, latent_dim).
        """
        B, S = token_ids.shape
        device = token_ids.device

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

        if self.vae_mode:
            mu = self.proj_mu(pooled)
            logvar = self.proj_logvar(pooled)
            return mu, logvar
        else:
            return self.proj(pooled)


# ═════════════════════════════════════════════════════════════════════════
# Predictor μ — MLP mapping masked latent → estimate of unmasked latent
# ═════════════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """Small MLP that predicts the unmasked-view latent from the masked-view
    latent.

    ``e_replay = PRED_mu(v_masked)`` where ``v_masked`` comes from
    ``ENC_psi`` (deterministic context encoder) and the target is
    ``v_replay`` sampled from ``ENC_phi`` (stochastic target encoder).

    Architecture: a few linear layers with GELU activation and LayerNorm.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 512,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        layers = []
        in_dim = latent_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim, bias=False))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, v_masked: torch.Tensor) -> torch.Tensor:
        """Predict unmasked-view latent from masked-view latent.

        Args:
            v_masked: (B, latent_dim) — context encoding from ENC_psi.

        Returns:
            e_replay: (B, latent_dim) — estimate of v_replay.
        """
        return self.net(v_masked)


# ═════════════════════════════════════════════════════════════════════════
# Decoder — autoregressive transformer that reconstructs tokens from latent
# ═════════════════════════════════════════════════════════════════════════

class JEPADecoder(nn.Module):
    """Autoregressive decoder that reconstructs the replay token sequence
    from the latent vector ``v_replay``.

    During training, teacher forcing is used (the full target sequence is
    fed in).  During inference, tokens are generated one-by-one until
    ``<eos>`` or ``max_new_tokens`` is reached.

    The latent vector is projected and prepended as the first "token"
    embedding, followed by a ``<bos>`` token and then the target sequence.
    Causal attention ensures tokens only attend to preceding tokens
    (including the latent prefix).
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = LATENT_DIM,
        pad_id: int = 0,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        ffn_activation: str = "gelu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.pad_id = pad_id
        self.d_model = d_model

        # Project latent to d_model so it can be prepended as a "token embedding".
        self.latent_proj = nn.Linear(latent_dim, d_model, bias=False)

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, max_seq_len,
                causal=True,  # autoregressive
                ffn_activation=ffn_activation,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

        # Weight tying: output projection shares weights with token embedding.
        self.lm_head.weight = self.token_embedding.weight

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
        self,
        v_replay: torch.Tensor,
        target_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Training forward pass with teacher forcing.

        Prepends the latent vector as a prefix "pseudo-token" before the
        target sequence, then predicts each token autoregressively.

        Sequence layout:
            [latent_proj(v)]  <bos>  t_0  t_1  ...  t_{M-1}  <eos>

        The loss is computed on positions t_0 … <eos> (standard next-token
        prediction, shifted by one).

        Args:
            v_replay:   (B, latent_dim) — latent from ENC_phi.
            target_ids: (B, M) int       — target token IDs (full replay).
            bos_id:     int              — <bos> token ID.
            eos_id:     int              — <eos> token ID.

        Returns:
            logits: (B, M+1, vocab_size) — next-token logits.
                logits[:, t, :] predicts target_ids[:, t] for t=0..M-1,
                and logits[:, M, :] predicts the <eos>.
        """
        B, M = target_ids.shape
        device = target_ids.device

        # Build decoder input: [latent_prefix, <bos>, target_ids]
        latent_emb = self.latent_proj(v_replay).unsqueeze(1)  # (B, 1, d_model)

        bos_emb = self.token_embedding(
            torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        )  # (B, 1, d_model)

        target_emb = self.token_embedding(target_ids)  # (B, M, d_model)

        # Concatenate: [latent | bos | target]
        x = torch.cat([latent_emb, bos_emb, target_emb], dim=1)  # (B, 1+1+M, d_model)

        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, 2+M, vocab_size)

        # Return logits for positions after <bos>:
        # logits[:, 2:] corresponds to predicting target_ids (and <eos> after
        # the last target token).  We return everything so the caller can
        # decide the loss mask.
        return logits[:, 1:, :]  # (B, 1+M, vocab_size), shifted by 1

    @torch.no_grad()
    def generate(
        self,
        v_replay: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 8192,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate tokens from the latent vector.

        Args:
            v_replay:       (B, latent_dim).
            bos_id, eos_id: special token IDs.
            max_new_tokens: maximum tokens to generate.
            temperature:    softmax temperature.

        Returns:
            generated: (B, max_new_tokens) int — token IDs (0-padded).
            lengths:   (B,) int               — actual generated lengths.
        """
        B = v_replay.shape[0]
        device = v_replay.device

        # Start with latent prefix + <bos>
        latent_emb = self.latent_proj(v_replay).unsqueeze(1)  # (B, 1, d_model)
        bos_emb = self.token_embedding(
            torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        )
        x = torch.cat([latent_emb, bos_emb], dim=1)  # (B, 2, d_model)

        # Process the prefix through all blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)

        generated = torch.full(
            (B, max_new_tokens), self.pad_id, dtype=torch.long, device=device
        )
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # First token prediction from the <bos> position
        logits = self.lm_head(x[:, -1:, :])  # (B, 1, vocab_size)

        for step in range(max_new_tokens):
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_token[done] = self.pad_id

            generated[:, step] = next_token
            lengths[~done] += 1
            done = done | (next_token == eos_id)

            if done.all():
                break

            # Embed the new token and run through blocks
            if step < max_new_tokens - 1:
                next_emb = self.token_embedding(next_token).unsqueeze(1)  # (B, 1, d_model)
                x = torch.cat([x, next_emb], dim=1)
                # Re-run transformer on the full sequence (simple but slow;
                # KV caching can be added later).
                for block in self.blocks:
                    x = block(x)
                x = self.ln_final(x)
                logits = self.lm_head(x[:, -1:, :])

        return generated, lengths


# ═════════════════════════════════════════════════════════════════════════
# Full JEPA model — combines encoders, predictor, decoder, and losses
# ═════════════════════════════════════════════════════════════════════════

class JEPAModel(nn.Module):
    """Top-level JEPA model for learning representations from raw replays.

    Parameters
    ----------
    vocab_size : int
        BPE vocabulary size.
    pad_id, mask_id, bos_id, eos_id : int
        Special token IDs.
    latent_dim : int
        Dimensionality of the latent space (v_replay, v_masked, e_replay).
    encoder_cfg, predictor_cfg, decoder_cfg : dict
        Sub-module configuration dicts (see individual classes).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        mask_id: int,
        bos_id: int,
        eos_id: int,
        latent_dim: int = LATENT_DIM,
        encoder_cfg: Optional[dict] = None,
        predictor_cfg: Optional[dict] = None,
        decoder_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.latent_dim = latent_dim

        enc_cfg = encoder_cfg or {}
        pred_cfg = predictor_cfg or {}
        dec_cfg = decoder_cfg or {}

        # Target encoder φ — stochastic (VAE mode), produces (mu, logvar).
        self.enc_phi = JEPAEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            vae_mode=True,  # outputs (mu, logvar) for VAE + JEPA target
            **enc_cfg,
        )

        # Context encoder ψ — deterministic, processes masked replays.
        self.enc_psi = JEPAEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            latent_dim=latent_dim,
            vae_mode=False,  # deterministic output
            **enc_cfg,
        )

        # Predictor μ — maps v_masked → e_replay
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            **pred_cfg,
        )

        # Decoder — reconstructs replay tokens from v_replay
        self.decoder = JEPADecoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            pad_id=pad_id,
            **dec_cfg,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample v_replay ~ N(mu, σ²) via reparameterisation.

        Args:
            mu:     (B, latent_dim).
            logvar: (B, latent_dim).

        Returns:
            v_replay: (B, latent_dim) — sampled latent.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        token_ids: torch.Tensor,       # (B, S) — full unmasked replay tokens
        masked_token_ids: torch.Tensor, # (B, S) — randomly masked version
        target_ids: torch.Tensor,       # (B, M) — full replay for decoder (may equal token_ids)
        mask: Optional[torch.Tensor] = None,  # (B,) bool — which examples to mask (unused for now)
    ) -> dict[str, torch.Tensor]:
        """Full forward pass returning all intermediate latents and logits.

        Args:
            token_ids:         Full (unmasked) replay token IDs.
            masked_token_ids:  Masked replay token IDs.
            target_ids:        Target sequence for reconstruction (typically = token_ids).
            mask:              Optional sample mask (not used currently).

        Returns:
            Dict with keys:
                mu, logvar     — (B, latent_dim) from ENC_phi.
                v_replay       — (B, latent_dim) sampled latent.
                v_masked       — (B, latent_dim) from ENC_psi.
                e_replay       — (B, latent_dim) predictor output.
                recon_logits   — (B, 1+M, vocab_size) decoder output.
        """
        # ── Target encoder φ: encode full replay → (mu, logvar) ──
        mu, logvar = self.enc_phi(token_ids)
        v_replay = self.reparameterize(mu, logvar)

        # ── Context encoder ψ: encode masked replay → v_masked ──
        v_masked = self.enc_psi(masked_token_ids)

        # ── Predictor μ: v_masked → e_replay ──
        e_replay = self.predictor(v_masked)

        # ── Decoder: reconstruct replay from v_replay ──
        recon_logits = self.decoder(v_replay, target_ids, self.bos_id, self.eos_id)

        return {
            "mu": mu,
            "logvar": logvar,
            "v_replay": v_replay,
            "v_masked": v_masked,
            "e_replay": e_replay,
            "recon_logits": recon_logits,
        }

    def save_checkpoint(self, path: str, **extra) -> None:
        ckpt = {"model_state_dict": self.state_dict(), **extra}
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None) -> dict:
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["model_state_dict"])
        return ckpt


# ═════════════════════════════════════════════════════════════════════════
# Loss computation
# ═════════════════════════════════════════════════════════════════════════

def compute_losses(
    outputs: dict[str, torch.Tensor],
    target_ids: torch.Tensor,
    mask_id: int,
    pad_id: int,
    beta_recon: float = 1.0,
    beta_kl: float = 0.001,
    ignore_loss_tokens: Optional[set[int]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined JEPA + VAE loss.

    Args:
        outputs: Dict from ``JEPAModel.forward()`` containing:
            - v_replay, e_replay: (B, latent_dim) — JEPA target & estimate.
            - mu, logvar:         (B, latent_dim) — VAE parameters.
            - recon_logits:       (B, 1+M, vocab_size) — decoder output.
        target_ids:     (B, M) int — ground-truth token IDs for reconstruction.
        mask_id:        Token ID for <mask> (ignored in reconstruction loss).
        pad_id:         Token ID for <pad> (ignored in reconstruction loss).
        beta_recon:     Weight for reconstruction loss.
        beta_kl:        Weight for KL divergence (β-VAE).
        ignore_loss_tokens: Additional token IDs to exclude from recon loss.

    Returns:
        total_loss: scalar tensor.
        metrics: dict with per-loss-component values.
    """
    if ignore_loss_tokens is None:
        ignore_loss_tokens = set()

    # ── 1. JEPA loss: MSE(v_replay, e_replay) ─────────────────────────
    # Detach v_replay so gradients only flow through the predictor branch.
    # This is standard in JEPA: the target branch (φ) is updated via EMA
    # or stop-gradient; here we stop-gradient φ w.r.t. the JEPA loss and
    # let it only receive gradients from the VAE loss.
    jepa_loss = F.mse_loss(outputs["e_replay"], outputs["v_replay"].detach())

    # ── 2. VAE reconstruction loss ────────────────────────────────────
    recon_logits = outputs["recon_logits"]  # (B, 1+M, V)
    # recon_logits[:, t, :] predicts target_ids[:, t] for t=0..M-1
    # and recon_logits[:, M, :] predicts <eos>.
    M = target_ids.shape[1]

    # Build targets including a final <eos> placeholder.
    # We'll compute loss only on non-pad, non-mask positions.
    # Target shape: (B, 1+M) — first position after <bos>, <eos> is appended.
    eos_id_tensor = torch.full(
        (target_ids.shape[0], 1),
        mask_id,  # placeholder — we'll mask it out if mask_id is in ignore set
        dtype=target_ids.dtype,
        device=target_ids.device,
    )
    full_targets = torch.cat([target_ids, eos_id_tensor], dim=1)  # (B, M+1)

    # Build loss mask: ignore pad, mask, and any user-specified tokens.
    loss_mask = torch.ones_like(full_targets, dtype=torch.bool)
    for tok_id in ignore_loss_tokens | {pad_id, mask_id}:
        loss_mask = loss_mask & (full_targets != tok_id)

    n_active = loss_mask.sum()
    if n_active == 0:
        recon_loss = torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
        recon_acc = 0.0
    else:
        recon_loss = F.cross_entropy(
            recon_logits[loss_mask],
            full_targets[loss_mask],
            reduction="mean",
        )
        with torch.no_grad():
            preds = recon_logits.argmax(dim=-1)
            correct = (preds[loss_mask] == full_targets[loss_mask]).sum().item()
            recon_acc = correct / n_active.item()

    # ── 3. KL divergence: D_KL(N(mu, σ²) || N(0, I)) ─────────────────
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    # KL = -0.5 * Σ (1 + logvar - mu² - exp(logvar))
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # ── Total loss ────────────────────────────────────────────────────
    total_loss = jepa_loss + beta_recon * recon_loss + beta_kl * kl_loss

    metrics = {
        "loss": total_loss.item(),
        "jepa_loss": jepa_loss.item(),
        "recon_loss": recon_loss.item() if n_active > 0 else 0.0,
        "recon_accuracy": recon_acc,
        "kl_loss": kl_loss.item(),
        "beta_recon": beta_recon,
        "beta_kl": beta_kl,
    }

    return total_loss, metrics
