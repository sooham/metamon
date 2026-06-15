f"""JEPA (Joint Embedding Predictive Architecture) model for raw replay learning.

Architecture overview
---------------------

- Your team encoding + Opponent team encoding
- battle encoding

- do contrastive learning between your team and opponent team
if a battle A uses team A, its embedding should be closer to  
the team A , than team B , which is not used in the battle A 

Partially Observable Markov Decision Process
- (true state i) -- action p1, p2 -> (probabilty distribution over true state i+1)
       |                                      |
       v                                      v
- (observed state_i ) --- action ---> probability distribution over state_{i+1}

- we see the observed state i at any time

- in a JEPA architecture
- encoding of a mutated observation and and the unmutated observation should be close together 
- encoding of the ground truth observation (state with all hidden pokemon opposing player policy)
-   we don't know this 100% though
-   can an encoding we predict from the T'th state have maximum similarity to
-   all the following states the replay?  

-   if we take the current state at timestep T (with same pokemon) and play the opponent policy randomly
-   the encoding for state T will average out all opponent and player policies, will focus only on modelling 
-   the pokemon in play
-   if we average out over unseen pokemon but keep the same player it will model only the opposing player policy

-  probability distribution over pokemon teams | current and all previous states 

--------------------
Current state (aggregation of all knowns over play): 
pokemon field in play (current pokemon in the field, opponent pokemon in play, weather, field effects ) , opponent pokemon team, 
your pokemon team

Ground truth state:
current state + unknown pokemon in opposing team  

- as the battle progresses , the future state contains more information about the ground truth

- so the encoding between a state currently known  
and it's prior states leading to the current state should be close 

- this should pick up variation in revealed future moves, status effects, HP drop etc.

Encoder_state( state at time step T-1)           Encoder_state(state at time step T)
     |                                                     |
     v                                                     v
   encoding_state_{t-1} --->  predictor_{action} ---->  encoding_{t}
     |                                                     |
     v                                                     v
Decoder_state(state at time step T-1)                  Decoder state for time step T
    should match exactly

--------------------


Given a parsed replay string R from one POV of a battle state into a sequence of token IDs
``s = (t_0, ..., t_{T-1})``, three modules operate on it:

1. **Encoder φ** — ``ENC_phi(s) → s_enc`
   A transformer encoder (bidirectional attention) that reads the observed state
   replay and produces a latent vector `s_enc`.  For the
   VAE regularisation, the encoder actually outputs ``(mu, logvar)`` and
   ``s_enc`` is sampled via reparameterisation::

       s_enc = mu + std * ε,   ε ~ N(0, I)

   This ``s_enc`` serves two purposes:
   - **JEPA target** — the predictor tries to estimate it from an earlier state in the battle.
   - **VAE latent** — the decoder reconstructs the original state from it and the KL
     loss pushes ``(mu, std)`` toward an isotropic Gaussian N(0, I).

3. **Predictor μ** — ``PRED(s_source) → s_dest``
   A small MLP based model that maps the s_source latent to an estimate ``s_dest``
   where s_dest is the state derived immediately from s_source

4. **Decoder** — ``DEC(s_enc) → reconstructed state``
   An autoregressive transformer decoder that reconstructs the original
   replay tokens from the latent ``s``.  Uses teacher forcing during
   training and autoregressive sampling at inference.

Losses
------

*JEPA (contrastive prediction) loss* — MSE between the target representation
and the predictor's estimate::

    L_jepa = || s_dest - PRED_psi(s_dest) ||²

*Reconstruction loss* — standard cross-entropy over the token vocabulary
(teacher-forced autoregressive decoding)::

    L_recon = -Σ_t log p(s_t | s_{<t}, s_source)

*KL divergence* — regularises the latent distribution toward an isotropic
Gaussian prior (β-VAE style)::

    L_kl = D_KL( N(mu, σ²) || N(0, I))

Total loss::

    L = L_jepa + β_recon * (L_recon for previous state and next state) + β_kl * (L_kl for previous state and next state)

File layout
-----------
- ``JEPAEncoder``       — transformer encoder (phi)
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

# TODO: calculate and find out
MAX_STATE_LENGTH: int = 5000

# Latent dimension (size of s_enc).
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
        # x is shape (batch, n_heads, sequence_length, attn_dim)
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
        # batch, sequence length, embed dimension
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # TODO: ablation study with XSA - exlusive self attention
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
    """Transformer encoder that maps a state token sequence to a latent vector.
    - ``ENC_phi`` (target encoder) — processes the full state
      Outputs ``(mu, logvar)`` for VAE regularisation; ``s_enc`` is
      sampled via reparameterisation.

    Architecture
    ------------
    ``token_ids`` → 
    token embedding + positional → 
    N × transformer blocks (bidirectional attention) → 
    mean pool over non-pad positions → 
    linear projection(s) → 
    latent vector(s).

    Parameters
    ----------
    vocab_size : int
        state vocabulary size (includes <pad>, <bos>, <eos>, etc.).
    pad_id : int
        Token ID for padding.
    latent_dim : int
        Dimensionality of the output latent vector.
    d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, theta, ffn_activation :
        Standard transformer hyperparameters.
    outputs ``(mu, logvar)`` for VAE reparameterisation.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        latent_dim: int = LATENT_DIM,
        d_model: int = 256, # TODO: the embedding dimension might be too small
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

        # Projection head: pooled representation → latent
        # For VAE mode we need two projections (mu and logvar).
        self.proj_mu = nn.Linear(d_model, latent_dim, bias=False)
        self.proj_logvar = nn.Linear(d_model, latent_dim, bias=False)

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
            token_ids: (B, S) int — state token IDs (padded with ``pad_id``).

        Returns:
            ``(mu, logvar)`` — both (B, latent_dim).
        """
        B, S = token_ids.shape
        device = token_ids.device

        # Token embeddings
        x = self.token_embedding(token_ids)  # (B, S, d_model)

        # Transformer blocks (bidirectional)
        # there is residual connection  inside block
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)  # (B, S, d_model)

        # Mean pool over non-pad positions
        # pad_mask = (token_ids != self.pad_id).unsqueeze(-1).float()  # (B, S, 1)
        # x_sum = (x * pad_mask).sum(dim=1)          # (B, d_model)
        # x_count = pad_mask.sum(dim=1).clamp(min=1) # (B, 1)
        # pooled = x_sum / x_count                   # (B, d_model)
        mu = self.proj_mu(x)
        logvar = self.proj_logvar(x)
        return mu, logvar


# ═════════════════════════════════════════════════════════════════════════
# Predictor μ — MLP mapping masked latent → estimate of unmasked latent
# ═════════════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """Small MLP that predicts the next state's latent from the previous states 
    latent.

    ``s_next = PRED_psi(s_prev)`` where ``s_prev`` comes from
    ``s_prev`` (deterministic context encoder) and the target is
    ``s_next`` . Both are sampled from ``ENC_phi`` (stochastic target encoder).

    Architecture: a few linear layers with GELU activation and LayerNorm.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 512,
        n_hidden: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert latent_dim <= hidden_dim
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

    def forward(self, s_prev: torch.Tensor) -> torch.Tensor:
        """Predict next state latent from previous state latent

        Args:
            s_prev: (B, latent_dim) — context encoding from ENC.

        Returns:
            s_next: (B, latent_dim) — estimate of s_next.
        """
        return self.net(s_prev)


# ═════════════════════════════════════════════════════════════════════════
# Decoder — autoregressive transformer that reconstructs tokens from latent
# ═════════════════════════════════════════════════════════════════════════

class JEPADecoder(nn.Module):
    """Autoregressive decoder that reconstructs the replay token sequence
    from the latent vector ``s_enc``.

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
        # TODO: double check the embedding dimension
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
        # self.lm_head.weight = self.token_embedding.weight

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
        s_enc: torch.Tensor,
        target_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        f"""Training forward pass with teacher forcing.

        Prepends the latent vector as a prefix "pseudo-token" before the
        target sequence, then predicts each token autoregressively.

        Sequence layout:
            [latent_proj(v)]  t_0  t_1  ...  t_{M-1} 
            t_0 is <bos> and t_{M-1} is <eos>

        The loss is computed on positions t_0 … t_{M-1} (standard next-token
        prediction, shifted by one).

        Args:
            s_enc:   (B, latent_dim)     — latent sampled from JEPAEncoder 
            target_ids: (B, M) int       — target token IDs of the original state that produced s_enc
            bos_id:     int              — <bos> token ID.
            eos_id:     int              — <eos> token ID.

        Returns:
            logits: (B, M+1, vocab_size) — next-token logits.
                logits[:, 0, :] predicts target_id for <bos>
                logits[:, t, :] predicts target_ids[:, t-1] for t=1..M,
                and logits[:, M+1, :] predicts the <eos>.
        """
        device = target_ids.device

        # Build decoder input: [latent_prefix, target_ids]
        latent_emb = self.latent_proj(s_enc).unsqueeze(1)  # (B, 1, d_model)

        target_emb = self.token_embedding(target_ids)  # (B, M+1, d_model)

        # Concatenate: [latent target]
        x = torch.cat([latent_emb, target_emb], dim=1)  # (B, M+1, d_model)

        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, 1+M, vocab_size)

        # Return logits for positions after s_enc 
        # logits[:, 1:] corresponds to predicting target_ids 
        # We return everything so the caller can decide the loss mask.
        return logits[:, 1:, :]  # (B, M+1, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        s_enc: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = MAX_STATE_LENGTH,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate tokens from the latent vector.

        Args:
            s_enc:       (B, latent_dim).
            bos_id, eos_id: special token IDs.
            max_new_tokens: maximum tokens to generate.
            temperature:    softmax temperature.

        Returns:
            generated: (B, max_new_tokens) int — token IDs (0-padded).
            lengths:   (B,) int               — actual generated lengths.
        """
        B = s_enc.shape[0]
        device = s_enc.device

        # Start with latent prefix
        latent_emb = self.latent_proj(s_enc).unsqueeze(1)  # (B, 1, d_model)
        x = latent_emb

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
                # TODO: KV caching
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
        vocabulary size.
    pad_id, mask_id, bos_id, eos_id : int
        Special token IDs.
    latent_dim : int
        Dimensionality of the latent space (s_enc).
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
            **enc_cfg,
        )

        # Predictor μ — maps v_masked → e_replay
        self.predictor = JEPAPredictor(
            latent_dim=latent_dim,
            **pred_cfg,
        )

        # Decoder — reconstructs replay tokens from s_enc
        self.decoder = JEPADecoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            pad_id=pad_id,
            **dec_cfg,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample s_enc ~ N(mu, σ²) via reparameterisation.

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
        prev_state_tokens: torch.Tensor, # (B, S) — prev state tokens
        next_state_tokens: torch.Tensor, # (B, S) — next state tokens  
    ) -> dict[str, torch.Tensor]:
        """Full forward pass returning all intermediate latents and logits.

        Args:
            token_ids:         Full (unmasked) replay token IDs.
ß
        Returns:
            Dict with keys:
                mu_prev, logvar_prev   — (B, latent_dim) from Encoder.
                mu_next, logvar_next   — (B, latent_dim) from Encoder.
                enc_prev, enc_next     — (B, latent_dim) sampled latent.
                predicted_next         — (B, latent_dim) predictor output.
                recon_logits           — (B, 1+M, vocab_size) decoder output.
        """
        # ── previous state encoder : encode previous state → (mu, logvar) ──
        mu_prev, logvar_prev = self.enc_phi(prev_state_tokens)
        prev_state_encoded = self.reparameterize(mu_prev, logvar_prev)

        # ── Predictor μ: s_prev → s_next ──
        predicted_next = self.predictor(prev_state_encoded)

        # ── next state encoder : encode next state → (mu, logvar) ──
        mu_next, logvar_next = self.enc_phi(next_state_tokens)
        next_state_encoded = self.reparameterize(mu_next, logvar_next)

        # ── Predictor μ: s_prev → s_next ──
        predicted_next = self.predictor(prev_state_encoded)

        # ── Decoder: reconstruct state from encoding, predict the logits from generation 
        recon_logits_prev = self.decoder(prev_state_encoded, prev_state_tokens,  self.bos_id, self.eos_id)
        recon_logits_next = self.decoder(next_state_encoded, next_state_tokens,  self.bos_id, self.eos_id)

        return {
            "mu_prev": mu_prev,
            "logvar_prev": logvar_prev,
            "enc_prev": prev_state_encoded,

            "mu_next": mu_next,
            "logvar_next": logvar_next,
            "enc_next": next_state_encoded,

            "predicted_next": predicted_next,
            "recon_logits_prev": recon_logits_prev,
            "recon_logits_next": recon_logits_next,
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
def compute_normal_prior_loss(mu, logvar):
    # KL = -0.5 * Σ (1 + logvar - mu² - exp(logvar)) (rao-blackwell)
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

def compute_recon_loss(recon_logits, actual_state_tokens, ignore_loss_tokens):
    # recon_logits[:, t, :] predicts actual_state_tokens[:, t] for t=0..M-1
    # and recon_logits[:, M, :] predicts <eos>.
    # We'll compute loss only on non-pad positions.
    # Target shape: (B, 1+M) 

    # Build loss mask: ignore pad, and any user-specified tokens.
    loss_mask = torch.ones_like(recon_logits, dtype=torch.bool)
    for tok_id in ignore_loss_tokens:
        loss_mask = loss_mask & (recon_logits != tok_id)

    n_active = loss_mask.sum()
    if n_active == 0:
        recon_loss = torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
    else:
        recon_loss = F.cross_entropy(
            recon_logits[loss_mask],
            actual_state_tokens[loss_mask],
            reduction="mean",
        )
        # TODO: this is not useful
        # with torch.no_grad():
        #     preds = recon_logits_prev.argmax(dim=-1)
        #     correct = (preds[loss_mask] == recon_logits_prev[loss_mask]).sum().item()
        #     recon_acc = correct / n_active.item()
    
    return recon_loss

def compute_losses(
    outputs: dict[str, torch.Tensor],
    prev_state_tokens: torch.Tensor,
    next_state_tokens: torch.Tensor,
    pad_id: int,
    beta_recon: float = 1.0,
    beta_kl: float = 0.001,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined JEPA + VAE loss.

    Args:
        outputs: Dict from ``JEPAModel.forward()`` containing:
            - enc_next, predicted_next: (B, latent_dim) — JEPA target & estimate.
            - mu_prev, logvar_prev, mu_next, logvar_next:         (B, latent_dim) — VAE parameters for the states
            - recon_logits_prev, recon_logits_next:       (B, 1+M, vocab_size) — decoder output.
        target_ids:     (B, M) int — ground-truth token IDs for reconstruction.
        pad_id:         Token ID for <pad> (ignored in reconstruction loss).
        beta_recon:     Weight for reconstruction loss.
        beta_kl:        Weight for KL divergence (β-VAE).

    Returns:
        total_loss: scalar tensor.
        metrics: dict with per-loss-component values.
    """
    # ── 1. JEPA loss: MSE(enc_next predicted_next) ─────────────────────────
    # Detach the enc_next so gradients only flow through the predictor branch.
    # This is standard in JEPA: the target branch (the branch which predicts enc_next) is updated via EMA
    # or stop-gradient; here we stop-gradient  w.r.t. the JEPA loss and
    # let it only receive gradients from the VAE loss.
    # TODO: right now, both branches for previous and next prediction are using the same weights, I am not sure if this
    # is completely correct
    jepa_loss = F.mse_loss(outputs["enc_next"].detach(), outputs["predicted_next"])

    # ── 2. VAE reconstruction loss ────────────────────────────────────
    recon_loss_prev = compute_recon_loss(outputs["recon_logits_prev"], prev_state_tokens, { pad_id }),  # (B, 1+M, V)
    recon_loss_next = compute_recon_loss(outputs["recon_logits_next"], next_state_tokens, { pad_id })


    # ── 3. KL divergence: D_KL(N(mu_prev, σ²) || N(0, I)) ─────────────────
    kl_loss_prev = compute_normal_prior_loss(outputs["mu_prev"], outputs["logvar_prev"])
    kl_loss_next = compute_normal_prior_loss(outputs["mu_next"], outputs["logvar_next"])

    # ── Total loss ────────────────────────────────────────────────────
    total_loss = jepa_loss + beta_recon * (recon_loss_next + recon_loss_prev) + beta_kl * (kl_loss_prev + kl_loss_next)

    metrics = {
        "loss": total_loss.item(),
        "jepa_loss": jepa_loss.item(),
        "recon_loss_next": recon_loss_next.item(), 
        "recon_loss_prev": recon_loss_prev.item(), 
        "kl_loss_next": kl_loss_next.item(),
        "kl_loss_prev": kl_loss_prev.item(),
        "beta_recon": beta_recon,
        "beta_kl": beta_kl,
    }

    return total_loss, metrics
