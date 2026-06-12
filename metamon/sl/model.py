"""Autoregressive transformer for next-state prediction in Pokémon battles.

Given the current state ``state[t]`` and the action ``action[t]``, the model
autoregressively predicts every token of ``state[t+1]``.

Prompt structure (variable-length per example, padded to batch max):
    <bos>  s₀  s₁  …  s_{L-1}  <eos>  <boa>  a  <eoa>  <bos>  t₀  t₁  …  t_{M-1}  <eos>  <pad>…
    ├──────── state[t] (L tokens) ───┤  ├─act─┤  ├────── state[t+1] (M tokens) ──────┤

The loss is only computed on the **state[t+1]** region (``t₀ … t_{M-1} <eos>``).
At inference time the prompt ends at the second ``<bos>`` and the model
generates until ``<eos>`` or until the maximum context length is exhausted.

Action indices are embedded through a dedicated learnable embedding table
(``action_emb``), not through the token vocabulary.  ``<pad>`` has a non-zero
ID passed as ``padding_idx`` to ``nn.Embedding`` so that its vector is zeroed.
Index 0 of the embedding is permanently unused (no token maps to it).

Constants (derived from WorldModelObservationSpace):
    MAX_STATE_LENGTH  = 312   # maximal token count for a fully-revealed state
    ACTION_OVERHEAD   = 5     # <eos><boa>A<eoa><bos> between state_t and state_next
    SAFETY_FACTOR     = 2.5   # configurable margin for context window
    MAX_CONTEXT_LENGTH = 832  # ceil((MAX_STATE_LENGTH + ACTION_OVERHEAD) * SAFETY_FACTOR / 64) * 64
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Layout constants (derived from WorldModelObservationSpace) ──────────

# Maximum token count for a single state when all Pokémon, moves, boosts,
# conditions, and effects are fully revealed.
MAX_STATE_LENGTH: int = 312

# Number of structural tokens separating state_t from state_next:
#   <eos> (1) + <boa> (1) + action (1) + <eoa> (1) + <bos> (1)
ACTION_OVERHEAD: int = 5

# Safety factor applied to the context window (configurable).
SAFETY_FACTOR: float = 2.5

# Maximum context length the model will ever process.
# Rounded to a multiple of 64 for GPU efficiency.
MAX_CONTEXT_LENGTH: int = (
    math.ceil((MAX_STATE_LENGTH + ACTION_OVERHEAD) * SAFETY_FACTOR / 64) * 64
)  # = 768


# ── RoPE ─────────────────────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al. 2021) applied per-head to Q and K."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                     # (max_seq_len, dim//2)
        self.register_buffer("cos_cached", freqs.cos())      # (max_seq_len, dim//2)
        self.register_buffer("sin_cached", freqs.sin())

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        """Apply RoPE to *x* in-place style (returns rotated tensor).

        Args:
            x: (B, H, S, D_head) — queries or keys.
            offset: starting position (used during autoregressive generation).
        """
        S = x.shape[-2]
        dtype = x.dtype
        # Cache bf16 copies on first use to avoid dtype cast every forward pass.
        if not hasattr(self, '_cos_bf16'):
            self._cos_bf16 = {}
            self._sin_bf16 = {}
        if dtype not in self._cos_bf16:
            self._cos_bf16[dtype] = self.cos_cached.to(dtype=dtype)
            self._sin_bf16[dtype] = self.sin_cached.to(dtype=dtype)
        cos = self._cos_bf16[dtype][offset : offset + S]  # (S, D//2)
        sin = self._sin_bf16[dtype][offset : offset + S]
        # interleave to full head dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)        # (S, D)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)


# ── Causal self-attention block ──────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

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
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(y)

    def forward_with_kv(
        self, x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass that returns (output, updated_kv_cache).

        When *kv_cache* is None (prefill), processes the full sequence
        with causal masking and returns all K,V for the cache.  When
        *kv_cache* is provided (decode), only the new token's K,V are
        computed and concatenated with the cached values — the new token
        attends to all prior tokens (no causal mask).

        Args:
            x: (B, S, D) — input embeddings (S=1 during decode).
            kv_cache: None or (k_cached, v_cached), each (B, H, S_cached, D_head).
            cache_pos: RoPE position offset (0 for prefill, prefix_len + step for decode).

        Returns:
            output: (B, S, D).
            new_cache: (k, v) to replace *kv_cache* for the next step.
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = self.rope(q, offset=cache_pos)
        k = self.rope(k, offset=cache_pos)

        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2)   # (B, H, S_cached+S, D_head)
            v = torch.cat([v_cached, v], dim=2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,  # no dropout during generation
            is_causal=(kv_cache is None),  # causal only during prefill
        )
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(y), (k, v)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with RoPE causal attention + SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        # SwiGLU FFN
        self.ffn_w1 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_w2 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        # SwiGLU
        gate = F.silu(self.ffn_w1(self.ln2(x)))
        up = self.ffn_w2(self.ln2(x))
        x = x + self.dropout(self.ffn_out(gate * up))
        return x

    def forward_with_kv(
        self, x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV cache — see ``CausalSelfAttention.forward_with_kv``."""
        attn_out, new_kv = self.attn.forward_with_kv(
            self.ln1(x), kv_cache=kv_cache, cache_pos=cache_pos
        )
        x = x + attn_out
        gate = F.silu(self.ffn_w1(self.ln2(x)))
        up = self.ffn_w2(self.ln2(x))
        x = x + self.dropout(self.ffn_out(gate * up))
        return x, new_kv


# ── Full model ───────────────────────────────────────────────────────────

class WorldModelTransformer(nn.Module):
    """Autoregressive decoder that predicts state[t+1] from state[t] + action[t].

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the text vocabulary (token IDs are 1-based;
        0 is unused; real tokens start at 1.  ``<pad>`` has a non-zero ID
        passed as ``padding_idx`` to zero its embedding vector.
    max_seq_len : int
        Maximum sequence length the RoPE cache and transformer can handle.
        Should be >= MAX_CONTEXT_LENGTH.
    d_model, n_heads, n_layers, d_ff, dropout, theta:
        Standard transformer hyperparameters.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        action_to_token_id: dict[int, int],
        max_seq_len: int = 1024,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self._action_to_token_id = action_to_token_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            vocab_size + 1, d_model, padding_idx=pad_id
        )  # +1 for unused index 0

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size + 1, bias=False)

        self.apply(self._init_weights)

        # Tie input embedding and output projection weights.
        # Must happen AFTER _init_weights so that the token_embedding
        # zeroing (index 0 + padding_idx) is not undone by the Linear
        # init path overwriting the shared tensor.
        self.lm_head.weight = self.token_embedding.weight

        # Precompute action-index → token-ID lookup table for fast vectorized
        # access in build_prompt.  Action indices range from -1 to 12.
        max_action = max(action_to_token_id.keys())
        min_action = min(action_to_token_id.keys())
        action_lookup = torch.zeros(max_action - min_action + 1, dtype=torch.long)
        for idx, tok in action_to_token_id.items():
            action_lookup[idx - min_action] = tok
        self.register_buffer("_action_lookup", action_lookup, persistent=False)
        # actions + _action_base gives the correct lookup index.
        # For min_action = -1: _action_base = 1, so action 0 → index 1.
        self._action_base = -min_action

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Zero the <pad> token embedding (non-zero ID, but its vector is unused).
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx] = 0.0
            # Index 0 in the token embedding is permanently unused (tokenizer
            # vocabulary starts at 1).
            with torch.no_grad():
                module.weight[0] = 0.0

    # ── Prompt builder ────────────────────────────────────────────────

    def build_prompt(self,
        state_t: torch.Tensor,
        state_next: torch.Tensor,
        actions: torch.Tensor,
        state_t_lengths: torch.Tensor,
        state_next_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        boa_id: int,
        eoa_id: int,
        pad_id: int,
        max_context: int = MAX_CONTEXT_LENGTH,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build variable-length prompts from batched state/action tensors.

        Each example i produces a prompt of length ``L_i + M_i + 7``:

            <bos> state_t[:L_i] <eos> <boa> ACTION <eoa> <bos> state_next[:M_i] <eos>

        Prompts are right-padded with *pad_id* to the batch maximum (capped
        at *max_context*).

        Args:
            state_t:           (B, max_L) int — padded state[t] tokens.
            state_next:        (B, max_M) int — padded state[t+1] tokens.
            actions:           (B,)      int — action indices (-1 … 12).
            state_t_lengths:   (B,)      int — actual token count per state_t.
            state_next_lengths:(B,)      int — actual token count per state_next.
            bos_id, eos_id, boa_id, eoa_id: special token IDs.
            pad_id:            token ID used for padding (default 0).
            max_context:       maximum total prompt length (truncation cap).

        Returns:
            token_ids:  (B, T) int — full prompt with right-padding.
            sn_start:   (B,)   int — index in *targets* (shifted) where
                           state_next prediction begins.
            sn_end:     (B,)   int — index in *targets* where state_next
                           prediction ends (inclusive, points to final <eos>).
        """
        B = state_t.shape[0]
        device = state_t.device
        L = state_t_lengths  # (B,)
        M = state_next_lengths  # (B,)

        # Clamp lengths so total fits in max_context.
        # We need L_i + M_i + 7 ≤ max_context, and both L_i, M_i are fit
        # within their respective padded tensors.
        max_L = state_t.shape[1]
        max_M = state_next.shape[1]
        L_clamped = torch.clamp(L, max=min(max_L, max_context - 7))
        M_budget = max_context - 7 - L_clamped
        M_clamped = torch.clamp(M, max=torch.clamp(M_budget, min=0, max=max_M))

        total_lens = L_clamped + M_clamped + 7  # (B,)
        # Always allocate to max_context so the output shape is static.
        # This avoids .item() (which causes a Dynamo graph break) and
        # the memory overhead is negligible: (B, max_context) int64 =
        # 256 × 832 × 8 = 1.7 MB per batch.
        T = max_context

        # ── Fully vectorized prompt construction ───────────────────
        # All operations are pure tensor ops — no Python loops, no
        # .item() calls, no graph breaks for torch.compile.
        #
        # Position of each element per row i:
        #   column     content
        #    0          <bos>
        #    1..L_i     state_t[i, 0:L_i-1]
        #    L_i+1      <eos>
        #    L_i+2      <boa>
        #    L_i+3      <action>
        #    L_i+4      <eoa>
        #    L_i+5      <bos>
        #    L_i+6..    state_next[i, 0:M_i-1]
        #    L_i+6+M_i  <eos>
        #    rest       <pad>

        pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        # 1) Fill with pad_id.
        token_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)

        # 2) <bos> at column 0.
        token_ids[:, 0] = bos_id

        # 3) state_t block: columns [1, 1+L_i).
        #    Gather from state_t using src index = pos - 1, masked to valid range.
        st_mask = (pos >= 1) & (pos < 1 + L_clamped.unsqueeze(1))  # (B, T)
        st_src = (pos - 1).clamp(min=0, max=state_t.shape[1] - 1).expand(B, -1)
        st_tokens = state_t.gather(1, st_src)  # (B, T) — values outside mask are clobbered
        token_ids = torch.where(st_mask, st_tokens, token_ids)

        # 4) <eos> at column L_i + 1.
        eos1_col = (L_clamped + 1).unsqueeze(1)  # (B, 1)
        eos1_mask = (pos == eos1_col) & (eos1_col < T)
        token_ids = torch.where(eos1_mask, eos_id, token_ids)

        # 5) <boa> at column L_i + 2.
        boa_col = (L_clamped + 2).unsqueeze(1)
        boa_mask = (pos == boa_col) & (boa_col < T)
        token_ids = torch.where(boa_mask, boa_id, token_ids)

        # 6) Action token at column L_i + 3.
        #    Look up action→token via precomputed buffer (no Python loop).
        act_col = (L_clamped + 3).unsqueeze(1)
        act_mask = (pos == act_col) & (act_col < T)
        action_indices = actions + self._action_base  # shift to lookup range
        action_tokens = self._action_lookup[action_indices]  # (B,)
        token_ids = torch.where(act_mask, action_tokens.unsqueeze(1), token_ids)

        # 7) <eoa> at column L_i + 4.
        eoa_col = (L_clamped + 4).unsqueeze(1)
        eoa_mask = (pos == eoa_col) & (eoa_col < T)
        token_ids = torch.where(eoa_mask, eoa_id, token_ids)

        # 8) <bos> at column L_i + 5.
        bos2_col = (L_clamped + 5).unsqueeze(1)
        bos2_mask = (pos == bos2_col) & (bos2_col < T)
        token_ids = torch.where(bos2_mask, bos_id, token_ids)

        # 9) state_next block: columns [L_i+6, L_i+6+M_i).
        sn_start_col = (L_clamped + 6).unsqueeze(1)  # (B, 1)
        sn_mask = (pos >= sn_start_col) & (pos < sn_start_col + M_clamped.unsqueeze(1))
        sn_src = (pos - sn_start_col).clamp(min=0, max=state_next.shape[1] - 1)
        sn_tokens = state_next.gather(1, sn_src)
        token_ids = torch.where(sn_mask, sn_tokens, token_ids)

        # 10) Final <eos> at column L_i + 6 + M_i.
        eos2_col = (L_clamped + 6 + M_clamped).unsqueeze(1)
        eos2_mask = (pos == eos2_col) & (eos2_col < T)
        token_ids = torch.where(eos2_mask, eos_id, token_ids)

        # sn_start / sn_end in *targets* (shifted by 1):
        # targets[t] = token_ids[t+1]
        # state_next[0]  is at token_ids[L_i+6] → targets[L_i+5]
        # final <eos>    is at token_ids[L_i+6+M_i] → targets[L_i+5+M_i]
        sn_start = (L_clamped + 5).long()
        sn_end = (L_clamped + 5 + M_clamped).long()

        return token_ids, sn_start, sn_end

    # ── Training forward ──────────────────────────────────────────────

    def forward(
        self,
        state_t: torch.Tensor,
        state_next: torch.Tensor,
        actions: torch.Tensor,
        state_t_lengths: torch.Tensor,
        state_next_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        boa_id: int,
        eoa_id: int,
        ignore_loss_tokens: Optional[set[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass (teacher forcing).

        Builds a variable-length prompt per example and computes logits over
        the entire sequence (shifted for next-token prediction).  The loss
        mask is True **only** for the state_next region (including its
        closing ``<eos>``), minus any tokens in *ignore_loss_tokens*.

        Args:
            state_t:            (B, max_L) int — padded current state.
            state_next:         (B, max_M) int — padded next state (target).
            actions:            (B,)      int — action indices (-1 … 12).
            state_t_lengths:    (B,)      int — actual token count per state_t.
            state_next_lengths: (B,)      int — actual token count per state_next.
            bos_id, eos_id, boa_id, eoa_id: special token IDs.
            ignore_loss_tokens: set of token IDs to exclude from the loss
                                (e.g. {pad_id, boa_id, eoa_id}).

        Returns:
            logits:    (B, T-1, V) — next-token logits.
            targets:   (B, T-1)   — target token IDs.
            loss_mask: (B, T-1)   — bool mask, True where loss is computed.
        """
        B = state_t.shape[0]
        device = state_t.device

        if ignore_loss_tokens is None:
            ignore_loss_tokens = set()

        # Build variable-length prompts
        token_ids, sn_start, sn_end = self.build_prompt(
            state_t=state_t,
            state_next=state_next,
            actions=actions,
            state_t_lengths=state_t_lengths,
            state_next_lengths=state_next_lengths,
            bos_id=bos_id,
            eos_id=eos_id,
            boa_id=boa_id,
            eoa_id=eoa_id,
            pad_id=self.pad_id,
            max_context=self.max_seq_len,
        )
        T = token_ids.shape[1]

        # ── Embeddings ──────────────────────────────────────────────
        x = self.token_embedding(token_ids)                    # (B, T, d_model)

        # ── Transformer ─────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)                               # (B, T, V)

        # ── Targets & loss mask ─────────────────────────────────────
        # Shift right: logits[t] predicts targets[t] = token_ids[t+1]
        logits = logits[:, :-1, :]                             # (B, T-1, V)
        targets = token_ids[:, 1:]                             # (B, T-1)

        # Loss mask: state_next region + final <eos> for each example.
        # Vectorized using column indices.
        cols = torch.arange(T - 1, device=device).unsqueeze(0)  # (1, T-1)
        sn_start_col = sn_start.unsqueeze(1)   # (B, 1)
        sn_end_col = sn_end.unsqueeze(1)       # (B, 1)
        loss_mask = (cols >= sn_start_col) & (cols <= sn_end_col)
        # Clamp end to valid range
        loss_mask = loss_mask & (cols < T - 1)

        # Exclude ignore_loss_tokens (pad, delimiter tokens, etc.)
        if ignore_loss_tokens:
            ignore_ids = torch.tensor(
                sorted(ignore_loss_tokens), dtype=torch.long, device=device
            )
            loss_mask = loss_mask & ~torch.isin(targets, ignore_ids)

        return logits, targets, loss_mask

    # ── KV-cache-aware token forward ────────────────────────────

    def _forward_tokens_with_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        cache_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Process *token_ids* through the transformer, returning logits and
        per-layer KV caches.

        Args:
            token_ids: (B, S) int — input token IDs (S=1 during decode).
            kv_caches: None (prefill) or list of (k, v) per layer (decode).
            cache_pos: RoPE position offset.

        Returns:
            logits:    (B, S, V).
            new_caches: list of (k, v) per layer (updated).
        """
        x = self.token_embedding(token_ids)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block.forward_with_kv(x, kv_cache=kv, cache_pos=cache_pos)
            new_caches.append(new_kv)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits, new_caches

    # ── Autoregressive generation ────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        state_t: torch.Tensor,
        actions: torch.Tensor,
        state_t_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        boa_id: int,
        eoa_id: int,
        max_new_tokens: int = 340,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate state[t+1] given state[t] and action.

        Uses KV caching: the prompt prefix is processed once (prefill),
        then each new token costs only one transformer forward pass.

        Args:
            state_t:         (B, max_L) int — padded current state.
            actions:         (B,)      int — action indices.
            state_t_lengths: (B,)      int — actual token count per state_t.
            bos_id, eos_id, boa_id, eoa_id: special token IDs.
            max_new_tokens:  maximum tokens to generate (including <eos>).
            temperature:     softmax temperature (1.0 = no change).

        Returns:
            generated: (B, max_new_tokens) int — generated tokens (0-padded).
            lengths:   (B,) int — actual generated length per example.
        """
        B = state_t.shape[0]
        max_L = state_t.shape[1]
        device = state_t.device
        L = state_t_lengths  # (B,)

        # ── Build prefix ──────────────────────────────────────────
        max_prefix_len = max_L + 6
        prefix_ids = torch.full((B, max_prefix_len), self.pad_id, dtype=torch.long, device=device)
        pos = torch.arange(max_prefix_len, device=device).unsqueeze(0)

        prefix_ids[:, 0] = bos_id
        st_mask = (pos >= 1) & (pos < 1 + L.unsqueeze(1))
        st_src = (pos - 1).clamp(min=0, max=max_L - 1).expand(B, -1)
        prefix_ids = torch.where(st_mask, state_t.gather(1, st_src), prefix_ids)

        eos1_col = (L + 1).unsqueeze(1)
        prefix_ids = torch.where((pos == eos1_col) & (eos1_col < max_prefix_len), eos_id, prefix_ids)
        boa_col = (L + 2).unsqueeze(1)
        prefix_ids = torch.where((pos == boa_col) & (boa_col < max_prefix_len), boa_id, prefix_ids)
        act_col = (L + 3).unsqueeze(1)
        action_tokens = self._action_lookup[actions + self._action_base]
        prefix_ids = torch.where((pos == act_col) & (act_col < max_prefix_len), action_tokens.unsqueeze(1), prefix_ids)
        eoa_col = (L + 4).unsqueeze(1)
        prefix_ids = torch.where((pos == eoa_col) & (eoa_col < max_prefix_len), eoa_id, prefix_ids)
        bos2_col = (L + 5).unsqueeze(1)
        prefix_ids = torch.where((pos == bos2_col) & (bos2_col < max_prefix_len), bos_id, prefix_ids)

        prefix_lens = L + 6
        max_prefix = int(prefix_lens.max().item())
        prefix_ids = prefix_ids[:, :max_prefix]

        # ── Guard ───────────────────────────────────────────────────
        min_safe = max_prefix + max_new_tokens
        if min_safe > self.max_seq_len:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) too small.  Prefix up to "
                f"{max_prefix} + {max_new_tokens} new tokens = {min_safe}.  "
                f"Increase max_seq_len to at least {min_safe}."
            )

        # ── Prefill: process full prefix, capture KV caches ──────
        logits, kv_caches = self._forward_tokens_with_kv(
            prefix_ids, kv_caches=None, cache_pos=0,
        )
        # logits shape: (B, max_prefix, V)

        generated = torch.full((B, max_new_tokens), self.pad_id, dtype=torch.long, device=device)
        generated_lengths = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            # Logits at the last position predict the next token.
            next_logits = logits[:, -1, :]  # (B, V)

            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_token[done] = self.pad_id

            generated[:, step] = next_token
            not_done = ~done
            generated_lengths[not_done] += 1
            done = done | (next_token == eos_id)
            if done.all():
                break

            # Decode: process the new token with KV cache.
            logits, kv_caches = self._forward_tokens_with_kv(
                next_token.unsqueeze(1),         # (B, 1)
                kv_caches=kv_caches,
                cache_pos=max_prefix + step,
            )

        return generated, generated_lengths

    # ── Checkpointing ────────────────────────────────────────────────

    def save_checkpoint(self, path: str, **extra) -> None:
        """Save model weights and optional extra data to a checkpoint file.

        Args:
            path: filesystem path for the checkpoint (e.g. ``checkpoint.pt``).
            **extra: additional key-value pairs stored alongside
                     ``model_state_dict`` (e.g. epoch, optimizer state).
        """
        ckpt = {"model_state_dict": self.state_dict(), **extra}
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location=None) -> dict:
        """Load state_dict into this model from a checkpoint file.

        Args:
            path: filesystem path to the checkpoint.
            map_location: device remap (passed to :func:`torch.load`).

        Returns:
            The full checkpoint dict (including any extra keys stored
            alongside ``model_state_dict``).
        """
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["model_state_dict"])
        return ckpt


# ── Token IDs whose loss is always ignored ──────────────────────────────
# These tokens are structural (they carry no Pokémon-state information) or
# are padding and should not contribute to the training objective.
#
# Callers may extend this set with additional tokens as needed.
DEFAULT_IGNORE_LOSS_TOKENS: set[int] = set()  # set at training time from tokenizer


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked cross-entropy loss.

    Args:
        logits:    (B, T, V) — model logits.
        targets:   (B, T)   — target token IDs.
        loss_mask: (B, T)   — True where loss should be computed.

    Returns:
        loss: scalar tensor.
        metrics: dict with 'loss', 'token_accuracy'.
    """
    n_active = loss_mask.sum()
    if n_active == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), {
            "loss": 0.0,
            "token_accuracy": 0.0,
        }

    loss = F.cross_entropy(
        logits[loss_mask],
        targets[loss_mask],
        reduction="mean",
    )

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds[loss_mask] == targets[loss_mask]).sum().item()
        acc = correct / n_active.item()

    return loss, {"loss": loss.item(), "token_accuracy": acc}
