"""Autoregressive transformer for next-state prediction in Pokémon battles.

Given the current state ``state[t]`` and the action ``action[t]``, the model
autoregressively predicts every token of ``state[t+1]``.

Prompt structure (training):
    <bos>  s₀  s₁  …  sₙ  <eos>  <boa>  a  <eoa>  <bos>  t₀  t₁  …  tₙ  <eos>
    ├────────── state[t] ──────────┤  ├─ act ─┤  ├────── state[t+1] ──────────┤

The loss is only computed on the **state[t+1]** region (``t₀ … tₙ <eos>``).
At inference time the prompt ends at the second ``<bos>`` and the model
generates until ``<eos>``.

UNKNOWN_TOKEN = 0 doubles as the padding token, so ``nn.Embedding(padding_idx=0)``
works without remapping.  Action indices are embedded through a dedicated
learnable embedding table (``action_emb``), not through the token vocabulary.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        cos = self.cos_cached[offset : offset + S]           # (S, D//2)
        sin = self.sin_cached[offset : offset + S]
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


# ── Full model ───────────────────────────────────────────────────────────

class WorldModelTransformer(nn.Module):
    """Autoregressive decoder that predicts state[t+1] from state[t] + action[t].

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the text vocabulary (token IDs are 1-based;
        0 = UNKNOWN_TOKEN / padding).
    max_seq_len : int
        Maximum prompt length (state + delimiters + action + state).
    d_model, n_heads, n_layers, d_ff, dropout, theta:
        Standard transformer hyperparameters.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 680,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        # Small learned embedding for action indices (-1 … 12 → 14 entries)
        self.num_actions = 14
        self.action_embedding = nn.Embedding(self.num_actions, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size + 1, bias=False)

        # Tie input embedding and output projection weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Keep padding_idx at zero
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx] = 0.0

    def _action_to_token(self, action_idx: torch.Tensor) -> torch.Tensor:
        """Map raw action index (-1 … 12) → action embedding index (0 … 13)."""
        return action_idx + 1  # -1→0, 0→1, …, 12→13

    def forward(
        self,
        state_t: torch.Tensor,
        state_next: torch.Tensor,
        actions: torch.Tensor,
        bos_id: int,
        eos_id: int,
        boa_id: int,
        eoa_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass (teacher forcing).

        Builds the full prompt:
            <bos> state_t <eos> <boa> action <eoa> <bos> state_next <eos>

        Returns logits over the *entire* sequence (shifted for next-token
        prediction), plus a loss mask that is True *only* for the state_next
        region.

        Args:
            state_t:    (B, S) int — current state token IDs.
            state_next: (B, S) int — next state token IDs (target).
            actions:    (B,)   int — action indices (-1 … 12).
            bos_id, eos_id, boa_id, eoa_id: special token IDs.

        Returns:
            logits: (B, T, V) — logits for each position (shifted).
            targets: (B, T) — target token IDs (shifted).
            loss_mask: (B, T) bool — True for positions in the state_next region.
        """
        B, S = state_t.shape
        device = state_t.device

        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        eos = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
        boa = torch.full((B, 1), boa_id, dtype=torch.long, device=device)
        eoa = torch.full((B, 1), eoa_id, dtype=torch.long, device=device)

        # Action is embedded via a separate action_embedding table, not the
        # text vocabulary.  We reserve a position for it between <boa>/<eoa>
        # and overwrite that position's embedding after the token lookup.
        action_idx = self._action_to_token(actions)           # (B,)
        ACTION_PLACEHOLDER = bos_id

        # Layout (T = 2*S + 7):
        #   pos 0:          <bos>
        #   pos 1..S:       state_t
        #   pos S+1:        <eos>
        #   pos S+2:        <boa>
        #   pos S+3:        ACTION  (placeholder — overwritten below)
        #   pos S+4:        <eoa>
        #   pos S+5:        <bos>
        #   pos S+6..2S+5:  state_next
        #   pos 2S+6:       <eos>
        token_ids = torch.cat([
            bos,
            state_t,
            eos,
            boa,
            torch.full((B, 1), ACTION_PLACEHOLDER, dtype=torch.long, device=device),
            eoa,
            bos,
            state_next,
            eos,
        ], dim=1)  # (B, T)

        T = token_ids.shape[1]

        # ── Embeddings ──────────────────────────────────────────────
        x = self.token_embedding(token_ids)                    # (B, T, d_model)
        action_pos = S + 3                                     # overwrite ACTION slot
        x[:, action_pos, :] = self.action_embedding(action_idx)

        # ── Transformer ─────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)                               # (B, T, V)

        # ── Targets & loss mask ─────────────────────────────────────
        # Shift right: predict token at position i from position i-1
        # logits[:, :-1, :] predicts targets[:, 1:]
        logits = logits[:, :-1, :]                             # (B, T-1, V)
        targets = token_ids[:, 1:]                             # (B, T-1)

        # Loss mask: only the state_next region + its final <eos>
        # In targets (shifted right): state_next[0] is at position S+5,
        # the final <eos> is at position 2S+5.
        sn_start = S + 5                                       # first token of state_next in targets
        loss_mask = torch.zeros(B, T - 1, dtype=torch.bool, device=device)
        loss_mask[:, sn_start:] = True                         # (B, T-1)

        # Also mask out padding tokens (0) within state_next
        # targets[:, sn_start:] contains state_next tokens shifted;
        # we should not compute loss on padding (0) positions
        pad_mask = targets != 0                                # (B, T-1)
        loss_mask = loss_mask & pad_mask

        return logits, targets, loss_mask

    @torch.no_grad()
    def generate(
        self,
        state_t: torch.Tensor,
        actions: torch.Tensor,
        bos_id: int,
        eos_id: int,
        boa_id: int,
        eoa_id: int,
        max_new_tokens: int = 340,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate state[t+1] given state[t] and action.

        Args:
            state_t: (B, S) int — current state token IDs.
            actions: (B,) int — action indices.
            bos_id, eos_id, boa_id, eoa_id: special token IDs.
            max_new_tokens: maximum tokens to generate (including <eos>).
            temperature: softmax temperature (1.0 = no change).

        Returns:
            (B, max_new_tokens) int — generated token IDs (padded with 0).
        """
        B, S = state_t.shape
        device = state_t.device

        action_idx = self._action_to_token(actions)
        ACTION_PLACEHOLDER = bos_id

        # Build prefix: <bos> st <eos> <boa> [ACTION] <eoa> <bos>
        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        eos = torch.full((B, 1), eos_id, dtype=torch.long, device=device)
        boa = torch.full((B, 1), boa_id, dtype=torch.long, device=device)
        eoa = torch.full((B, 1), eoa_id, dtype=torch.long, device=device)

        prefix_ids = torch.cat([
            bos, state_t, eos, boa,
            torch.full((B, 1), ACTION_PLACEHOLDER, dtype=torch.long, device=device),
            eoa, bos,
        ], dim=1)  # (B, S+6)

        generated = torch.full((B, max_new_tokens), 0, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # KV cache not implemented — for simplicity we re-encode the full
        # sequence each step.  For a 680-token sequence this is still fast.
        for step in range(max_new_tokens):
            token_ids = torch.cat([prefix_ids, generated[:, :step]], dim=1)
            # Cap at max_seq_len
            if token_ids.shape[1] > self.max_seq_len:
                token_ids = token_ids[:, -self.max_seq_len:]

            x = self.token_embedding(token_ids)
            # Overwrite action position
            action_pos = 1 + S + 1 + 1
            if action_pos < x.shape[1]:
                x[:, action_pos, :] = self.action_embedding(action_idx)

            for block in self.blocks:
                x = block(x)
            x = self.ln_final(x)
            logits = self.lm_head(x[:, -1, :])                 # (B, V)

            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            next_token[done] = 0

            generated[:, step] = next_token
            done = done | (next_token == eos_id)
            if done.all():
                break

        return generated


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
