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

        total_lens = L + M + 7  # (B,) — total prompt length per example
        # Cap at max_context
        total_lens = torch.clamp(total_lens, max=max_context)
        T = int(total_lens.max().item())  # batch max for stacking

        # We'll build each example's prompt separately, then stack.
        # This is O(B * T) which is fine for typical batch sizes (≤ 64).
        token_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        # sn_start[i] = position in targets where state_next region begins
        sn_start = torch.empty(B, dtype=torch.long, device=device)
        sn_end = torch.empty(B, dtype=torch.long, device=device)

        for i in range(B):
            li = min(L[i].item(), max_context - 7)  # state_t tokens we can fit
            mi = min(M[i].item(), max_context - 7 - li)  # remaining budget for state_next

            # Pack prompt (0-indexed):
            #  0:         <bos>
            #  1 .. li:   state_t
            #  li+1:      <eos>
            #  li+2:      <boa>
            #  li+3:      <action_X>  — action as a regular token
            #  li+4:      <eoa>
            #  li+5:      <bos>
            #  li+6 .. li+5+mi: state_next
            #  li+6+mi:   <eos>
            t = 0
            token_ids[i, t] = bos_id; t += 1
            token_ids[i, t:t+li] = state_t[i, :li]; t += li
            token_ids[i, t] = eos_id; t += 1
            token_ids[i, t] = boa_id; t += 1
            action_idx = actions[i].item()
            token_ids[i, t] = self._action_to_token_id[action_idx]
            t += 1
            token_ids[i, t] = eoa_id; t += 1
            token_ids[i, t] = bos_id; t += 1
            token_ids[i, t:t+mi] = state_next[i, :mi]; t += mi
            token_ids[i, t] = eos_id; t += 1

            # sn_start / sn_end in *targets* (shifted by 1):
            # targets[t] = token_ids[t+1]
            # state_next[0]  is at token_ids[li+6] → targets[li+5]
            # state_next[-1] is at token_ids[li+5+mi] → targets[li+4+mi]
            # final <eos>    is at token_ids[li+6+mi] → targets[li+5+mi]
            sn_start[i] = li + 5
            sn_end[i] = li + 5 + mi  # inclusive index of final <eos> in targets

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

        # Loss mask: state_next region + final <eos> for each example
        loss_mask = torch.zeros(B, T - 1, dtype=torch.bool, device=device)
        for i in range(B):
            a = sn_start[i].item()
            b = sn_end[i].item()
            if a < T - 1:
                loss_mask[i, a:min(b + 1, T - 1)] = True

        # Exclude ignore_loss_tokens (pad, delimiter tokens, etc.)
        if ignore_loss_tokens:
            ignore_mask = torch.zeros(B, T - 1, dtype=torch.bool, device=device)
            for tid in ignore_loss_tokens:
                ignore_mask = ignore_mask | (targets == tid)
            loss_mask = loss_mask & ~ignore_mask

        return logits, targets, loss_mask

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

        # Build prefix per example: <bos> st[:L_i] <eos> <boa> <action_X> <eoa> <bos>
        # Max prefix length = max_L + 6
        max_prefix_len = max_L + 6
        prefix_ids = torch.full((B, max_prefix_len), self.pad_id, dtype=torch.long, device=device)
        prefix_lens = torch.empty(B, dtype=torch.long, device=device)

        for i in range(B):
            li = L[i].item()
            action_tok = self._action_to_token_id[actions[i].item()]
            t = 0
            prefix_ids[i, t] = bos_id; t += 1
            prefix_ids[i, t:t+li] = state_t[i, :li]; t += li
            prefix_ids[i, t] = eos_id; t += 1
            prefix_ids[i, t] = boa_id; t += 1
            prefix_ids[i, t] = action_tok
            t += 1
            prefix_ids[i, t] = eoa_id; t += 1
            prefix_ids[i, t] = bos_id; t += 1
            prefix_lens[i] = t

        max_prefix = int(prefix_lens.max().item())
        prefix_ids = prefix_ids[:, :max_prefix]  # trim batch to max prefix

        # Guard: sliding-window truncation must not drop the action token.
        # The action sits at position li+3 in the prefix; when total > max_seq_len
        # the leftmost tokens are cut, potentially removing the action silently.
        min_safe = max_L + 6 + max_new_tokens
        if min_safe > self.max_seq_len:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) too small for generation. "
                f"The prefix needs up to {max_L + 6} tokens, plus up to "
                f"{max_new_tokens} generated tokens = {min_safe}. "
                f"Increase max_seq_len to at least {min_safe}."
            )

        generated = torch.full((B, max_new_tokens), self.pad_id, dtype=torch.long, device=device)
        generated_lengths = torch.zeros(B, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # KV cache not implemented — re-encode full sequence each step.
        for step in range(max_new_tokens):
            token_ids = torch.cat([prefix_ids, generated[:, :step]], dim=1)
            # Cap at max_seq_len (sliding window — keep last max_seq_len tokens)
            if token_ids.shape[1] > self.max_seq_len:
                token_ids = token_ids[:, -self.max_seq_len:]

            x = self.token_embedding(token_ids)

            for block in self.blocks:
                x = block(x)
            x = self.ln_final(x)
            logits = self.lm_head(x[:, -1, :])                 # (B, V)

            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            next_token[done] = self.pad_id

            generated[:, step] = next_token
            # Track length for examples not yet done
            not_done = ~done
            generated_lengths[not_done] += 1
            done = done | (next_token == eos_id)
            if done.all():
                break

        return generated, generated_lengths


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
