"""Causal self-attention components for GPT-2."""

import math

import torch
from torch import Tensor, nn

from gpt2_124m.config import GPT2Config


class CausalAttentionHead(nn.Module):
    """One masked self-attention head operating on GPT-2 input representations."""

    def __init__(self, config: GPT2Config) -> None:
        """Create trainable query, key, and value projections for one attention head."""
        super().__init__()
        if config.emb_dim % config.n_heads != 0:
            raise ValueError("config.emb_dim must be divisible by config.n_heads.")

        self.config = config
        self.head_dim = config.emb_dim // config.n_heads
        self.query = nn.Linear(config.emb_dim, self.head_dim, bias=config.qkv_bias)
        self.key = nn.Linear(config.emb_dim, self.head_dim, bias=config.qkv_bias)
        self.value = nn.Linear(config.emb_dim, self.head_dim, bias=config.qkv_bias)
        self.attn_dropout = nn.Dropout(config.drop_rate)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return contextualized values, optionally with pre-dropout attention weights."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch_size, sequence_length, emb_dim]; "
                f"got {tuple(x.shape)}."
            )

        _, sequence_length, emb_dim = x.shape
        if emb_dim != self.config.emb_dim:
            raise ValueError(
                f"x has emb_dim {emb_dim}, but config.emb_dim is {self.config.emb_dim}."
            )
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) exceeds the configured context_length "
                f"({self.config.context_length})."
            )

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vectors = self.attn_dropout(attention_weights) @ values

        if return_attention_weights:
            return context_vectors, attention_weights
        return context_vectors


class MultiHeadCausalAttention(nn.Module):
    """GPT-2-style fused causal self-attention across all configured heads."""

    def __init__(self, config: GPT2Config) -> None:
        """Create fused QKV and output projections compatible with GPT-2 terminology."""
        super().__init__()
        if config.emb_dim % config.n_heads != 0:
            raise ValueError("config.emb_dim must be divisible by config.n_heads.")

        self.config = config
        self.num_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.qkv_bias)
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
        self.attn_dropout = nn.Dropout(config.drop_rate)
        self.resid_dropout = nn.Dropout(config.drop_rate)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return all-head causal attention output, optionally with pre-dropout weights."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch_size, sequence_length, emb_dim]; "
                f"got {tuple(x.shape)}."
            )

        batch_size, sequence_length, emb_dim = x.shape
        if emb_dim != self.config.emb_dim:
            raise ValueError(
                f"x has emb_dim {emb_dim}, but config.emb_dim is {self.config.emb_dim}."
            )
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) exceeds the configured context_length "
                f"({self.config.context_length})."
            )

        queries, keys, values = self.c_attn(x).split(self.config.emb_dim, dim=-1)
        queries = queries.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vectors = self.attn_dropout(attention_weights) @ values

        context_vectors = context_vectors.transpose(1, 2).contiguous()
        context_vectors = context_vectors.reshape(batch_size, sequence_length, self.config.emb_dim)
        output = self.resid_dropout(self.c_proj(context_vectors))

        if return_attention_weights:
            return output, attention_weights
        return output
