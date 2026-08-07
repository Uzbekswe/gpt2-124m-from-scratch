"""Shared neural-network layers for the GPT-2 architecture."""

import math

import torch
from torch import Tensor, nn

from gpt2_124m.attention import MultiHeadCausalAttention
from gpt2_124m.config import GPT2Config


class GPT2LayerNorm(nn.Module):
    """Layer normalization across the final GPT-2 embedding dimension."""

    def __init__(self, config: GPT2Config) -> None:
        """Create GPT-2-compatible affine normalization parameters."""
        super().__init__()
        self.emb_dim = config.emb_dim
        self.epsilon = config.layer_norm_epsilon
        self.weight = nn.Parameter(torch.ones(config.emb_dim))
        self.bias = nn.Parameter(torch.zeros(config.emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize independently across the final embedding dimension."""
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point tensor.")
        if x.ndim == 0 or x.shape[-1] != self.emb_dim:
            raise ValueError(
                f"x must have final dimension {self.emb_dim}; "
                f"got shape {tuple(x.shape)}."
            )

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        weight = self.weight.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype)
        return weight * normalized + bias


class GPT2GELU(nn.Module):
    """GPT-2's tanh approximation to the Gaussian error linear unit."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply the GPT-2-compatible GELU activation elementwise."""
        tanh_input = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(tanh_input))


class GPT2MLP(nn.Module):
    """GPT-2's position-wise feed-forward network."""

    def __init__(self, config: GPT2Config) -> None:
        """Create the GPT-2 Small-compatible expansion, activation, and projection layers."""
        super().__init__()
        self.config = config
        self.hidden_dim = 4 * config.emb_dim
        self.c_fc = nn.Linear(config.emb_dim, self.hidden_dim, bias=True)
        self.gelu = GPT2GELU()
        self.c_proj = nn.Linear(self.hidden_dim, config.emb_dim, bias=True)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the same GPT-2 MLP independently to each token representation."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch_size, sequence_length, emb_dim]; "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.config.emb_dim:
            raise ValueError(
                f"x has emb_dim {x.shape[-1]}, but config.emb_dim is {self.config.emb_dim}."
            )

        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class GPT2Block(nn.Module):
    """One GPT-2 pre-layer-normalization transformer block."""

    def __init__(self, config: GPT2Config) -> None:
        """Create GPT-2-compatible normalization, attention, and MLP submodules."""
        super().__init__()
        self.config = config
        self.ln_1 = GPT2LayerNorm(config)
        self.attn = MultiHeadCausalAttention(config)
        self.ln_2 = GPT2LayerNorm(config)
        self.mlp = GPT2MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """Apply GPT-2's two pre-norm sublayers and their residual connections."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch_size, sequence_length, emb_dim]; "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.config.emb_dim:
            raise ValueError(
                f"x has emb_dim {x.shape[-1]}, but config.emb_dim is {self.config.emb_dim}."
            )

        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = shortcut + x

        shortcut = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return shortcut + x
