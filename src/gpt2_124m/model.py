"""Complete GPT-2 Small language model assembly."""

import math

from torch import Tensor, nn

from gpt2_124m.config import GPT2Config
from gpt2_124m.embeddings import GPT2Embeddings
from gpt2_124m.layers import GPT2Block, GPT2LayerNorm


class GPT2Model(nn.Module):
    """The original GPT-2 decoder-only language-model architecture."""

    def __init__(self, config: GPT2Config) -> None:
        """Create a randomly initialized GPT-2 model with tied input/output token weights."""
        super().__init__()
        self.config = config
        self.embeddings = GPT2Embeddings(config)
        self.h = nn.ModuleList(GPT2Block(config) for _ in range(config.n_layers))
        self.ln_f = GPT2LayerNorm(config)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.lm_head.weight = self.embeddings.wte.weight
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Initialize random GPT-2 weights for from-scratch training.

        Linear and embedding weights use a normal distribution with standard deviation 0.02;
        linear biases are zero. LayerNorm keeps its conventional unit scale and zero bias.
        Attention and MLP ``c_proj`` weights use GPT-2's residual scaling
        ``0.02 / sqrt(2 * n_layers)``. This method is for random initialization only and
        must not be called after official weights have been imported.
        """
        initialized_weights: set[int] = set()
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if id(module.weight) not in initialized_weights:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    initialized_weights.add(id(module.weight))
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, GPT2LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        residual_projection_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.h:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=residual_projection_std)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=residual_projection_std)

    def count_trainable_parameters(self) -> int:
        """Return the number of unique trainable parameters, respecting weight tying."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return raw vocabulary logits with shape ``[batch, sequence, vocab_size]``."""
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, sequence_length]; "
                f"got {tuple(input_ids.shape)}."
            )
        sequence_length = input_ids.shape[1]
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) exceeds the configured context_length "
                f"({self.config.context_length})."
            )

        x = self.embeddings(input_ids)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
