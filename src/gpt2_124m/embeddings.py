"""GPT-2 token and learned absolute positional embeddings."""

import torch
from torch import Tensor, nn

from gpt2_124m.config import GPT2Config


class GPT2Embeddings(nn.Module):
    """Convert GPT-2 token IDs into input representations for later transformer blocks."""

    def __init__(self, config: GPT2Config) -> None:
        """Create GPT-2-compatible token and learned absolute position embedding tables."""
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.emb_dim)
        self.wpe = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.emb_dim,
        )
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return token and position embeddings with shape ``[batch, sequence, emb_dim]``."""
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

        positions = torch.arange(sequence_length, device=input_ids.device)
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(positions)
        return self.drop(token_embeddings + position_embeddings)
