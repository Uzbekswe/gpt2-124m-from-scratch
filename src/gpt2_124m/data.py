"""In-memory token windows for next-token prediction."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create overlapping input and next-token target sequences from token IDs."""

    def __init__(self, token_ids: Sequence[int], context_length: int, stride: int) -> None:
        """Store token IDs and the windowing rules for next-token prediction."""
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer.")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer.")

        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids
        ):
            raise ValueError("token_ids must contain integers only.")
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must be non-negative.")

        self._token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        """Return the number of complete input-target windows."""
        available_starts = len(self._token_ids) - self.context_length
        if available_starts <= 0:
            return 0
        return (available_starts - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return an input window and its one-token-shifted target window."""
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(f"window index out of range: {index!r}")

        start = index * self.stride
        stop = start + self.context_length
        return self._token_ids[start:stop], self._token_ids[start + 1 : stop + 1]
