"""Thin wrapper around tiktoken's official GPT-2 encoding."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken

END_OF_TEXT_TOKEN = "<|endoftext|>"


class GPT2Tokenizer:
    """Encode and decode text with the GPT-2-compatible tiktoken vocabulary."""

    def __init__(self) -> None:
        try:
            import tiktoken
        except ImportError as error:
            raise ImportError(
                "GPT-2 tokenization requires tiktoken. "
                'Install it with `python -m pip install -e ".[dev]"` or `.[train]`. '
            ) from error
        self._encoding = tiktoken.get_encoding("gpt2")

    @property
    def encoding(self) -> "tiktoken.Encoding":
        """Return the underlying official GPT-2 encoding."""
        return self._encoding

    @property
    def end_of_text_token_id(self) -> int:
        """Return GPT-2's token ID for ``<|endoftext|>``."""
        return self._encoding.eot_token

    def encode(self, text: str) -> list[int]:
        """Convert text to GPT-2 token IDs, allowing ``<|endoftext|>`` explicitly."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string; got {type(text).__name__}.")

        return self._encoding.encode(text, allowed_special={END_OF_TEXT_TOKEN})

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert GPT-2 token IDs back to text."""
        return self._encoding.decode(list(token_ids))
