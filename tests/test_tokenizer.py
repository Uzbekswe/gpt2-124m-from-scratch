"""Tests for the GPT-2-compatible tokenizer wrapper."""

from gpt2_124m.tokenizer import END_OF_TEXT_TOKEN, GPT2Tokenizer


def test_encode_then_decode_round_trips_text() -> None:
    """Ordinary in-memory text round-trips through the official GPT-2 encoding."""
    tokenizer = GPT2Tokenizer()
    text = "Hello, GPT-2!"

    token_ids = tokenizer.encode(text)

    assert token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert tokenizer.decode(token_ids) == text


def test_end_of_text_special_token_is_supported() -> None:
    """The GPT-2 document-boundary token encodes and decodes explicitly."""
    tokenizer = GPT2Tokenizer()
    text = f"First document.{END_OF_TEXT_TOKEN}Second document."

    token_ids = tokenizer.encode(text)

    assert token_ids.count(tokenizer.end_of_text_token_id) == 1
    assert tokenizer.decode(token_ids) == text
