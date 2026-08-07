"""Tests for GPT-2 configuration blueprints."""

from dataclasses import FrozenInstanceError

import pytest

from gpt2_124m.config import GPT2_DEBUG_CONFIG, GPT2Config


def test_gpt2_small_defaults_are_exact() -> None:
    """The default blueprint matches the original GPT-2 Small architecture."""
    config = GPT2Config()

    assert config.vocab_size == 50_257
    assert config.context_length == 1_024
    assert config.emb_dim == 768
    assert config.n_heads == 12
    assert config.n_layers == 12
    assert config.drop_rate == 0.1
    assert config.qkv_bias is True
    assert config.layer_norm_epsilon == 1e-5

    with pytest.raises(FrozenInstanceError):
        config.emb_dim = 1  # type: ignore[misc]


def test_debug_configuration_is_valid_and_small() -> None:
    """The test-only configuration meets the same structural constraints."""
    assert GPT2_DEBUG_CONFIG.emb_dim % GPT2_DEBUG_CONFIG.n_heads == 0
    assert GPT2_DEBUG_CONFIG.context_length < GPT2Config().context_length
    assert GPT2_DEBUG_CONFIG.n_layers < GPT2Config().n_layers


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 0}, "vocab_size must be a positive integer"),
        ({"context_length": -1}, "context_length must be a positive integer"),
        ({"emb_dim": 10, "n_heads": 3}, "emb_dim must be divisible by n_heads"),
        ({"n_heads": 0}, "n_heads must be a positive integer"),
        ({"n_layers": 0}, "n_layers must be a positive integer"),
        ({"drop_rate": 1.0}, "drop_rate must be a number"),
        ({"layer_norm_epsilon": 0.0}, "layer_norm_epsilon must be a positive number"),
        ({"qkv_bias": "yes"}, "qkv_bias must be a boolean"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    """Invalid structural or numeric values fail immediately and clearly."""
    with pytest.raises(ValueError, match=message):
        GPT2Config(**kwargs)  # type: ignore[arg-type]
