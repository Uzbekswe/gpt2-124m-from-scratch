"""Tests for one causal self-attention head."""

from dataclasses import replace

import pytest
import torch

from gpt2_124m.attention import CausalAttentionHead
from gpt2_124m.config import GPT2_DEBUG_CONFIG


def test_attention_head_returns_context_vectors_with_head_dimension() -> None:
    """One head maps embeddings to its share of the model representation."""
    attention = CausalAttentionHead(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)

    output = attention(x)

    assert output.shape == (2, 5, GPT2_DEBUG_CONFIG.emb_dim // GPT2_DEBUG_CONFIG.n_heads)


def test_attention_head_can_return_attention_weights_for_inspection() -> None:
    """The inspection option exposes one attention distribution per sequence position."""
    attention = CausalAttentionHead(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)

    _, attention_weights = attention(x, return_attention_weights=True)

    assert attention_weights.shape == (2, 5, 5)


def test_attention_weights_are_zero_for_future_positions() -> None:
    """Causal masking removes every future-token probability exactly."""
    attention = CausalAttentionHead(GPT2_DEBUG_CONFIG)
    attention.eval()
    x = torch.randn(1, 5, GPT2_DEBUG_CONFIG.emb_dim)

    _, attention_weights = attention(x, return_attention_weights=True)

    assert torch.count_nonzero(attention_weights.triu(diagonal=1)) == 0


def test_future_tokens_do_not_change_earlier_outputs_in_eval_mode() -> None:
    """Positions can use only themselves and preceding input representations."""
    attention = CausalAttentionHead(GPT2_DEBUG_CONFIG)
    attention.eval()
    x = torch.randn(1, 5, GPT2_DEBUG_CONFIG.emb_dim)
    changed_future_x = x.clone()
    changed_future_x[:, 3:] += 10.0

    original_output = attention(x)
    changed_output = attention(changed_future_x)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_attention_head_rejects_sequences_longer_than_its_context_window() -> None:
    """The reusable causal mask has the same maximum length as the configuration."""
    attention = CausalAttentionHead(GPT2_DEBUG_CONFIG)
    x = torch.randn(1, GPT2_DEBUG_CONFIG.context_length + 1, GPT2_DEBUG_CONFIG.emb_dim)

    with pytest.raises(ValueError, match="exceeds the configured context_length"):
        attention(x)


def test_gradients_reach_query_key_and_value_projections() -> None:
    """All three trainable projections participate in one head's output."""
    attention = CausalAttentionHead(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 4, GPT2_DEBUG_CONFIG.emb_dim)

    attention(x).sum().backward()

    for projection in (attention.query, attention.key, attention.value):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_attention_dropout_is_active_in_train_mode_and_disabled_in_eval_mode() -> None:
    """Attention-weight dropout is stochastic only while training."""
    config = replace(GPT2_DEBUG_CONFIG, drop_rate=0.5)
    attention = CausalAttentionHead(config)
    x = torch.ones(1, 4, config.emb_dim)
    with torch.no_grad():
        for projection in (attention.query, attention.key, attention.value):
            projection.weight.fill_(1.0)
            projection.bias.zero_()

    attention.train()
    torch.manual_seed(1)
    train_output_one = attention(x)
    torch.manual_seed(2)
    train_output_two = attention(x)

    attention.eval()
    eval_output_one = attention(x)
    eval_output_two = attention(x)

    assert not torch.equal(train_output_one, train_output_two)
    torch.testing.assert_close(eval_output_one, eval_output_two)
