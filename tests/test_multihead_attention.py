"""Tests for GPT-2-style fused multi-head causal attention."""

from dataclasses import replace

import pytest
import torch

from gpt2_124m.attention import MultiHeadCausalAttention
from gpt2_124m.config import GPT2_DEBUG_CONFIG, GPT2Config


def test_multihead_attention_returns_model_embedding_dimension() -> None:
    """Recombined heads return one full model representation per token."""
    attention = MultiHeadCausalAttention(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)

    output = attention(x)

    assert output.shape == (2, 5, GPT2_DEBUG_CONFIG.emb_dim)


def test_multihead_attention_can_return_per_head_weights() -> None:
    """The inspection option exposes a separate score distribution for every head."""
    attention = MultiHeadCausalAttention(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)

    _, attention_weights = attention(x, return_attention_weights=True)

    assert attention_weights.shape == (2, GPT2_DEBUG_CONFIG.n_heads, 5, 5)


def test_multihead_attention_weights_are_zero_for_future_positions() -> None:
    """Every head applies the same causal restriction to future token positions."""
    attention = MultiHeadCausalAttention(GPT2_DEBUG_CONFIG)
    attention.eval()
    x = torch.randn(1, 5, GPT2_DEBUG_CONFIG.emb_dim)

    _, attention_weights = attention(x, return_attention_weights=True)

    assert torch.count_nonzero(attention_weights.triu(diagonal=1)) == 0


def test_future_inputs_do_not_change_earlier_multihead_outputs_in_eval_mode() -> None:
    """No head permits a position to depend on future input representations."""
    attention = MultiHeadCausalAttention(GPT2_DEBUG_CONFIG)
    attention.eval()
    x = torch.randn(1, 5, GPT2_DEBUG_CONFIG.emb_dim)
    changed_future_x = x.clone()
    changed_future_x[:, 3:] += 10.0

    original_output = attention(x)
    changed_output = attention(changed_future_x)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_gradients_reach_fused_qkv_and_output_projections() -> None:
    """Both GPT-2-style linear layers participate in the output computation graph."""
    attention = MultiHeadCausalAttention(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 4, GPT2_DEBUG_CONFIG.emb_dim)

    attention(x).sum().backward()

    assert attention.c_attn.weight.grad is not None
    assert attention.c_proj.weight.grad is not None
    assert torch.count_nonzero(attention.c_attn.weight.grad) > 0
    assert torch.count_nonzero(attention.c_proj.weight.grad) > 0


def test_exact_gpt2_small_head_dimensions_are_supported() -> None:
    """GPT-2 Small uses twelve 64-dimensional heads and a 768-dimensional output."""
    attention = MultiHeadCausalAttention(GPT2Config())

    assert attention.num_heads == 12
    assert attention.head_dim == 64
    assert attention.c_attn.in_features == 768
    assert attention.c_attn.out_features == 3 * 768
    assert attention.c_proj.out_features == 768


def test_multihead_attention_rejects_bad_embedding_and_sequence_dimensions() -> None:
    """Inputs must match the configured embedding and context dimensions."""
    attention = MultiHeadCausalAttention(GPT2_DEBUG_CONFIG)
    wrong_embedding_x = torch.randn(1, 4, GPT2_DEBUG_CONFIG.emb_dim + 1)
    long_sequence_x = torch.randn(
        1,
        GPT2_DEBUG_CONFIG.context_length + 1,
        GPT2_DEBUG_CONFIG.emb_dim,
    )

    with pytest.raises(ValueError, match="x has emb_dim"):
        attention(wrong_embedding_x)
    with pytest.raises(ValueError, match="exceeds the configured context_length"):
        attention(long_sequence_x)


def test_multihead_dropout_changes_train_behavior_and_is_disabled_in_eval_mode() -> None:
    """Attention and output dropout are active only while the module trains."""
    config = replace(GPT2_DEBUG_CONFIG, drop_rate=0.5)
    attention = MultiHeadCausalAttention(config)
    x = torch.ones(1, 4, config.emb_dim)
    with torch.no_grad():
        attention.c_attn.weight.fill_(1.0)
        attention.c_attn.bias.zero_()
        attention.c_proj.weight.fill_(1.0)
        attention.c_proj.bias.zero_()

    attention.train()
    torch.manual_seed(1)
    train_output = attention(x)

    attention.eval()
    eval_output_one = attention(x)
    eval_output_two = attention(x)

    assert not torch.equal(train_output, eval_output_one)
    torch.testing.assert_close(eval_output_one, eval_output_two)
