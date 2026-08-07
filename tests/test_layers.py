"""Tests for shared GPT-2 neural-network layers."""

import pytest
import torch
import torch.nn.functional as functional

from gpt2_124m.config import GPT2_DEBUG_CONFIG
from gpt2_124m.layers import GPT2LayerNorm


@pytest.mark.parametrize(
    "shape",
    [
        (2, GPT2_DEBUG_CONFIG.emb_dim),
        (2, 3, GPT2_DEBUG_CONFIG.emb_dim),
    ],
)
def test_layer_norm_preserves_supported_input_shapes_and_dtype(shape: tuple[int, ...]) -> None:
    """LayerNorm supports batches with or without a sequence dimension."""
    layer_norm = GPT2LayerNorm(GPT2_DEBUG_CONFIG)
    x = torch.randn(shape, dtype=torch.float32)

    output = layer_norm(x)

    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_default_layer_norm_has_zero_mean_and_unit_variance_per_embedding() -> None:
    """Default affine parameters leave each normalized embedding standardized."""
    layer_norm = GPT2LayerNorm(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim)

    output = layer_norm(x)

    torch.testing.assert_close(output.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        output.var(dim=-1, unbiased=False),
        torch.ones(2, 3),
        atol=1e-4,
        rtol=0.0,
    )


def test_learned_weight_and_bias_change_layer_norm_output() -> None:
    """The affine parameters can scale and shift normalized values."""
    layer_norm = GPT2LayerNorm(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, GPT2_DEBUG_CONFIG.emb_dim)
    default_output = layer_norm(x)
    with torch.no_grad():
        layer_norm.weight.fill_(2.0)
        layer_norm.bias.fill_(3.0)

    output = layer_norm(x)

    torch.testing.assert_close(output, 2.0 * default_output + 3.0)


def test_gradients_reach_input_weight_and_bias() -> None:
    """LayerNorm remains differentiable for its input and affine parameters."""
    layer_norm = GPT2LayerNorm(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim, requires_grad=True)
    loss = (layer_norm(x) * torch.randn_like(x)).sum()

    loss.backward()

    assert x.grad is not None
    assert layer_norm.weight.grad is not None
    assert layer_norm.bias.grad is not None
    assert torch.count_nonzero(x.grad) > 0
    assert torch.count_nonzero(layer_norm.weight.grad) > 0
    assert torch.count_nonzero(layer_norm.bias.grad) > 0


def test_layer_norm_rejects_an_incorrect_final_embedding_dimension() -> None:
    """The final input dimension must match the configured embedding size."""
    layer_norm = GPT2LayerNorm(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim + 1)

    with pytest.raises(ValueError, match="final dimension"):
        layer_norm(x)


def test_layer_norm_matches_pytorch_functional_layer_norm() -> None:
    """The custom implementation matches PyTorch's LayerNorm calculation."""
    layer_norm = GPT2LayerNorm(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim)
    with torch.no_grad():
        layer_norm.weight.copy_(torch.randn_like(layer_norm.weight))
        layer_norm.bias.copy_(torch.randn_like(layer_norm.bias))

    output = layer_norm(x)
    expected = functional.layer_norm(
        x,
        normalized_shape=(GPT2_DEBUG_CONFIG.emb_dim,),
        weight=layer_norm.weight,
        bias=layer_norm.bias,
        eps=layer_norm.epsilon,
    )

    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
