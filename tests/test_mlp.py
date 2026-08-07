"""Tests for GPT-2 GELU and feed-forward network layers."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional

from gpt2_124m.config import GPT2_DEBUG_CONFIG, GPT2Config
from gpt2_124m.layers import GPT2GELU, GPT2MLP


def test_gpt2_gelu_matches_pytorch_tanh_approximation() -> None:
    """The custom activation follows GPT-2's tanh GELU formula."""
    gelu = GPT2GELU()
    x = torch.linspace(-4.0, 4.0, steps=101)

    output = gelu(x)
    expected = functional.gelu(x, approximate="tanh")

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    torch.testing.assert_close(output, expected)


def test_mlp_preserves_batch_sequence_and_embedding_dimensions() -> None:
    """Every token receives a full-size transformed embedding."""
    mlp = GPT2MLP(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)

    output = mlp(x)

    assert output.shape == x.shape


def test_mlp_expands_then_contracts_the_embedding_dimension() -> None:
    """GPT-2's first projection is four times wider, then returns to model width."""
    mlp = GPT2MLP(GPT2_DEBUG_CONFIG)

    assert mlp.c_fc.in_features == GPT2_DEBUG_CONFIG.emb_dim
    assert mlp.c_fc.out_features == 4 * GPT2_DEBUG_CONFIG.emb_dim
    assert mlp.c_proj.in_features == 4 * GPT2_DEBUG_CONFIG.emb_dim
    assert mlp.c_proj.out_features == GPT2_DEBUG_CONFIG.emb_dim


def test_gradients_reach_all_mlp_projection_parameters() -> None:
    """Both MLP projections and their biases participate in backpropagation."""
    mlp = GPT2MLP(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim)
    loss = (mlp(x) * torch.randn_like(x)).sum()

    loss.backward()

    for projection in (mlp.c_fc, mlp.c_proj):
        assert projection.weight.grad is not None
        assert projection.bias.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0
        assert torch.count_nonzero(projection.bias.grad) > 0


def test_mlp_dropout_changes_train_output_and_is_disabled_in_eval_mode() -> None:
    """The final MLP dropout is stochastic only during training."""
    config = replace(GPT2_DEBUG_CONFIG, drop_rate=0.5)
    mlp = GPT2MLP(config)
    x = torch.ones(1, 4, config.emb_dim)
    with torch.no_grad():
        mlp.c_fc.weight.fill_(1.0)
        mlp.c_fc.bias.zero_()
        mlp.c_proj.weight.fill_(1.0)
        mlp.c_proj.bias.zero_()

    mlp.train()
    torch.manual_seed(1)
    train_output = mlp(x)

    mlp.eval()
    eval_output_one = mlp(x)
    eval_output_two = mlp(x)

    assert not torch.equal(train_output, eval_output_one)
    torch.testing.assert_close(eval_output_one, eval_output_two)


def test_exact_gpt2_small_mlp_dimensions_are_supported() -> None:
    """GPT-2 Small uses the exact 768 -> 3072 -> 768 MLP dimensions."""
    mlp = GPT2MLP(GPT2Config())

    assert mlp.c_fc.in_features == 768
    assert mlp.c_fc.out_features == 3_072
    assert mlp.c_proj.in_features == 3_072
    assert mlp.c_proj.out_features == 768


def test_mlp_rejects_an_incorrect_final_embedding_dimension() -> None:
    """The MLP input must match the configured model embedding dimension."""
    mlp = GPT2MLP(GPT2_DEBUG_CONFIG)
    x = torch.randn(1, 2, GPT2_DEBUG_CONFIG.emb_dim + 1)

    with pytest.raises(ValueError, match="x has emb_dim"):
        mlp(x)
