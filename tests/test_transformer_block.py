"""Tests for one GPT-2 pre-layer-normalization transformer block."""

import torch

from gpt2_124m.attention import MultiHeadCausalAttention
from gpt2_124m.config import GPT2_DEBUG_CONFIG, GPT2Config
from gpt2_124m.layers import GPT2MLP, GPT2Block, GPT2LayerNorm


def test_transformer_block_preserves_batch_sequence_and_embedding_shape() -> None:
    """One transformer block returns a full representation for every input token."""
    block = GPT2Block(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)

    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_does_not_modify_its_input_in_place() -> None:
    """Residual paths use new tensors and leave the caller's input unchanged."""
    block = GPT2Block(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 5, GPT2_DEBUG_CONFIG.emb_dim)
    original_x = x.clone()

    block(x)

    torch.testing.assert_close(x, original_x)


def test_gradients_reach_all_transformer_block_sublayers() -> None:
    """Both normalization, attention, and MLP stages participate in backpropagation."""
    block = GPT2Block(GPT2_DEBUG_CONFIG)
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim)
    loss = (block(x) * torch.randn_like(x)).sum()

    loss.backward()

    parameters = (
        block.ln_1.weight,
        block.ln_1.bias,
        block.attn.c_attn.weight,
        block.attn.c_proj.weight,
        block.ln_2.weight,
        block.ln_2.bias,
        block.mlp.c_fc.weight,
        block.mlp.c_proj.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_future_tokens_do_not_change_earlier_block_outputs_in_eval_mode() -> None:
    """The block preserves the causal behavior supplied by its attention submodule."""
    block = GPT2Block(GPT2_DEBUG_CONFIG)
    block.eval()
    x = torch.randn(1, 5, GPT2_DEBUG_CONFIG.emb_dim)
    changed_future_x = x.clone()
    changed_future_x[:, 3:] += 10.0

    original_output = block(x)
    changed_output = block(changed_future_x)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_zero_attention_and_mlp_parameters_leave_only_residual_paths() -> None:
    """Zeroed sublayers make both residual additions return the original input exactly."""
    block = GPT2Block(GPT2_DEBUG_CONFIG)
    block.eval()
    x = torch.randn(2, 3, GPT2_DEBUG_CONFIG.emb_dim)
    with torch.no_grad():
        for module in (block.attn, block.mlp):
            for parameter in module.parameters():
                parameter.zero_()

    output = block(x)

    assert torch.equal(output, x)


def test_transformer_block_supports_debug_and_exact_gpt2_small_configurations() -> None:
    """The reusable block supports both test and exact production architecture values."""
    debug_block = GPT2Block(GPT2_DEBUG_CONFIG)
    production_block = GPT2Block(GPT2Config())

    assert debug_block.attn.num_heads == GPT2_DEBUG_CONFIG.n_heads
    assert production_block.attn.num_heads == 12
    assert production_block.attn.head_dim == 64
    assert production_block.mlp.hidden_dim == 3_072


def test_transformer_block_uses_gpt2_compatible_submodule_names() -> None:
    """Future checkpoint loading can address GPT-2 block components by standard names."""
    block = GPT2Block(GPT2_DEBUG_CONFIG)

    assert tuple(block._modules) == ("ln_1", "attn", "ln_2", "mlp")
    assert isinstance(block.ln_1, GPT2LayerNorm)
    assert isinstance(block.attn, MultiHeadCausalAttention)
    assert isinstance(block.ln_2, GPT2LayerNorm)
    assert isinstance(block.mlp, GPT2MLP)
