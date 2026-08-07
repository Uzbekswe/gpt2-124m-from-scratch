"""Tests for the complete GPT-2 language model assembly."""

import math

import pytest
import torch

from gpt2_124m.config import GPT2_DEBUG_CONFIG, GPT2Config
from gpt2_124m.layers import GPT2Block
from gpt2_124m.model import GPT2Model


def test_model_returns_one_raw_logit_vector_per_input_token() -> None:
    """A batch of GPT-2 token IDs maps to one vocabulary-sized score vector per token."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(input_ids)

    assert logits.shape == (2, 3, GPT2_DEBUG_CONFIG.vocab_size)


def test_model_contains_distinct_transformer_blocks_for_every_configured_layer() -> None:
    """Each transformer layer owns independent parameters rather than sharing one block."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)

    assert len(model.h) == GPT2_DEBUG_CONFIG.n_layers
    assert all(isinstance(block, GPT2Block) for block in model.h)
    assert len({id(block) for block in model.h}) == GPT2_DEBUG_CONFIG.n_layers


def test_language_model_head_and_token_embeddings_share_the_same_parameter() -> None:
    """GPT-2 ties output scores directly to the learned input token table."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)

    assert model.lm_head.weight is model.embeddings.wte.weight
    assert model.lm_head.weight.data_ptr() == model.embeddings.wte.weight.data_ptr()


def test_model_accepts_context_length_and_rejects_longer_sequences() -> None:
    """The model has exactly the configured learned-position context limit."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    valid_input_ids = torch.zeros((1, GPT2_DEBUG_CONFIG.context_length), dtype=torch.long)
    long_input_ids = torch.zeros((1, GPT2_DEBUG_CONFIG.context_length + 1), dtype=torch.long)

    assert model(valid_input_ids).shape == (
        1,
        GPT2_DEBUG_CONFIG.context_length,
        GPT2_DEBUG_CONFIG.vocab_size,
    )
    with pytest.raises(ValueError, match="exceeds the configured context_length"):
        model(long_input_ids)


def test_future_token_ids_do_not_change_earlier_logits_in_eval_mode() -> None:
    """Causal attention preserves next-token prediction's no-future-information rule."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    changed_future_ids = torch.tensor([[1, 2, 3, 99, 100]], dtype=torch.long)

    original_logits = model(input_ids)
    changed_logits = model(changed_future_ids)

    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])


def test_gradients_reach_embeddings_blocks_final_norm_and_tied_output_head() -> None:
    """All major model stages participate in end-to-end backpropagation."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    logits = model(input_ids)
    loss = (logits * torch.randn_like(logits)).sum()

    loss.backward()

    parameters = (
        model.embeddings.wte.weight,
        model.h[0].attn.c_attn.weight,
        model.ln_f.weight,
        model.lm_head.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


def test_exact_gpt2_small_parameter_count_is_124_439_808() -> None:
    """GPT-2 Small's count depends on sharing the token embedding and output-head weights."""
    model = GPT2Model(GPT2Config())

    assert model.count_trainable_parameters() == 124_439_808


def test_model_outputs_finite_raw_logits_not_normalized_probabilities() -> None:
    """The model leaves normalization for a later loss or sampling operation."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    model.eval()
    logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits.sum(dim=-1), torch.ones_like(logits.sum(dim=-1)))


def test_default_initialization_has_reasonable_gpt2_statistics() -> None:
    """Random initialization follows the documented standard and residual scales."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    embedding_std = model.embeddings.wte.weight.std().item()
    residual_std = model.h[0].attn.c_proj.weight.std().item()
    expected_residual_std = 0.02 / math.sqrt(2 * GPT2_DEBUG_CONFIG.n_layers)

    assert abs(model.embeddings.wte.weight.mean().item()) < 0.005
    assert 0.015 < embedding_std < 0.025
    assert abs(residual_std - expected_residual_std) < 0.005
    assert torch.equal(model.ln_f.weight, torch.ones_like(model.ln_f.weight))
    assert torch.equal(model.ln_f.bias, torch.zeros_like(model.ln_f.bias))
