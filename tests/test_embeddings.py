"""Tests for GPT-2 input embeddings."""

from dataclasses import replace

import pytest
import torch

from gpt2_124m.config import GPT2_DEBUG_CONFIG
from gpt2_124m.embeddings import GPT2Embeddings


def test_embeddings_return_batch_sequence_and_embedding_dimensions() -> None:
    """A batch of token IDs becomes one representation vector per token."""
    embeddings = GPT2Embeddings(GPT2_DEBUG_CONFIG)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    output = embeddings(input_ids)

    assert output.shape == (2, 3, GPT2_DEBUG_CONFIG.emb_dim)


def test_embedding_output_uses_a_floating_point_dtype() -> None:
    """Embedding lookup converts integer token IDs into floating-point representations."""
    embeddings = GPT2Embeddings(GPT2_DEBUG_CONFIG)

    output = embeddings(torch.tensor([[1, 2]], dtype=torch.long))

    assert torch.is_floating_point(output)


def test_embeddings_reject_sequences_longer_than_the_context_window() -> None:
    """A learned position table cannot represent positions beyond the configured limit."""
    embeddings = GPT2Embeddings(GPT2_DEBUG_CONFIG)
    input_ids = torch.zeros((1, GPT2_DEBUG_CONFIG.context_length + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds the configured context_length"):
        embeddings(input_ids)


def test_embeddings_reject_input_ids_without_batch_and_sequence_dimensions() -> None:
    """The embedding module accepts only a batch of token sequences."""
    embeddings = GPT2Embeddings(GPT2_DEBUG_CONFIG)

    with pytest.raises(ValueError, match="must have shape"):
        embeddings(torch.tensor([1, 2, 3], dtype=torch.long))


def test_same_token_at_different_positions_has_different_representations() -> None:
    """Learned absolute positions distinguish repeated tokens in a sequence."""
    embeddings = GPT2Embeddings(GPT2_DEBUG_CONFIG)
    embeddings.eval()
    with torch.no_grad():
        embeddings.wte.weight.zero_()
        embeddings.wpe.weight.zero_()
        embeddings.wpe.weight[1].fill_(1.0)

    output = embeddings(torch.tensor([[7, 7]], dtype=torch.long))

    assert not torch.equal(output[:, 0], output[:, 1])


def test_gradients_reach_token_and_position_embedding_tables() -> None:
    """Both learned embedding tables participate in the computation graph."""
    embeddings = GPT2Embeddings(GPT2_DEBUG_CONFIG)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    embeddings(input_ids).sum().backward()

    assert embeddings.wte.weight.grad is not None
    assert embeddings.wpe.weight.grad is not None
    assert torch.count_nonzero(embeddings.wte.weight.grad) > 0
    assert torch.count_nonzero(embeddings.wpe.weight.grad) > 0


def test_dropout_changes_train_outputs_and_is_disabled_in_eval_mode() -> None:
    """Embedding dropout is stochastic for training and deterministic for evaluation."""
    config = replace(GPT2_DEBUG_CONFIG, drop_rate=0.5)
    embeddings = GPT2Embeddings(config)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        embeddings.wte.weight.fill_(1.0)
        embeddings.wpe.weight.fill_(1.0)

    embeddings.train()
    torch.manual_seed(1)
    train_output_one = embeddings(input_ids)
    torch.manual_seed(2)
    train_output_two = embeddings(input_ids)

    embeddings.eval()
    eval_output_one = embeddings(input_ids)
    eval_output_two = embeddings(input_ids)

    assert not torch.equal(train_output_one, train_output_two)
    torch.testing.assert_close(eval_output_one, eval_output_two)
