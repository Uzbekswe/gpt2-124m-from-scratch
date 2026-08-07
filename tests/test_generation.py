"""Tests for greedy and sampled autoregressive token-ID generation."""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from gpt2_124m.generation import generate


@dataclass(frozen=True, slots=True)
class FakeGenerationConfig:
    """Small context and vocabulary dimensions for a deterministic fake model."""

    context_length: int = 4
    vocab_size: int = 8


class IncrementModel(nn.Module):
    """Predict the token ID after each final input token and record call lengths."""

    def __init__(self, config: FakeGenerationConfig = FakeGenerationConfig()) -> None:
        super().__init__()
        self.config = config
        self.call_sequence_lengths: list[int] = []

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return deterministic logits whose argmax increments each input token ID."""
        self.call_sequence_lengths.append(input_ids.shape[1])
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.config.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        next_ids = (input_ids + 1) % self.config.vocab_size
        logits.scatter_(dim=-1, index=next_ids.unsqueeze(-1), value=10.0)
        return logits


class CandidateModel(nn.Module):
    """Return fixed candidate logits with known top-k token IDs for every position."""

    def __init__(self) -> None:
        super().__init__()
        self.config = FakeGenerationConfig(vocab_size=5)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Repeat fixed vocabulary scores for every batch item and sequence position."""
        scores = torch.tensor([-2.0, 4.0, 3.0, 1.0, 0.0], device=input_ids.device)
        return scores.expand(input_ids.shape[0], input_ids.shape[1], -1)


def test_greedy_generation_appends_expected_token_ids() -> None:
    """Argmax generation deterministically appends the fake model's highest-logit tokens."""
    model = IncrementModel()
    input_ids = torch.tensor([[1, 2], [6, 7]], dtype=torch.long)

    generated_ids = generate(model, input_ids, max_new_tokens=3)

    expected = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 1, 2]], dtype=torch.long)
    assert torch.equal(generated_ids, expected)


def test_generation_returns_expected_length_and_zero_steps_returns_prompt() -> None:
    """Generation appends exactly its requested number of tokens, including zero."""
    model = IncrementModel()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    generated_ids = generate(model, input_ids, max_new_tokens=4)
    unchanged_ids = generate(model, input_ids, max_new_tokens=0)

    assert generated_ids.shape == (1, 6)
    assert torch.equal(unchanged_ids, input_ids)


def test_sampling_with_the_same_seeded_generator_is_reproducible() -> None:
    """An explicit generator makes temperature sampling repeatable for callers and tests."""
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    first_generator = torch.Generator(device="cpu").manual_seed(123)
    second_generator = torch.Generator(device="cpu").manual_seed(123)

    first_ids = generate(
        CandidateModel(),
        input_ids,
        max_new_tokens=5,
        temperature=0.75,
        do_sample=True,
        generator=first_generator,
    )
    second_ids = generate(
        CandidateModel(),
        input_ids,
        max_new_tokens=5,
        temperature=0.75,
        do_sample=True,
        generator=second_generator,
    )

    assert torch.equal(first_ids, second_ids)


def test_top_k_sampling_only_selects_allowed_candidates() -> None:
    """Top-k filtering excludes all token IDs outside the two highest-logit choices."""
    generated_ids = generate(
        CandidateModel(),
        torch.tensor([[0]], dtype=torch.long),
        max_new_tokens=30,
        top_k=2,
        do_sample=True,
        generator=torch.Generator(device="cpu").manual_seed(4),
    )

    assert set(generated_ids[0, 1:].tolist()).issubset({1, 2})


def test_top_k_one_matches_greedy_generation() -> None:
    """A one-token sampling pool is equivalent to selecting the argmax greedily."""
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    greedy_ids = generate(CandidateModel(), input_ids, max_new_tokens=4)
    top_one_ids = generate(
        CandidateModel(),
        input_ids,
        max_new_tokens=4,
        top_k=1,
        do_sample=True,
        generator=torch.Generator(device="cpu").manual_seed(99),
    )

    assert torch.equal(top_one_ids, greedy_ids)


@pytest.mark.parametrize(
    ("input_ids", "temperature", "top_k", "error", "message"),
    [
        (
            torch.tensor([1, 2], dtype=torch.long),
            1.0,
            None,
            ValueError,
            "input_ids must have shape",
        ),
        (torch.tensor([[1.0]], dtype=torch.float32), 1.0, None, TypeError, "dtype torch.long"),
        (torch.tensor([[1]], dtype=torch.long), 0.0, None, ValueError, "temperature"),
        (torch.tensor([[1]], dtype=torch.long), -1.0, None, ValueError, "temperature"),
        (torch.tensor([[1]], dtype=torch.long), 1.0, 0, ValueError, "top_k"),
        (torch.tensor([[1]], dtype=torch.long), 1.0, 1.5, ValueError, "top_k"),
        (torch.tensor([[1]], dtype=torch.long), 1.0, 9, ValueError, "cannot exceed"),
    ],
)
def test_generation_rejects_invalid_inputs(
    input_ids: Tensor,
    temperature: float,
    top_k: int | float | None,
    error: type[Exception],
    message: str,
) -> None:
    """Input shape/type and sampling controls are validated before model inference."""
    with pytest.raises(error, match=message):
        generate(
            IncrementModel(),
            input_ids,
            max_new_tokens=1,
            temperature=temperature,
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_generation_crops_model_inputs_to_the_context_length() -> None:
    """The generated history may grow, while each model call stays within GPT-2's context."""
    model = IncrementModel(FakeGenerationConfig(context_length=3))
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    generated_ids = generate(model, input_ids, max_new_tokens=4)

    assert generated_ids.shape == (1, 8)
    assert model.call_sequence_lengths == [3, 3, 3, 3]


@pytest.mark.parametrize("starts_in_train_mode", [True, False])
def test_generation_restores_the_original_model_mode(starts_in_train_mode: bool) -> None:
    """Inference does not accidentally leave a caller's model in a different mode."""
    model = IncrementModel()
    model.train(starts_in_train_mode)

    generate(model, torch.tensor([[1]], dtype=torch.long), max_new_tokens=1)

    assert model.training is starts_in_train_mode
