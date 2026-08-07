"""Autoregressive GPT-2 token-ID generation utilities."""

from math import isfinite
from numbers import Real

import torch
from torch import Tensor, nn


def select_next_token(
    final_logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select one token ID per row from final-position vocabulary logits.

    Greedy mode returns the highest-logit token. Sampling mode first applies temperature
    scaling and optional top-k filtering, then samples from the resulting probabilities.
    """
    _validate_final_logits(final_logits)
    _validate_temperature(temperature)
    _validate_top_k(top_k, vocab_size=final_logits.shape[-1])
    if not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a boolean.")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")

    if not do_sample:
        return torch.argmax(final_logits, dim=-1, keepdim=True)

    sampling_logits = final_logits / temperature
    if top_k is not None:
        sampling_logits = _filter_top_k(sampling_logits, top_k)
    probabilities = torch.softmax(sampling_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def generate(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively append token IDs using a GPT-2-compatible language model.

    The returned tensor retains the entire prompt and generated sequence, while each model
    call is cropped to the model's finite learned-position context window.
    """
    _validate_input_ids(input_ids)
    _validate_max_new_tokens(max_new_tokens)
    _validate_temperature(temperature)
    context_length, vocab_size = _get_model_dimensions(model)
    _validate_top_k(top_k, vocab_size=vocab_size)
    if not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a boolean.")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")

    if input_ids.numel() > 0 and (input_ids.min() < 0 or input_ids.max() >= vocab_size):
        raise ValueError(f"input_ids must contain token IDs in [0, {vocab_size}).")
    if max_new_tokens == 0:
        return input_ids

    original_training_state = model.training
    generated_ids = input_ids
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                model_input_ids = generated_ids[:, -context_length:]
                logits = model(model_input_ids)
                _validate_model_logits(logits, model_input_ids.shape[0], vocab_size)
                next_token_ids = select_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                    generator=generator,
                )
                generated_ids = torch.cat((generated_ids, next_token_ids), dim=1)
    finally:
        model.train(original_training_state)

    return generated_ids


def _filter_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Replace all but each row's top-k logits with negative infinity."""
    top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
    filtered_logits = torch.full_like(logits, -torch.inf)
    return filtered_logits.scatter(dim=-1, index=top_indices, src=top_logits)


def _validate_input_ids(input_ids: object) -> None:
    """Ensure generation starts from a non-empty long token-ID batch."""
    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids must be a torch.Tensor.")
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]; "
            f"got {tuple(input_ids.shape)}."
        )
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have non-empty batch and sequence dimensions.")
    if input_ids.dtype != torch.long:
        raise TypeError(f"input_ids must have dtype torch.long; got {input_ids.dtype}.")


def _validate_max_new_tokens(value: object) -> None:
    """Reject invalid numbers of autoregressive generation steps."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_new_tokens must be a non-negative integer.")


def _validate_temperature(value: object) -> None:
    """Reject invalid temperature scales before sampling or greedy selection."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("temperature must be a positive finite number.")


def _validate_top_k(top_k: object, *, vocab_size: int) -> None:
    """Ensure top-k can retain at least one, but no more than all, vocabulary tokens."""
    if top_k is None:
        return
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer or None.")
    if top_k > vocab_size:
        raise ValueError(f"top_k ({top_k}) cannot exceed the vocabulary size ({vocab_size}).")


def _get_model_dimensions(model: nn.Module) -> tuple[int, int]:
    """Read the context and vocabulary dimensions required by a GPT-2-like model."""
    config = getattr(model, "config", None)
    context_length = getattr(config, "context_length", None)
    vocab_size = getattr(config, "vocab_size", None)
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise TypeError("model must expose positive context_length and vocab_size in model.config.")
    return context_length, vocab_size


def _validate_final_logits(final_logits: object) -> None:
    """Ensure one vocabulary-logit row is available for each batch item."""
    if not isinstance(final_logits, Tensor):
        raise TypeError("final_logits must be a torch.Tensor.")
    if final_logits.ndim != 2:
        raise ValueError(
            "final_logits must have shape [batch_size, vocab_size]; "
            f"got {tuple(final_logits.shape)}."
        )
    if final_logits.shape[1] == 0:
        raise ValueError("final_logits must include at least one vocabulary logit.")
    if not torch.is_floating_point(final_logits):
        raise TypeError("final_logits must be floating-point values.")


def _validate_model_logits(logits: object, batch_size: int, vocab_size: int) -> None:
    """Check that a model call returned logits compatible with its advertised vocabulary."""
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError(
            "model must return logits with shape [batch_size, sequence_length, vocab_size]."
        )
    if logits.shape[0] != batch_size or logits.shape[1] == 0 or logits.shape[2] != vocab_size:
        raise ValueError(
            "model returned logits incompatible with the generation input or vocabulary."
        )
    if not torch.is_floating_point(logits):
        raise TypeError("model logits must be floating-point values.")
