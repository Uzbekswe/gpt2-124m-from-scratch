"""Official GPT-2 Small weight import and numerical compatibility verification."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gpt2_124m.generation import generate
from gpt2_124m.model import GPT2Model
from gpt2_124m.tokenizer import GPT2Tokenizer

OFFICIAL_GPT2_MODEL_ID = "openai-community/gpt2"
"""The Hugging Face model ID for the original GPT-2 Small checkpoint."""

FLOAT32_COMPATIBILITY_RTOL = 1e-4
FLOAT32_COMPATIBILITY_ATOL = 1e-4


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Evidence produced by an official GPT-2 numerical compatibility check."""

    input_token_ids: tuple[int, ...]
    logits_shape: tuple[int, ...]
    max_absolute_difference: float
    rtol: float
    atol: float
    passed: bool
    greedy_token_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _TensorMapping:
    """One source tensor and its destination parameter, including layout conversion."""

    source_key: str
    destination: Tensor
    transpose: bool = False


def load_official_gpt2_weights(
    model: GPT2Model,
    reference_state_dict: Mapping[str, Tensor],
) -> None:
    """Load official GPT-2 Small tensors into this project's matching model architecture."""
    _validate_official_gpt2_small_config(model)
    _load_gpt2_style_weights(model, reference_state_dict)


def download_official_gpt2_reference(
    *,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Download the optional Hugging Face GPT-2 reference model only when called.

    Transformers is intentionally imported here rather than at package import time, keeping
    ordinary model use and offline unit tests independent of Hugging Face dependencies.
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as error:
        raise ImportError(
            "Official GPT-2 verification requires the optional dependency. "
            'Install it with `python -m pip install -e ".[verify]"`. '
        ) from error

    reference_model = GPT2LMHeadModel.from_pretrained(OFFICIAL_GPT2_MODEL_ID)
    if device is not None:
        reference_model.to(device)
    return reference_model


def verify_official_gpt2_compatibility(
    model: GPT2Model,
    reference_model: nn.Module,
    *,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 5,
    rtol: float = FLOAT32_COMPATIBILITY_RTOL,
    atol: float = FLOAT32_COMPATIBILITY_ATOL,
) -> CompatibilityReport:
    """Import official weights and compare float32 logits plus deterministic greedy tokens.

    ``rtol=atol=1e-4`` is the documented float32 tolerance. It allows harmless numerical
    ordering differences while requiring close agreement for every vocabulary logit.
    """
    _validate_non_negative_integer(max_new_tokens, name="max_new_tokens")
    _validate_non_negative_float(rtol, name="rtol")
    _validate_non_negative_float(atol, name="atol")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string.")

    model_device = _module_device(model)
    reference_device = _module_device(reference_model)
    if model_device != reference_device:
        raise ValueError(
            "model and reference_model must be on the same device for numerical comparison."
        )

    load_official_gpt2_weights(model, reference_model.state_dict())
    tokenizer = GPT2Tokenizer()
    input_token_ids = tuple(tokenizer.encode(prompt))
    input_ids = torch.tensor([input_token_ids], dtype=torch.long, device=model_device)

    model_was_training = model.training
    reference_was_training = reference_model.training
    model.eval()
    reference_model.eval()
    try:
        with torch.inference_mode():
            custom_logits = model(input_ids)
            reference_logits = _extract_logits(reference_model(input_ids))
            if custom_logits.shape != reference_logits.shape:
                raise AssertionError(
                    "GPT-2 logits have different shapes: "
                    f"{tuple(custom_logits.shape)} and {tuple(reference_logits.shape)}."
                )
            max_absolute_difference = (custom_logits - reference_logits).abs().max().item()
            torch.testing.assert_close(custom_logits, reference_logits, rtol=rtol, atol=atol)

            custom_generated_ids = generate(model, input_ids, max_new_tokens=max_new_tokens)
            reference_generated_ids = _greedy_generate_reference(
                reference_model,
                input_ids,
                max_new_tokens=max_new_tokens,
                context_length=model.config.context_length,
            )
            if not torch.equal(custom_generated_ids, reference_generated_ids):
                raise AssertionError(
                    "GPT-2 greedy generation token IDs do not match the reference."
                )
    finally:
        model.train(model_was_training)
        reference_model.train(reference_was_training)

    return CompatibilityReport(
        input_token_ids=input_token_ids,
        logits_shape=tuple(custom_logits.shape),
        max_absolute_difference=max_absolute_difference,
        rtol=rtol,
        atol=atol,
        passed=True,
        greedy_token_ids=tuple(tuple(row.tolist()) for row in custom_generated_ids),
    )


def _load_gpt2_style_weights(
    model: GPT2Model,
    reference_state_dict: Mapping[str, Tensor],
) -> None:
    """Copy a GPT-2-layout mapping after complete validation.

    This parameterized implementation is intentionally separate from the public strict
    importer so offline tests can verify all mapping rules with ``GPT2_DEBUG_CONFIG``.
    """
    if not isinstance(reference_state_dict, Mapping):
        raise TypeError("reference_state_dict must be a mapping of tensor names to tensors.")
    if model.lm_head.weight is not model.embeddings.wte.weight:
        raise ValueError("model must preserve GPT-2's tied lm_head and token-embedding weights.")

    mappings = _build_tensor_mappings(model)
    sources = _validate_source_tensors(reference_state_dict, mappings)
    with torch.no_grad():
        for mapping, source in zip(mappings, sources, strict=True):
            value = source.transpose(0, 1) if mapping.transpose else source
            mapping.destination.copy_(value.to(device=mapping.destination.device))


def _validate_official_gpt2_small_config(model: GPT2Model) -> None:
    """Reject models whose architecture cannot accept the official GPT-2 Small checkpoint."""
    expected_values = {
        "vocab_size": 50_257,
        "context_length": 1_024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "qkv_bias": True,
    }
    mismatches = [
        f"{name}={getattr(model.config, name)!r} (expected {expected!r})"
        for name, expected in expected_values.items()
        if getattr(model.config, name) != expected
    ]
    if mismatches:
        raise ValueError(
            "model.config must exactly match the original GPT-2 Small architecture; "
            f"found {', '.join(mismatches)}."
        )


def _build_tensor_mappings(model: GPT2Model) -> list[_TensorMapping]:
    """Describe every official GPT-2 tensor copied into this project's module layout."""
    mappings = [
        _TensorMapping("transformer.wte.weight", model.embeddings.wte.weight),
        _TensorMapping("transformer.wpe.weight", model.embeddings.wpe.weight),
    ]
    for layer_index, block in enumerate(model.h):
        prefix = f"transformer.h.{layer_index}"
        mappings.extend(
            [
                _TensorMapping(f"{prefix}.ln_1.weight", block.ln_1.weight),
                _TensorMapping(f"{prefix}.ln_1.bias", block.ln_1.bias),
                _TensorMapping(f"{prefix}.attn.c_attn.weight", block.attn.c_attn.weight, True),
                _TensorMapping(f"{prefix}.attn.c_attn.bias", block.attn.c_attn.bias),
                _TensorMapping(f"{prefix}.attn.c_proj.weight", block.attn.c_proj.weight, True),
                _TensorMapping(f"{prefix}.attn.c_proj.bias", block.attn.c_proj.bias),
                _TensorMapping(f"{prefix}.ln_2.weight", block.ln_2.weight),
                _TensorMapping(f"{prefix}.ln_2.bias", block.ln_2.bias),
                _TensorMapping(f"{prefix}.mlp.c_fc.weight", block.mlp.c_fc.weight, True),
                _TensorMapping(f"{prefix}.mlp.c_fc.bias", block.mlp.c_fc.bias),
                _TensorMapping(f"{prefix}.mlp.c_proj.weight", block.mlp.c_proj.weight, True),
                _TensorMapping(f"{prefix}.mlp.c_proj.bias", block.mlp.c_proj.bias),
            ]
        )
    mappings.extend(
        [
            _TensorMapping("transformer.ln_f.weight", model.ln_f.weight),
            _TensorMapping("transformer.ln_f.bias", model.ln_f.bias),
        ]
    )
    return mappings


def _validate_source_tensors(
    reference_state_dict: Mapping[str, Tensor],
    mappings: list[_TensorMapping],
) -> list[Tensor]:
    """Validate all expected source keys and shapes before changing any model parameter."""
    sources: list[Tensor] = []
    for mapping in mappings:
        if mapping.source_key not in reference_state_dict:
            raise KeyError(f"reference_state_dict is missing required key: {mapping.source_key}")
        source = reference_state_dict[mapping.source_key]
        if not isinstance(source, Tensor):
            raise TypeError(f"reference tensor {mapping.source_key} must be a torch.Tensor.")
        expected_shape = (
            tuple(reversed(mapping.destination.shape))
            if mapping.transpose
            else tuple(mapping.destination.shape)
        )
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                f"reference tensor {mapping.source_key} has shape {tuple(source.shape)}, "
                f"but expected {expected_shape}."
            )
        sources.append(source)
    return sources


def _extract_logits(model_output: object) -> Tensor:
    """Extract Hugging Face-style ``.logits`` from a reference model result."""
    logits = getattr(model_output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("reference_model must return an object with a tensor logits attribute.")
    return logits


def _greedy_generate_reference(
    reference_model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
) -> Tensor:
    """Generate reference tokens with the exact greedy, cropped procedure used by our model."""
    generated_ids = input_ids
    for _ in range(max_new_tokens):
        logits = _extract_logits(reference_model(generated_ids[:, -context_length:]))
        next_token_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat((generated_ids, next_token_ids), dim=1)
    return generated_ids


def _module_device(module: nn.Module) -> torch.device:
    """Return a module's parameter device, or CPU for a parameter-free invalid reference."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _validate_non_negative_integer(value: object, *, name: str) -> None:
    """Reject non-integer or negative public count arguments."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_non_negative_float(value: object, *, name: str) -> None:
    """Reject invalid public numerical tolerances."""
    if isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative number.")
