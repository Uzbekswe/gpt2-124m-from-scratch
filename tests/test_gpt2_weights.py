"""Offline tests for official GPT-2 tensor mapping and layout conversion."""

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from gpt2_124m.config import GPT2_DEBUG_CONFIG
from gpt2_124m.gpt2_weights import (
    _build_tensor_mappings,
    _load_gpt2_style_weights,
    load_official_gpt2_weights,
)
from gpt2_124m.model import GPT2Model


def _synthetic_reference_state_dict(model: GPT2Model) -> dict[str, Tensor]:
    """Create a tiny GPT-2-layout state dictionary with distinct, checkable values."""
    state_dict: dict[str, Tensor] = {}
    for index, mapping in enumerate(_build_tensor_mappings(model), start=1):
        source_shape = (
            tuple(reversed(mapping.destination.shape))
            if mapping.transpose
            else tuple(mapping.destination.shape)
        )
        state_dict[mapping.source_key] = (
            torch.arange(mapping.destination.numel(), dtype=torch.float32).reshape(source_shape)
            + index * 1_000.0
        )
    return state_dict


def _load_debug_synthetic_weights(model: GPT2Model, state_dict: Mapping[str, Tensor]) -> None:
    """Exercise the parameterized copier without pretending DEBUG_CONFIG is official GPT-2."""
    _load_gpt2_style_weights(model, state_dict)


def test_directly_mapped_embeddings_and_normalization_tensors_copy_correctly() -> None:
    """Embedding and LayerNorm tensors retain their official layout without transposition."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    state_dict = _synthetic_reference_state_dict(model)

    _load_debug_synthetic_weights(model, state_dict)

    torch.testing.assert_close(model.embeddings.wte.weight, state_dict["transformer.wte.weight"])
    torch.testing.assert_close(model.embeddings.wpe.weight, state_dict["transformer.wpe.weight"])
    torch.testing.assert_close(model.h[0].ln_1.weight, state_dict["transformer.h.0.ln_1.weight"])
    torch.testing.assert_close(model.ln_f.bias, state_dict["transformer.ln_f.bias"])


def test_conv1d_projection_weights_are_transposed_for_torch_linear() -> None:
    """All Conv1D projection matrices switch from [in, out] to Linear's [out, in]."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    state_dict = _synthetic_reference_state_dict(model)

    _load_debug_synthetic_weights(model, state_dict)

    block = model.h[0]
    for source_key, destination in (
        ("transformer.h.0.attn.c_attn.weight", block.attn.c_attn.weight),
        ("transformer.h.0.attn.c_proj.weight", block.attn.c_proj.weight),
        ("transformer.h.0.mlp.c_fc.weight", block.mlp.c_fc.weight),
        ("transformer.h.0.mlp.c_proj.weight", block.mlp.c_proj.weight),
    ):
        torch.testing.assert_close(destination, state_dict[source_key].transpose(0, 1))


def test_biases_and_layer_norm_tensors_are_not_transposed() -> None:
    """Only Conv1D matrices transpose; one-dimensional biases and norms copy unchanged."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    state_dict = _synthetic_reference_state_dict(model)

    _load_debug_synthetic_weights(model, state_dict)

    torch.testing.assert_close(
        model.h[0].attn.c_attn.bias,
        state_dict["transformer.h.0.attn.c_attn.bias"],
    )
    torch.testing.assert_close(
        model.h[0].mlp.c_proj.bias,
        state_dict["transformer.h.0.mlp.c_proj.bias"],
    )
    torch.testing.assert_close(model.h[0].ln_2.bias, state_dict["transformer.h.0.ln_2.bias"])


def test_missing_reference_tensor_fails_clearly() -> None:
    """Importing never silently leaves a required official GPT-2 parameter untouched."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    state_dict = _synthetic_reference_state_dict(model)
    state_dict.pop("transformer.h.1.mlp.c_proj.bias")

    with pytest.raises(KeyError, match="missing required key"):
        _load_debug_synthetic_weights(model, state_dict)


def test_wrong_reference_tensor_shape_fails_clearly() -> None:
    """Every source shape is checked before copying any parameter into the model."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    state_dict = _synthetic_reference_state_dict(model)
    state_dict["transformer.h.0.attn.c_attn.weight"] = torch.zeros(1, 1)

    with pytest.raises(ValueError, match="has shape"):
        _load_debug_synthetic_weights(model, state_dict)


def test_official_importer_rejects_a_non_gpt2_small_configuration() -> None:
    """The public importer refuses test-sized or otherwise incompatible architectures."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)

    with pytest.raises(ValueError, match="exactly match the original GPT-2 Small"):
        load_official_gpt2_weights(model, {})


def test_weight_tying_remains_intact_after_import() -> None:
    """Importing token embeddings preserves the shared output-head parameter storage."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    state_dict = _synthetic_reference_state_dict(model)

    _load_debug_synthetic_weights(model, state_dict)

    assert model.lm_head.weight is model.embeddings.wte.weight
    assert model.lm_head.weight.data_ptr() == model.embeddings.wte.weight.data_ptr()
    torch.testing.assert_close(model.lm_head.weight, state_dict["transformer.wte.weight"])
