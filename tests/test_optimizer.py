"""Tests for AdamW configuration and one GPT-2 training step."""

import math

import pytest
import torch
from torch import nn

from gpt2_124m.config import GPT2_DEBUG_CONFIG, TrainingConfig
from gpt2_124m.model import GPT2Model
from gpt2_124m.training import configure_optimizer, train_step


def _debug_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return one small, valid next-token prediction batch."""
    return (
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
        torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"weight_decay": -0.1}, "weight_decay must be non-negative"),
        ({"beta1": 1.0}, "beta1 must be in the range"),
        ({"beta2": -0.1}, "beta2 must be in the range"),
        ({"grad_clip_norm": 0.0}, "grad_clip_norm must be positive"),
    ],
)
def test_training_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    """Optimizer settings must stay within valid numerical ranges."""
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_optimizer_includes_every_trainable_parameter_exactly_once() -> None:
    """Tied weights and all remaining model parameters appear in exactly one group."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    optimizer = configure_optimizer(model, TrainingConfig())
    optimized_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

    assert len(optimized_ids) == len(set(optimized_ids))
    assert set(optimized_ids) == trainable_ids


def test_optimizer_excludes_biases_and_layer_norm_from_weight_decay() -> None:
    """One-dimensional bias and normalization parameters use the no-decay group."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    config = TrainingConfig(weight_decay=0.1)
    optimizer = configure_optimizer(model, config)
    no_decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] == 0.0)
    no_decay_ids = {id(parameter) for parameter in no_decay_group["params"]}

    assert id(model.h[0].attn.c_attn.bias) in no_decay_ids
    assert id(model.h[0].ln_1.weight) in no_decay_ids
    assert id(model.ln_f.bias) in no_decay_ids


def test_optimizer_applies_weight_decay_to_matrix_weights() -> None:
    """Embedding and projection matrices belong to the decay parameter group."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    config = TrainingConfig(weight_decay=0.1)
    optimizer = configure_optimizer(model, config)
    decay_group = next(
        group for group in optimizer.param_groups if group["weight_decay"] == config.weight_decay
    )
    decay_ids = {id(parameter) for parameter in decay_group["params"]}

    assert id(model.embeddings.wte.weight) in decay_ids
    assert id(model.h[0].mlp.c_fc.weight) in decay_ids


def test_configure_optimizer_rejects_a_model_without_trainable_parameters() -> None:
    """Creating an optimizer for a fully frozen model would be a configuration error."""
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="no trainable parameters"):
        configure_optimizer(model, TrainingConfig())


def test_train_step_returns_finite_metrics_and_updates_a_parameter() -> None:
    """One step computes loss, clips gradients, and changes trainable model weights."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    optimizer = configure_optimizer(model, TrainingConfig())
    original_weight = model.h[0].attn.c_attn.weight.detach().clone()

    metrics = train_step(
        model,
        _debug_batch(),
        optimizer,
        device="cpu",
        grad_clip_norm=1.0,
    )

    assert math.isfinite(metrics.loss)
    assert math.isfinite(metrics.grad_norm)
    assert not torch.equal(model.h[0].attn.c_attn.weight, original_weight)


def test_train_step_creates_adamw_optimizer_state() -> None:
    """AdamW allocates its moment estimates when the first update is performed."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    optimizer = configure_optimizer(model, TrainingConfig())

    train_step(model, _debug_batch(), optimizer, device="cpu", grad_clip_norm=1.0)

    assert optimizer.state


def test_train_step_switches_the_model_to_train_mode() -> None:
    """A training update enables dropout and other train-mode model behavior."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    model.eval()
    optimizer = configure_optimizer(model, TrainingConfig())

    train_step(model, _debug_batch(), optimizer, device="cpu", grad_clip_norm=1.0)

    assert model.training


@pytest.mark.parametrize("grad_clip_norm", [0.0, -1.0, True, float("nan")])
def test_train_step_rejects_invalid_gradient_clip_norm(grad_clip_norm: object) -> None:
    """Gradient clipping must have a finite, strictly positive threshold."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    optimizer = configure_optimizer(model, TrainingConfig())

    with pytest.raises(ValueError, match="grad_clip_norm"):
        train_step(
            model,
            _debug_batch(),
            optimizer,
            device="cpu",
            grad_clip_norm=grad_clip_norm,  # type: ignore[arg-type]
        )
