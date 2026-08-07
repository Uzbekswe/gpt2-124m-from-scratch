"""Tests for the local multi-step GPT-2 training loop."""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gpt2_124m.config import GPT2_DEBUG_CONFIG, LocalLoopConfig, TrainingConfig
from gpt2_124m.model import GPT2Model
from gpt2_124m.training import TrainingHistory, configure_optimizer, train_model_simple


def _loader(batch_size: int = 1) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Return a small reusable DataLoader with valid GPT-2 input-target pairs."""
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    target_ids = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
    return DataLoader(TensorDataset(input_ids, target_ids), batch_size=batch_size, shuffle=False)


def _empty_loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Return an empty DataLoader with otherwise valid tensor shapes and dtypes."""
    input_ids = torch.empty((0, 3), dtype=torch.long)
    target_ids = torch.empty((0, 3), dtype=torch.long)
    return DataLoader(TensorDataset(input_ids, target_ids), batch_size=1)


def _run_loop(
    max_steps: int,
    eval_every: int,
) -> tuple[GPT2Model, torch.optim.AdamW, TrainingHistory]:
    """Build DEBUG-config training components and run a requested number of updates."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    training_config = TrainingConfig()
    optimizer = configure_optimizer(model, training_config)
    history = train_model_simple(
        model,
        _loader(),
        _loader(),
        optimizer,
        device="cpu",
        training_config=training_config,
        loop_config=LocalLoopConfig(max_steps=max_steps, eval_every=eval_every),
    )
    return model, optimizer, history


def test_loop_performs_exactly_max_steps_and_records_one_loss_per_step() -> None:
    """Each requested update produces exactly one training loss record."""
    model, optimizer, history = _run_loop(max_steps=3, eval_every=2)

    assert history.train_steps == [1, 2, 3]
    assert len(history.train_losses) == 3
    assert optimizer.state[model.h[0].attn.c_attn.weight]["step"].item() == 3


def test_loop_evaluates_on_schedule_and_after_the_final_step() -> None:
    """Evaluation occurs at cadence boundaries and always includes the final update."""
    _, _, history = _run_loop(max_steps=5, eval_every=2)

    assert history.validation_steps == [2, 4, 5]
    assert len(history.validation_losses) == 3


def test_loop_changes_model_parameters() -> None:
    """Repeated training steps update trainable GPT-2 parameters."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    training_config = TrainingConfig()
    optimizer = configure_optimizer(model, training_config)
    original_weight = model.h[0].attn.c_attn.weight.detach().clone()

    train_model_simple(
        model,
        _loader(),
        _loader(),
        optimizer,
        device="cpu",
        training_config=training_config,
        loop_config=LocalLoopConfig(max_steps=2, eval_every=1),
    )

    assert not torch.equal(model.h[0].attn.c_attn.weight, original_weight)


def test_loop_restarts_a_short_train_loader_without_caching_batches() -> None:
    """A one-batch loader can provide more updates by creating fresh iterators."""
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    training_config = TrainingConfig()
    optimizer = configure_optimizer(model, training_config)
    one_batch_loader = _loader(batch_size=2)

    history = train_model_simple(
        model,
        one_batch_loader,
        _loader(),
        optimizer,
        device="cpu",
        training_config=training_config,
        loop_config=LocalLoopConfig(max_steps=3, eval_every=2),
    )

    assert history.train_steps == [1, 2, 3]
    assert optimizer.state[model.h[0].attn.c_attn.weight]["step"].item() == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_steps": 0, "eval_every": 1}, "max_steps must be a positive integer"),
        ({"max_steps": 1, "eval_every": 0}, "eval_every must be a positive integer"),
        ({"max_steps": 1, "eval_every": 1, "eval_batches": 0}, "eval_batches"),
    ],
)
def test_local_loop_config_rejects_invalid_values(kwargs: dict[str, int], message: str) -> None:
    """Local update and validation cadence values must be positive counts."""
    with pytest.raises(ValueError, match=message):
        LocalLoopConfig(**kwargs)


def test_empty_train_and_validation_loaders_fail_clearly() -> None:
    """The loop reports which required DataLoader has no batches to provide."""
    training_config = TrainingConfig()

    train_model = GPT2Model(GPT2_DEBUG_CONFIG)
    train_optimizer = configure_optimizer(train_model, training_config)
    with pytest.raises(ValueError, match="train_loader"):
        train_model_simple(
            train_model,
            _empty_loader(),
            _loader(),
            train_optimizer,
            device="cpu",
            training_config=training_config,
            loop_config=LocalLoopConfig(max_steps=1, eval_every=1),
        )

    val_model = GPT2Model(GPT2_DEBUG_CONFIG)
    val_optimizer = configure_optimizer(val_model, training_config)
    with pytest.raises(ValueError, match="val_loader"):
        train_model_simple(
            val_model,
            _loader(),
            _empty_loader(),
            val_optimizer,
            device="cpu",
            training_config=training_config,
            loop_config=LocalLoopConfig(max_steps=1, eval_every=1),
        )


def test_local_training_and_validation_losses_are_finite() -> None:
    """The loop returns finite Python-number histories suitable for later plotting."""
    _, _, history = _run_loop(max_steps=3, eval_every=2)

    assert all(math.isfinite(loss) for loss in history.train_losses)
    assert all(math.isfinite(loss) for loss in history.validation_losses)
