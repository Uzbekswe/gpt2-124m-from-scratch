"""Tests for portable, reproducible local-training checkpoints."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gpt2_124m.checkpoint import RestoredTrainingState, load_checkpoint, save_checkpoint
from gpt2_124m.config import GPT2_DEBUG_CONFIG, GPT2Config, LocalLoopConfig, TrainingConfig
from gpt2_124m.model import GPT2Model
from gpt2_124m.training import TrainingHistory, configure_optimizer, train_step


def _make_model_and_optimizer(config: GPT2Config = GPT2_DEBUG_CONFIG) -> tuple[
    GPT2Model, torch.optim.AdamW
]:
    """Build a small model and its AdamW optimizer for checkpoint tests."""
    model = GPT2Model(config)
    optimizer = configure_optimizer(model, TrainingConfig())
    return model, optimizer


def _batch(offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Make one deterministic debug-sized next-token batch."""
    input_ids = torch.tensor(
        [
            [offset, offset + 1, offset + 2, offset + 3],
            [offset + 4, offset + 5, offset + 6, offset + 7],
        ],
        dtype=torch.long,
    )
    return input_ids, input_ids + 1


def _history() -> TrainingHistory:
    """Return a representative local loop history."""
    return TrainingHistory(
        train_steps=[1],
        train_losses=[1.25],
        validation_steps=[1],
        validation_losses=[1.5],
    )


def _assert_nested_equal(actual: object, expected: object) -> None:
    """Compare state dictionaries containing nested tensors and Python values."""
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, type(actual))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_checkpoint_is_created_and_restores_identical_eval_logits(tmp_path: Path) -> None:
    """A checkpoint restores parameters exactly enough to reproduce eval-mode logits."""
    torch.manual_seed(123)
    model, optimizer = _make_model_and_optimizer()
    train_step(model, _batch(1), optimizer, device="cpu", grad_clip_norm=1.0)
    model.eval()
    with torch.inference_mode():
        logits_at_save_time = model(_batch(1)[0]).clone()

    checkpoint_path = tmp_path / "nested" / "local_resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        completed_step=1,
        history=_history(),
    )

    restored_model, restored_optimizer = _make_model_and_optimizer()
    load_checkpoint(
        str(checkpoint_path),
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )
    restored_model.eval()
    with torch.inference_mode():
        restored_logits = restored_model(_batch(1)[0])

    assert checkpoint_path.is_file()
    torch.testing.assert_close(restored_logits, logits_at_save_time)


def test_checkpoint_restores_optimizer_step_history_and_configurations(tmp_path: Path) -> None:
    """Optimizer moments and local metadata are all returned from the saved checkpoint."""
    model, optimizer = _make_model_and_optimizer()
    train_step(model, _batch(1), optimizer, device="cpu", grad_clip_norm=1.0)
    expected_optimizer_state = deepcopy(optimizer.state_dict())
    training_config = TrainingConfig(learning_rate=1e-3)
    loop_config = LocalLoopConfig(max_steps=3, eval_every=2, eval_batches=1)
    expected_history = _history()
    checkpoint_path = tmp_path / "metadata.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        completed_step=1,
        history=expected_history,
        training_config=training_config,
        loop_config=loop_config,
    )

    restored_model, restored_optimizer = _make_model_and_optimizer()
    restored_state = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )

    assert isinstance(restored_state, RestoredTrainingState)
    assert restored_state.completed_step == 1
    assert restored_state.history == expected_history
    assert restored_state.training_config == training_config
    assert restored_state.loop_config == loop_config
    _assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer_state)


def test_checkpoint_rejects_a_model_with_a_different_gpt2_configuration(tmp_path: Path) -> None:
    """Loading checks architecture metadata before accepting incompatible tensors."""
    model, optimizer = _make_model_and_optimizer()
    checkpoint_path = tmp_path / "debug.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        completed_step=0,
        history=TrainingHistory(),
    )
    different_config = replace(GPT2_DEBUG_CONFIG, emb_dim=64, n_heads=4)
    different_model, different_optimizer = _make_model_and_optimizer(different_config)

    with pytest.raises(ValueError, match="GPT2Config does not match"):
        load_checkpoint(
            checkpoint_path,
            model=different_model,
            optimizer=different_optimizer,
            map_location="cpu",
        )


def test_checkpoint_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    """Missing paths and non-checkpoint bytes receive understandable errors."""
    model, optimizer = _make_model_and_optimizer()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_checkpoint(
            tmp_path / "missing.pt",
            model=model,
            optimizer=optimizer,
            map_location="cpu",
        )

    malformed_path = tmp_path / "malformed.pt"
    malformed_path.write_bytes(b"not a PyTorch checkpoint")
    with pytest.raises(ValueError, match="malformed checkpoint"):
        load_checkpoint(
            malformed_path,
            model=model,
            optimizer=optimizer,
            map_location="cpu",
        )


def test_checkpoint_restores_rng_for_reproducible_next_training_step(tmp_path: Path) -> None:
    """The original and restored runs match after the same dropout-using next batch."""
    torch.manual_seed(456)
    dropout_debug_config = replace(GPT2_DEBUG_CONFIG, drop_rate=0.2)
    model, optimizer = _make_model_and_optimizer(dropout_debug_config)
    first_batch = _batch(1)
    next_batch = _batch(9)
    train_step(model, first_batch, optimizer, device="cpu", grad_clip_norm=1.0)
    checkpoint_path = tmp_path / "reproducible_resume.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        completed_step=1,
        history=_history(),
        training_config=TrainingConfig(),
        loop_config=LocalLoopConfig(max_steps=2, eval_every=1),
    )

    train_step(model, next_batch, optimizer, device="cpu", grad_clip_norm=1.0)

    restored_model, restored_optimizer = _make_model_and_optimizer(dropout_debug_config)
    load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )
    train_step(restored_model, next_batch, restored_optimizer, device="cpu", grad_clip_norm=1.0)

    for original_parameter, restored_parameter in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(original_parameter, restored_parameter, rtol=1e-5, atol=1e-6)
