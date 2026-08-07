"""Versioned, atomic checkpoints for reproducible local GPT-2 training resumes."""

import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import Tensor, nn

from gpt2_124m.config import GPT2Config, LocalLoopConfig, TrainingConfig
from gpt2_124m.training import TrainingHistory

CHECKPOINT_SCHEMA_VERSION = 1
"""The supported on-disk checkpoint schema version."""

ConfigType = TypeVar("ConfigType", TrainingConfig, LocalLoopConfig)


@dataclass(frozen=True, slots=True)
class RestoredTrainingState:
    """Serializable local-training state restored from a checkpoint."""

    completed_step: int
    history: TrainingHistory
    training_config: TrainingConfig | None
    loop_config: LocalLoopConfig | None


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_step: int,
    history: TrainingHistory,
    training_config: TrainingConfig | None = None,
    loop_config: LocalLoopConfig | None = None,
) -> Path:
    """Atomically save serializable state needed for a reproducible local resume."""
    checkpoint_path = _validate_checkpoint_path(path)
    _validate_completed_step(completed_step)
    model_config = _get_model_config(model)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "gpt2_config": asdict(model_config),
                "training_config": asdict(training_config) if training_config is not None else None,
                "loop_config": asdict(loop_config) if loop_config is not None else None,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_step": completed_step,
                "history": asdict(history),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": _get_cuda_rng_states(),
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: torch.device | str | None = None,
) -> RestoredTrainingState:
    """Load model, optimizer, metadata, and PyTorch RNG state into existing objects."""
    checkpoint_path = _validate_checkpoint_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except (OSError, pickle.UnpicklingError, RuntimeError, EOFError) as error:
        raise ValueError(f"malformed checkpoint: could not load {checkpoint_path}") from error

    if not isinstance(checkpoint, Mapping):
        raise ValueError("malformed checkpoint: expected a dictionary payload.")
    _validate_schema_version(checkpoint)
    _validate_required_keys(checkpoint)

    saved_config = _deserialize_gpt2_config(checkpoint["gpt2_config"])
    if saved_config != _get_model_config(model):
        raise ValueError("checkpoint GPT2Config does not match model.config.")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("malformed checkpoint: invalid model state dictionary.") from error
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise ValueError("malformed checkpoint: invalid optimizer state dictionary.") from error

    completed_step = _validate_completed_step(checkpoint["completed_step"])
    history = _deserialize_history(checkpoint["history"])
    training_config = _deserialize_optional_config(checkpoint["training_config"], TrainingConfig)
    loop_config = _deserialize_optional_config(checkpoint["loop_config"], LocalLoopConfig)
    _restore_rng_states(checkpoint["torch_rng_state"], checkpoint["cuda_rng_states"])

    return RestoredTrainingState(
        completed_step=completed_step,
        history=history,
        training_config=training_config,
        loop_config=loop_config,
    )


def _validate_checkpoint_path(path: str | Path) -> Path:
    """Normalize a checkpoint path and require the documented file extension."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".pt":
        raise ValueError("checkpoint paths must use the .pt extension.")
    return checkpoint_path


def _get_model_config(model: nn.Module) -> GPT2Config:
    """Return the GPT-2 configuration attached to a compatible model instance."""
    config = getattr(model, "config", None)
    if not isinstance(config, GPT2Config):
        raise TypeError("model must expose a GPT2Config as model.config.")
    return config


def _validate_completed_step(value: object) -> int:
    """Return a valid non-negative global update count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("completed_step must be a non-negative integer.")
    return value


def _get_cuda_rng_states() -> list[Tensor] | None:
    """Capture CUDA generators when CUDA is available, keeping checkpoint tensors portable."""
    if not torch.cuda.is_available():
        return None
    return [state.cpu() for state in torch.cuda.get_rng_state_all()]


def _validate_schema_version(checkpoint: Mapping[str, Any]) -> None:
    """Reject checkpoints written by a schema this project does not understand."""
    schema_version = checkpoint.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema version: "
            f"{schema_version!r}; expected {CHECKPOINT_SCHEMA_VERSION}."
        )


def _validate_required_keys(checkpoint: Mapping[str, Any]) -> None:
    """Ensure the checkpoint contains every field needed for a local resume."""
    required_keys = {
        "gpt2_config",
        "training_config",
        "loop_config",
        "model_state_dict",
        "optimizer_state_dict",
        "completed_step",
        "history",
        "torch_rng_state",
        "cuda_rng_states",
    }
    missing_keys = sorted(required_keys.difference(checkpoint))
    if missing_keys:
        raise ValueError(f"malformed checkpoint: missing required keys {missing_keys}.")


def _deserialize_gpt2_config(value: object) -> GPT2Config:
    """Rebuild a validated GPT-2 configuration from its plain saved dictionary."""
    if not isinstance(value, Mapping):
        raise ValueError("malformed checkpoint: gpt2_config must be a dictionary.")
    try:
        return GPT2Config(**dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("malformed checkpoint: invalid gpt2_config.") from error


def _deserialize_optional_config(
    value: object,
    config_type: type[ConfigType],
) -> ConfigType | None:
    """Rebuild one optional validated configuration from a plain saved dictionary."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("malformed checkpoint: saved configuration must be a dictionary or None.")
    try:
        return config_type(**dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("malformed checkpoint: invalid saved configuration.") from error


def _deserialize_history(value: object) -> TrainingHistory:
    """Rebuild history with independent lists, not a checkpoint-owned mutable mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("malformed checkpoint: history must be a dictionary.")

    history_keys = {
        "train_steps",
        "train_losses",
        "validation_steps",
        "validation_losses",
    }
    if set(value) != history_keys:
        raise ValueError("malformed checkpoint: history has unexpected fields.")
    if not all(isinstance(value[key], list) for key in history_keys):
        raise ValueError("malformed checkpoint: history fields must be lists.")

    return TrainingHistory(
        train_steps=list(value["train_steps"]),
        train_losses=list(value["train_losses"]),
        validation_steps=list(value["validation_steps"]),
        validation_losses=list(value["validation_losses"]),
    )


def _restore_rng_states(cpu_rng_state: object, cuda_rng_states: object) -> None:
    """Restore PyTorch random generators after object state has been loaded."""
    if not isinstance(cpu_rng_state, Tensor) or cpu_rng_state.dtype != torch.uint8:
        raise ValueError("malformed checkpoint: torch_rng_state must be a uint8 tensor.")
    torch.set_rng_state(cpu_rng_state.cpu())

    if cuda_rng_states is None or not torch.cuda.is_available():
        return
    if not isinstance(cuda_rng_states, (list, tuple)) or not all(
        isinstance(state, Tensor) and state.dtype == torch.uint8 for state in cuda_rng_states
    ):
        raise ValueError("malformed checkpoint: cuda_rng_states must be uint8 tensors or None.")
    torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])
