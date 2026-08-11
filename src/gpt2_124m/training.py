"""Language-model loss and read-only validation evaluation utilities."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from math import isfinite
from numbers import Real

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from gpt2_124m.config import LocalLoopConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class TrainingStepMetrics:
    """Scalar metrics produced by one optimizer-backed training step."""

    loss: float
    grad_norm: float


@dataclass(slots=True)
class TrainingHistory:
    """Serializable losses and update indices collected by a local training loop."""

    train_steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    validation_steps: list[int] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)


def compute_language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute next-token cross-entropy from raw vocabulary logits and token-ID targets."""
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch_size, sequence_length, vocab_size]; "
            f"got {tuple(logits.shape)}."
        )
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be a floating-point tensor.")
    if targets.ndim != 2:
        raise ValueError(
            "targets must have shape [batch_size, sequence_length]; "
            f"got {tuple(targets.shape)}."
        )
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits and targets must have matching batch and sequence dimensions; "
            f"got {tuple(logits.shape[:2])} and {tuple(targets.shape)}."
        )
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if targets.dtype not in integer_dtypes:
        raise TypeError("targets must contain integer token IDs.")

    vocab_size = logits.shape[-1]
    if targets.numel() > 0 and (targets.min() < 0 or targets.max() >= vocab_size):
        raise ValueError(f"target IDs must be in [0, {vocab_size}).")

    flattened_logits = logits.reshape(-1, vocab_size)
    flattened_targets = targets.reshape(-1).to(dtype=torch.long)
    return functional.cross_entropy(flattened_logits, flattened_targets)


def evaluate_loss(
    model: nn.Module,
    dataloader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device | str,
    max_batches: int | None = None,
) -> float:
    """Return mean next-token loss without creating gradients or changing model weights."""
    if (
        max_batches is not None
        and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0)
    ):
        raise ValueError("max_batches must be a positive integer or None.")

    original_training_state = model.training
    losses: list[float] = []
    model.eval()
    iterator = iter(dataloader)
    try:
        batches = iterator if max_batches is None else islice(iterator, max_batches)
        with torch.inference_mode():
            for input_ids, target_ids in batches:
                logits = model(input_ids.to(device))
                loss = compute_language_model_loss(logits, target_ids.to(device))
                losses.append(loss.item())
    finally:
        _close_if_possible(iterator)
        model.train(original_training_state)

    if not losses:
        raise ValueError("dataloader must provide at least one batch for evaluation.")
    return sum(losses) / len(losses)


def configure_optimizer(model: nn.Module, training_config: TrainingConfig) -> torch.optim.AdamW:
    """Create AdamW with decay only on trainable matrix and tensor weights."""
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    seen_parameter_ids: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen_parameter_ids:
            continue
        seen_parameter_ids.add(id(parameter))
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    if not decay_parameters and not no_decay_parameters:
        raise ValueError("model has no trainable parameters.")

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": training_config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
    )


def train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    grad_clip_norm: float,
) -> TrainingStepMetrics:
    """Run one loss, backward, gradient-clipping, and AdamW update sequence."""
    if (
        isinstance(grad_clip_norm, bool)
        or not isinstance(grad_clip_norm, Real)
        or not isfinite(grad_clip_norm)
        or grad_clip_norm <= 0.0
    ):
        raise ValueError("grad_clip_norm must be a positive finite number.")

    input_ids, target_ids = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids.to(device))
    loss = compute_language_model_loss(logits, target_ids.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("training loss is non-finite.")

    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return TrainingStepMetrics(loss=loss.item(), grad_norm=float(gradient_norm))


def train_model_simple(
    model: nn.Module,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    val_loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    training_config: TrainingConfig,
    loop_config: LocalLoopConfig,
) -> TrainingHistory:
    """Run a fixed number of local training updates with periodic read-only validation."""
    history = TrainingHistory()
    train_iterator = iter(train_loader)
    try:
        for step in range(1, loop_config.max_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                _close_if_possible(train_iterator)
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as error:
                    raise ValueError("train_loader must provide at least one batch.") from error

            metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                grad_clip_norm=training_config.grad_clip_norm,
            )
            history.train_steps.append(step)
            history.train_losses.append(metrics.loss)

            if step % loop_config.eval_every == 0 or step == loop_config.max_steps:
                try:
                    validation_loss = evaluate_loss(
                        model,
                        val_loader,
                        device,
                        max_batches=loop_config.eval_batches,
                    )
                except ValueError as error:
                    if str(error) == "dataloader must provide at least one batch for evaluation.":
                        raise ValueError("val_loader must provide at least one batch.") from error
                    raise
                history.validation_steps.append(step)
                history.validation_losses.append(validation_loss)
    finally:
        _close_if_possible(train_iterator)

    return history


def _close_if_possible(resource: object | None) -> None:
    """Close an early-stopped iterable before interpreter shutdown when it supports ``close``."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()
