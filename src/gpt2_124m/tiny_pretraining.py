"""A cost-capped, artifact-producing training proof for exact GPT-2 Small."""

import json
import platform
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from gpt2_124m.checkpoint import save_checkpoint
from gpt2_124m.config import GPT2Config, LocalLoopConfig, TinyPretrainingConfig
from gpt2_124m.fineweb import FineWebEduIterableDataset, TokenizerLike
from gpt2_124m.generation import generate
from gpt2_124m.model import GPT2Model
from gpt2_124m.tokenizer import GPT2Tokenizer
from gpt2_124m.training import (
    TrainingHistory,
    configure_optimizer,
    evaluate_loss,
    train_step,
)

EXPECTED_PARAMETER_COUNT = 124_439_808


@dataclass(frozen=True, slots=True)
class TinyTrainingResult:
    """Paths and scalar results from one completed tiny pretraining proof."""

    completed_steps: int
    initial_train_loss: float
    final_train_loss: float
    final_validation_loss: float | None
    output_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    sample_path: Path
    summary_path: Path


class TinyTrainingDeadlineExceeded(RuntimeError):
    """Signal that the configurable wall-clock deadline ended a tiny training run."""

    def __init__(self, reason: str, summary_path: Path | None = None) -> None:
        super().__init__(reason)
        self.summary_path = summary_path


@dataclass(frozen=True, slots=True)
class _Deadline:
    """A cooperative, monotonic wall-clock deadline safe for Unix and local CPU execution."""

    started_at: float
    max_runtime_seconds: int | None

    @classmethod
    def start(cls, max_runtime_seconds: int | None) -> "_Deadline":
        return cls(started_at=time.monotonic(), max_runtime_seconds=max_runtime_seconds)

    def check(self, phase: str) -> None:
        if self.max_runtime_seconds is None:
            return
        elapsed_seconds = time.monotonic() - self.started_at
        if elapsed_seconds >= self.max_runtime_seconds:
            raise TinyTrainingDeadlineExceeded(
                f"max_runtime_seconds ({self.max_runtime_seconds}) expired during {phase}.",
            )


class _TruncatedSequenceDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Expose shorter training sequences without changing GPT-2's 1,024-position model."""

    def __init__(
        self,
        source: Iterable[tuple[Tensor, Tensor]],
        *,
        sequence_length: int,
    ) -> None:
        super().__init__()
        self.source = source
        self.sequence_length = sequence_length

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for input_ids, target_ids in self.source:
            if input_ids.ndim != 1 or target_ids.ndim != 1:
                raise ValueError("streamed examples must be rank-1 token tensors.")
            if input_ids.shape != target_ids.shape:
                raise ValueError("streamed input and target tensors must have matching shapes.")
            if input_ids.shape[0] < self.sequence_length:
                continue
            yield input_ids[: self.sequence_length], target_ids[: self.sequence_length]


class _BatchedIterable(Iterable[tuple[Tensor, Tensor]]):
    """Batch a stream in-process and close its iterator when a fixed-step run stops early."""

    def __init__(self, source: Iterable[tuple[Tensor, Tensor]], *, batch_size: int) -> None:
        self.source = source
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        iterator = iter(self.source)
        input_batch: list[Tensor] = []
        target_batch: list[Tensor] = []
        try:
            for input_ids, target_ids in iterator:
                input_batch.append(input_ids)
                target_batch.append(target_ids)
                if len(input_batch) == self.batch_size:
                    yield torch.stack(input_batch), torch.stack(target_batch)
                    input_batch.clear()
                    target_batch.clear()
            if input_batch:
                yield torch.stack(input_batch), torch.stack(target_batch)
        finally:
            _close_if_possible(iterator)


def build_fineweb_tiny_dataloaders(
    config: TinyPretrainingConfig,
    *,
    model_config: GPT2Config = GPT2Config(),
) -> tuple[
    Iterable[tuple[Tensor, Tensor]],
    Iterable[tuple[Tensor, Tensor]],
    TokenizerLike,
]:
    """Build closable lazy FineWeb-Edu batch streams without opening remote data yet."""
    _validate_sequence_length(config.sequence_length, model_config)
    try:
        tokenizer: TokenizerLike = GPT2Tokenizer()
    except ImportError as error:
        raise ImportError(
            "Tiny FineWeb training requires tiktoken and datasets. "
            'Install them with `python -m pip install -e ".[train]"`. '
        ) from error

    dataset_kwargs = {
        "config": model_config,
        "tokenizer": tokenizer,
        "validation_fraction": config.validation_fraction,
        "dataset_name": config.dataset_name,
        "configuration": config.dataset_configuration,
        "revision": config.dataset_revision,
        "text_field": config.text_field,
        "document_id_field": config.document_id_field,
    }
    train_dataset = FineWebEduIterableDataset(
        split="train",
        max_documents=config.train_max_documents,
        **dataset_kwargs,
    )
    validation_dataset = FineWebEduIterableDataset(
        split="validation",
        max_documents=config.validation_max_documents,
        **dataset_kwargs,
    )
    return (
        _BatchedIterable(
            _TruncatedSequenceDataset(train_dataset, sequence_length=config.sequence_length),
            batch_size=config.batch_size,
        ),
        _BatchedIterable(
            _TruncatedSequenceDataset(validation_dataset, sequence_length=config.sequence_length),
            batch_size=config.batch_size,
        ),
        tokenizer,
    )


def run_fineweb_tiny_pretraining(config: TinyPretrainingConfig) -> TinyTrainingResult:
    """Run the public exact-GPT-2 entry point on lazily streamed FineWeb-Edu."""
    model_config = GPT2Config()
    try:
        train_loader, validation_loader, tokenizer = build_fineweb_tiny_dataloaders(
            config,
            model_config=model_config,
        )
    except Exception as error:
        _write_initialization_failure_summary(config, model_config, error)
        raise
    return run_tiny_pretraining(
        config,
        train_loader=train_loader,
        validation_loader=validation_loader,
        tokenizer=tokenizer,
        model_config=model_config,
    )


def run_tiny_pretraining(
    config: TinyPretrainingConfig,
    *,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    validation_loader: Iterable[tuple[Tensor, Tensor]],
    tokenizer: TokenizerLike,
    model_config: GPT2Config = GPT2Config(),
) -> TinyTrainingResult:
    """Train a supplied GPT-2 configuration for a tiny fixed step budget and save artifacts.

    The public FineWeb entry point always supplies the exact GPT-2 Small configuration. The
    ``model_config`` argument exists so CPU tests can exercise this workflow with the existing
    test-only debug configuration.
    """
    if not isinstance(config, TinyPretrainingConfig):
        raise TypeError("config must be a TinyPretrainingConfig.")
    if not isinstance(model_config, GPT2Config):
        raise TypeError("model_config must be a GPT2Config.")
    _validate_sequence_length(config.sequence_length, model_config)
    _set_seed(config.seed)
    device = _select_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model: GPT2Model | None = None
    optimizer: torch.optim.Optimizer | None = None
    parameter_count: int | None = None
    history = TrainingHistory()
    loop_config = LocalLoopConfig(
        max_steps=config.max_steps,
        eval_every=config.eval_every,
        eval_batches=config.eval_batches,
    )
    config_path = output_dir / "training_config.json"
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint_final.pt"
    sample_path = output_dir / "generated_sample.txt"
    summary_path = output_dir / "training_summary.json"
    _write_json(
        config_path,
        {
            "tiny_pretraining_config": asdict(config),
            "gpt2_config": asdict(model_config),
        },
    )

    deadline = _Deadline.start(config.max_runtime_seconds)
    initial_train_loss: float | None = None
    final_validation_loss: float | None = None
    completed_steps = 0
    status = "failed"
    reason: str | None = None
    timeout_error: TinyTrainingDeadlineExceeded | None = None
    checkpoint_error: str | None = None
    train_iterator: Iterator[tuple[Tensor, Tensor]] | None = None

    try:
        deadline.check("model initialization")
        model = GPT2Model(model_config).to(device)
        parameter_count = model.count_trainable_parameters()
        if model_config == GPT2Config() and parameter_count != EXPECTED_PARAMETER_COUNT:
            raise AssertionError(
                "expected "
                f"{EXPECTED_PARAMETER_COUNT:,} trainable parameters, got {parameter_count:,}."
            )
        optimizer = configure_optimizer(model, config.optimizer)
        deadline.check("initial train-loss evaluation")
        initial_train_loss = evaluate_loss(model, train_loader, device, max_batches=1)
        _append_metric(
            metrics_path,
            {"event": "initial_train_loss", "step": 0, "loss": initial_train_loss},
        )
        deadline.check("initial train-loss evaluation")
        train_iterator = iter(train_loader)

        for step in range(1, config.max_steps + 1):
            deadline.check(f"before optimizer step {step}")
            batch, train_iterator = _next_batch(train_iterator, train_loader, name="train_loader")
            metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                grad_clip_norm=config.optimizer.grad_clip_norm,
            )
            completed_steps = step
            history.train_steps.append(step)
            history.train_losses.append(metrics.loss)
            if step % config.log_every == 0 or step == config.max_steps:
                _append_metric(
                    metrics_path,
                    {
                        "event": "train",
                        "step": step,
                        "loss": metrics.loss,
                        "grad_norm": metrics.grad_norm,
                    },
                )
            deadline.check(f"after optimizer step {step}")

            if step % config.eval_every == 0 or step == config.max_steps:
                final_validation_loss = evaluate_loss(
                    model,
                    validation_loader,
                    device,
                    max_batches=config.eval_batches,
                )
                history.validation_steps.append(step)
                history.validation_losses.append(final_validation_loss)
                _append_metric(
                    metrics_path,
                    {"event": "validation", "step": step, "loss": final_validation_loss},
                )
                deadline.check(f"validation after optimizer step {step}")

        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            completed_step=completed_steps,
            history=history,
            training_config=config.optimizer,
            loop_config=loop_config,
        )
        deadline.check("final checkpoint")
        generated_text = _generate_sample(model, tokenizer, config, device)
        sample_path.write_text(generated_text + "\n", encoding="utf-8")
        deadline.check("sample generation")
        status = "completed"
    except TinyTrainingDeadlineExceeded as error:
        status = "timed_out"
        reason = str(error)
        timeout_error = error
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        _close_if_possible(train_iterator)
        should_save_partial_checkpoint = (
            model is not None
            and optimizer is not None
            and completed_steps > 0
            and not checkpoint_path.exists()
        )
        if should_save_partial_checkpoint:
            try:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    completed_step=completed_steps,
                    history=history,
                    training_config=config.optimizer,
                    loop_config=loop_config,
                )
            except Exception as error:
                checkpoint_error = f"{type(error).__name__}: {error}"
        duration_seconds = time.monotonic() - deadline.started_at
        summary = {
            "status": status,
            "reason": reason,
            "parameter_count": parameter_count,
            "device": _device_info(device),
            "dataset": {
                "name": config.dataset_name,
                "configuration": config.dataset_configuration,
                "revision": config.dataset_revision,
                "train_split": "train",
                "validation_split": "validation",
                "validation_fraction": config.validation_fraction,
            },
            "completed_steps": completed_steps,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": history.train_losses[-1] if history.train_losses else None,
            "final_validation_loss": final_validation_loss,
            "wall_clock_seconds": duration_seconds,
            "checkpoint_error": checkpoint_error,
            "artifact_paths": {
                "training_config": str(config_path),
                "metrics": str(metrics_path),
                "checkpoint": str(checkpoint_path),
                "generated_sample": str(sample_path),
                "summary": str(summary_path),
            },
        }
        _write_json(summary_path, summary)

    if timeout_error is not None:
        raise TinyTrainingDeadlineExceeded(str(timeout_error), summary_path)
    if initial_train_loss is None or not history.train_losses:
        raise RuntimeError("tiny training finished without an initial and final train loss.")
    return TinyTrainingResult(
        completed_steps=completed_steps,
        initial_train_loss=initial_train_loss,
        final_train_loss=history.train_losses[-1],
        final_validation_loss=final_validation_loss,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        sample_path=sample_path,
        summary_path=summary_path,
    )


def _next_batch(
    iterator: Iterator[tuple[Tensor, Tensor]],
    dataloader: Iterable[tuple[Tensor, Tensor]],
    *,
    name: str,
) -> tuple[tuple[Tensor, Tensor], Iterator[tuple[Tensor, Tensor]]]:
    """Return the next batch, restarting re-iterable streaming loaders when needed."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        try:
            return next(iterator), iterator
        except StopIteration as error:
            raise ValueError(f"{name} must provide at least one batch.") from error


def _close_if_possible(resource: object | None) -> None:
    """Close early-stopped generator stacks before native-library interpreter teardown."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _write_initialization_failure_summary(
    config: TinyPretrainingConfig,
    model_config: GPT2Config,
    error: Exception,
) -> None:
    """Record a concise artifact when optional tokenizer/data setup fails before training starts."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    _write_json(
        summary_path,
        {
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "parameter_count": EXPECTED_PARAMETER_COUNT
            if model_config == GPT2Config()
            else None,
            "device": {"selected_device": config.device},
            "dataset": {
                "name": config.dataset_name,
                "configuration": config.dataset_configuration,
                "revision": config.dataset_revision,
                "train_split": "train",
                "validation_split": "validation",
                "validation_fraction": config.validation_fraction,
            },
            "completed_steps": 0,
            "initial_train_loss": None,
            "final_train_loss": None,
            "final_validation_loss": None,
            "wall_clock_seconds": 0.0,
            "artifact_paths": {
                "training_config": str(output_dir / "training_config.json"),
                "metrics": str(output_dir / "metrics.jsonl"),
                "checkpoint": str(output_dir / "checkpoint_final.pt"),
                "generated_sample": str(output_dir / "generated_sample.txt"),
                "summary": str(summary_path),
            },
        },
    )


def _generate_sample(
    model: GPT2Model,
    tokenizer: TokenizerLike,
    config: TinyPretrainingConfig,
    device: torch.device,
) -> str:
    """Generate a short post-training sample without modifying model parameters."""
    prompt_ids = tokenizer.encode(config.prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token.")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device.type).manual_seed(config.seed)
    generated_ids = generate(
        model,
        input_ids,
        config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        do_sample=config.do_sample,
        generator=generator,
    )
    return tokenizer.decode(generated_ids[0].tolist())


def _set_seed(seed: int) -> None:
    """Seed CPU and CUDA generators used by initialization, dropout, and sampling."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(requested_device: str) -> torch.device:
    """Resolve the explicit CPU/CUDA choice or an automatic CUDA-first selection."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested_device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid device: {requested_device!r}.") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device string.")
    return device


def _validate_sequence_length(sequence_length: int, model_config: GPT2Config) -> None:
    """Keep short proof sequences within the exact model's learned position range."""
    if sequence_length > model_config.context_length:
        raise ValueError(
            f"sequence_length ({sequence_length}) exceeds model context_length "
            f"({model_config.context_length})."
        )


def _append_metric(path: Path, record: dict[str, object]) -> None:
    """Append one JSONL event so metrics remain readable if a later step fails."""
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a small human-readable JSON artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _device_info(device: torch.device) -> dict[str, object]:
    """Record device evidence without requiring CUDA on CPU test machines."""
    info: dict[str, object] = {
        "selected_device": str(device),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_bytes": properties.total_memory,
            }
        )
    return info
