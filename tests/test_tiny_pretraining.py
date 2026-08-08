"""CPU-fast checks for the artifact-producing tiny pretraining workflow."""

import json
from pathlib import Path

import pytest
import torch

from gpt2_124m import tiny_pretraining
from gpt2_124m.checkpoint import load_checkpoint
from gpt2_124m.config import GPT2_DEBUG_CONFIG, TinyPretrainingConfig, TrainingConfig
from gpt2_124m.model import GPT2Model
from gpt2_124m.tiny_pretraining import TinyTrainingDeadlineExceeded, run_tiny_pretraining
from gpt2_124m.training import configure_optimizer, train_step


class FakeTokenizer:
    """Small deterministic tokenizer substitute used only by CPU workflow tests."""

    end_of_text_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _batch(offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[offset, offset + 1, offset + 2, offset + 3]], dtype=torch.long)
    return input_ids, input_ids + 1


def _config(output_dir: Path) -> TinyPretrainingConfig:
    return TinyPretrainingConfig(
        output_dir=str(output_dir),
        max_steps=2,
        batch_size=1,
        sequence_length=4,
        eval_every=1,
        eval_batches=1,
        log_every=1,
        max_new_tokens=2,
        top_k=4,
        do_sample=False,
        device="cpu",
    )


def test_one_optimizer_step_changes_a_debug_model_parameter() -> None:
    """The shared training step performs a real optimizer update on CPU."""
    torch.manual_seed(7)
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    optimizer = configure_optimizer(model, TrainingConfig())
    original_weight = model.embeddings.wte.weight.detach().clone()

    train_step(model, _batch(1), optimizer, device="cpu", grad_clip_norm=1.0)

    assert not torch.equal(model.embeddings.wte.weight, original_weight)


def test_tiny_pretraining_writes_metrics_summary_sample_and_checkpoint(tmp_path: Path) -> None:
    """A complete tiny CPU proof creates every artifact required by the future cloud run."""
    result = run_tiny_pretraining(
        _config(tmp_path),
        train_loader=[_batch(1), _batch(9)],
        validation_loader=[_batch(17)],
        tokenizer=FakeTokenizer(),
        model_config=GPT2_DEBUG_CONFIG,
    )

    assert result.completed_steps == 2
    assert result.checkpoint_path.is_file()
    assert result.metrics_path.is_file()
    assert result.sample_path.is_file()
    assert result.summary_path.is_file()

    metric_records = [json.loads(line) for line in result.metrics_path.read_text().splitlines()]
    assert [record["event"] for record in metric_records] == [
        "initial_train_loss",
        "train",
        "validation",
        "train",
        "validation",
    ]
    summary = json.loads(result.summary_path.read_text())
    assert summary["status"] == "completed"
    assert summary["reason"] is None
    assert summary["completed_steps"] == 2
    assert summary["parameter_count"] == GPT2Model(GPT2_DEBUG_CONFIG).count_trainable_parameters()
    assert summary["final_validation_loss"] is not None
    assert Path(summary["artifact_paths"]["checkpoint"]) == result.checkpoint_path


def test_tiny_pretraining_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    """The final artifact contains model and optimizer state usable by the existing loader."""
    result = run_tiny_pretraining(
        _config(tmp_path),
        train_loader=[_batch(1)],
        validation_loader=[_batch(9)],
        tokenizer=FakeTokenizer(),
        model_config=GPT2_DEBUG_CONFIG,
    )
    restored_model = GPT2Model(GPT2_DEBUG_CONFIG)
    restored_optimizer = configure_optimizer(restored_model, TrainingConfig())

    restored_state = load_checkpoint(
        result.checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        map_location="cpu",
    )

    assert restored_state.completed_step == 2
    assert restored_state.training_config == _config(tmp_path).optimizer


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_runtime_seconds": 0},
        {"batch_size": 0},
        {"sequence_length": 0},
        {"validation_fraction": 1.0},
        {"top_k": 0},
        {"device": ""},
    ],
)
def test_tiny_pretraining_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    """Unsafe budgets and malformed data/sampling options fail before work begins."""
    with pytest.raises(ValueError):
        TinyPretrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_tiny_pretraining_rejects_sequence_beyond_model_context(tmp_path: Path) -> None:
    """Short-run data cannot exceed the model's learned positional-embedding table."""
    config = TinyPretrainingConfig(
        output_dir=str(tmp_path),
        sequence_length=GPT2_DEBUG_CONFIG.context_length + 1,
    )
    with pytest.raises(ValueError, match="exceeds model context_length"):
        run_tiny_pretraining(
            config,
            train_loader=[_batch(1)],
            validation_loader=[_batch(9)],
            tokenizer=FakeTokenizer(),
            model_config=GPT2_DEBUG_CONFIG,
        )


def test_tiny_pretraining_writes_a_timeout_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monotonic deadline stops safely and records why no optimizer step completed."""
    clock_values = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr(tiny_pretraining.time, "monotonic", lambda: next(clock_values))
    config = TinyPretrainingConfig(
        output_dir=str(tmp_path),
        max_runtime_seconds=1,
        max_steps=2,
        sequence_length=4,
        max_new_tokens=1,
        top_k=4,
        do_sample=False,
        device="cpu",
    )

    with pytest.raises(TinyTrainingDeadlineExceeded, match="max_runtime_seconds") as error:
        run_tiny_pretraining(
            config,
            train_loader=[_batch(1)],
            validation_loader=[_batch(9)],
            tokenizer=FakeTokenizer(),
            model_config=GPT2_DEBUG_CONFIG,
        )

    summary_path = error.value.summary_path
    assert summary_path is not None and summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "timed_out"
    assert "max_runtime_seconds" in summary["reason"]
    assert summary["completed_steps"] == 0
