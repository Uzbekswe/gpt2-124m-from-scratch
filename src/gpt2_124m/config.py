"""Validated configuration blueprints for GPT-2."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real


@dataclass(frozen=True, slots=True)
class GPT2Config:
    """Immutable hyperparameters for the original GPT-2 Small architecture."""

    vocab_size: int = 50_257
    context_length: int = 1_024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = True
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        """Reject values that cannot define a valid transformer configuration."""
        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
        }
        for name, value in positive_int_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")

        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                "emb_dim must be divisible by n_heads so every attention head has the same size."
            )

        if (
            isinstance(self.drop_rate, bool)
            or not isinstance(self.drop_rate, Real)
            or not 0.0 <= self.drop_rate < 1.0
        ):
            raise ValueError("drop_rate must be a number in the range [0.0, 1.0).")

        if (
            isinstance(self.layer_norm_epsilon, bool)
            or not isinstance(self.layer_norm_epsilon, Real)
            or self.layer_norm_epsilon <= 0.0
        ):
            raise ValueError("layer_norm_epsilon must be a positive number.")

        if not isinstance(self.qkv_bias, bool):
            raise ValueError("qkv_bias must be a boolean.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Validated optimization settings for local GPT-2 training steps."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Reject optimizer settings outside AdamW's supported numeric ranges."""
        numeric_fields = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "grad_clip_norm": self.grad_clip_norm,
        }
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number; got {value!r}.")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in the range [0.0, 1.0).")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in the range [0.0, 1.0).")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive.")


@dataclass(frozen=True, slots=True)
class LocalLoopConfig:
    """Validated controls for a simple local training-and-evaluation loop."""

    max_steps: int
    eval_every: int
    eval_batches: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid update counts and evaluation cadence values."""
        for name, value in {
            "max_steps": self.max_steps,
            "eval_every": self.eval_every,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")

        if self.eval_batches is not None and (
            isinstance(self.eval_batches, bool)
            or not isinstance(self.eval_batches, int)
            or self.eval_batches <= 0
        ):
            raise ValueError("eval_batches must be a positive integer or None.")


@dataclass(frozen=True, slots=True)
class TinyPretrainingConfig:
    """Validated, deliberately small controls for one end-to-end pretraining proof."""

    output_dir: str = "artifacts/tiny-pretrain"
    seed: int = 1_337
    max_runtime_seconds: int | None = 180
    max_steps: int = 3
    batch_size: int = 1
    sequence_length: int = 128
    eval_every: int = 2
    eval_batches: int = 1
    log_every: int = 1
    train_max_documents: int | None = 128
    validation_max_documents: int | None = 4
    validation_fraction: float = 0.005
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_configuration: str = "sample-10BT"
    dataset_revision: str = "v1.0.0"
    text_field: str = "text"
    document_id_field: str = "id"
    prompt: str = "Once upon a time"
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_k: int | None = 40
    do_sample: bool = True
    device: str = "auto"
    optimizer: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        """Reject unsafe tiny-run values before a model or remote stream is created."""
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty string.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        for name, value in {
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "eval_every": self.eval_every,
            "eval_batches": self.eval_batches,
            "log_every": self.log_every,
            "max_new_tokens": self.max_new_tokens,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if self.max_runtime_seconds is not None and (
            isinstance(self.max_runtime_seconds, bool)
            or not isinstance(self.max_runtime_seconds, int)
            or self.max_runtime_seconds <= 0
        ):
            raise ValueError("max_runtime_seconds must be a positive integer or None.")
        for name, value in {
            "train_max_documents": self.train_max_documents,
            "validation_max_documents": self.validation_max_documents,
        }.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None.")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, Real)
            or not isfinite(self.validation_fraction)
            or not 0.0 <= self.validation_fraction < 1.0
        ):
            raise ValueError("validation_fraction must be a finite number in [0.0, 1.0).")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not isfinite(self.temperature)
            or self.temperature <= 0.0
        ):
            raise ValueError("temperature must be a positive finite number.")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None.")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be a boolean.")
        for name, value in {
            "dataset_name": self.dataset_name,
            "dataset_configuration": self.dataset_configuration,
            "dataset_revision": self.dataset_revision,
            "text_field": self.text_field,
            "document_id_field": self.document_id_field,
            "prompt": self.prompt,
            "device": self.device,
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.optimizer, TrainingConfig):
            raise TypeError("optimizer must be a TrainingConfig.")


# Test-only configuration that keeps future unit tests fast on a small CPU workload.
GPT2_DEBUG_CONFIG = GPT2Config(
    vocab_size=128,
    context_length=16,
    emb_dim=32,
    n_heads=4,
    n_layers=2,
    drop_rate=0.0,
)
