"""Validated configuration blueprints for GPT-2."""

from dataclasses import dataclass
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


# Test-only configuration that keeps future unit tests fast on a small CPU workload.
GPT2_DEBUG_CONFIG = GPT2Config(
    vocab_size=128,
    context_length=16,
    emb_dim=32,
    n_heads=4,
    n_layers=2,
    drop_rate=0.0,
)
