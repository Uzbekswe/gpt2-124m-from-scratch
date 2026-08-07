"""Local-only environment checks for future GPT-2 VESSL runs."""

import json
import platform
from dataclasses import asdict, dataclass

import torch

import gpt2_124m
from gpt2_124m.config import GPT2Config

GPT2_SMALL_TRAINABLE_PARAMETER_COUNT = 124_439_808
"""Exact original GPT-2 Small trainable count with tied input/output embeddings."""


@dataclass(frozen=True, slots=True)
class LocalPreflightReport:
    """Serializable facts about the local environment, with no cloud side effects."""

    python_version: str
    pytorch_version: str
    cuda_available: bool
    selected_device: str
    package_import_status: dict[str, bool]
    gpt2_small_trainable_parameter_count: int


def collect_local_preflight_report() -> LocalPreflightReport:
    """Collect local runtime facts without contacting VESSL or downloading anything."""
    config = GPT2Config()
    parameter_count = _count_trainable_parameters(config)
    if parameter_count != GPT2_SMALL_TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError(
            "the GPT-2 Small parameter-count formula no longer matches the model spec."
        )

    cuda_available = torch.cuda.is_available()
    return LocalPreflightReport(
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_available=cuda_available,
        selected_device="cuda" if cuda_available else "cpu",
        package_import_status={"gpt2_124m": bool(gpt2_124m.__version__), "torch": True},
        gpt2_small_trainable_parameter_count=parameter_count,
    )


def _count_trainable_parameters(config: GPT2Config) -> int:
    """Calculate the exact tied-weight GPT-2 parameter count without allocating the 124M model."""
    embedding_parameters = config.vocab_size * config.emb_dim
    positional_embedding_parameters = config.context_length * config.emb_dim
    layer_norm_parameters = 2 * config.emb_dim
    attention_parameters = (
        config.emb_dim * (3 * config.emb_dim)
        + (3 * config.emb_dim)
        + config.emb_dim * config.emb_dim
        + config.emb_dim
    )
    mlp_parameters = (
        config.emb_dim * (4 * config.emb_dim)
        + (4 * config.emb_dim)
        + (4 * config.emb_dim) * config.emb_dim
        + config.emb_dim
    )
    block_parameters = 2 * layer_norm_parameters + attention_parameters + mlp_parameters
    final_layer_norm_parameters = layer_norm_parameters
    return (
        embedding_parameters
        + positional_embedding_parameters
        + config.n_layers * block_parameters
        + final_layer_norm_parameters
    )


def main() -> None:
    """Print the local-only preflight report as JSON for a terminal user or future script."""
    print(json.dumps(asdict(collect_local_preflight_report()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
