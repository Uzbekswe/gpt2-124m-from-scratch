#!/usr/bin/env python3
"""Run one GPU-only forward/backward smoke check for the exact GPT-2 Small model."""

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gpt2_124m.config import GPT2Config
from gpt2_124m.model import GPT2Model
from gpt2_124m.training import compute_language_model_loss

EXPECTED_PARAMETER_COUNT = 124_439_808


@dataclass(frozen=True, slots=True)
class SmokeReport:
    """Serializable evidence produced by the one-pass GPU smoke check."""

    python_version: str
    pytorch_version: str
    cuda_version: str | None
    gpu_name: str
    gpu_memory_bytes: int
    selected_device: str
    parameter_count: int
    smoke_loss: float
    gpu_memory_allocated_bytes: int
    gradients_finite: bool
    success: bool


def run_smoke_test() -> SmokeReport:
    """Run exactly one model forward/backward pass without an optimizer or weight update."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this VESSL GPU smoke test.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    device_properties = torch.cuda.get_device_properties(device)
    model = GPT2Model(GPT2Config()).to(device)
    parameter_count = model.count_trainable_parameters()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_PARAMETER_COUNT:,} trainable parameters, got {parameter_count:,}."
        )

    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, 8),
        device=device,
        dtype=torch.long,
    )
    target_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, 8),
        device=device,
        dtype=torch.long,
    )
    logits = model(input_ids)
    loss = compute_language_model_loss(logits, target_ids)
    if not torch.isfinite(loss):
        raise FloatingPointError("smoke loss is non-finite.")
    loss.backward()

    gradients_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not gradients_finite:
        raise FloatingPointError("one or more smoke-test gradients are missing or non-finite.")

    return SmokeReport(
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=device_properties.name,
        gpu_memory_bytes=device_properties.total_memory,
        selected_device=str(device),
        parameter_count=parameter_count,
        smoke_loss=loss.item(),
        gpu_memory_allocated_bytes=torch.cuda.memory_allocated(device),
        gradients_finite=gradients_finite,
        success=True,
    )


def log_to_vessl_if_available(report: SmokeReport) -> bool:
    """Log required numeric smoke metrics when the optional VESSL SDK exists in the image."""
    try:
        import vessl
    except ImportError:
        return False

    vessl.log(
        step=0,
        payload={
            "parameter_count": report.parameter_count,
            "smoke_loss": report.smoke_loss,
            "gpu_memory_allocated": report.gpu_memory_allocated_bytes,
            "success": 1,
        },
    )
    return True


def write_report(report: SmokeReport, output_dir: Path) -> Path:
    """Write the completed JSON smoke report to the VESSL-exported output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    return report_path


def main() -> None:
    """Parse the local output path, run the smoke check, log metrics, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/output"),
        help="directory that receives smoke_report.json (default: /output)",
    )
    args = parser.parse_args()

    report = run_smoke_test()
    vessl_logged = log_to_vessl_if_available(report)
    report_path = write_report(report, args.output_dir)
    print(json.dumps({**asdict(report), "vessl_logged": vessl_logged}, indent=2, sort_keys=True))
    print(f"smoke report written to {report_path}")


if __name__ == "__main__":
    main()
