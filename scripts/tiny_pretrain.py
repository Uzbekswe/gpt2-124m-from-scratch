#!/usr/bin/env python3
"""Run the intentionally tiny, streamed FineWeb-Edu pretraining proof."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpt2_124m.config import TinyPretrainingConfig, TrainingConfig
from gpt2_124m.tiny_pretraining import TinyTrainingDeadlineExceeded, run_fineweb_tiny_pretraining


def _load_config(path: Path | None) -> dict[str, Any]:
    """Load a JSON configuration, starting from the validated tiny-run defaults."""
    values: dict[str, Any] = asdict(TinyPretrainingConfig())
    if path is None:
        return values
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read config file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"config file is not valid JSON: {path}") from error
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a JSON object.")
    unknown_fields = sorted(set(loaded).difference(values))
    if unknown_fields:
        raise ValueError(f"config file has unknown fields: {unknown_fields}.")
    for key, value in loaded.items():
        if key == "optimizer":
            if not isinstance(value, dict):
                raise ValueError("optimizer must be a JSON object.")
            values["optimizer"].update(value)
        else:
            values[key] = value
    return values


def _build_parser() -> argparse.ArgumentParser:
    """Expose all cost, data, optimization, and artifact controls as CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional JSON configuration file")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-runtime-seconds", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--train-max-documents", type=int)
    parser.add_argument("--validation-max-documents", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-configuration")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--text-field")
    parser.add_argument("--document-id-field")
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--do-sample", action="store_true", default=None)
    sampling_group.add_argument("--greedy", action="store_false", dest="do_sample")
    return parser


def parse_config(argv: list[str] | None = None) -> TinyPretrainingConfig:
    """Merge the optional JSON config with explicit CLI overrides and validate the result."""
    args = _build_parser().parse_args(argv)
    values = _load_config(args.config)
    optimizer_values = values.pop("optimizer")
    assert isinstance(optimizer_values, dict)
    optimizer_overrides = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "grad_clip_norm": args.grad_clip_norm,
    }
    optimizer_values.update(
        {key: value for key, value in optimizer_overrides.items() if value is not None}
    )
    for field_name, value in vars(args).items():
        if field_name in {"config", *optimizer_overrides} or value is None:
            continue
        values[field_name] = value
    values["optimizer"] = TrainingConfig(**optimizer_values)
    return TinyPretrainingConfig(**values)


def main() -> None:
    """Run the tiny pretraining proof and print the final JSON summary path."""
    config = parse_config()
    try:
        result = run_fineweb_tiny_pretraining(config)
    except TinyTrainingDeadlineExceeded as error:
        print(error.summary_path)
        raise SystemExit(2) from error
    print(result.summary_path)


if __name__ == "__main__":
    main()
