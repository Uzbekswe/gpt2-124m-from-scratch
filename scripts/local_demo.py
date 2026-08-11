#!/usr/bin/env python3
"""Run a fast, deterministic CPU demonstration of training and generation."""

import argparse
import json

import torch

from gpt2_124m.config import GPT2_DEBUG_CONFIG, TrainingConfig
from gpt2_124m.generation import generate
from gpt2_124m.model import GPT2Model
from gpt2_124m.training import configure_optimizer, train_step


def run_demo(*, seed: int, steps: int) -> dict[str, object]:
    """Train the test-sized model on synthetic token IDs, then greedily generate IDs."""
    torch.manual_seed(seed)
    model = GPT2Model(GPT2_DEBUG_CONFIG)
    optimizer = configure_optimizer(model, TrainingConfig())
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    target_ids = input_ids + 1
    losses: list[float] = []

    for _ in range(steps):
        metrics = train_step(
            model,
            (input_ids, target_ids),
            optimizer,
            device="cpu",
            grad_clip_norm=1.0,
        )
        losses.append(metrics.loss)

    prompt_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated_ids = generate(model, prompt_ids, max_new_tokens=6)
    return {
        "configuration": "GPT2_DEBUG_CONFIG",
        "purpose": "synthetic CPU smoke demo, not a language-quality result",
        "seed": seed,
        "steps": steps,
        "parameter_count": model.count_trainable_parameters(),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "prompt_ids": prompt_ids[0].tolist(),
        "generated_ids": generated_ids[0].tolist(),
    }


def main() -> None:
    """Parse demo controls and print machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    print(json.dumps(run_demo(seed=args.seed, steps=args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
