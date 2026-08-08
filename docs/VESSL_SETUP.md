# VESSL Cloud setup and future tiny-training plan

This repository uses modern VESSL Cloud and the `vesslctl` CLI. It does not use the legacy
`vessl` CLI, legacy projects/experiments/runs, or run YAML files.

## Local development and Cloud images

The project supports Python 3.10 and 3.11. Develop locally with Python 3.11. Before any
Cloud job, inspect the selected image's Python and PyTorch versions and run the local test
suite.

## Modern Cloud workflow

1. Authenticate outside the repository with `vesslctl auth login`.
2. Confirm scope with `vesslctl auth status`, `vesslctl org list`, `vesslctl team list`, and
   `vesslctl config show`.
3. Inspect available resources with `vesslctl billing show`, `vesslctl cluster list`, and
   `vesslctl resource-spec list`.
4. Create dedicated object volumes only after approval. Upload a filtered local source volume
   with `vesslctl volume upload`, then mount it into a job with `--object-volume`.
5. Submit a job only after reviewing its exact `vesslctl job create` command and receiving
   explicit approval.

Credentials remain in the VESSL CLI configuration outside this repository. Never place cloud
tokens, GitHub tokens, or other credentials in source files, JSON configuration, volumes, or
documentation examples.

## Future tiny-training job plan — not submitted

Milestone 15a supplies `scripts/tiny_pretrain.py` and `configs/tiny_pretrain.json`. A later,
explicitly approved Milestone 15b job will mount a filtered source volume at
`/workspace/gpt2-124m` and a separate output volume at `/output`. It will install the optional
`train` dependencies, then run a fixed small step budget:

```sh
set -eu
python -m pip install --timeout 30 --retries 1 --no-build-isolation -e ".[train]"
python scripts/tiny_pretrain.py \
  --config configs/tiny_pretrain.json \
  --output-dir /output/tiny-pretrain-15b \
  --max-runtime-seconds 180
```

The expected persistent artifacts are `/output/tiny-pretrain-15b/training_config.json`,
`/output/tiny-pretrain-15b/metrics.jsonl`, `/output/tiny-pretrain-15b/checkpoint_final.pt`,
`/output/tiny-pretrain-15b/generated_sample.txt`, and
`/output/tiny-pretrain-15b/training_summary.json`. This plan is intentionally not a submitted
job and must be reviewed again for its current resource price, maximum runtime, and data limits
before launch.

The Python deadline is cooperative: it stops at safe checkpoints before or after model work and
writes a `timed_out` summary. A blocked Hugging Face network operation cannot be preempted by
Python safely, but its failure is reported with the original exception as context.
