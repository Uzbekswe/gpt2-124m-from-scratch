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

## Milestone 15b outcome and one planned clean-exit verification job

The approved Milestone 15b A100 job completed all three optimizer updates and exported every
expected artifact. At interpreter shutdown it then raised
`Fatal Python error: PyGILState_Release`; the Cloud job was terminated to stop billing. The log
shows `datasets==5.0.1`, `pyarrow==25.0.0`, `hf-xet==1.6.0`, `aiohttp==3.14.3`, and related native
extensions, but has no native backtrace that can identify one responsible binary. Do not claim a
specific library as the root cause.

The likely failure boundary is the bounded Hugging Face Parquet stream being finalized alongside
Arrow/network worker resources. The project now closes every early-stopped stream iterator before
interpreter teardown, removes the tiny run's unnecessary DataLoader wrapper, and constrains the
direct Cloud streaming stack with `configs/requirements/tiny-pretrain-cloud.txt`.

The following is a **not submitted** single-job verification plan. It must be reviewed and
explicitly approved again before use. It mounts a filtered source volume at
`/workspace/gpt2-124m` and a separate output volume at `/output`, installs the pinned optional
training dependencies, and keeps the fixed three-step workload:

```sh
set -eu
python -m pip install --timeout 30 --retries 1 --no-build-isolation \
  -c configs/requirements/tiny-pretrain-cloud.txt -e ".[train]"
python scripts/tiny_pretrain.py \
  --config configs/tiny_pretrain.json \
  --output-dir /output/tiny-pretrain-15c-clean-exit \
  --max-runtime-seconds 180 \
  --force-clean-exit
```

The expected persistent artifacts are `/output/tiny-pretrain-15c-clean-exit/training_config.json`,
`/output/tiny-pretrain-15c-clean-exit/metrics.jsonl`,
`/output/tiny-pretrain-15c-clean-exit/checkpoint_final.pt`,
`/output/tiny-pretrain-15c-clean-exit/generated_sample.txt`, and
`/output/tiny-pretrain-15c-clean-exit/training_summary.json`. This plan is intentionally not a
submitted job and must be reviewed again for its current resource price, maximum runtime, and
data limits before launch.

Using your own approved volumes and the pinned A100 image, the **not submitted** verification
command template is:

```sh
vesslctl job create \
  --name gpt2-124m-tiny-pretrain-15c-clean-exit \
  --resource-spec REPLACE_WITH_RESOURCE_SPEC \
  --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385 \
  --object-volume REPLACE_WITH_SOURCE_VOLUME:/workspace/gpt2-124m \
  --object-volume REPLACE_WITH_OUTPUT_VOLUME:/output \
  --working-dir /workspace/gpt2-124m \
  --tag gpt2-124m \
  --tag tiny-pretrain \
  --tag milestone-15c \
  --cmd 'set -eu; python -m pip install --timeout 30 --retries 1 --no-build-isolation -c configs/requirements/tiny-pretrain-cloud.txt -e ".[train]"; python scripts/tiny_pretrain.py --config configs/tiny_pretrain.json --output-dir /output/tiny-pretrain-15c-clean-exit --max-runtime-seconds 180 --force-clean-exit'
```

Before any approval of that command, refresh your source volume with the reviewed 15c commit.
Do not submit a retry if this verification job fails.

The Python deadline is cooperative: it stops at safe checkpoints before or after model work and
writes a `timed_out` summary. A blocked Hugging Face network operation cannot be preempted by
Python safely, but its failure is reported with the original exception as context. The
`--force-clean-exit` option is only for this single-purpose Cloud batch process. It is reached
only after success, all artifact writes have closed, and explicit stream cleanup has completed;
it bypasses remaining Python finalizers that caused the 15b fault.
