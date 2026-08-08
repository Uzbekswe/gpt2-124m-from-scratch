# Exact GPT-2 Small (124M) Reimplementation

This project is an educational, from-scratch reimplementation of the original GPT-2
Small architecture. It implements the exact 124,439,808-parameter model and supports
both compatibility verification with official GPT-2 weights and later training the
identical architecture from random initialization.

## Local setup

The project supports Python 3.10 and 3.11. Local development uses Python 3.11. The
controlled VESSL GPU smoke run uses VESSL's available Python 3.10 CUDA image.

```bash
cd /Users/uzbekswe/gpt2-124m-from-scratch
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the foundation checks:

```bash
python -c "import gpt2_124m; print(gpt2_124m.__version__)"
python -m pytest
python -m ruff check .
```

The package source lives in `src/gpt2_124m`; tests live in `tests`; future notebooks
and configuration files will live in `notebooks` and `configs`.

## Tiny Training Run

The tiny training run is an end-to-end proof, not a useful pretrained model. It keeps the
exact 124,439,808-parameter GPT-2 Small architecture, streams a bounded amount of
FineWeb-Edu, makes a few AdamW updates, evaluates, saves a checkpoint, and generates one
short sample.

Install only the optional streaming dependencies, then run it locally or in an approved
GPU job:

```bash
python -m pip install -e ".[train]"
python scripts/tiny_pretrain.py \
  --config configs/tiny_pretrain.json \
  --output-dir artifacts/tiny-pretrain
```

The JSON configuration and CLI flags control the data source, document caps, sequence and
batch sizes, optimizer settings, logging/evaluation cadence, random seed, sampling, device,
output path, and a cooperative `max_runtime_seconds` deadline (180 seconds by default). The run
writes `training_config.json`, `metrics.jsonl`, `checkpoint_final.pt`,
`generated_sample.txt`, and `training_summary.json`. The summary records `completed`, `failed`,
or `timed_out`, with its reason and completed optimizer steps. FineWeb-Edu is streamed only when
the command starts; no dataset files are committed to Git.

## Official GPT-2 compatibility mode

Compatibility mode imports weights from `openai-community/gpt2` into this project's
independent implementation, then compares raw logits and deterministic greedy token IDs.
Self-trained mode uses the same architecture but starts with this project's own random
initialization; it does not use official weights.

Transformers is an optional verification-only dependency. It is not used to implement,
train, or generate with this package:

```bash
python -m pip install -e ".[dev,verify]"
```

Run the opt-in online compatibility check:

```bash
RUN_OFFICIAL_GPT2_INTEGRATION=1 python -m pytest tests/test_official_gpt2_integration.py -q
```

The command downloads official weights into Hugging Face's local cache. Downloaded
weights, caches, and checkpoints are ignored by Git and must never be committed. A
matching result demonstrates architectural implementation fidelity; it does not prove
that this project independently reproduced GPT-2's original training run.
