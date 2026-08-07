# Exact GPT-2 Small (124M) Reimplementation

This project is an educational, from-scratch reimplementation of the original GPT-2
Small architecture. It implements the exact 124,439,808-parameter model and supports
both compatibility verification with official GPT-2 weights and later training the
identical architecture from random initialization.

## Local setup

This project targets Python 3.11.

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
