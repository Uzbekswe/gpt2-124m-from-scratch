# GPT-2 Small Portfolio Demo (5–7 Minutes)

## 0:00–0:45 — State the project clearly

Open [README.md](../README.md). Use this introduction:

> I reimplemented the original GPT-2 Small architecture in fresh PyTorch code. The model
> has the exact 12-block, 12-head, 768-dimensional structure and 124,439,808 tied-weight
> parameters. I tested the components locally and validated the complete training path on
> an A100, but I do not claim that a three-step run produced a useful pretrained model.

Point to “What this project proves” to distinguish implementation, local tests, Cloud
execution, and model quality.

## 0:45–2:15 — Walk through the architecture

Show the README Mermaid diagram, then open these files:

1. [`config.py`](../src/gpt2_124m/config.py): exact production defaults and a separate
   debug configuration used to keep unit tests fast.
2. [`attention.py`](../src/gpt2_124m/attention.py): one fused `c_attn` projection, causal
   mask, 12-way head reshape, and `c_proj`, matching GPT-2 weight-loading terminology.
3. [`layers.py`](../src/gpt2_124m/layers.py): custom LayerNorm, tanh GELU, 4x MLP, and the
   two pre-norm residual paths.
4. [`model.py`](../src/gpt2_124m/model.py): learned embeddings, 12 distinct blocks, final
   LayerNorm, raw vocabulary logits, and exact token/output weight tying.

Explain that causality is tested by changing future tokens and confirming earlier outputs
do not change. Weight tying reduces the parameter count because the vocabulary output head
reuses the token-embedding parameter instead of allocating a second matrix.

## 2:15–3:15 — Show verification, not just code

Open [`test_model.py`](../tests/test_model.py) and point to the exact parameter-count,
weight-tying, causal-isolation, and gradient tests. Then show the broader suite:

```bash
python -m pytest
python -m ruff check .
```

If demo time is constrained, run focused checks instead:

```bash
python -m pytest tests/test_model.py tests/test_gpt2_weights.py -q
```

Explain that most tests use `GPT2_DEBUG_CONFIG` for speed, while the production-count test
instantiates the exact model.

## 3:15–4:15 — Explain official GPT-2 compatibility

Open [`gpt2_weights.py`](../src/gpt2_124m/gpt2_weights.py). Highlight:

- every required official tensor is mapped and shape-checked before copying;
- GPT-2 Conv1D matrices are transposed into PyTorch `nn.Linear` layout;
- token/output weight tying is preserved; and
- the optional online check compares float32 logits and greedy token IDs with the reference.

Offline tests use synthetic GPT-2-layout tensors, so normal verification needs no checkpoint
download. The real reference comparison is opt-in:

```bash
RUN_OFFICIAL_GPT2_INTEGRATION=1 \
  python -m pytest tests/test_official_gpt2_integration.py -q
```

Be explicit that this command downloads official weights and verifies compatibility; it
does not make those weights an independently trained result.

## 4:15–5:30 — Present the A100 evidence

Open the [Milestone 15c experiment report](experiments/15c-vessl-a100-tiny-pretraining.md).
Explain the two Cloud validation levels:

- smoke job `job-b7sx700lxrx3`: exact parameter count, finite loss and gradients, artifact;
- clean tiny-training job `job-nri2sb9n1w1i`: three optimizer steps, validation,
  checkpoint, sample, summary, and a succeeded Cloud state on an A100-SXM4-80GB.

The tiny job moved train loss from 10.9330 to 9.8814 and recorded validation loss 9.9406.
Say immediately that three points are operational evidence, not a convergence curve or a
quality benchmark.

Mention the engineering lesson: the first attempt finished its artifacts but exposed a
native Python finalization fault. The clean run used deterministic iterator cleanup,
pinned streaming dependencies, bounded setup/runtime controls, and a success-only batch
exit after artifacts closed.

## 5:30–6:15 — End with candid limitations

- The random model received only three updates; its generated sample is essentially random.
- The work does not reproduce WebText, OpenAI's token/compute budget, or GPT-2 quality.
- The Cloud checkpoint remains private and is not a downloadable portfolio model.
- A serious pretraining run needs a defined token budget, scheduler/mixed precision,
  throughput measurements, durable data-position resume, longer validation, and a model card.

## Likely interviewer questions

**Why exactly 124,439,808 parameters?**

It is the sum of the specified embeddings, 12 independently parameterized attention/MLP
blocks, and normalization parameters, with the vocabulary output matrix tied to the token
embedding matrix. The test suite asserts the exact total.

**Why implement custom LayerNorm and GELU?**

They make the GPT-2 equations and compatibility choices explicit. Tests compare them
against PyTorch's compatible operations within float32 tolerance.

**How do you know attention is causal?**

The mask sets future scores to negative infinity before softmax, and tests verify both zero
future attention weights and unchanged earlier outputs when later tokens change.

**Why fuse query, key, and value?**

One `c_attn` projection is efficient and matches the official GPT-2 tensor layout, making
strict weight import simpler than coordinating separate per-head modules.

**Did you train GPT-2?**

I trained the exact GPT-2 Small architecture for three bounded updates to validate the full
system. I did not train it to GPT-2 quality, and I describe the result as a systems proof.

**What made the checkpoint resumable?**

It stores model and AdamW states, completed step, history, configurations, and PyTorch CPU
and CUDA random-number states. Exact arbitrary streaming-data position is a separate
large-scale data-pipeline concern.

**What would you do next with more budget?**

Define a token and cost budget first, add mixed precision and a scheduled learning rate,
measure throughput, implement durable stream-position resume, validate regularly, and
publish only well-documented checkpoints and original evaluation charts.
