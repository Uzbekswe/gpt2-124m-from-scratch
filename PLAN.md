# Portfolio Scope and Project Status

This repository is an independent educational reimplementation of GPT-2 Small (124M),
with a deliberately honest boundary between architectural correctness, systems validation,
and model quality.

## Portfolio thesis

The project demonstrates two complementary skills:

1. rebuilding a known decoder-only Transformer in clean PyTorch modules; and
2. connecting that implementation to data streaming, training, checkpointing, validation,
   generation, and a bounded GPU execution path.

The portfolio claim is not that this project reproduced GPT-2's training or produced a useful
language model. The three-update FineWeb-Edu run is an end-to-end systems proof.

## Completed scope

### Architecture and local package

- GPT-2 BPE tokenization with the 50,257-token vocabulary.
- Learned token and positional embeddings with a 1,024-token context window.
- Twelve pre-LayerNorm Transformer blocks, twelve causal attention heads, 768-dimensional
  embeddings, GPT-2 GELU, residual paths, and tied input/output weights.
- Official GPT-2 tensor-name mapping, including Conv1D-to-Linear transposes.
- Exact production parameter-count assertion: **124,439,808 trainable parameters**.
- Cross-entropy loss, AdamW grouping, gradient clipping, local training, evaluation,
  checkpoint save/load, RNG restoration, and greedy/temperature/top-k generation.

### Data and execution evidence

- Lazy FineWeb-Edu streaming with pinned dataset metadata, deterministic document-ID hashing,
  `<|endoftext|>` boundaries, and continuous sequence packing.
- A CUDA forward/backward smoke script and a real three-update FineWeb-Edu training proof on
  an A100, with validation, checkpointing, generation, and artifact summaries.
- Offline unit and integration coverage for tensor shapes, causality, gradients, data behavior,
  weight tying, checkpoint restoration, and synthetic official-weight mapping.
- Local CPU demo, MIT license, third-party attribution, and GitHub Actions coverage for Python
  3.10 and 3.11.

## Explicitly not implemented

These items remain outside the current portfolio claim:

- A long-running production pretraining study targeting one billion tokens or 40 GPU-hours.
- Gradient accumulation to a 262,144-token effective batch.
- BF16/FP16 mixed-precision execution, warmup, cosine learning-rate decay, and throughput
  instrumentation for a serious training run.
- Durable FineWeb stream-position resume. Current checkpoints resume model, optimizer, history,
  configuration, and RNG state, but not an arbitrary remote data cursor.
- Public `gpt2_124m.train`, `generate`, `import_weights`, and `verify_compatibility` modules.
  The current supported entry points are the scripts documented in `README.md`.
- A released trained checkpoint or claims about GPT-2-quality language generation.
- Educational notebooks, loss/perplexity/throughput charts, a model card for released weights,
  and an interactive web generation demo.
- A side-by-side compatibility report comparing every parameter tensor and seeded sampling.

## Evidence boundary

The official-weight path verifies architectural compatibility by comparing mapped tensors,
float32 logits, and deterministic greedy generation. The optional online check downloads the
reference checkpoint; the normal test suite uses synthetic tensors and does not require it.

The VESSL experiment verifies orchestration and clean artifact production. Its three updates are
not enough to infer convergence, generalization, perplexity, throughput, or useful language
ability.

## Next expansion, only with a new budget decision

If this project is later extended beyond portfolio cleanup, the next milestone should define a
token and cost budget first, then add mixed precision, scheduled learning rates, effective-batch
training, regular validation, throughput/memory measurement, durable data resume, checkpoint
retention, and a model card before publishing any weights.

## Attribution boundary

The implementation is written independently and is informed by Sebastian Raschka's *Build a
Large Language Model (From Scratch)*. GPT-2, PyTorch, tiktoken, Transformers, and FineWeb-Edu
retain their own licenses and attribution requirements; see `docs/ATTRIBUTION.md`.
