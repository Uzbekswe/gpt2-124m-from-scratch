# Exact GPT-2 124M Reimplementation

## Summary

Build one model only: the original GPT-2 Small/124M architecture.

The repository will demonstrate two separate achievements:

1. **Compatibility mode:** load official GPT-2 weights and numerically match reference outputs.
2. **From-scratch mode:** initialize the same 124M architecture with random weights and run a genuine pretraining experiment on VESSL.

There will be no TinyGPT, architectural modifications, or claims of inventing a new model.

## Architecture and Repository

Implement the following using PyTorch:

- GPT-2 BPE tokenizer with 50,257 tokens.
- Token and learned absolute positional embeddings.
- 1,024-token context window.
- 12 transformer blocks.
- 768-dimensional embeddings.
- 12 causal-attention heads.
- LayerNorm, GELU, feed-forward network, dropout, and residual connections.
- Query, key, value, and output projections with GPT-2-compatible biases.
- Final LayerNorm and language-model output head.
- Weight tying between token embeddings and output head.
- Exactly **124,439,808 trainable parameters**.

Use:

```text
src/gpt2_124m/
├── config.py
├── tokenizer.py
├── data.py
├── attention.py
├── layers.py
├── model.py
├── generation.py
├── training.py
├── checkpoint.py
└── gpt2_weights.py
```

Notebooks will teach and demonstrate the package; they will not contain a second copy of the implementation.

## Development Phases

### 1. Model implementation

Rebuild Chapters 2-4 as clean modules:

- Tokenization and input-target construction.
- Causal self-attention.
- Multi-head attention.
- Transformer blocks.
- Complete 12-block GPT-2 model.
- Parameter counting and tensor-shape inspection.

Add a tiny debug configuration strictly for unit tests, but all published model results use the 124M configuration.

### 2. Training and generation

Rebuild Chapter 5:

- Cross-entropy loss and validation loss.
- AdamW optimizer and learning-rate scheduling.
- Gradient accumulation and clipping.
- Mixed-precision training.
- Greedy, temperature, and top-k generation.
- Checkpoint save, resume, and inference.
- Deterministic seeds and reproducible configurations.

Public commands:

```bash
python -m gpt2_124m.train --config configs/pretrain.yaml
python -m gpt2_124m.generate --checkpoint CHECKPOINT --prompt "Once upon a time"
python -m gpt2_124m.import_weights --source openai-community/gpt2
python -m gpt2_124m.verify_compatibility
```

### 3. Official GPT-2 compatibility

- Download official GPT-2 Small weights through a dedicated importer.
- Convert/transplant the weights into the custom PyTorch classes.
- Handle GPT-2's transposed projection matrices correctly.
- Compare parameters and logits against a reference implementation.
- Verify generated tokens for fixed prompts and random seeds.
- Keep Hugging Face Transformers as an optional verification dependency—not as the implementation.

The README will describe this as an independent reimplementation of GPT-2, not a new model. GPT-2 Small is documented as the 124M member of the original family: <https://huggingface.co/openai-community/gpt2>.

### 4. VESSL pretraining experiment

Train the exact same 124M architecture from random initialization:

- Dataset: streamed `HuggingFaceFW/fineweb-edu`, `sample-10BT`.
- Tokenizer: GPT-2 BPE.
- Sequence length: 1,024 tokens.
- Documents separated with `<|endoftext|>` and packed into complete sequences.
- Validation: deterministic document-ID hash split, with 0.5% reserved.
- Target: up to 1 billion training tokens or 40 GPU-hours, whichever comes first.
- Hardware target: one VESSL GPU with at least 24 GB VRAM.
- Precision: BF16 when supported, otherwise FP16.
- Effective batch: 262,144 tokens using gradient accumulation.
- Optimizer: AdamW with weight decay, warmup, cosine decay, and gradient clipping.
- Save resumable checkpoints and samples at regular intervals.

FineWeb-Edu provides GPT-2-tokenized sample subsets and uses the ODC-By license: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>.

This run proves that the complete pipeline works. It will not be described as reproducing OpenAI's original training because WebText, token count, and compute are different.

## Portfolio Deliverables

- Clean, tested Python package.
- Educational notebooks covering data, attention, architecture, training, and inference.
- Original GPT-2 architecture diagram.
- VESSL run configuration and reproducibility guide.
- Training and validation loss curves.
- Perplexity, throughput, memory, and tokens-processed charts.
- Generated samples from random initialization, intermediate checkpoints, final checkpoint, and official GPT-2.
- Side-by-side official-weight compatibility report.
- Minimal interactive generation demo.
- Model card documenting dataset, compute, intended use, limitations, and licensing.
- Clear attribution to Sebastian Raschka, OpenAI GPT-2, and FineWeb-Edu.
- No copied charts, results, or book visuals presented as original work.

## Tests and Acceptance Criteria

- Causal masking prevents future-token leakage.
- Input targets are shifted exactly one token.
- Every module passes shape and gradient tests.
- Weight tying shares the same underlying parameter.
- A debug configuration can overfit a tiny batch.
- Save/resume matches uninterrupted training.
- Seeded generation is deterministic.
- The production configuration reports exactly 124,439,808 parameters.
- Imported official weights produce reference-matching logits within float32 tolerance.
- CPU tests pass in GitHub Actions.
- VESSL smoke training, checkpoint resume, validation, and generation all complete successfully.

## Fixed Assumptions

- Repository name: `gpt2-124m-from-scratch`.
- Implementation is written fresh, with upstream code used only for study and verification.
- Official weights demonstrate compatibility.
- Randomly initialized weights demonstrate personal training experience.
- The separate smarter architecture you plan later remains completely outside this repository.
