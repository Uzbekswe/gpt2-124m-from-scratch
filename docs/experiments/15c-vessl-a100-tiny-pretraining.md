# Milestone 15c: VESSL A100 Tiny-Pretraining Verification

## Purpose and success criteria

This experiment tested whether the repository's real pretraining path could complete on a
Cloud GPU and exit cleanly. It was deliberately limited to three optimizer updates. Success
meant that the exact GPT-2 Small model could stream and pack FineWeb-Edu text, train with
finite losses, evaluate a deterministic validation split, save a resumable checkpoint,
generate a sample, persist all expected artifacts, and leave VESSL in a succeeded state.

This is an end-to-end systems validation. It is not meaningful language-model pretraining
and does not demonstrate GPT-2-quality text.

## Architecture and environment

| Item | Verified setting |
| --- | --- |
| Model | Fresh PyTorch GPT-2 Small implementation |
| Parameters | **124,439,808 trainable** |
| Transformer shape | 12 blocks, 12 heads, 768 embedding dimensions |
| Attention head size | 64 dimensions |
| Context/vocabulary | 1,024 positions / 50,257 tokens |
| Output weights | Tied to token embeddings |
| Cloud scope | Private VESSL Cloud scope |
| Job | Private job identifier omitted |
| GPU | NVIDIA A100-SXM4-80GB |
| Container | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385` |
| Source revision | Git commit `22375ca9ec66a6bacfce7674d01f85b8af10f3fc` |

The application asserted the exact parameter count before training. The run used the same
`GPT2Config()` and `GPT2Model` classes exercised by the local tests; the shorter 128-token
training sequences reduced the cost of the proof without changing the model's learned
1,024-position architecture.

## Data and experiment design

The job streamed `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, revision
`v1.0.0`. Streaming was appropriate because the experiment needed only a bounded number
of documents and did not need to download or persist the full dataset. The data path:

1. assigns documents to train or validation by SHA-256 of the stable document ID;
2. tokenizes with the official GPT-2 `tiktoken` encoding;
3. appends one `<|endoftext|>` token per document; and
4. packs a continuous next-token stream while preserving transitions between chunks.

The configuration used a batch size of 1, sequence length 128, fixed seed 1337, AdamW
learning rate `3e-4`, weight decay `0.1`, and gradient clipping at `1.0`. It capped training
at exactly three optimizer steps, evaluated at steps 2 and 3, and allowed one validation
batch per evaluation. The source limits were 128 selected training documents and 4 selected
validation documents. These are execution bounds, not a reported training-token total.

## Results

| Result | Value |
| --- | ---: |
| VESSL state | Succeeded |
| Total Cloud duration | 1m 3s |
| Training script runtime | 22.02s |
| Completed optimizer steps | 3 |
| Initial train loss | 10.9330 |
| Final train loss | 9.8814 |
| Final validation loss | 9.9406 |
| Parameter count | 124,439,808 |

The decreasing loss over three updates confirms that gradients, AdamW updates, and the
streamed batches were connected correctly. It is too little evidence to infer convergence,
generalization, final perplexity, or useful language ability.

## Artifacts

The successful job wrote these files under the Cloud artifact path
`/output/tiny-pretrain-15c-clean-exit/`:

| Artifact | Purpose |
| --- | --- |
| `training_config.json` | Exact model, data, optimizer, and run controls |
| `metrics.jsonl` | Step-indexed train and validation measurements |
| `checkpoint_final.pt` | Model, optimizer, step, history, configuration, and RNG state |
| `generated_sample.txt` | Short post-run sample; expected to be essentially random |
| `training_summary.json` | Status, hardware, duration, losses, step count, and artifact paths |

The checkpoint remains a private VESSL artifact. It is not committed to Git, publicly
downloadable from this repository, or presented as a useful pretrained model.

## Failure and fix: 15b to 15c

The first tiny-training job completed all three optimizer steps and wrote every artifact. After
application work finished, Python raised a
`PyGILState_Release` fatal error during interpreter finalization, and the Cloud job had to
be manually terminated to stop it remaining active. The available trace did not identify
one faulty native binary, so attributing the defect specifically to `pyarrow`, `datasets`,
or another individual package would overstate the evidence. The failure boundary involved
interpreter shutdown with the streamed-dataset/native-extension subsystem loaded.

Milestone 15c made the smallest targeted runtime changes:

- explicitly close early-stopped dataset, token, packer, and batch iterators;
- pin the direct streaming runtime dependencies for reproducibility;
- bound package-install retries and add a cooperative training deadline; and
- allow `--force-clean-exit` only on the successful one-purpose batch path, after artifact
  files are written and explicit cleanup has completed.

The verification job then reached VESSL state `succeeded` with all artifacts present. The
guard does not turn failures or timeouts into success: those paths still report their
status and exit non-zero.

## Reproducing the workflow safely

No VESSL credential, GitHub token, dataset file, or model checkpoint is needed to inspect
the implementation and run its offline tests:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

To exercise the same tiny FineWeb-Edu entry point on suitable local GPU hardware, install
the pinned training extras and use the checked-in configuration:

```bash
python -m pip install --no-build-isolation \
  -c configs/requirements/tiny-pretrain-cloud.txt -e ".[train]"
python scripts/tiny_pretrain.py \
  --config configs/tiny_pretrain.json \
  --output-dir artifacts/tiny-pretrain \
  --max-runtime-seconds 180
```

This command requires Hugging Face network access and streams public FineWeb-Edu records.
Cloud reproduction additionally requires the reader's own VESSL account, private
credentials stored outside Git, a reviewed GPU resource, and an approved artifact volume;
the workflow is documented in [VESSL Cloud setup](../VESSL_SETUP.md). The repository does
not publish the original Cloud volumes or checkpoint as part of reproducibility.

## Limitations and meaningful next steps

Three steps at batch size 1 and sequence length 128 are enough to validate orchestration,
not to train a language model. A meaningful pretraining study would require a stated token
budget, many more optimizer updates, effective-batch design, mixed precision, a learning-rate
schedule, throughput and memory measurement, regular validation, robust data-stream resume,
checkpoint retention, and a responsible compute budget. Any released weights would also
need a model card covering FineWeb-Edu's ODC-By attribution, intended uses, limitations, and
evaluation. None of those future outcomes is implied by this experiment.
