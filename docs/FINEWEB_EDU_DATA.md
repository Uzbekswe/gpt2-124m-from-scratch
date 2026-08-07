# FineWeb-Edu streaming data pipeline

Milestone 14b uses the public `HuggingFaceFW/fineweb-edu` dataset with the
`sample-10BT` configuration and pinned `v1.0.0` revision. Its source records use the
`text` field for content and the stable `id` field for deterministic split assignment.

FineWeb-Edu is licensed under ODC-By. Any future dataset use, artifacts, model card, or
portfolio materials must retain the required attribution. Dataset files are streamed and
are never committed to this repository. See the [FineWeb-Edu dataset
card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for its current metadata
and license.

Each document receives exactly one GPT-2 `<|endoftext|>` token. This marks a boundary so
the model can learn that unrelated documents do not form one continuous natural document.
The pipeline assigns documents to train or validation through SHA-256 of the stable `id`,
not Python's process-randomized `hash()`. A document therefore stays in the same split on
every worker and future run.

The packer collects 1,025 consecutive token IDs for GPT-2's 1,024-token context: the first
1,024 are inputs and the shifted 1,024 are targets. It then advances by 1,024 IDs, keeping
the final target token as the first input token of the next sequence. This preserves the
next-token transition at every chunk boundary without holding the full dataset in memory.

Install the optional streaming dependency only when it is needed:

```bash
python -m pip install -e ".[data]"
```
