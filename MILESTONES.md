# Execution Checklist

- Complete only one milestone at a time.
- Run focused tests and Ruff before marking a milestone complete.
- Explain the concepts and changed files concisely.
- Do not start the next milestone until I approve.
- Do not push to GitHub or use VESSL/cloud compute without my explicit approval.

1. [x] Professional project foundation
2. [x] GPT-2 configuration object
3. [x] GPT-2 tokenizer + input/target dataset
4. [x] Token and positional embeddings
5. [x] One causal self-attention head
6. [x] Multi-head causal attention
7. [x] Layer normalization
8. [x] GELU + feed-forward network
9. [x] Transformer block
10. [x] Full 12-block GPT-2 Small model, output head, weight tying, and exact parameter count
11a. [x] Language-model loss and validation evaluation
11b. [x] AdamW optimizer and one training step
11c. [x] Local multi-step training loop
11d. [x] Checkpoint save, load, and reproducible local resume
12. [x] Text generation: greedy, temperature, and top-k
13. [x] Official GPT-2 weight import and numerical compatibility verification (offline checks)
    - [ ] Optional online official-weight verification (requires the `verify` dependency)
14a. [x] VESSL Cloud preflight and run-configuration scaffolding
14b. [x] FineWeb-Edu streaming, tokenization, packing, and validation split
14c. [ ] **Next:** VESSL GPU smoke training run
14d. [ ] Main VESSL pretraining run, artifacts, and metrics
15. [ ] Portfolio documentation, original charts, demo, and model card
