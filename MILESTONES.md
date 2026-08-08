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
14c. [x] VESSL GPU smoke training run
14d. [ ] Main VESSL pretraining run, artifacts, and metrics
15a. [x] Tiny cost-capped pretraining configuration, artifacts, and local verification
15a.2 [x] Bounded tiny-training timeout and streamed-data failure safety
15b. [x] One cost-capped VESSL Cloud tiny pretraining job
    - The 3-step application completed and exported all artifacts, but the Python process hit a
      native shutdown fault and its Cloud job was terminated to stop billing.
15c. [x] Diagnose the VESSL native Python shutdown fault and verify one clean exit
    - Job `job-nri2sb9n1w1i` succeeded after deterministic stream cleanup, pinned streaming
      dependencies, bounded runtime controls, and a batch-only clean-exit guard reached only
      after artifacts were written successfully.
16. [x] Portfolio documentation, experiment evidence, and interview demo
    - Original training charts and a model card remain optional future work if a meaningful
      checkpoint is trained and released; the 3-step systems proof is not presented as one.
