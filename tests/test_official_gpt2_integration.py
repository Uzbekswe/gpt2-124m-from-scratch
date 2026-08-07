"""Opt-in online verification against the official GPT-2 Small reference checkpoint."""

import os

import pytest

from gpt2_124m.config import GPT2Config
from gpt2_124m.gpt2_weights import (
    download_official_gpt2_reference,
    verify_official_gpt2_compatibility,
)
from gpt2_124m.model import GPT2Model

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OFFICIAL_GPT2_INTEGRATION") != "1",
    reason="Set RUN_OFFICIAL_GPT2_INTEGRATION=1 to download and verify official GPT-2 weights.",
)


def test_official_gpt2_logits_and_greedy_tokens_match() -> None:
    """Downloaded official tensors reproduce reference logits and deterministic token IDs."""
    pytest.importorskip("transformers")
    model = GPT2Model(GPT2Config())
    reference_model = download_official_gpt2_reference(device="cpu")

    report = verify_official_gpt2_compatibility(model, reference_model, max_new_tokens=5)

    assert report.passed
    assert report.logits_shape == (1, len(report.input_token_ids), 50_257)
