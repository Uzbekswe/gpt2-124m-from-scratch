"""Offline tests for FineWeb-Edu split, tokenization, and continuous packing utilities."""

import importlib.util
import sys
import types
from dataclasses import replace
from itertools import islice

import pytest
import torch

from gpt2_124m.config import GPT2_DEBUG_CONFIG
from gpt2_124m.fineweb import (
    DEFAULT_VALIDATION_FRACTION,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    FineWebEduIterableDataset,
    document_split,
    iter_document_token_ids,
    pack_token_ids,
    stream_fineweb_edu_documents,
)


class FakeTokenizer:
    """Encode synthetic test strings into small, easy-to-inspect token sequences."""

    end_of_text_token_id = 99

    def __init__(self, token_ids_by_text: dict[str, list[int]]) -> None:
        self.token_ids_by_text = token_ids_by_text

    def encode(self, text: str) -> list[int]:
        """Return a fresh token list so the pipeline can append its EOT token."""
        return list(self.token_ids_by_text[text])


class ClosableDocumentIterator:
    """Synthetic remote-source stand-in that records explicit early-stop cleanup."""

    def __init__(self, documents: list[dict[str, str]]) -> None:
        self.documents = iter(documents)
        self.closed = False

    def __iter__(self) -> "ClosableDocumentIterator":
        return self

    def __next__(self) -> dict[str, str]:
        return next(self.documents)

    def close(self) -> None:
        self.closed = True


def _document_id_for_split(split: str, index: int = 0) -> str:
    """Find a deterministic synthetic ID assigned to the requested rare/common split."""
    candidate_index = index
    while True:
        candidate = f"synthetic-document-{candidate_index}"
        if document_split(candidate) == split:
            return candidate
        candidate_index += 1


def _documents_for_both_splits() -> list[dict[str, str]]:
    """Build long enough train and validation documents for independent packing tests."""
    return [
        {"id": _document_id_for_split(TRAIN_SPLIT), "text": "train-one"},
        {"id": _document_id_for_split(TRAIN_SPLIT, 1), "text": "train-two"},
        {"id": _document_id_for_split(VALIDATION_SPLIT), "text": "validation-one"},
        {"id": _document_id_for_split(VALIDATION_SPLIT, 1), "text": "validation-two"},
    ]


def test_document_ids_have_stable_non_overlapping_split_assignments() -> None:
    """SHA-256-based assignment is repeatable and each document has exactly one split."""
    document_ids = [f"document-{index}" for index in range(2_000)]
    first_assignments = [document_split(document_id) for document_id in document_ids]
    second_assignments = [document_split(document_id) for document_id in document_ids]

    assert first_assignments == second_assignments
    train_ids = {
        document_id
        for document_id, split in zip(document_ids, first_assignments, strict=True)
        if split == TRAIN_SPLIT
    }
    validation_ids = set(document_ids) - train_ids
    assert train_ids.isdisjoint(validation_ids)
    assert validation_ids


@pytest.mark.parametrize(
    ("document_id", "validation_fraction", "error", "message"),
    [
        ("", DEFAULT_VALIDATION_FRACTION, ValueError, "document_id"),
        (123, DEFAULT_VALIDATION_FRACTION, ValueError, "document_id"),
        ("valid", -0.1, ValueError, "validation_fraction"),
        ("valid", 1.0, ValueError, "validation_fraction"),
        ("valid", float("nan"), ValueError, "validation_fraction"),
    ],
)
def test_split_validation_rejects_invalid_document_ids_and_fractions(
    document_id: object,
    validation_fraction: float,
    error: type[Exception],
    message: str,
) -> None:
    """The stable split utility fails early for invalid identifiers or fractions."""
    with pytest.raises(error, match=message):
        document_split(document_id, validation_fraction=validation_fraction)  # type: ignore[arg-type]


def test_document_tokenization_appends_exactly_one_eot_per_selected_document() -> None:
    """Each full selected document ends with one EOT ID, including an empty token sequence."""
    train_id = _document_id_for_split(TRAIN_SPLIT)
    documents = [{"id": train_id, "text": "first"}, {"id": train_id + "-2", "text": "second"}]
    tokenizer = FakeTokenizer({"first": [1, 2], "second": [3]})
    selected_documents = [
        document
        for document in documents
        if document_split(document["id"]) == TRAIN_SPLIT
    ]

    token_ids = list(
        iter_document_token_ids(selected_documents, tokenizer=tokenizer, split=TRAIN_SPLIT)
    )

    assert token_ids.count(tokenizer.end_of_text_token_id) == len(selected_documents)
    assert token_ids[-1] == tokenizer.end_of_text_token_id


def test_packer_returns_long_shifted_context_windows_and_preserves_boundaries() -> None:
    """Two adjacent chunks share the boundary token needed for the next prediction."""
    chunks = list(pack_token_ids(range(9), context_length=4))

    assert len(chunks) == 2
    first_inputs, first_targets = chunks[0]
    second_inputs, second_targets = chunks[1]
    assert first_inputs.dtype == torch.long
    assert first_targets.dtype == torch.long
    assert first_inputs.shape == (4,)
    assert first_targets.shape == (4,)
    assert torch.equal(first_targets, first_inputs + 1)
    assert torch.equal(second_targets, second_inputs + 1)
    assert first_targets[-1].item() == second_inputs[0].item()


def test_train_and_validation_iterable_datasets_keep_token_streams_separate() -> None:
    """Separate dataset instances select only their own stable document split before packing."""
    documents = _documents_for_both_splits()
    tokenizer = FakeTokenizer(
        {
            "train-one": [10, 11, 12],
            "train-two": [13, 14, 15],
            "validation-one": [20, 21, 22],
            "validation-two": [23, 24, 25],
        }
    )
    config = replace(GPT2_DEBUG_CONFIG, context_length=4)

    def source_factory() -> object:
        """Return a new synthetic source for each independently iterated split dataset."""
        return iter(documents)

    train_dataset = FineWebEduIterableDataset(
        split=TRAIN_SPLIT,
        config=config,
        tokenizer=tokenizer,
        document_source_factory=source_factory,
    )
    validation_dataset = FineWebEduIterableDataset(
        split=VALIDATION_SPLIT,
        config=config,
        tokenizer=tokenizer,
        document_source_factory=source_factory,
    )

    train_inputs, _ = next(iter(train_dataset))
    validation_inputs, _ = next(iter(validation_dataset))

    assert set(train_inputs.tolist()).issubset({10, 11, 12, 13, 14, 15, 99})
    assert set(validation_inputs.tolist()).issubset({20, 21, 22, 23, 24, 25, 99})


def test_early_stopped_dataset_iterator_closes_its_document_source() -> None:
    """Fixed-step runs release a live streaming source rather than defer it to Python shutdown."""
    train_id = _document_id_for_split(TRAIN_SPLIT)
    source = ClosableDocumentIterator([{"id": train_id, "text": "train"}])
    dataset = FineWebEduIterableDataset(
        split=TRAIN_SPLIT,
        config=replace(GPT2_DEBUG_CONFIG, context_length=4),
        tokenizer=FakeTokenizer({"train": [1, 2, 3, 4]}),
        document_source_factory=lambda: source,
    )
    iterator = iter(dataset)

    next(iterator)
    iterator.close()

    assert source.closed


def test_document_and_token_limits_stop_before_a_partial_document() -> None:
    """Smoke-run limits cap selected complete documents and never remove their final EOT token."""
    train_id = _document_id_for_split(TRAIN_SPLIT)
    documents = [
        {"id": train_id, "text": "one"},
        {"id": train_id + "-next", "text": "two"},
    ]
    selected_documents = [
        document for document in documents if document_split(document["id"]) == TRAIN_SPLIT
    ]
    tokenizer = FakeTokenizer({"one": [1, 2, 3], "two": [4, 5, 6]})

    one_document_tokens = list(
        iter_document_token_ids(
            selected_documents,
            tokenizer=tokenizer,
            split=TRAIN_SPLIT,
            max_documents=1,
        )
    )
    four_token_limit = list(
        iter_document_token_ids(
            selected_documents,
            tokenizer=tokenizer,
            split=TRAIN_SPLIT,
            max_tokens=4,
        )
    )

    assert one_document_tokens == [1, 2, 3, 99]
    assert four_token_limit == [1, 2, 3, 99]


@pytest.mark.parametrize("split", ["test", "", 3])
def test_token_iterator_rejects_invalid_split_names(split: object) -> None:
    """Only the two hash-defined splits are valid for the FineWeb streaming pipeline."""
    with pytest.raises(ValueError, match="split"):
        list(iter_document_token_ids([], tokenizer=FakeTokenizer({}), split=split))  # type: ignore[arg-type]


def test_actual_streaming_without_datasets_dependency_has_a_clear_install_error() -> None:
    """The optional dependency remains lazy and is required only for actual remote streaming."""
    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets is installed; this offline dependency-error check is not applicable.")

    with pytest.raises(ImportError, match="optional data dependency"):
        stream_fineweb_edu_documents()


def test_streaming_network_failure_has_a_clear_project_specific_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote initialization explains that streamed FineWeb requires Hugging Face access."""
    fake_datasets = types.ModuleType("datasets")

    def failing_load_dataset(*args: object, **kwargs: object) -> object:
        raise OSError("synthetic network outage")

    fake_datasets.load_dataset = failing_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="Hugging Face network access") as error:
        list(stream_fineweb_edu_documents())

    assert isinstance(error.value.__cause__, OSError)


def test_fineweb_iterable_dataset_does_not_open_remote_data_until_iteration() -> None:
    """Constructing a dataset stays offline; its source factory runs only when iteration starts."""
    source_started = False

    def source_factory() -> list[dict[str, str]]:
        nonlocal source_started
        source_started = True
        return []

    dataset = FineWebEduIterableDataset(
        split=TRAIN_SPLIT,
        config=GPT2_DEBUG_CONFIG,
        tokenizer=FakeTokenizer({}),
        document_source_factory=source_factory,
    )
    assert not source_started

    assert list(islice(dataset, 1)) == []
    assert source_started
