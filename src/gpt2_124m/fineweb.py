"""Lazy FineWeb-Edu streaming, deterministic splitting, and GPT-2 sequence packing."""

import hashlib
import itertools
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from math import isfinite
from numbers import Real
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from gpt2_124m.config import GPT2Config
from gpt2_124m.tokenizer import GPT2Tokenizer

FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIGURATION = "sample-10BT"
FINEWEB_EDU_REVISION = "v1.0.0"
FINEWEB_EDU_TEXT_FIELD = "text"
FINEWEB_EDU_DOCUMENT_ID_FIELD = "id"
DEFAULT_VALIDATION_FRACTION = 0.005

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
SUPPORTED_SPLITS = frozenset({TRAIN_SPLIT, VALIDATION_SPLIT})


class TokenizerLike(Protocol):
    """The small tokenizer interface required by the streaming document iterator."""

    @property
    def end_of_text_token_id(self) -> int:
        """Return the integer GPT-2 document-boundary token ID."""

    def encode(self, text: str) -> list[int]:
        """Encode one document into token IDs."""


DocumentSourceFactory = Callable[[], Iterable[Mapping[str, object]]]


def document_split(
    document_id: str,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> str:
    """Assign one stable document ID to exactly one deterministic train/validation split."""
    _validate_document_id(document_id)
    _validate_validation_fraction(validation_fraction)
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    normalized_hash = int.from_bytes(digest, byteorder="big") / (1 << 256)
    return VALIDATION_SPLIT if normalized_hash < validation_fraction else TRAIN_SPLIT


def iter_document_token_ids(
    documents: Iterable[Mapping[str, object]],
    *,
    tokenizer: TokenizerLike,
    split: str,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    text_field: str = FINEWEB_EDU_TEXT_FIELD,
    document_id_field: str = FINEWEB_EDU_DOCUMENT_ID_FIELD,
    max_documents: int | None = None,
    max_tokens: int | None = None,
) -> Iterator[int]:
    """Lazily yield selected-document GPT-2 IDs with one EOT token per complete document.

    ``max_tokens`` limits complete tokenized documents: if the next document plus its EOT
    would exceed the limit, iteration stops instead of emitting a truncated document.
    """
    _validate_split(split)
    _validate_validation_fraction(validation_fraction)
    _validate_field_name(text_field, name="text_field")
    _validate_field_name(document_id_field, name="document_id_field")
    _validate_optional_limit(max_documents, name="max_documents")
    _validate_optional_limit(max_tokens, name="max_tokens")
    eot_token_id = tokenizer.end_of_text_token_id
    _validate_token_id(eot_token_id, name="tokenizer.end_of_text_token_id")

    selected_documents = 0
    emitted_tokens = 0
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError("each document must be a mapping containing text and id fields.")
        document_id = _required_string_field(document, document_id_field)
        if document_split(document_id, validation_fraction=validation_fraction) != split:
            continue
        if max_documents is not None and selected_documents >= max_documents:
            return

        text = _required_string_field(document, text_field)
        document_token_ids = tokenizer.encode(text)
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in document_token_ids
        ):
            raise TypeError("tokenizer.encode must return integer token IDs.")
        if any(token_id < 0 for token_id in document_token_ids):
            raise ValueError("tokenizer.encode must return non-negative token IDs.")
        document_token_ids.append(eot_token_id)

        if max_tokens is not None and emitted_tokens + len(document_token_ids) > max_tokens:
            return
        selected_documents += 1
        emitted_tokens += len(document_token_ids)
        yield from document_token_ids


def pack_token_ids(
    token_ids: Iterable[int],
    *,
    context_length: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Pack a continuous token stream into non-overlapping input windows with one-token overlap."""
    _validate_context_length(context_length)
    buffer: deque[int] = deque()
    for token_id in token_ids:
        _validate_token_id(token_id, name="token_ids item")
        buffer.append(token_id)
        if len(buffer) < context_length + 1:
            continue

        window = list(buffer)
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        target_ids = torch.tensor(window[1:], dtype=torch.long)
        yield input_ids, target_ids

        for _ in range(context_length):
            buffer.popleft()


def stream_fineweb_edu_documents(
    *,
    dataset_name: str = FINEWEB_EDU_DATASET,
    configuration: str = FINEWEB_EDU_CONFIGURATION,
    revision: str = FINEWEB_EDU_REVISION,
) -> Iterable[Mapping[str, object]]:
    """Request the remote FineWeb-Edu train stream only when a caller begins iteration."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "FineWeb-Edu streaming requires the optional data dependency. "
            'Install it with `python -m pip install -e ".[data]"`. '
        ) from error

    def documents() -> Iterator[Mapping[str, object]]:
        stream: Iterable[Mapping[str, object]] | None = None
        try:
            stream = load_dataset(
                dataset_name,
                name=configuration,
                split=TRAIN_SPLIT,
                streaming=True,
                revision=revision,
            )
            yield from stream
        except Exception as error:
            raise RuntimeError(
                "Could not initialize or read the FineWeb-Edu stream. "
                "Hugging Face network access is required for streamed training."
            ) from error
        finally:
            # ``datasets`` uses native Arrow and network resources while streaming. Closing the
            # iterator here lets those resources stop before Python interpreter finalization.
            _close_if_possible(stream)

    return documents()


class FineWebEduIterableDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """A lazy, split-isolated FineWeb-Edu source ready for a future PyTorch DataLoader."""

    def __init__(
        self,
        *,
        split: str,
        config: GPT2Config = GPT2Config(),
        tokenizer: TokenizerLike | None = None,
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        max_documents: int | None = None,
        max_tokens: int | None = None,
        dataset_name: str = FINEWEB_EDU_DATASET,
        configuration: str = FINEWEB_EDU_CONFIGURATION,
        revision: str = FINEWEB_EDU_REVISION,
        text_field: str = FINEWEB_EDU_TEXT_FIELD,
        document_id_field: str = FINEWEB_EDU_DOCUMENT_ID_FIELD,
        document_source_factory: DocumentSourceFactory | None = None,
    ) -> None:
        """Store streaming settings without touching Hugging Face data or VESSL resources."""
        super().__init__()
        _validate_split(split)
        _validate_validation_fraction(validation_fraction)
        _validate_optional_limit(max_documents, name="max_documents")
        _validate_optional_limit(max_tokens, name="max_tokens")
        _validate_field_name(text_field, name="text_field")
        _validate_field_name(document_id_field, name="document_id_field")
        _validate_field_name(dataset_name, name="dataset_name")
        _validate_field_name(configuration, name="configuration")
        _validate_field_name(revision, name="revision")
        if not isinstance(config, GPT2Config):
            raise TypeError("config must be a GPT2Config.")
        if document_source_factory is not None and not callable(document_source_factory):
            raise TypeError("document_source_factory must be callable or None.")

        self.split = split
        self.config = config
        self.tokenizer = tokenizer if tokenizer is not None else GPT2Tokenizer()
        self.validation_fraction = validation_fraction
        self.max_documents = max_documents
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name
        self.configuration = configuration
        self.revision = revision
        self.text_field = text_field
        self.document_id_field = document_id_field
        self.document_source_factory = document_source_factory

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Open one document stream and pack only this dataset instance's selected split."""
        documents = self._documents_for_current_worker()
        token_ids = iter_document_token_ids(
            documents,
            tokenizer=self.tokenizer,
            split=self.split,
            validation_fraction=self.validation_fraction,
            text_field=self.text_field,
            document_id_field=self.document_id_field,
            max_documents=self.max_documents,
            max_tokens=self.max_tokens,
        )
        packed_examples = pack_token_ids(token_ids, context_length=self.config.context_length)
        try:
            yield from packed_examples
        finally:
            # Fixed-step training stops before the remote stream is exhausted. Explicitly closing
            # every generator runs Hugging Face/Arrow cleanup while Python is fully operational.
            _close_if_possible(packed_examples)
            _close_if_possible(token_ids)
            _close_if_possible(documents)

    def _documents_for_current_worker(self) -> Iterable[Mapping[str, object]]:
        """Create a lazy source and shard source documents across DataLoader workers when used."""
        if self.document_source_factory is None:
            documents = stream_fineweb_edu_documents(
                dataset_name=self.dataset_name,
                configuration=self.configuration,
                revision=self.revision,
            )
        else:
            documents = self.document_source_factory()

        worker_info = get_worker_info()
        if worker_info is None:
            return documents
        return itertools.islice(documents, worker_info.id, None, worker_info.num_workers)


def _validate_document_id(document_id: object) -> None:
    """Require a non-empty stable string identifier before deterministic hashing."""
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string.")


def _validate_split(split: object) -> None:
    """Reject names other than this pipeline's train and validation streams."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split must be one of {sorted(SUPPORTED_SPLITS)}; got {split!r}.")


def _validate_validation_fraction(value: object) -> None:
    """Require a finite fraction that leaves at least some documents for train splitting."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or not 0.0 <= value < 1.0
    ):
        raise ValueError("validation_fraction must be a finite number in the range [0.0, 1.0).")


def _validate_optional_limit(value: object, *, name: str) -> None:
    """Require positive document/token caps when an optional smoke-run limit is set."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None.")


def _validate_context_length(context_length: object) -> None:
    """Require a positive next-token prediction sequence length."""
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length <= 0
    ):
        raise ValueError("context_length must be a positive integer.")


def _validate_field_name(value: object, *, name: str) -> None:
    """Require non-empty field names and pinned dataset metadata strings."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_token_id(token_id: object, *, name: str) -> None:
    """Reject token IDs that cannot safely enter a PyTorch long tensor."""
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _required_string_field(document: Mapping[str, object], field_name: str) -> str:
    """Read one required text or ID field with an actionable validation error."""
    if field_name not in document:
        raise KeyError(f"document is missing required field {field_name!r}.")
    value = document[field_name]
    if not isinstance(value, str):
        raise TypeError(f"document field {field_name!r} must be a string.")
    if field_name == FINEWEB_EDU_DOCUMENT_ID_FIELD:
        _validate_document_id(value)
    return value


def _close_if_possible(resource: object | None) -> None:
    """Close an iterator/resource when it exposes the standard generator ``close`` method."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()
