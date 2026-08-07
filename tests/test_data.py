"""Tests for next-token input-target window construction."""

import torch

from gpt2_124m.data import TokenWindowDataset


def test_input_and_target_have_an_exact_one_token_shift() -> None:
    """Each target token is the immediate successor of its input token."""
    dataset = TokenWindowDataset(token_ids=[10, 11, 12, 13, 14, 15], context_length=3, stride=1)

    inputs, targets = dataset[0]

    assert inputs.tolist() == [10, 11, 12]
    assert targets.tolist() == [11, 12, 13]


def test_dataset_length_respects_context_length_and_stride() -> None:
    """Only complete windows are counted, and stride sets each window's start."""
    dataset = TokenWindowDataset(token_ids=list(range(10)), context_length=4, stride=3)

    assert len(dataset) == 2
    assert dataset[1][0].tolist() == [3, 4, 5, 6]


def test_returned_windows_are_long_tensors_with_context_shape() -> None:
    """Model-ready windows use the integer tensor dtype and requested length."""
    dataset = TokenWindowDataset(token_ids=[0, 1, 2, 3, 4], context_length=3, stride=1)

    inputs, targets = dataset[0]

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert inputs.shape == (3,)
    assert targets.shape == (3,)
