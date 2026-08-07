"""Tests for language-model loss and validation evaluation."""

import pytest
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from gpt2_124m.training import compute_language_model_loss, evaluate_loss


class TinyLogitModel(nn.Module):
    """Return deterministic logits selected by the first token in each input row."""

    def __init__(self) -> None:
        super().__init__()
        self.logits_by_marker = nn.Parameter(
            torch.tensor(
                [
                    [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
                    [[-1.0, 0.0, 2.0], [2.0, 0.0, -1.0]],
                ]
            )
        )
        self.grad_enabled_during_forward: bool | None = None

    def forward(self, input_ids: Tensor) -> Tensor:
        """Look up one fixed `[sequence, vocabulary]` logit tensor per batch row."""
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return self.logits_by_marker[input_ids[:, 0]]


def test_language_model_loss_matches_pytorch_cross_entropy() -> None:
    """The utility flattens batch and sequence dimensions exactly as PyTorch expects."""
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 2]], dtype=torch.long)

    loss = compute_language_model_loss(logits, targets)
    expected = functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))

    torch.testing.assert_close(loss, expected)


def test_confident_correct_logits_have_lower_loss_than_confident_wrong_logits() -> None:
    """Cross-entropy rewards placing the highest score on the next-token target."""
    targets = torch.tensor([[0, 1]], dtype=torch.long)
    correct_logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
    wrong_logits = torch.tensor([[[-10.0, 10.0], [10.0, -10.0]]])

    assert compute_language_model_loss(correct_logits, targets) < compute_language_model_loss(
        wrong_logits,
        targets,
    )


@pytest.mark.parametrize(
    ("logits", "targets", "error", "message"),
    [
        (
            torch.randn(2, 3),
            torch.zeros((2, 3), dtype=torch.long),
            ValueError,
            "logits must have shape",
        ),
        (
            torch.randn(2, 3, 4),
            torch.zeros(2, dtype=torch.long),
            ValueError,
            "targets must have shape",
        ),
        (
            torch.randn(2, 3, 4),
            torch.zeros((2, 2), dtype=torch.long),
            ValueError,
            "matching batch and sequence",
        ),
        (
            torch.randn(1, 2, 4),
            torch.tensor([[0.0, 1.0]]),
            TypeError,
            "integer token IDs",
        ),
        (
            torch.randn(1, 2, 4),
            torch.tensor([[0, 4]], dtype=torch.long),
            ValueError,
            "target IDs must be in",
        ),
    ],
)
def test_language_model_loss_rejects_invalid_inputs(
    logits: Tensor,
    targets: Tensor,
    error: type[Exception],
    message: str,
) -> None:
    """Shape, type, and vocabulary-range mistakes fail before loss computation."""
    with pytest.raises(error, match=message):
        compute_language_model_loss(logits, targets)


def test_evaluate_loss_averages_known_batch_losses() -> None:
    """Validation evaluation returns the arithmetic mean of each evaluated batch loss."""
    model = TinyLogitModel()
    batches = [
        (torch.tensor([[0, 0]]), torch.tensor([[0, 1]])),
        (torch.tensor([[1, 1]]), torch.tensor([[2, 0]])),
    ]
    with torch.no_grad():
        expected_losses = [
            compute_language_model_loss(model.logits_by_marker[0:1], batches[0][1]).item(),
            compute_language_model_loss(model.logits_by_marker[1:2], batches[1][1]).item(),
        ]

    loss = evaluate_loss(model, batches, device="cpu")

    assert loss == pytest.approx(sum(expected_losses) / len(expected_losses))


def test_evaluate_loss_uses_inference_mode_and_creates_no_parameter_gradients() -> None:
    """Validation evaluation neither records gradients nor populates parameter gradients."""
    model = TinyLogitModel()
    batches = [(torch.tensor([[0, 0]]), torch.tensor([[0, 1]]))]

    evaluate_loss(model, batches, device="cpu")

    assert model.grad_enabled_during_forward is False
    assert model.logits_by_marker.grad is None


def test_evaluate_loss_restores_a_model_that_started_in_train_mode() -> None:
    """Validation does not accidentally leave a training model in evaluation mode."""
    model = TinyLogitModel()
    model.train()
    batches = [(torch.tensor([[0, 0]]), torch.tensor([[0, 1]]))]

    evaluate_loss(model, batches, device="cpu")

    assert model.training


def test_evaluate_loss_preserves_a_model_that_started_in_eval_mode() -> None:
    """Validation leaves an already-evaluating model in evaluation mode."""
    model = TinyLogitModel()
    model.eval()
    batches = [(torch.tensor([[0, 0]]), torch.tensor([[0, 1]]))]

    evaluate_loss(model, batches, device="cpu")

    assert not model.training


@pytest.mark.parametrize("max_batches", [0, -1, True, "1"])
def test_evaluate_loss_rejects_invalid_max_batches(max_batches: object) -> None:
    """The batch limit must be a positive integer when it is provided."""
    model = TinyLogitModel()
    batches = [(torch.tensor([[0, 0]]), torch.tensor([[0, 1]]))]

    with pytest.raises(ValueError, match="max_batches"):
        evaluate_loss(model, batches, device="cpu", max_batches=max_batches)  # type: ignore[arg-type]
