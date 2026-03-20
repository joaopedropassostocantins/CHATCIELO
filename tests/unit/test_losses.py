"""Unit tests for src/training/losses.py."""
from __future__ import annotations

import torch
import pytest

from src.training.losses import (
    CombinedPreferenceLoss,
    LabelSmoothingCrossEntropy,
    MarginRankingLoss,
)


class TestLabelSmoothingCrossEntropy:
    def test_loss_is_positive(self):
        loss_fn = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.1)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_loss_is_scalar(self):
        loss_fn = LabelSmoothingCrossEntropy()
        logits = torch.randn(4, 3)
        targets = torch.zeros(4, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert loss.shape == torch.Size([])

    def test_perfect_prediction_low_loss(self):
        """Very confident correct predictions should yield low loss."""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.0)
        logits = torch.tensor([[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]])
        targets = torch.tensor([0, 0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1

    def test_loss_is_finite(self):
        loss_fn = LabelSmoothingCrossEntropy()
        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)


class TestMarginRankingLoss:
    def test_all_tie_returns_zero(self):
        loss_fn = MarginRankingLoss()
        probs = torch.softmax(torch.randn(4, 3), dim=-1)
        targets = torch.full((4,), 2, dtype=torch.long)  # all tie
        loss = loss_fn(probs, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_non_negative(self):
        loss_fn = MarginRankingLoss()
        probs = torch.softmax(torch.randn(8, 3), dim=-1)
        targets = torch.randint(0, 3, (8,))
        loss = loss_fn(probs, targets)
        assert loss.item() >= 0


class TestCombinedPreferenceLoss:
    def test_returns_dict_with_keys(self):
        loss_fn = CombinedPreferenceLoss()
        logits = torch.randn(4, 3)
        probs = torch.softmax(logits, dim=-1)
        targets = torch.randint(0, 3, (4,))
        result = loss_fn(logits, probs, targets)
        assert "loss" in result
        assert "cls_loss" in result
        assert "rank_loss" in result

    def test_total_loss_is_finite(self):
        loss_fn = CombinedPreferenceLoss()
        logits = torch.randn(8, 3)
        probs = torch.softmax(logits, dim=-1)
        targets = torch.randint(0, 3, (8,))
        result = loss_fn(logits, probs, targets)
        assert torch.isfinite(result["loss"])

    def test_weights_sum_preserved(self):
        """cls_weight=0.7 means total ≈ 0.7*cls + 0.3*rank."""
        loss_fn = CombinedPreferenceLoss(cls_weight=0.7)
        logits = torch.randn(4, 3)
        probs = torch.softmax(logits, dim=-1)
        targets = torch.zeros(4, dtype=torch.long)
        result = loss_fn(logits, probs, targets)
        expected = 0.7 * result["cls_loss"] + 0.3 * result["rank_loss"]
        assert abs(result["loss"].item() - expected.item()) < 1e-5
