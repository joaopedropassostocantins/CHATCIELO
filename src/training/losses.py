"""
Loss functions for preference ranking training.

Implements:
  1. LabelSmoothingCrossEntropy — primary classification loss
  2. MarginRankingLoss — auxiliary pairwise ranking loss
  3. CombinedPreferenceLoss — weighted combination of both

MATHEMATICAL DEFINITIONS
────────────────────────
Label Smoothing CrossEntropy:
  ỹ_c = (1-ε)·1[c=y] + ε/C
  L_cls = -Σ_c ỹ_c · log(softmax(l)_c)

Margin Ranking Loss (for A vs B binary case):
  Given score_a = P(A wins), score_b = P(B wins), y ∈ {-1, +1}:
  L_rank = max(0, -y · (score_a - score_b) + margin)

Combined:
  L = α · L_cls + (1-α) · L_rank
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing.

    Args:
        num_classes: Number of output classes.
        smoothing: Label smoothing factor ε ∈ [0, 1).

    Returns:
        Scalar loss tensor.

    Validation Metrics:
        - smoothing=0.0 must equal standard CrossEntropyLoss exactly.
        - Loss must be >= 0.
    """

    def __init__(self, num_classes: int = 3, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy.

        Args:
            logits: Raw predictions, shape (B, num_classes).
            targets: Ground-truth class indices, shape (B,).

        Returns:
            Scalar mean loss over the batch.
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)

        loss = (-smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


class MarginRankingLoss(nn.Module):
    """Pairwise margin ranking loss on preference scores.

    Converts the 3-class output into a binary A-vs-B score for ranking.

    Args:
        margin: Minimum required margin between P(A) and P(B).

    Returns:
        Scalar loss tensor.
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin
        self._loss = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute margin ranking loss.

        Args:
            probabilities: Softmax probabilities, shape (B, 3).
                           Column 0 = P(A), column 1 = P(B), column 2 = P(tie).
            targets: Ground-truth labels in {0, 1, 2}, shape (B,).
                     Tie examples are excluded from ranking loss.

        Returns:
            Scalar mean ranking loss.
        """
        # Only consider non-tie examples
        mask = targets != 2
        if mask.sum() == 0:
            return torch.tensor(0.0, device=probabilities.device, requires_grad=True)

        probs = probabilities[mask]
        tgts = targets[mask]

        score_a = probs[:, 0]
        score_b = probs[:, 1]

        # y=+1 means A should rank higher, y=-1 means B should rank higher
        y = torch.where(tgts == 0, torch.ones_like(tgts, dtype=torch.float), -torch.ones_like(tgts, dtype=torch.float))

        return self._loss(score_a, score_b, y)


class CombinedPreferenceLoss(nn.Module):
    """Weighted combination of classification + ranking losses.

    Args:
        num_classes: Number of output classes.
        cls_weight: Weight α for classification loss. Ranking weight = 1-α.
        smoothing: Label smoothing for classification loss.
        margin: Margin for ranking loss.

    Returns:
        Scalar combined loss.
    """

    def __init__(
        self,
        num_classes: int = 3,
        cls_weight: float = 0.7,
        smoothing: float = 0.1,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.rank_weight = 1.0 - cls_weight
        self.cls_loss = LabelSmoothingCrossEntropy(num_classes, smoothing)
        self.rank_loss = MarginRankingLoss(margin)

    def forward(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: Raw model logits, shape (B, num_classes).
            probabilities: softmax(logits), shape (B, num_classes).
            targets: Ground-truth labels, shape (B,).

        Returns:
            Dict with keys: 'loss' (total), 'cls_loss', 'rank_loss'.
        """
        l_cls = self.cls_loss(logits, targets)
        l_rank = self.rank_loss(probabilities, targets)
        total = self.cls_weight * l_cls + self.rank_weight * l_rank
        return {"loss": total, "cls_loss": l_cls, "rank_loss": l_rank}
