"""
Evaluation metrics for the preference ranking model.

Implements:
  - Accuracy (exact match)
  - Macro-AUC (one-vs-rest)
  - Log-loss (multiclass)
  - NDCG@K (for ranking evaluation)
  - Preference score calibration (ECE)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    ndcg_score,
    roc_auc_score,
)


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        labels: Ground-truth class indices, shape (N,). Values in {0, 1, 2}.
        probabilities: Predicted probabilities, shape (N, 3).
                       Columns: [P(A wins), P(B wins), P(tie)].

    Returns:
        Dict with keys: accuracy, auc, log_loss, preference_score_mean.

    Raises:
        ValueError: If labels and probabilities have incompatible shapes.

    Validation Metrics:
        - auc is macro-averaged one-vs-rest ROC AUC.
        - log_loss is sklearn multiclass log-loss.
        - preference_score_mean: mean max probability (confidence proxy).
    """
    if len(labels) == 0:
        return {"accuracy": 0.0, "auc": 0.0, "log_loss": 0.0, "preference_score_mean": 0.0}

    predictions = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(labels, predictions))

    try:
        auc = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc = 0.0

    ll = float(log_loss(labels, probabilities, labels=[0, 1, 2]))

    # Mean confidence (max probability per row) — proxy for model certainty
    pref_score_mean = float(probabilities.max(axis=1).mean())

    return {
        "accuracy": accuracy,
        "auc": auc,
        "log_loss": ll,
        "preference_score_mean": pref_score_mean,
    }


def compute_ndcg(
    relevance_scores: np.ndarray,
    predicted_scores: np.ndarray,
    k: int = 10,
) -> float:
    """Compute NDCG@K for ranking evaluation.

    Used when evaluating ranked lists of responses (not just pairwise).

    Args:
        relevance_scores: True relevance scores, shape (1, N) or (N,).
        predicted_scores: Model-predicted scores, shape (1, N) or (N,).
        k: Cutoff rank.

    Returns:
        NDCG@K score in [0, 1].

    Validation Metrics:
        - Result must be in [0, 1].
    """
    if relevance_scores.ndim == 1:
        relevance_scores = relevance_scores.reshape(1, -1)
    if predicted_scores.ndim == 1:
        predicted_scores = predicted_scores.reshape(1, -1)
    return float(ndcg_score(relevance_scores, predicted_scores, k=k))


def compute_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) for the winning class.

    Measures how well confidence scores are calibrated.
    Well-calibrated model: P(correct | confidence=0.8) ≈ 0.8.

    Args:
        labels: True class indices, shape (N,).
        probabilities: Predicted probabilities, shape (N, 3).
        n_bins: Number of confidence bins.

    Returns:
        ECE in [0, 1]. Lower is better. Acceptance criterion: ECE < 0.05.

    Validation Metrics:
        - ECE < 0.05 is the production acceptance threshold.
    """
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)
