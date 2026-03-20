"""Unit tests for src/evaluation/metrics.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_ece, compute_metrics, compute_ndcg


class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 0, 1])
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        m = compute_metrics(labels, probs)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        labels = np.array([0, 1, 2])
        probs = np.ones((3, 3)) / 3
        m = compute_metrics(labels, probs)
        assert "accuracy" in m
        assert "auc" in m
        assert "log_loss" in m
        assert "preference_score_mean" in m

    def test_empty_input_returns_zeros(self):
        m = compute_metrics(np.array([]), np.array([]).reshape(0, 3))
        assert m["accuracy"] == 0.0

    def test_preference_score_in_range(self):
        labels = np.array([0, 1, 2])
        probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        m = compute_metrics(labels, probs)
        assert 0.0 <= m["preference_score_mean"] <= 1.0

    def test_random_uniform_auc_near_half(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=200)
        probs = np.ones((200, 3)) / 3
        m = compute_metrics(labels, probs)
        # Uniform predictions → AUC ≈ 0.5
        assert 0.4 < m["auc"] < 0.6


class TestComputeEce:
    def test_perfect_calibration(self):
        """Model always predicts 1.0 for correct class → ECE ≈ 0."""
        labels = np.array([0, 1, 2, 0, 1])
        probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        ece = compute_ece(labels, probs)
        assert ece == pytest.approx(0.0, abs=1e-5)

    def test_ece_in_range(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 3, 100)
        probs = rng.dirichlet([1, 1, 1], size=100)
        ece = compute_ece(labels, probs)
        assert 0.0 <= ece <= 1.0


class TestComputeNdcg:
    def test_perfect_ranking(self):
        rel = np.array([3, 2, 1, 0])
        ndcg = compute_ndcg(rel, rel, k=4)
        assert ndcg == pytest.approx(1.0)

    def test_result_in_range(self):
        rel = np.array([1, 0, 1, 0])
        pred = np.array([0.9, 0.6, 0.3, 0.1])
        ndcg = compute_ndcg(rel, pred, k=4)
        assert 0.0 <= ndcg <= 1.0
