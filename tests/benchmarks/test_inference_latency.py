"""
Performance Benchmarks — P99 latency must be < 300ms.

Criterion: benchmark.stats["max"] < 0.300 (seconds)

Run with:
    pytest tests/benchmarks/ --benchmark-only --benchmark-histogram
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config.settings import MerchantSegment
from src.data.dataset import build_input_text
from src.data.preprocessing import scrub_pii
from src.evaluation.metrics import compute_metrics
from src.features.feature_engineering import compute_auxiliary_features


# ── Fixture: fast predictor mock ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_predictor():
    """Predictor that returns deterministic results without model loading."""
    from src.inference.predictor import Predictor, PreferenceResult

    predictor = MagicMock(spec=Predictor)

    def _predict(prompt, response_a, response_b, segment=MerchantSegment.VAREJO):
        return PreferenceResult(
            prob_a_wins=0.6,
            prob_b_wins=0.3,
            prob_tie=0.1,
            winner="A",
            latency_ms=10.0,
        )

    predictor.predict = _predict
    return predictor


# ── Benchmark: PII scrubbing ──────────────────────────────────────────────────

def test_scrub_pii_latency(benchmark):
    """scrub_pii must process a typical message in < 5ms."""
    text = (
        "Preciso de ajuda com meu CPF 123.456.789-09 "
        "e cartão 4111 1111 1111 1111. "
        "Email: joao@cielo.com.br"
    )
    result = benchmark(scrub_pii, text)
    assert "123.456.789-09" not in result
    # P99 criterion (enforced by pytest-benchmark internally via --benchmark-max-time)
    assert benchmark.stats["max"] < 0.005  # 5ms


# ── Benchmark: Feature engineering ───────────────────────────────────────────

def test_feature_engineering_latency(benchmark):
    """compute_auxiliary_features must complete in < 10ms."""
    result = benchmark(
        compute_auxiliary_features,
        "Qual o limite de parcelamento para lojistas MEI?",
        "O limite é R$ 1.000 por transação.",
        "Você pode parcelar em até 12x sem juros.",
        MerchantSegment.MEI,
    )
    assert result.shape == (12,)
    assert benchmark.stats["max"] < 0.010  # 10ms


# ── Benchmark: Input text construction ───────────────────────────────────────

def test_build_input_text_latency(benchmark):
    """build_input_text must complete in < 1ms."""
    result = benchmark(
        build_input_text,
        "Qual o prazo de liquidação?",
        "O prazo é D+1.",
        "O prazo é D+2 para débito.",
        MerchantSegment.VAREJO,
    )
    assert "<prompt>" in result
    assert benchmark.stats["max"] < 0.001  # 1ms


# ── Benchmark: Full inference mock (P99 < 300ms) ─────────────────────────────

def test_inference_end_to_end_latency(benchmark, mock_predictor):
    """Full inference pipeline must have P99 < 300ms.

    This tests the orchestration overhead (PII scrub + feature eng + predict).
    In production, the model forward pass is the dominant cost.
    """

    def run_inference():
        prompt = "Como funciona o limite de crédito para Corporate?"
        response_a = "O limite é calculado com base no faturamento anual."
        response_b = "Você pode solicitar aumento de limite pelo portal."
        return mock_predictor.predict(
            prompt=scrub_pii(prompt),
            response_a=scrub_pii(response_a),
            response_b=scrub_pii(response_b),
            segment=MerchantSegment.CORPORATE,
        )

    result = benchmark(run_inference)
    assert result.winner in {"A", "B", "tie"}

    # P99 latency criterion
    assert benchmark.stats["max"] < 0.300, (
        f"P99 latency {benchmark.stats['max']*1000:.1f}ms exceeds 300ms threshold. "
        "Optimization required before merging."
    )


# ── Benchmark: Batch metrics computation ─────────────────────────────────────

def test_metrics_computation_latency(benchmark):
    """compute_metrics for 1000 examples must complete in < 100ms."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=1000)
    raw = rng.random((1000, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)

    result = benchmark(compute_metrics, labels, probs)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert benchmark.stats["max"] < 0.100  # 100ms
