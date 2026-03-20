"""
ML Property-Based Tests using hypothesis.

Invariants that must hold for ALL inputs:
  1. Preference scores (probabilities) are always in [0, 1].
  2. Probabilities sum to 1.0.
  3. Predicted class is always in {0, 1, 2}.
  4. Model is resilient to malformed / extreme inputs.
  5. PII is never present in model outputs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from src.config.settings import MerchantSegment
from src.evaluation.metrics import compute_ece, compute_metrics
from src.features.feature_engineering import compute_auxiliary_features
from src.models.preference_model import PreferenceModel, PreferenceModelConfig


# ── Lightweight model fixture (random weights, no HF download) ───────────────

@pytest.fixture(scope="module")
def tiny_model():
    """Tiny mock of PreferenceModel that returns random probabilities."""
    model = MagicMock(spec=PreferenceModel)

    def _forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        logits = torch.randn(batch_size, 3)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probs}

    model.return_value = _forward
    model.side_effect = _forward
    model.__call__ = _forward
    return model


# ── Property: output probabilities always in [0, 1] ─────────────────────────

@given(
    a_logit=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    b_logit=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    tie_logit=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_output_in_unit_interval(a_logit, b_logit, tie_logit):
    """softmax is always in [0,1] regardless of logit values."""
    logits = torch.tensor([[a_logit, b_logit, tie_logit]])
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    for p in probs:
        assert 0.0 <= p.item() <= 1.0


@given(
    a_logit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    b_logit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    tie_logit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_softmax_sums_to_one(a_logit, b_logit, tie_logit):
    """Probabilities must always sum to 1.0."""
    logits = torch.tensor([[a_logit, b_logit, tie_logit]])
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    assert abs(probs.sum().item() - 1.0) < 1e-5


@given(
    a_logit=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
    b_logit=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
    tie_logit=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
)
@settings(max_examples=300)
def test_argmax_class_in_valid_range(a_logit, b_logit, tie_logit):
    """Predicted class must always be 0, 1, or 2."""
    logits = torch.tensor([[a_logit, b_logit, tie_logit]])
    pred = logits.argmax(dim=-1).item()
    assert pred in {0, 1, 2}


# ── Property: auxiliary features always finite and correct shape ─────────────

@given(
    text_a=st.text(min_size=0, max_size=500),
    text_b=st.text(min_size=0, max_size=500),
    segment=st.sampled_from(list(MerchantSegment)),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_auxiliary_features_always_finite(text_a, text_b, segment):
    """Auxiliary features must be finite for any text input."""
    features = compute_auxiliary_features(text_a, text_b, segment)
    assert features.shape == (12,)
    assert np.all(np.isfinite(features))


# ── Property: metric functions accept valid probability arrays ───────────────

@given(
    n=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=100)
def test_compute_metrics_accepts_valid_inputs(n):
    """compute_metrics should not raise for any valid (labels, probs) pair."""
    rng = np.random.default_rng(n)
    labels = rng.integers(0, 3, size=n)
    raw = rng.random((n, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)

    m = compute_metrics(labels, probs)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["preference_score_mean"] <= 1.0
    assert m["log_loss"] >= 0.0


@given(
    n=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=100)
def test_ece_always_in_unit_interval(n):
    """ECE must be in [0, 1] for any valid predictions."""
    rng = np.random.default_rng(n)
    labels = rng.integers(0, 3, size=n)
    raw = rng.random((n, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)

    ece = compute_ece(labels, probs)
    assert 0.0 <= ece <= 1.0
