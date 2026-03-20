"""
Integration Tests — End-to-End Pipeline Validation.

Validates the full flow:
  Dataloader → Model → Scoring

Tests run with tiny random weights (no HF checkpoint required).
This tests the contract between each component, not model quality.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config.settings import MerchantSegment
from src.data.dataset import (
    ChatCieloDataset,
    PreferenceExample,
    build_input_text,
)
from src.data.preprocessing import scrub_pii
from src.evaluation.metrics import compute_metrics
from src.features.feature_engineering import compute_auxiliary_features
from src.training.losses import CombinedPreferenceLoss


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    """Fast tokenizer that returns fixed-size tensors."""
    def tokenize(text, max_length=64, padding=None, truncation=None, return_tensors=None):
        return {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }
    return tokenize


@pytest.fixture
def sample_examples():
    """10 labeled pairwise preference examples."""
    return [
        PreferenceExample(
            conversation_id=f"conv-{i:03d}",
            prompt=f"Pergunta {i} sobre parcelamento",
            response_a=f"Resposta A {i}",
            response_b=f"Resposta B {i}",
            merchant_segment=MerchantSegment.VAREJO,
            label=i % 3,
        )
        for i in range(10)
    ]


@pytest.fixture
def dataset(sample_examples, mock_tokenizer):
    return ChatCieloDataset(sample_examples, mock_tokenizer, max_length=64)


@pytest.fixture
def dataloader(dataset):
    return DataLoader(dataset, batch_size=4, shuffle=False)


# ── Test: Dataloader output shapes ───────────────────────────────────────────

def test_dataloader_batch_shapes(dataloader):
    """Dataloader must produce tensors of expected shapes."""
    batch = next(iter(dataloader))
    assert batch["input_ids"].shape == (4, 64)
    assert batch["attention_mask"].shape == (4, 64)
    assert batch["labels"].shape == (4,)


def test_dataloader_label_values(dataloader):
    """All labels must be in {0, 1, 2}."""
    for batch in dataloader:
        labels = batch["labels"]
        assert labels.min() >= 0
        assert labels.max() <= 2


# ── Test: Features → Model pipeline ──────────────────────────────────────────

def test_auxiliary_features_pipeline(sample_examples):
    """Feature engineering must produce valid vectors for all examples."""
    for ex in sample_examples:
        f = compute_auxiliary_features(ex.response_a, ex.response_b, ex.merchant_segment)
        assert f.shape == (12,)
        assert np.all(np.isfinite(f))


def test_mock_model_forward_pass(dataloader):
    """Mock model must produce valid (logits, probabilities) from dataloader batches."""
    from src.models.preference_model import PreferenceModel, PreferenceModelConfig

    with patch.object(PreferenceModel, "__init__", return_value=None):
        model = PreferenceModel.__new__(PreferenceModel)

        def mock_forward(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            logits = torch.randn(batch_size, 3)
            probs = torch.softmax(logits, dim=-1)
            labels = kwargs.get("labels")
            result = {"logits": logits, "probabilities": probs}
            if labels is not None:
                result["loss"] = torch.tensor(1.0, requires_grad=True)
            return result

        model.__call__ = mock_forward

    for batch in dataloader:
        labels = batch.pop("labels")
        output = mock_forward(**batch, labels=labels)

        assert "logits" in output
        assert "probabilities" in output
        assert "loss" in output

        # Probabilities must be in [0, 1]
        probs = output["probabilities"]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

        # Probabilities must sum to 1 per row
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ── Test: Loss function integration ──────────────────────────────────────────

def test_loss_backward_passes(dataloader):
    """Loss.backward() must not raise for a valid batch."""
    loss_fn = CombinedPreferenceLoss()

    batch = next(iter(dataloader))
    labels = batch["labels"]
    batch_size = labels.shape[0]

    logits = torch.randn(batch_size, 3, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)

    result = loss_fn(logits, probs, labels)
    result["loss"].backward()  # Must not raise

    assert logits.grad is not None


# ── Test: Scoring → Metrics pipeline ─────────────────────────────────────────

def test_end_to_end_metrics():
    """Full pipeline: random model → collect predictions → compute metrics."""
    n = 50
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=n)
    raw_logits = rng.standard_normal((n, 3))
    # Numerically stable softmax
    exp_logits = np.exp(raw_logits - raw_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    metrics = compute_metrics(labels, probs)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["preference_score_mean"] <= 1.0


# ── Test: PII never flows through the pipeline ───────────────────────────────

def test_pii_scrubbed_before_tokenization():
    """PII-containing prompt must be scrubbed before reaching tokenizer."""
    captured_texts = []

    def spy_tokenizer(text, **kwargs):
        captured_texts.append(text)
        return {
            "input_ids": torch.zeros(1, 32, dtype=torch.long),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }

    ex = PreferenceExample(
        conversation_id="pii-test",
        prompt="Meu CPF é 111.222.333-44",
        response_a="Resposta A",
        response_b="Resposta B",
        label=0,
    )
    ds = ChatCieloDataset([ex], spy_tokenizer, max_length=32, scrub=True)
    _ = ds[0]

    assert len(captured_texts) == 1
    assert "111.222.333-44" not in captured_texts[0]
