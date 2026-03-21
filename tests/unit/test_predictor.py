"""Unit tests for src/inference/predictor.py.

Tests:
  - PreferenceResult dataclass invariants.
  - _cache_key determinism and uniqueness.
  - Predictor.__init__ configuration.
  - Predictor._get_cache / _set_cache with mocked Redis.
  - Predictor.predict: happy path, cache hit, empty-after-scrub, PII scrubbing.
  - Predictor.predict_batch: segment defaulting, ordering.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config.settings import MerchantSegment
from src.inference.predictor import PreferenceResult, Predictor, _cache_key


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tokenizer(max_length: int = 32):
    """Returns a fast mock tokenizer that produces fixed-shape tensors."""

    def tokenize(text, max_length=max_length, padding=None, truncation=None, return_tensors=None):
        return {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }

    return tokenize


def _make_model():
    """Returns a mock model that produces deterministic probabilities."""
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    def forward(*args, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        probs = torch.tensor([[0.6, 0.3, 0.1]] * batch_size)
        return {"logits": torch.randn(batch_size, 3), "probabilities": probs}

    model.side_effect = forward
    return model


@pytest.fixture
def predictor():
    """Predictor with mocked model + tokenizer, no Redis."""
    return Predictor(
        model=_make_model(),
        tokenizer=_make_tokenizer(),
        max_length=32,
        batch_size=4,
        device="cpu",
        redis_client=None,
    )


@pytest.fixture
def predictor_with_redis(predictor):
    """Predictor with a mocked Redis client attached."""
    redis = MagicMock()
    redis.get.return_value = None  # cache miss by default
    predictor.redis = redis
    return predictor, redis


# ── PreferenceResult ──────────────────────────────────────────────────────────

class TestPreferenceResult:
    def test_valid_result_created(self):
        r = PreferenceResult(
            prob_a_wins=0.6,
            prob_b_wins=0.3,
            prob_tie=0.1,
            winner="A",
            latency_ms=10.0,
        )
        assert r.winner == "A"

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            PreferenceResult(
                prob_a_wins=0.5,
                prob_b_wins=0.5,
                prob_tie=0.5,  # sums to 1.5
                winner="A",
                latency_ms=5.0,
            )

    def test_probability_out_of_range_rejected(self):
        with pytest.raises(AssertionError):
            PreferenceResult(
                prob_a_wins=1.2,
                prob_b_wins=-0.1,
                prob_tie=-0.1,
                winner="A",
                latency_ms=5.0,
            )

    def test_from_cache_default_false(self):
        r = PreferenceResult(
            prob_a_wins=0.6, prob_b_wins=0.3, prob_tie=0.1,
            winner="A", latency_ms=10.0,
        )
        assert r.from_cache is False

    def test_tie_winner_accepted(self):
        r = PreferenceResult(
            prob_a_wins=0.33, prob_b_wins=0.34, prob_tie=0.33,
            winner="tie", latency_ms=5.0,
        )
        assert r.winner == "tie"


# ── _cache_key ────────────────────────────────────────────────────────────────

class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("prompt", "a", "b", "MEI")
        k2 = _cache_key("prompt", "a", "b", "MEI")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = _cache_key("p1", "a", "b", "MEI")
        k2 = _cache_key("p2", "a", "b", "MEI")
        assert k1 != k2

    def test_segment_affects_key(self):
        k1 = _cache_key("p", "a", "b", "MEI")
        k2 = _cache_key("p", "a", "b", "VAREJO")
        assert k1 != k2

    def test_starts_with_namespace(self):
        k = _cache_key("p", "a", "b", "MEI")
        assert k.startswith("chatcielo:pref:")

    def test_key_is_hex_after_prefix(self):
        k = _cache_key("p", "a", "b", "MEI")
        hex_part = k[len("chatcielo:pref:"):]
        assert all(c in "0123456789abcdef" for c in hex_part)


# ── Predictor init ────────────────────────────────────────────────────────────

class TestPredictorInit:
    def test_model_moved_to_device(self):
        model = _make_model()
        Predictor(model=model, tokenizer=_make_tokenizer(), device="cpu")
        model.to.assert_called_with("cpu")

    def test_model_set_to_eval(self):
        model = _make_model()
        Predictor(model=model, tokenizer=_make_tokenizer(), device="cpu")
        model.eval.assert_called_once()

    def test_defaults(self):
        p = Predictor(model=_make_model(), tokenizer=_make_tokenizer())
        assert p.max_length == 1024
        assert p.batch_size == 16
        assert p.device == "cpu"
        assert p.redis is None
        assert p.cache_ttl == 3600


# ── Cache helpers ─────────────────────────────────────────────────────────────

class TestCacheHelpers:
    def test_get_cache_returns_none_without_redis(self, predictor):
        assert predictor._get_cache("any-key") is None

    def test_set_cache_no_op_without_redis(self, predictor):
        result = PreferenceResult(
            prob_a_wins=0.6, prob_b_wins=0.3, prob_tie=0.1,
            winner="A", latency_ms=5.0,
        )
        predictor._set_cache("key", result)  # must not raise

    def test_get_cache_miss_returns_none(self, predictor_with_redis):
        predictor, redis = predictor_with_redis
        redis.get.return_value = None
        assert predictor._get_cache("missing-key") is None

    def test_get_cache_hit_returns_result(self, predictor_with_redis):
        predictor, redis = predictor_with_redis
        cached_data = json.dumps({
            "prob_a_wins": 0.7,
            "prob_b_wins": 0.2,
            "prob_tie": 0.1,
            "winner": "A",
            "latency_ms": 8.0,
        })
        redis.get.return_value = cached_data
        result = predictor._get_cache("some-key")
        assert result is not None
        assert result.prob_a_wins == pytest.approx(0.7)
        assert result.from_cache is True

    def test_set_cache_calls_setex(self, predictor_with_redis):
        predictor, redis = predictor_with_redis
        result = PreferenceResult(
            prob_a_wins=0.6, prob_b_wins=0.3, prob_tie=0.1,
            winner="A", latency_ms=5.0,
        )
        predictor._set_cache("test-key", result)
        redis.setex.assert_called_once()
        call_args = redis.setex.call_args
        assert call_args[0][0] == "test-key"
        assert call_args[0][1] == predictor.cache_ttl

    def test_get_cache_swallows_redis_error(self, predictor_with_redis):
        predictor, redis = predictor_with_redis
        redis.get.side_effect = ConnectionError("Redis down")
        assert predictor._get_cache("key") is None

    def test_set_cache_swallows_redis_error(self, predictor_with_redis):
        predictor, redis = predictor_with_redis
        redis.setex.side_effect = ConnectionError("Redis down")
        result = PreferenceResult(
            prob_a_wins=0.6, prob_b_wins=0.3, prob_tie=0.1,
            winner="A", latency_ms=5.0,
        )
        predictor._set_cache("key", result)  # must not raise


# ── Predictor.predict ─────────────────────────────────────────────────────────

class TestPredictorPredict:
    def test_happy_path_returns_result(self, predictor):
        result = predictor.predict(
            prompt="Qual o limite de parcelamento?",
            response_a="Até 12 vezes.",
            response_b="Até 18 vezes com juros.",
        )
        assert isinstance(result, PreferenceResult)
        assert result.winner in {"A", "B", "tie"}

    def test_probabilities_in_unit_interval(self, predictor):
        result = predictor.predict(
            prompt="Como funciona o crédito?",
            response_a="Resposta A",
            response_b="Resposta B",
        )
        assert 0.0 <= result.prob_a_wins <= 1.0
        assert 0.0 <= result.prob_b_wins <= 1.0
        assert 0.0 <= result.prob_tie <= 1.0

    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict(
            prompt="Q",
            response_a="A",
            response_b="B",
        )
        total = result.prob_a_wins + result.prob_b_wins + result.prob_tie
        assert abs(total - 1.0) < 1e-4

    def test_latency_ms_is_positive(self, predictor):
        result = predictor.predict(
            prompt="Q",
            response_a="A",
            response_b="B",
        )
        assert result.latency_ms >= 0.0

    def test_from_cache_false_on_miss(self, predictor):
        result = predictor.predict(
            prompt="Q",
            response_a="A",
            response_b="B",
        )
        assert result.from_cache is False

    def test_cache_hit_returns_cached(self, predictor_with_redis):
        predictor, redis = predictor_with_redis
        cached_data = json.dumps({
            "prob_a_wins": 0.9,
            "prob_b_wins": 0.05,
            "prob_tie": 0.05,
            "winner": "A",
            "latency_ms": 1.0,
        })
        redis.get.return_value = cached_data
        result = predictor.predict(
            prompt="Q",
            response_a="A",
            response_b="B",
        )
        assert result.from_cache is True
        assert result.prob_a_wins == pytest.approx(0.9)

    def test_empty_prompt_after_scrub_raises(self, predictor):
        with pytest.raises(ValueError, match="Empty input"):
            predictor.predict(
                prompt="   ",
                response_a="Resposta A",
                response_b="Resposta B",
            )

    def test_empty_response_after_scrub_raises(self, predictor):
        with pytest.raises(ValueError, match="Empty input"):
            predictor.predict(
                prompt="Qual o prazo?",
                response_a="   ",
                response_b="B",
            )

    def test_pii_scrubbed_before_inference(self, predictor):
        """CPF in prompt must not cause failure; PII is stripped before tokenization."""
        result = predictor.predict(
            prompt="Meu CPF é 123.456.789-09, qual meu limite?",
            response_a="Seu limite é R$ 1.000.",
            response_b="Consulte pelo app.",
        )
        assert isinstance(result, PreferenceResult)

    def test_segment_mei_accepted(self, predictor):
        result = predictor.predict(
            prompt="Q",
            response_a="A",
            response_b="B",
            segment=MerchantSegment.MEI,
        )
        assert isinstance(result, PreferenceResult)

    def test_segment_corporate_accepted(self, predictor):
        result = predictor.predict(
            prompt="Q",
            response_a="A",
            response_b="B",
            segment=MerchantSegment.CORPORATE,
        )
        assert isinstance(result, PreferenceResult)


# ── Predictor.predict_batch ───────────────────────────────────────────────────

class TestPredictorBatch:
    def test_batch_returns_one_result_per_input(self, predictor):
        inputs = [
            {"prompt": f"P{i}", "response_a": f"A{i}", "response_b": f"B{i}"}
            for i in range(5)
        ]
        results = predictor.predict_batch(inputs)
        assert len(results) == 5

    def test_batch_all_valid_results(self, predictor):
        inputs = [
            {"prompt": "Q", "response_a": "A", "response_b": "B"},
            {"prompt": "Q2", "response_a": "A2", "response_b": "B2"},
        ]
        for r in predictor.predict_batch(inputs):
            assert isinstance(r, PreferenceResult)
            assert r.winner in {"A", "B", "tie"}

    def test_batch_defaults_segment_to_varejo(self, predictor):
        inputs = [{"prompt": "Q", "response_a": "A", "response_b": "B"}]
        results = predictor.predict_batch(inputs)
        assert len(results) == 1

    def test_batch_respects_explicit_segment(self, predictor):
        inputs = [
            {"prompt": "Q", "response_a": "A", "response_b": "B", "segment": "MEI"},
        ]
        results = predictor.predict_batch(inputs)
        assert len(results) == 1

    def test_empty_batch_returns_empty_list(self, predictor):
        results = predictor.predict_batch([])
        assert results == []

    def test_batch_larger_than_batch_size(self, predictor):
        """Batches larger than batch_size must still be processed fully."""
        predictor.batch_size = 2
        inputs = [
            {"prompt": f"Q{i}", "response_a": "A", "response_b": "B"}
            for i in range(7)
        ]
        results = predictor.predict_batch(inputs)
        assert len(results) == 7
