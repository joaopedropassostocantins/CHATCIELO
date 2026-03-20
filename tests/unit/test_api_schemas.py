"""Unit tests for API request/response schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.requests import BatchPreferenceRequest, FeedbackRequest, PreferenceRequest
from src.api.schemas.responses import PreferenceResponse


class TestPreferenceRequest:
    def test_valid_request(self):
        r = PreferenceRequest(
            prompt="Qual o limite?",
            response_a="R$ 1.000",
            response_b="R$ 2.000",
            merchant_segment="MEI",
        )
        assert r.merchant_segment == "MEI"

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            PreferenceRequest(prompt="   ", response_a="A", response_b="B")

    def test_invalid_segment_rejected(self):
        with pytest.raises(ValidationError):
            PreferenceRequest(
                prompt="P",
                response_a="A",
                response_b="B",
                merchant_segment="UNKNOWN",
            )

    def test_default_segment_is_varejo(self):
        r = PreferenceRequest(prompt="P", response_a="A", response_b="B")
        assert r.merchant_segment == "VAREJO"

    def test_max_length_prompt(self):
        PreferenceRequest(prompt="x" * 4096, response_a="A", response_b="B")

    def test_exceeds_max_length_prompt(self):
        with pytest.raises(ValidationError):
            PreferenceRequest(prompt="x" * 4097, response_a="A", response_b="B")


class TestBatchPreferenceRequest:
    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            BatchPreferenceRequest(items=[])

    def test_too_many_items_rejected(self):
        items = [
            {"prompt": "P", "response_a": "A", "response_b": "B"}
            for _ in range(33)
        ]
        with pytest.raises(ValidationError):
            BatchPreferenceRequest(items=items)


class TestPreferenceResponse:
    def test_probabilities_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PreferenceResponse(
                prob_a_wins=1.5,
                prob_b_wins=0.0,
                prob_tie=0.0,
                winner="A",
                latency_ms=10.0,
            )
