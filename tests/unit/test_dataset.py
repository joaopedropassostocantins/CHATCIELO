"""Unit tests for src/data/dataset.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config.settings import MerchantSegment
from src.data.dataset import (
    LABEL_MAP,
    NUM_LABELS,
    ChatCieloDataset,
    PreferenceExample,
    build_input_text,
)


class TestBuildInputText:
    def test_contains_segment_prefix(self):
        text = build_input_text("prompt", "a", "b", MerchantSegment.MEI)
        assert "[MEI]" in text

    def test_contains_all_tags(self):
        text = build_input_text("p", "a", "b", MerchantSegment.VAREJO)
        assert "<prompt>" in text
        assert "<response_a>" in text
        assert "<response_b>" in text

    def test_segment_conditioning_varejo(self):
        text = build_input_text("p", "a", "b", MerchantSegment.VAREJO)
        assert "[VAREJO]" in text

    def test_segment_conditioning_corporate(self):
        text = build_input_text("p", "a", "b", MerchantSegment.CORPORATE)
        assert "[CORPORATE]" in text

    def test_prompt_content_present(self):
        text = build_input_text("minha pergunta", "resp a", "resp b")
        assert "minha pergunta" in text
        assert "resp a" in text
        assert "resp b" in text


class TestLabelMap:
    def test_model_a_maps_to_zero(self):
        assert LABEL_MAP["model_a"] == 0

    def test_model_b_maps_to_one(self):
        assert LABEL_MAP["model_b"] == 1

    def test_tie_maps_to_two(self):
        assert LABEL_MAP["tie"] == 2
        assert LABEL_MAP["tie (bothbad)"] == 2

    def test_num_labels_is_three(self):
        assert NUM_LABELS == 3


class TestChatCieloDataset:
    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.return_value = {
            "input_ids": torch.zeros(1, 32, dtype=torch.long),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        return tok

    @pytest.fixture
    def example(self):
        return PreferenceExample(
            conversation_id="conv-001",
            prompt="Como parcelar?",
            response_a="Em até 12x.",
            response_b="Em até 18x com juros.",
            merchant_segment=MerchantSegment.MEI,
            label=0,
        )

    def test_len(self, mock_tokenizer, example):
        ds = ChatCieloDataset([example, example], mock_tokenizer, max_length=32)
        assert len(ds) == 2

    def test_getitem_returns_tensors(self, mock_tokenizer, example):
        ds = ChatCieloDataset([example], mock_tokenizer, max_length=32)
        item = ds[0]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

    def test_label_present_when_training(self, mock_tokenizer, example):
        ds = ChatCieloDataset([example], mock_tokenizer)
        item = ds[0]
        assert "labels" in item
        assert item["labels"].item() == 0

    def test_no_label_when_inference(self, mock_tokenizer):
        ex = PreferenceExample(
            conversation_id="inf-001",
            prompt="Pergunta",
            response_a="A",
            response_b="B",
            label=None,
        )
        ds = ChatCieloDataset([ex], mock_tokenizer)
        item = ds[0]
        assert "labels" not in item

    def test_pii_scrubbing_applied(self, example):
        """PII in prompt must not reach tokenizer input."""
        dirty_example = PreferenceExample(
            conversation_id="pii-001",
            prompt="CPF: 111.222.333-44",
            response_a="OK",
            response_b="Não",
            label=1,
        )
        calls = []

        def mock_tok(text, **kwargs):
            calls.append(text)
            return {
                "input_ids": torch.zeros(1, 32, dtype=torch.long),
                "attention_mask": torch.ones(1, 32, dtype=torch.long),
            }

        ds = ChatCieloDataset([dirty_example], mock_tok, scrub=True)
        ds[0]
        assert "111.222.333-44" not in calls[0]
