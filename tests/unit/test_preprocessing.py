"""Unit tests for src/data/preprocessing.py."""
from __future__ import annotations

import pytest

from src.data.preprocessing import contains_pii, normalize_text, pseudonymize, scrub_pii


class TestScrubPii:
    def test_cpf_dotted(self):
        result = scrub_pii("Meu CPF é 123.456.789-09.")
        assert "123.456.789-09" not in result
        assert "[CPF_REDACTED]" in result

    def test_cpf_plain(self):
        result = scrub_pii("CPF 12345678909 cadastrado.")
        assert "12345678909" not in result

    def test_card_number(self):
        result = scrub_pii("Cartão: 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in result
        assert "[CARD_REDACTED]" in result

    def test_email(self):
        result = scrub_pii("Contato: joao@cielo.com.br")
        assert "joao@cielo.com.br" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_phone_br(self):
        result = scrub_pii("Ligue: (11) 99999-0000")
        assert "(11) 99999-0000" not in result
        assert "[PHONE_REDACTED]" in result

    def test_clean_text_unchanged(self):
        text = "Como funciona o parcelamento no débito?"
        assert scrub_pii(text) == text

    def test_multiple_pii_types(self):
        text = "CPF: 111.222.333-44 email: a@b.com"
        result = scrub_pii(text)
        assert "111.222.333-44" not in result
        assert "a@b.com" not in result

    def test_empty_string(self):
        assert scrub_pii("") == ""


class TestContainsPii:
    def test_detects_cpf(self):
        assert contains_pii("CPF: 111.222.333-44") is True

    def test_clean_text(self):
        assert contains_pii("Texto limpo sem PII") is False

    def test_after_scrub(self):
        raw = "CPF: 111.222.333-44"
        assert contains_pii(scrub_pii(raw)) is False


class TestPseudonymize:
    def test_deterministic(self):
        a = pseudonymize("user-123", "salt-abc")
        b = pseudonymize("user-123", "salt-abc")
        assert a == b

    def test_different_salts_produce_different_hashes(self):
        a = pseudonymize("user-123", "salt-abc")
        b = pseudonymize("user-123", "salt-xyz")
        assert a != b

    def test_output_is_64_chars(self):
        result = pseudonymize("anything", "salt")
        assert len(result) == 64

    def test_different_values_produce_different_hashes(self):
        a = pseudonymize("user-1", "salt")
        b = pseudonymize("user-2", "salt")
        assert a != b


class TestNormalizeText:
    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_edges(self):
        assert normalize_text("  hello  ") == "hello"

    def test_removes_control_chars(self):
        assert normalize_text("hello\x00world") == "helloworld"
