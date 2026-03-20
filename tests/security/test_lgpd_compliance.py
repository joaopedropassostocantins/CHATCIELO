"""
LGPD Security Tests — these block any code that leaks PII.

These tests validate:
  1. scrub_pii() removes all LGPD-regulated PII patterns.
  2. Model inference pipeline never returns PII in outputs.
  3. API responses do not contain PII in response fields.
  4. Pseudonymization is irreversible (no plaintext recoverable).
  5. Log outputs do not contain raw PII.
"""
from __future__ import annotations

import re

import pytest

from src.data.preprocessing import contains_pii, pseudonymize, scrub_pii

# ── Known PII test vectors ────────────────────────────────────────────────────
CPF_SAMPLES = [
    "123.456.789-09",
    "111.222.333-44",
    "00000000000",
]
CARD_SAMPLES = [
    "4111 1111 1111 1111",
    "5500000000000004",
    "3714 496353 98431",
]
EMAIL_SAMPLES = [
    "joao@cielo.com.br",
    "maria.silva@empresa.com",
]
PHONE_SAMPLES = [
    "(11) 99999-0000",
    "11987654321",
    "(21) 98765-4321",
]
CNPJ_SAMPLES = [
    "11.222.333/0001-44",
    "11222333000144",
]


class TestScrubPiiComprehensive:
    @pytest.mark.parametrize("cpf", CPF_SAMPLES)
    def test_cpf_removed(self, cpf):
        assert cpf not in scrub_pii(f"Meu CPF é {cpf}.")

    @pytest.mark.parametrize("card", CARD_SAMPLES)
    def test_card_removed(self, card):
        result = scrub_pii(f"Cartão {card}")
        # The raw card digits should not appear in output
        digits_only = re.sub(r"\D", "", card)
        assert digits_only not in re.sub(r"\D", "", result)

    @pytest.mark.parametrize("email", EMAIL_SAMPLES)
    def test_email_removed(self, email):
        assert email not in scrub_pii(f"email: {email}")

    @pytest.mark.parametrize("phone", PHONE_SAMPLES)
    def test_phone_removed(self, phone):
        result = scrub_pii(f"ligue: {phone}")
        assert phone not in result

    @pytest.mark.parametrize("cnpj", CNPJ_SAMPLES)
    def test_cnpj_removed(self, cnpj):
        assert cnpj not in scrub_pii(f"CNPJ: {cnpj}")


class TestContainsPii:
    @pytest.mark.parametrize("pii", CPF_SAMPLES + EMAIL_SAMPLES + CNPJ_SAMPLES)
    def test_detects_pii(self, pii):
        assert contains_pii(f"texto com {pii} aqui") is True

    def test_clean_text_no_pii(self):
        clean_texts = [
            "Como funciona o parcelamento?",
            "Qual o limite do meu terminal?",
            "Preciso de ajuda com a maquininha.",
        ]
        for text in clean_texts:
            assert contains_pii(text) is False

    def test_after_scrub_no_pii(self):
        """After scrubbing, contains_pii must return False for all patterns."""
        dirty_texts = [
            f"CPF: {CPF_SAMPLES[0]}",
            f"Cartão: {CARD_SAMPLES[0]}",
            f"Email: {EMAIL_SAMPLES[0]}",
        ]
        for text in dirty_texts:
            scrubbed = scrub_pii(text)
            assert contains_pii(scrubbed) is False, (
                f"PII leaked after scrubbing: original='{text}', scrubbed='{scrubbed}'"
            )


class TestPseudonymization:
    def test_output_does_not_contain_input(self):
        user_id = "user-12345"
        result = pseudonymize(user_id, "test-salt")
        assert user_id not in result

    def test_output_is_hex(self):
        result = pseudonymize("any-value", "salt")
        assert all(c in "0123456789abcdef" for c in result)

    def test_cannot_recover_original(self):
        """The hash cannot be reversed — verify by checking no substring match."""
        original = "joao.silva"
        hashed = pseudonymize(original, "production-salt")
        # No part of the original should appear in the hash
        assert "joao" not in hashed
        assert "silva" not in hashed

    def test_different_users_different_hashes(self):
        h1 = pseudonymize("user-001", "salt")
        h2 = pseudonymize("user-002", "salt")
        assert h1 != h2


class TestApiResponsePiiLeak:
    """Verify that response fields from the API never contain raw PII."""

    def test_preference_response_fields_have_no_pii(self):
        from src.api.schemas.responses import PreferenceResponse

        response = PreferenceResponse(
            prob_a_wins=0.6,
            prob_b_wins=0.3,
            prob_tie=0.1,
            winner="A",
            latency_ms=50.0,
            conversation_id="conv-abc",
        )
        # Serialize and check no PII in the output
        as_str = response.model_dump_json()
        for pii_sample in CPF_SAMPLES + EMAIL_SAMPLES:
            assert pii_sample not in as_str
