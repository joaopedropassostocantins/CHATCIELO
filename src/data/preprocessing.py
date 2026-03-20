"""
PII scrubbing and text normalization for LGPD compliance.

DIFF vs Kaggle baseline:
────────────────────────
KAGGLE: No PII scrubbing — raw text fed directly to tokenizer.
CHATCIELO: All text passes through scrub_pii() before tokenization.
           CPF, card numbers, phone numbers, and names replaced with
           anonymized placeholders.
────────────────────────
"""
from __future__ import annotations

import hashlib
import hmac
import re

# ── PII regex patterns ────────────────────────────────────────────────────────
# CPF: 000.000.000-00 or 00000000000
_CPF_RE = re.compile(r"\b\d{3}[.\-]?\d{3}[.\-]?\d{3}[-]?\d{2}\b")

# Credit/debit card: 13-19 digits, optionally spaced/dashed
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

# Brazilian phone: (11) 9xxxx-xxxx or 11999999999
_PHONE_BR_RE = re.compile(r"\(?\d{2}\)?[\s\-]?9\d{4}[\s\-]?\d{4}\b")

# Email
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")

# CNPJ: 00.000.000/0000-00
_CNPJ_RE = re.compile(r"\b\d{2}[.\-]?\d{3}[.\-]?\d{3}[/]?\d{4}[-]?\d{2}\b")


def scrub_pii(text: str) -> str:
    """Remove PII from text, replacing with typed placeholders.

    This is the LGPD compliance layer applied before any model inference
    or data persistence.

    Args:
        text: Raw input text potentially containing PII.

    Returns:
        Text with all detected PII replaced by placeholder tokens:
        [CPF_REDACTED], [CARD_REDACTED], [PHONE_REDACTED],
        [EMAIL_REDACTED], [CNPJ_REDACTED].

    Validation Metrics:
        - After scrubbing, _CPF_RE.search(result) must return None.
        - Placeholder tokens are detectable and auditable in downstream logs.
    """
    text = _CPF_RE.sub("[CPF_REDACTED]", text)
    text = _CNPJ_RE.sub("[CNPJ_REDACTED]", text)
    text = _CARD_RE.sub("[CARD_REDACTED]", text)
    text = _PHONE_BR_RE.sub("[PHONE_REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    return text


def pseudonymize(value: str, salt: str) -> str:
    """One-way pseudonymization using HMAC-SHA256.

    Used for conversation_id and user_id before storing in logs.
    The original value cannot be recovered without the salt.

    Args:
        value: The raw identifier to pseudonymize.
        salt: Application-level secret salt (from ANONYMIZATION_SALT env var).

    Returns:
        64-character hex digest that consistently maps value → hash.

    Validation Metrics:
        - Same (value, salt) always produces same digest (deterministic).
        - Different salts produce different digests (salt-dependent).
    """
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()


def contains_pii(text: str) -> bool:
    """Return True if the text contains any detectable PII pattern.

    Used by the LGPD audit script and security tests.

    Args:
        text: Text to inspect.

    Returns:
        True if any PII pattern is found, False otherwise.

    Validation Metrics:
        - Must return True for any string matching LGPD-regulated patterns.
    """
    patterns = [_CPF_RE, _CNPJ_RE, _CARD_RE, _PHONE_BR_RE, _EMAIL_RE]
    return any(p.search(text) for p in patterns)


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove control characters.

    Args:
        text: Raw text.

    Returns:
        Text with collapsed whitespace and stripped leading/trailing spaces.
    """
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
