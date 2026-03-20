"""
Feature engineering for merchant-aware preference ranking.

DIFF vs Kaggle baseline:
────────────────────────
KAGGLE: No auxiliary features — pure text-to-label mapping.
CHATCIELO: Adds structured merchant features injected as model prefix:
    + Merchant segment one-hot (MEI/VAREJO/CORPORATE)
    + Conversation turn count
    + Response length delta (|len_a - len_b|)
    + Lexical overlap score (Jaccard between responses)
    + Average sentence length per response (proxy for formality)
────────────────────────

Mathematical definition:
    Given (p, r_a, r_b, s) where p=prompt, r_a/r_b=responses, s=segment:

    f_struct(r) = [len(r), avg_sent_len(r), type_token_ratio(r)]
    f_compare   = [len_delta, jaccard(r_a, r_b), cosine_sim(e_a, e_b)]
    f_segment   = one_hot(s, |Segments|)

    x_aux = concat([f_struct(r_a), f_struct(r_b), f_compare, f_segment])
    dim(x_aux) = 3 + 3 + 3 + 3 = 12
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.config.settings import MerchantSegment

SEGMENT_INDEX: Dict[MerchantSegment, int] = {
    MerchantSegment.MEI: 0,
    MerchantSegment.VAREJO: 1,
    MerchantSegment.CORPORATE: 2,
}

AUX_FEATURE_DIM = 12


def _avg_sentence_length(text: str) -> float:
    """Compute average word count per sentence (formality proxy).

    Args:
        text: Input text.

    Returns:
        Mean sentence length in words. Returns 0.0 for empty text.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0
    return float(np.mean([len(s.split()) for s in sentences]))


def _type_token_ratio(text: str) -> float:
    """Compute Type-Token Ratio (lexical diversity proxy).

    Args:
        text: Input text.

    Returns:
        |unique_words| / |total_words|. Returns 0.0 for empty text.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard index in [0, 1]. Returns 0.0 if both are empty.

    Validation Metrics:
        - Result must be in [0, 1].
    """
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _struct_features(text: str) -> List[float]:
    """Extract structural features from a single response.

    Args:
        text: Response text.

    Returns:
        List of 3 floats: [length, avg_sentence_length, type_token_ratio].
    """
    return [
        float(len(text)),
        _avg_sentence_length(text),
        _type_token_ratio(text),
    ]


def compute_auxiliary_features(
    response_a: str,
    response_b: str,
    segment: MerchantSegment,
) -> np.ndarray:
    """Compute the 12-dimensional auxiliary feature vector.

    Used as optional input to a late-fusion layer on top of the cross-encoder.

    Args:
        response_a: First candidate response (PII-scrubbed).
        response_b: Second candidate response (PII-scrubbed).
        segment: Cielo merchant segment.

    Returns:
        np.ndarray of shape (12,) with dtype float32.
        Layout: [struct_a(3), struct_b(3), compare(3), segment_onehot(3)]

    Validation Metrics:
        - Output shape must be (AUX_FEATURE_DIM,) = (12,).
        - All values must be finite (no NaN / Inf).
        - segment_onehot sums to 1.
    """
    f_a = _struct_features(response_a)
    f_b = _struct_features(response_b)

    len_delta = abs(len(response_a) - len(response_b)) / max(
        len(response_a) + len(response_b), 1
    )
    jaccard = _jaccard(response_a, response_b)
    # Sentence count difference as formality signal
    sent_delta = abs(_avg_sentence_length(response_a) - _avg_sentence_length(response_b))

    f_compare = [len_delta, jaccard, sent_delta]

    segment_onehot = [0.0, 0.0, 0.0]
    segment_onehot[SEGMENT_INDEX[segment]] = 1.0

    features = np.array(
        f_a + f_b + f_compare + segment_onehot,
        dtype=np.float32,
    )

    # Clip to finite range
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    assert features.shape == (AUX_FEATURE_DIM,), f"Expected ({AUX_FEATURE_DIM},), got {features.shape}"
    return features
