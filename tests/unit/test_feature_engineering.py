"""Unit tests for src/features/feature_engineering.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.config.settings import MerchantSegment
from src.features.feature_engineering import (
    AUX_FEATURE_DIM,
    compute_auxiliary_features,
    _jaccard,
    _type_token_ratio,
    _avg_sentence_length,
)


class TestAuxiliaryFeatures:
    def test_output_shape(self):
        f = compute_auxiliary_features("resp A", "resp B", MerchantSegment.VAREJO)
        assert f.shape == (AUX_FEATURE_DIM,)

    def test_output_dtype(self):
        f = compute_auxiliary_features("resp A", "resp B", MerchantSegment.MEI)
        assert f.dtype == np.float32

    def test_no_nan_or_inf(self):
        f = compute_auxiliary_features("", "", MerchantSegment.CORPORATE)
        assert np.all(np.isfinite(f))

    def test_segment_onehot_mei(self):
        f = compute_auxiliary_features("a", "b", MerchantSegment.MEI)
        # Last 3 elements are segment one-hot: MEI=index 0
        assert f[-3] == 1.0
        assert f[-2] == 0.0
        assert f[-1] == 0.0

    def test_segment_onehot_varejo(self):
        f = compute_auxiliary_features("a", "b", MerchantSegment.VAREJO)
        assert f[-3] == 0.0
        assert f[-2] == 1.0
        assert f[-1] == 0.0

    def test_segment_onehot_corporate(self):
        f = compute_auxiliary_features("a", "b", MerchantSegment.CORPORATE)
        assert f[-3] == 0.0
        assert f[-2] == 0.0
        assert f[-1] == 1.0

    def test_segment_onehot_sums_to_one(self):
        for seg in MerchantSegment:
            f = compute_auxiliary_features("a", "b", seg)
            assert abs(f[-3:].sum() - 1.0) < 1e-5

    def test_dim_constant_is_12(self):
        assert AUX_FEATURE_DIM == 12


class TestJaccard:
    def test_identical_texts(self):
        assert _jaccard("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_texts(self):
        assert _jaccard("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        result = _jaccard("a b c", "b c d")
        assert 0.0 < result < 1.0

    def test_empty_both(self):
        assert _jaccard("", "") == 0.0

    def test_result_in_range(self):
        result = _jaccard("the quick brown fox", "the lazy dog")
        assert 0.0 <= result <= 1.0


class TestTypeTokenRatio:
    def test_all_unique(self):
        result = _type_token_ratio("one two three four")
        assert result == pytest.approx(1.0)

    def test_all_repeated(self):
        result = _type_token_ratio("the the the the")
        assert result == pytest.approx(0.25)

    def test_empty(self):
        assert _type_token_ratio("") == 0.0


class TestAvgSentenceLength:
    def test_single_sentence(self):
        result = _avg_sentence_length("hello world how are you")
        assert result == pytest.approx(5.0)

    def test_empty(self):
        assert _avg_sentence_length("") == 0.0
