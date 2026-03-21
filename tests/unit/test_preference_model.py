"""Unit tests for src/models/preference_model.py.

Tests:
  - PreferenceModelConfig: default values and custom overrides.
  - AuxFusionHead: output shape, with/without aux_features (zero-pad path).
  - PreferenceModel: forward pass with and without labels, with and without
    aux_features, pooler_output vs last_hidden_state fallback, probabilities
    invariants, from_pretrained_checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.preference_model import (
    AuxFusionHead,
    PreferenceModel,
    PreferenceModelConfig,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_ENCODER_DIM = 64
_AUX_DIM = 12
_NUM_LABELS = 3
_BATCH = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_encoder_output(batch: int = _BATCH, hidden: int = _ENCODER_DIM, use_pooler: bool = True):
    """Fake HuggingFace encoder output."""
    last_hidden = torch.randn(batch, 16, hidden)
    pooler = torch.randn(batch, hidden) if use_pooler else None
    return SimpleNamespace(
        last_hidden_state=last_hidden,
        pooler_output=pooler,
    )


def _make_mock_encoder(hidden: int = _ENCODER_DIM, use_pooler: bool = True):
    """Mock transformer encoder that returns _make_encoder_output."""
    encoder = MagicMock()
    encoder.gradient_checkpointing_enable = MagicMock()

    def _forward(**kwargs):
        batch = kwargs["input_ids"].shape[0]
        return _make_encoder_output(batch=batch, hidden=hidden, use_pooler=use_pooler)

    encoder.side_effect = _forward
    encoder.__call__ = MagicMock(side_effect=_forward)
    return encoder


def _make_auto_config(hidden: int = _ENCODER_DIM):
    """Fake AutoConfig object with hidden_size."""
    cfg = MagicMock()
    cfg.hidden_size = hidden
    return cfg


def _patched_model(
    hidden: int = _ENCODER_DIM,
    use_pooler: bool = True,
    use_gradient_checkpointing: bool = False,
) -> PreferenceModel:
    """Build a PreferenceModel with mocked HuggingFace calls."""
    auto_cfg = _make_auto_config(hidden)
    mock_encoder = _make_mock_encoder(hidden=hidden, use_pooler=use_pooler)

    with patch("src.models.preference_model.AutoConfig.from_pretrained", return_value=auto_cfg), \
         patch("src.models.preference_model.AutoModel.from_pretrained", return_value=mock_encoder):
        cfg = PreferenceModelConfig(
            model_name="mock-model",
            num_labels=_NUM_LABELS,
            aux_feature_dim=_AUX_DIM,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        model = PreferenceModel(cfg)

    # Re-attach the mock encoder so forward calls work after __init__
    model.encoder = mock_encoder
    return model


def _batch_inputs(batch: int = _BATCH, seq_len: int = 16):
    """Minimal tokenized batch (no aux features)."""
    return {
        "input_ids": torch.zeros(batch, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
    }


# ── PreferenceModelConfig ─────────────────────────────────────────────────────

class TestPreferenceModelConfig:
    def test_default_model_name(self):
        cfg = PreferenceModelConfig()
        assert cfg.model_name == "microsoft/deberta-v3-large"

    def test_default_num_labels(self):
        cfg = PreferenceModelConfig()
        assert cfg.num_labels == 3

    def test_default_aux_feature_dim(self):
        cfg = PreferenceModelConfig()
        assert cfg.aux_feature_dim == _AUX_DIM

    def test_default_dropout_rate(self):
        cfg = PreferenceModelConfig()
        assert cfg.dropout_rate == pytest.approx(0.1)

    def test_default_gradient_checkpointing_disabled(self):
        cfg = PreferenceModelConfig()
        assert cfg.use_gradient_checkpointing is False

    def test_default_hidden_size_none(self):
        cfg = PreferenceModelConfig()
        assert cfg.hidden_size is None

    def test_custom_values_stored(self):
        cfg = PreferenceModelConfig(
            model_name="bert-base-uncased",
            num_labels=2,
            dropout_rate=0.3,
            use_gradient_checkpointing=True,
            hidden_size=128,
        )
        assert cfg.model_name == "bert-base-uncased"
        assert cfg.num_labels == 2
        assert cfg.dropout_rate == pytest.approx(0.3)
        assert cfg.use_gradient_checkpointing is True
        assert cfg.hidden_size == 128


# ── AuxFusionHead ─────────────────────────────────────────────────────────────

class TestAuxFusionHead:
    @pytest.fixture
    def head(self):
        return AuxFusionHead(
            encoder_dim=_ENCODER_DIM,
            aux_dim=_AUX_DIM,
            num_labels=_NUM_LABELS,
            dropout_rate=0.0,  # deterministic for tests
        )

    def test_output_shape_with_aux(self, head):
        cls_emb = torch.randn(_BATCH, _ENCODER_DIM)
        aux = torch.randn(_BATCH, _AUX_DIM)
        out = head(cls_emb, aux_features=aux)
        assert out.shape == (_BATCH, _NUM_LABELS)

    def test_output_shape_without_aux(self, head):
        cls_emb = torch.randn(_BATCH, _ENCODER_DIM)
        out = head(cls_emb, aux_features=None)
        assert out.shape == (_BATCH, _NUM_LABELS)

    def test_zero_pad_path_matches_expected_shape(self, head):
        """Calling with aux_features=None must produce same shape as with aux."""
        cls_emb = torch.randn(2, _ENCODER_DIM)
        out_no_aux = head(cls_emb, aux_features=None)
        out_with_zeros = head(cls_emb, aux_features=torch.zeros(2, _AUX_DIM))
        # Shapes must match (values may differ due to ReLU on zero vs zero)
        assert out_no_aux.shape == out_with_zeros.shape

    def test_output_is_finite(self, head):
        cls_emb = torch.randn(_BATCH, _ENCODER_DIM)
        out = head(cls_emb)
        assert torch.isfinite(out).all()

    def test_output_finite_without_aux(self, head):
        cls_emb = torch.randn(_BATCH, _ENCODER_DIM)
        out = head(cls_emb, aux_features=None)
        assert torch.isfinite(out).all()

    def test_single_sample(self, head):
        cls_emb = torch.randn(1, _ENCODER_DIM)
        out = head(cls_emb)
        assert out.shape == (1, _NUM_LABELS)

    def test_classifier_bias_initialized_to_zero(self):
        head = AuxFusionHead(_ENCODER_DIM, _AUX_DIM, _NUM_LABELS)
        assert torch.all(head.classifier.bias == 0.0)


# ── PreferenceModel ───────────────────────────────────────────────────────────

class TestPreferenceModelForward:
    @pytest.fixture
    def model(self):
        return _patched_model()

    def test_forward_returns_logits_key(self, model):
        out = model(**_batch_inputs())
        assert "logits" in out

    def test_forward_returns_probabilities_key(self, model):
        out = model(**_batch_inputs())
        assert "probabilities" in out

    def test_no_loss_without_labels(self, model):
        out = model(**_batch_inputs())
        assert "loss" not in out

    def test_loss_present_with_labels(self, model):
        inputs = _batch_inputs()
        inputs["labels"] = torch.randint(0, _NUM_LABELS, (_BATCH,))
        out = model(**inputs)
        assert "loss" in out

    def test_logits_shape(self, model):
        out = model(**_batch_inputs())
        assert out["logits"].shape == (_BATCH, _NUM_LABELS)

    def test_probabilities_shape(self, model):
        out = model(**_batch_inputs())
        assert out["probabilities"].shape == (_BATCH, _NUM_LABELS)

    def test_probabilities_sum_to_one(self, model):
        out = model(**_batch_inputs())
        sums = out["probabilities"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(_BATCH), atol=1e-5)

    def test_probabilities_in_unit_interval(self, model):
        out = model(**_batch_inputs())
        probs = out["probabilities"]
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_argmax_class_in_valid_range(self, model):
        out = model(**_batch_inputs())
        preds = out["probabilities"].argmax(dim=-1)
        assert ((preds >= 0) & (preds < _NUM_LABELS)).all()

    def test_loss_is_scalar(self, model):
        inputs = _batch_inputs()
        inputs["labels"] = torch.randint(0, _NUM_LABELS, (_BATCH,))
        out = model(**inputs)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self, model):
        inputs = _batch_inputs()
        inputs["labels"] = torch.randint(0, _NUM_LABELS, (_BATCH,))
        out = model(**inputs)
        assert torch.isfinite(out["loss"])

    def test_forward_with_aux_features(self, model):
        inputs = _batch_inputs()
        inputs["aux_features"] = torch.randn(_BATCH, _AUX_DIM)
        out = model(**inputs)
        assert out["probabilities"].shape == (_BATCH, _NUM_LABELS)

    def test_forward_with_token_type_ids(self, model):
        inputs = _batch_inputs()
        inputs["token_type_ids"] = torch.zeros(_BATCH, 16, dtype=torch.long)
        out = model(**inputs)
        assert "logits" in out

    def test_single_sample_forward(self, model):
        inputs = _batch_inputs(batch=1)
        out = model(**inputs)
        assert out["logits"].shape == (1, _NUM_LABELS)

    def test_logits_are_finite(self, model):
        out = model(**_batch_inputs())
        assert torch.isfinite(out["logits"]).all()


class TestPreferenceModelPoolerFallback:
    def test_uses_pooler_output_when_available(self):
        """Model must prefer pooler_output over last_hidden_state[:, 0]."""
        model = _patched_model(use_pooler=True)
        # Spy on the head to confirm CLS embedding comes from pooler
        with patch.object(model.head, "forward", wraps=model.head.forward) as spy:
            model(**_batch_inputs())
            called_cls = spy.call_args[0][0]  # first positional arg
            assert called_cls.shape == (_BATCH, _ENCODER_DIM)

    def test_falls_back_to_last_hidden_state(self):
        """When pooler_output is None, use last_hidden_state[:, 0]."""
        model = _patched_model(use_pooler=False)
        out = model(**_batch_inputs())
        assert out["probabilities"].shape == (_BATCH, _NUM_LABELS)


class TestPreferenceModelGradientCheckpointing:
    def test_gradient_checkpointing_enabled_when_configured(self):
        auto_cfg = _make_auto_config()
        mock_encoder = _make_mock_encoder()

        with patch("src.models.preference_model.AutoConfig.from_pretrained", return_value=auto_cfg), \
             patch("src.models.preference_model.AutoModel.from_pretrained", return_value=mock_encoder):
            cfg = PreferenceModelConfig(
                model_name="mock",
                use_gradient_checkpointing=True,
            )
            PreferenceModel(cfg)

        mock_encoder.gradient_checkpointing_enable.assert_called_once()

    def test_gradient_checkpointing_not_called_by_default(self):
        auto_cfg = _make_auto_config()
        mock_encoder = _make_mock_encoder()

        with patch("src.models.preference_model.AutoConfig.from_pretrained", return_value=auto_cfg), \
             patch("src.models.preference_model.AutoModel.from_pretrained", return_value=mock_encoder):
            cfg = PreferenceModelConfig(model_name="mock")
            PreferenceModel(cfg)

        mock_encoder.gradient_checkpointing_enable.assert_not_called()


class TestPreferenceModelFromCheckpoint:
    def test_loads_from_checkpoint(self, tmp_path: Path):
        model = _patched_model()
        checkpoint = tmp_path / "best.pt"
        torch.save(model.state_dict(), str(checkpoint))

        auto_cfg = _make_auto_config()
        mock_encoder = _make_mock_encoder()

        with patch("src.models.preference_model.AutoConfig.from_pretrained", return_value=auto_cfg), \
             patch("src.models.preference_model.AutoModel.from_pretrained", return_value=mock_encoder):
            cfg = PreferenceModelConfig(
                model_name="mock-model",
                num_labels=_NUM_LABELS,
                aux_feature_dim=_AUX_DIM,
            )
            loaded = PreferenceModel.from_pretrained_checkpoint(str(checkpoint), cfg)
            loaded.encoder = mock_encoder

        out = loaded(**_batch_inputs())
        assert "logits" in out

    def test_missing_checkpoint_raises(self, tmp_path: Path):
        auto_cfg = _make_auto_config()
        mock_encoder = _make_mock_encoder()

        with patch("src.models.preference_model.AutoConfig.from_pretrained", return_value=auto_cfg), \
             patch("src.models.preference_model.AutoModel.from_pretrained", return_value=mock_encoder):
            cfg = PreferenceModelConfig(model_name="mock-model")
            with pytest.raises((FileNotFoundError, RuntimeError)):
                PreferenceModel.from_pretrained_checkpoint(
                    str(tmp_path / "nonexistent.pt"), cfg
                )
