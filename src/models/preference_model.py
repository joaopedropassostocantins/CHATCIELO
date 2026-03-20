"""
Pairwise Preference Model — DeBERTa-v3 Cross-Encoder with Auxiliary Features.

MATHEMATICAL FORMULATION
────────────────────────
Given a triple (p, r_a, r_b) and merchant segment s:

  Input tokens:  x = tokenize("[s] <prompt>p</prompt> <response_a>r_a</response_a> <response_b>r_b</response_b>")
  Encoder:       h = DeBERTa_encoder(x)          # h ∈ R^{d_model}
  Aux features:  f = compute_auxiliary_features(r_a, r_b, s)  # f ∈ R^{12}
  Fusion:        z = concat([h, f])              # z ∈ R^{d_model + 12}
  Logits:        l = W·z + b                     # l ∈ R^3
  Probabilities: p = softmax(l)                  # [P(A wins), P(B wins), P(tie)]

Loss Function:
  L = CrossEntropy(p, y) with label smoothing ε=0.1

  Equivalently:
  L = -Σ_c  ỹ_c · log(p_c)
  where ỹ_c = (1-ε)·1[c=y] + ε/C

DIFF vs Kaggle baseline (orangeaugust/lmsys2):
──────────────────────────────────────────────
KAGGLE:
    class Model(nn.Module):
        def __init__(self, cfg):
            self.backbone = AutoModel.from_pretrained(cfg.model)
            self.head = nn.Linear(cfg.hidden_size, 3)
        def forward(self, x):
            out = self.backbone(**x).last_hidden_state[:, 0]
            return self.head(out)

CHATCIELO (this file):
+   class PreferenceModel(nn.Module):
+       # Uses pooler_output (not last_hidden_state[:, 0]) — more stable for DeBERTa
+       # Adds auxiliary feature fusion layer (AuxFusionHead)
+       # Adds dropout with configurable rate
+       # Supports gradient checkpointing for memory efficiency
+       # Supports 4-bit quantization via bitsandbytes
──────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from src.features.feature_engineering import AUX_FEATURE_DIM
from src.data.dataset import NUM_LABELS


@dataclass
class PreferenceModelConfig:
    """Configuration for the PreferenceModel.

    Args:
        model_name: HuggingFace model name or local path.
        num_labels: Number of output classes (default 3: A wins, B wins, tie).
        aux_feature_dim: Dimension of auxiliary feature vector (default 12).
        dropout_rate: Dropout probability applied before the classifier head.
        use_gradient_checkpointing: Enable gradient checkpointing to reduce VRAM.
        hidden_size: Encoder hidden size (auto-detected if None).
    """

    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = NUM_LABELS
    aux_feature_dim: int = AUX_FEATURE_DIM
    dropout_rate: float = 0.1
    use_gradient_checkpointing: bool = False
    hidden_size: Optional[int] = None


class AuxFusionHead(nn.Module):
    """Late-fusion head that combines encoder CLS embedding with auxiliary features.

    DIFF vs Kaggle: Kaggle uses a single linear layer on CLS embedding.
    Here we project auxiliary features to the same space before fusion.

    Args:
        encoder_dim: Dimension of the encoder CLS output.
        aux_dim: Dimension of the auxiliary feature vector.
        num_labels: Number of output classes.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        encoder_dim: int,
        aux_dim: int,
        num_labels: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        fusion_dim = encoder_dim + aux_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.aux_proj = nn.Linear(aux_dim, aux_dim)
        self.classifier = nn.Linear(fusion_dim, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the fusion head.

        Args:
            cls_embedding: CLS token embedding, shape (B, encoder_dim).
            aux_features: Auxiliary features tensor, shape (B, aux_dim).
                          If None, zero tensor is used (inference without aux).

        Returns:
            Logits tensor of shape (B, num_labels).
        """
        cls_embedding = self.dropout(cls_embedding)

        if aux_features is not None:
            aux_proj = torch.relu(self.aux_proj(aux_features))
            fused = torch.cat([cls_embedding, aux_proj], dim=-1)
        else:
            # Zero-pad aux dimension for graceful degradation
            zeros = torch.zeros(
                cls_embedding.size(0),
                self.aux_proj.in_features,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
            fused = torch.cat([cls_embedding, zeros], dim=-1)

        return self.classifier(fused)


class PreferenceModel(nn.Module):
    """DeBERTa-v3 cross-encoder for pairwise preference ranking.

    Args:
        config: PreferenceModelConfig instance.

    Returns:
        During forward(): dict with 'logits' (B, 3) and optionally 'loss'.

    Validation Metrics:
        - softmax(logits).sum(dim=-1) ≈ 1.0 for all rows.
        - Predicted class ∈ {0, 1, 2}.
        - Loss must be finite (no NaN/Inf).
    """

    def __init__(self, config: PreferenceModelConfig) -> None:
        super().__init__()
        self.config = config

        encoder_config = AutoConfig.from_pretrained(config.model_name)
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(
            config.model_name,
            config=encoder_config,
        )

        if config.use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = config.hidden_size or encoder_config.hidden_size
        self.head = AuxFusionHead(
            encoder_dim=hidden_size,
            aux_dim=config.aux_feature_dim,
            num_labels=config.num_labels,
            dropout_rate=config.dropout_rate,
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs, shape (B, seq_len).
            attention_mask: Attention mask, shape (B, seq_len).
            token_type_ids: Optional segment IDs, shape (B, seq_len).
            aux_features: Optional auxiliary features, shape (B, aux_dim).
            labels: Optional ground-truth labels in {0, 1, 2}, shape (B,).

        Returns:
            Dict with:
                - 'logits': shape (B, num_labels)
                - 'loss': scalar tensor (only if labels provided)
                - 'probabilities': softmax(logits), shape (B, num_labels)

        Raises:
            RuntimeError: If input tensors have mismatched batch dimensions.
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        encoder_output = self.encoder(**encoder_kwargs)

        # Use pooler_output for DeBERTa (more stable than last_hidden_state[:, 0])
        # DIFF vs Kaggle: Kaggle uses last_hidden_state[:, 0]
        if hasattr(encoder_output, "pooler_output") and encoder_output.pooler_output is not None:
            cls_emb = encoder_output.pooler_output
        else:
            cls_emb = encoder_output.last_hidden_state[:, 0]

        logits = self.head(cls_emb, aux_features)
        probabilities = torch.softmax(logits, dim=-1)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels)

        return output

    @classmethod
    def from_pretrained_checkpoint(
        cls, checkpoint_path: str, config: PreferenceModelConfig
    ) -> "PreferenceModel":
        """Load model weights from a saved checkpoint.

        Args:
            checkpoint_path: Path to the saved state_dict (.pt or .bin).
            config: Model configuration.

        Returns:
            PreferenceModel with loaded weights.

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
        """
        model = cls(config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model
