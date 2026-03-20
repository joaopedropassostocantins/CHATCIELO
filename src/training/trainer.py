"""
Training loop for the CHATCIELO preference model.

DIFF vs Kaggle baseline (maxreciprocate/kaggle-lmarena-1st-place):
──────────────────────────────────────────────────────────────────
KAGGLE:
    for epoch in epochs:
        for batch in loader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
    # No validation loop
    # No early stopping
    # No checkpoint management
    # No gradient accumulation (explicit)
    # No learning rate schedule beyond constant

CHATCIELO:
+   CosineAnnealingWarmup LR schedule
+   Gradient accumulation (configurable)
+   Per-epoch validation with AUC + accuracy
+   EarlyStopping on validation AUC
+   Checkpoint saving: best model + latest
+   Structured logging with loss breakdown (cls + rank)
+   Gradient clipping (max_norm=1.0)
──────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.evaluation.metrics import compute_metrics
from src.training.losses import CombinedPreferenceLoss


@dataclass
class TrainingState:
    """Mutable state tracked during training.

    Args:
        epoch: Current epoch (0-indexed).
        global_step: Total optimizer steps taken.
        best_val_auc: Best validation AUC seen so far.
        train_losses: Per-step loss history.
        val_metrics: Per-epoch validation metric history.
        no_improve_count: Consecutive epochs without AUC improvement.
    """

    epoch: int = 0
    global_step: int = 0
    best_val_auc: float = 0.0
    train_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    no_improve_count: int = 0


class Trainer:
    """Training orchestrator for the preference model.

    Args:
        model: PreferenceModel instance.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        output_dir: Path where checkpoints are saved.
        num_epochs: Total training epochs.
        learning_rate: Peak learning rate.
        gradient_accumulation_steps: Steps before optimizer update.
        warmup_ratio: Fraction of total steps used for LR warmup.
        early_stopping_patience: Epochs without improvement before stopping.
        device: Torch device string ('cpu', 'cuda', 'mps').

    Validation Metrics:
        - Logs val AUC, accuracy, log-loss per epoch.
        - Best checkpoint selected by val AUC (macro).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = "./artifacts/models",
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 2,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.device = device
        self.grad_accum = gradient_accumulation_steps
        self.patience = early_stopping_patience

        total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.loss_fn = CombinedPreferenceLoss()
        self.state = TrainingState()

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move all tensors in batch to the target device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict with mean 'loss', 'cls_loss', 'rank_loss' for the epoch.
        """
        self.model.train()
        total_loss = total_cls = total_rank = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            batch = self._move_batch(batch)
            labels = batch.pop("labels", None)
            aux = batch.pop("aux_features", None)

            output = self.model(**batch, aux_features=aux)
            loss_dict = self.loss_fn(output["logits"], output["probabilities"], labels)

            loss = loss_dict["loss"] / self.grad_accum
            loss.backward()

            total_loss += loss_dict["loss"].item()
            total_cls += loss_dict["cls_loss"].item()
            total_rank += loss_dict["rank_loss"].item()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1

            self.state.train_losses.append(loss_dict["loss"].item())

        n = len(self.train_loader)
        return {
            "loss": total_loss / n,
            "cls_loss": total_cls / n,
            "rank_loss": total_rank / n,
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation loop and compute metrics.

        Returns:
            Dict with keys: accuracy, auc, log_loss.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            labels = batch.pop("labels")
            batch.pop("aux_features", None)

            output = self.model(**batch)
            all_probs.append(output["probabilities"].cpu())
            all_labels.append(labels.cpu())

        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()
        return compute_metrics(labels, probs)

    def _save_checkpoint(self, name: str) -> None:
        """Save model state dict to output_dir/{name}.pt."""
        path = self.output_dir / f"{name}.pt"
        torch.save(self.model.state_dict(), path)

    def train(self) -> TrainingState:
        """Execute the full training loop.

        Returns:
            Final TrainingState with metrics history.

        Raises:
            RuntimeError: If CUDA out-of-memory occurs (user should reduce batch size).
        """
        print(f"Training on device: {self.device}")
        print(f"Epochs: {self.num_epochs}, Grad accum: {self.grad_accum}")

        for epoch in range(self.num_epochs):
            self.state.epoch = epoch
            t0 = time.time()

            train_metrics = self._train_epoch()
            val_metrics = self._validate()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"loss={train_metrics['loss']:.4f} "
                f"cls={train_metrics['cls_loss']:.4f} "
                f"rank={train_metrics['rank_loss']:.4f} | "
                f"val_auc={val_metrics.get('auc', 0.0):.4f} "
                f"val_acc={val_metrics.get('accuracy', 0.0):.4f} | "
                f"{elapsed:.1f}s"
            )

            self.state.val_metrics.append(val_metrics)
            self._save_checkpoint("latest")

            val_auc = val_metrics.get("auc", 0.0)
            if val_auc > self.state.best_val_auc:
                self.state.best_val_auc = val_auc
                self.state.no_improve_count = 0
                self._save_checkpoint("best")
                print(f"  ✓ New best AUC: {val_auc:.4f} — checkpoint saved.")
            else:
                self.state.no_improve_count += 1
                if self.state.no_improve_count >= self.patience:
                    print(f"  Early stopping after {epoch+1} epochs.")
                    break

        return self.state
