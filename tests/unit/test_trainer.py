"""Unit tests for src/training/trainer.py."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, TrainingState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dataloader(n_batches: int = 3, batch_size: int = 4) -> DataLoader:
    """Tiny deterministic DataLoader — no real model required."""
    seq_len = 32
    n = n_batches * batch_size
    ds = TensorDataset(
        torch.zeros(n, seq_len, dtype=torch.long),   # input_ids
        torch.ones(n, seq_len, dtype=torch.long),    # attention_mask
        torch.randint(0, 3, (n,)),                   # labels
    )

    def collate(batch):
        ids, masks, labels = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)


def _make_mock_model() -> MagicMock:
    """Mock PreferenceModel that returns random logits without HF download."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])
    model.train.return_value = None
    model.eval.return_value = None
    model.to.return_value = model

    def _forward(**kwargs):
        b = kwargs["input_ids"].shape[0]
        logits = torch.randn(b, 3, requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probs}

    model.side_effect = _forward
    model.__call__ = MagicMock(side_effect=_forward)
    return model


def _make_trainer(
    tmp_path: Path,
    val_loader: DataLoader | None = None,
    patience: int = 2,
    epochs: int = 1,
) -> Trainer:
    model = _make_mock_model()
    train_loader = _make_dataloader()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(tmp_path),
        num_epochs=epochs,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        early_stopping_patience=patience,
        device="cpu",
    )
    # Replace optimizer/scheduler with stubs so no real param update needed
    trainer.optimizer = MagicMock()
    trainer.optimizer.zero_grad.return_value = None
    trainer.optimizer.step.return_value = None
    trainer.scheduler = MagicMock()
    trainer.scheduler.step.return_value = None
    return trainer


# ── Tests: TrainingState dataclass ───────────────────────────────────────────

class TestTrainingState:
    def test_default_values(self):
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_val_auc == 0.0
        assert state.train_losses == []
        assert state.val_metrics == []
        assert state.no_improve_count == 0

    def test_state_is_mutable(self):
        state = TrainingState()
        state.epoch = 5
        state.train_losses.append(0.42)
        assert state.epoch == 5
        assert state.train_losses == [0.42]


# ── Tests: Trainer initialization ────────────────────────────────────────────

class TestTrainerInit:
    def test_output_dir_created(self, tmp_path):
        subdir = tmp_path / "checkpoints"
        _make_trainer(subdir)
        assert subdir.exists()

    def test_state_initialized(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.state.global_step == 0
        assert trainer.state.best_val_auc == 0.0

    def test_device_set(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.device == "cpu"

    def test_grad_accum_set(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.grad_accum == 1


# ── Tests: _validate() ────────────────────────────────────────────────────────

class TestValidate:
    def test_returns_empty_without_val_loader(self, tmp_path):
        trainer = _make_trainer(tmp_path, val_loader=None)
        result = trainer._validate()
        assert result == {}

    def test_returns_metrics_with_val_loader(self, tmp_path):
        val_loader = _make_dataloader(n_batches=2)
        trainer = _make_trainer(tmp_path, val_loader=val_loader)

        with patch("src.training.trainer.compute_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "accuracy": 0.6,
                "auc": 0.7,
                "log_loss": 0.9,
                "preference_score_mean": 0.5,
            }
            result = trainer._validate()

        assert "accuracy" in result
        assert "auc" in result
        assert result["auc"] == pytest.approx(0.7)

    def test_validate_does_not_mutate_state(self, tmp_path):
        val_loader = _make_dataloader(n_batches=2)
        trainer = _make_trainer(tmp_path, val_loader=val_loader)
        step_before = trainer.state.global_step

        with patch("src.training.trainer.compute_metrics", return_value={"auc": 0.5, "accuracy": 0.5, "log_loss": 1.0, "preference_score_mean": 0.5}):
            trainer._validate()

        assert trainer.state.global_step == step_before


# ── Tests: _save_checkpoint() ─────────────────────────────────────────────────

class TestSaveCheckpoint:
    def test_checkpoint_file_created(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with patch("src.training.trainer.torch.save") as mock_save:
            trainer._save_checkpoint("best")
        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        assert str(args[1]).endswith("best.pt")

    def test_checkpoint_latest_name(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with patch("src.training.trainer.torch.save") as mock_save:
            trainer._save_checkpoint("latest")
        args = mock_save.call_args[0]
        assert "latest.pt" in str(args[1])


# ── Tests: Full training loop ─────────────────────────────────────────────────

class TestTrainLoop:
    def test_train_returns_training_state(self, tmp_path):
        trainer = _make_trainer(tmp_path, epochs=1)
        with patch("src.training.trainer.torch.save"):
            state = trainer.train()
        assert isinstance(state, TrainingState)

    def test_global_step_increments(self, tmp_path):
        trainer = _make_trainer(tmp_path, epochs=1)
        with patch("src.training.trainer.torch.save"):
            state = trainer.train()
        # grad_accum=1 → one step per batch → 3 steps for 3 batches
        assert state.global_step == 3

    def test_train_losses_recorded(self, tmp_path):
        trainer = _make_trainer(tmp_path, epochs=1)
        with patch("src.training.trainer.torch.save"):
            state = trainer.train()
        assert len(state.train_losses) > 0
        assert all(isinstance(v, float) for v in state.train_losses)

    def test_val_metrics_recorded_per_epoch(self, tmp_path):
        val_loader = _make_dataloader(n_batches=2)
        trainer = _make_trainer(tmp_path, val_loader=val_loader, epochs=2)
        with patch("src.training.trainer.torch.save"), \
             patch("src.training.trainer.compute_metrics", return_value={
                 "accuracy": 0.5, "auc": 0.6, "log_loss": 1.0, "preference_score_mean": 0.5,
             }):
            state = trainer.train()
        assert len(state.val_metrics) == 2

    def test_best_checkpoint_saved_on_auc_improve(self, tmp_path):
        val_loader = _make_dataloader(n_batches=2)
        trainer = _make_trainer(tmp_path, val_loader=val_loader, epochs=1)
        saved_names = []

        def capture_save(state_dict, path):
            saved_names.append(Path(path).stem)

        with patch("src.training.trainer.torch.save", side_effect=capture_save), \
             patch("src.training.trainer.compute_metrics", return_value={
                 "accuracy": 0.8, "auc": 0.9, "log_loss": 0.5, "preference_score_mean": 0.8,
             }):
            trainer.train()

        assert "best" in saved_names

    def test_early_stopping_fires(self, tmp_path):
        """Early stopping must halt training when AUC does not improve."""
        val_loader = _make_dataloader(n_batches=2)
        trainer = _make_trainer(tmp_path, val_loader=val_loader, patience=1, epochs=5)
        # AUC = 0.0 every epoch → no improvement → stops after epoch 2
        with patch("src.training.trainer.torch.save"), \
             patch("src.training.trainer.compute_metrics", return_value={
                 "accuracy": 0.3, "auc": 0.0, "log_loss": 2.0, "preference_score_mean": 0.3,
             }):
            state = trainer.train()

        # Should stop before completing all 5 epochs
        assert state.epoch < 4

    def test_no_early_stopping_without_val_loader(self, tmp_path):
        """Without val_loader, all epochs must complete."""
        trainer = _make_trainer(tmp_path, val_loader=None, patience=1, epochs=3)
        with patch("src.training.trainer.torch.save"):
            state = trainer.train()
        assert state.epoch == 2   # 0-indexed, epoch 2 = 3rd epoch


# ── Tests: _move_batch() ──────────────────────────────────────────────────────

class TestMoveBatch:
    def test_tensors_moved_to_device(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = {
            "input_ids": torch.zeros(4, 32, dtype=torch.long),
            "attention_mask": torch.ones(4, 32, dtype=torch.long),
        }
        moved = trainer._move_batch(batch)
        assert "input_ids" in moved
        assert "attention_mask" in moved

    def test_all_keys_preserved(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = {"a": torch.zeros(2), "b": torch.ones(2)}
        moved = trainer._move_batch(batch)
        assert set(moved.keys()) == {"a", "b"}
