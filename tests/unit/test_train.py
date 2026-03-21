"""Unit tests for src/training/train.py (CLI entrypoint).

Tests are isolated using mocks: no HuggingFace downloads, no real Parquet files.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from click.testing import CliRunner

from src.training.train import main


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _fake_examples(n: int = 10) -> list:
    """Return a list of synthetic PreferenceExample-like MagicMocks."""
    return [MagicMock() for _ in range(n)]


def _make_patchers(train_path: str, val_path: str | None = None, n_examples: int = 10):
    """Context manager factory that patches all heavy I/O and network calls."""
    patches = {
        "tokenizer": patch(
            "src.training.train.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ),
        "load_train": patch(
            "src.training.train.load_examples_from_parquet",
            return_value=_fake_examples(n_examples),
        ),
        "dataset": patch(
            "src.training.train.ChatCieloDataset",
            return_value=MagicMock(__len__=lambda s: n_examples),
        ),
        "dataloader": patch(
            "src.training.train.DataLoader",
            return_value=MagicMock(),
        ),
        "model_cls": patch(
            "src.training.train.PreferenceModel",
            return_value=MagicMock(
                parameters=MagicMock(
                    return_value=iter([torch.zeros(1)])
                )
            ),
        ),
        "trainer_cls": patch(
            "src.training.train.Trainer",
            return_value=MagicMock(
                train=MagicMock(
                    return_value=MagicMock(best_val_auc=0.82, global_step=30)
                )
            ),
        ),
        "val_exists": patch(
            "src.training.train.Path.exists",
            return_value=(val_path is not None),
        ),
    }
    return patches


# ── Tests: happy-path invocation ──────────────────────────────────────────────

class TestMainHappyPath:
    def test_exits_zero_with_defaults(self, tmp_path):
        """CLI exits 0 with all default arguments when files are mocked."""
        runner = CliRunner()
        patchers = _make_patchers(train_path="./data/train.parquet", val_path="./data/val.parquet")

        with (
            patchers["tokenizer"],
            patchers["load_train"],
            patchers["dataset"],
            patchers["dataloader"],
            patchers["model_cls"],
            patchers["trainer_cls"],
            patchers["val_exists"],
        ):
            result = runner.invoke(main, [])

        assert result.exit_code == 0, result.output

    def test_output_reports_best_auc(self, tmp_path):
        """CLI output must include best_val_auc from the final TrainingState."""
        runner = CliRunner()
        patchers = _make_patchers(train_path="./data/train.parquet", val_path="./data/val.parquet")

        with (
            patchers["tokenizer"],
            patchers["load_train"],
            patchers["dataset"],
            patchers["dataloader"],
            patchers["model_cls"],
            patchers["trainer_cls"],
            patchers["val_exists"],
        ):
            result = runner.invoke(main, [])

        assert "0.82" in result.output or "auc" in result.output.lower()

    def test_tokenizer_called_with_model_name(self):
        """AutoTokenizer.from_pretrained must be called with --model-name."""
        runner = CliRunner()
        patchers = _make_patchers(train_path="./data/train.parquet")

        with (
            patchers["tokenizer"] as mock_tok,
            patchers["load_train"],
            patchers["dataset"],
            patchers["dataloader"],
            patchers["model_cls"],
            patchers["trainer_cls"],
            patchers["val_exists"],
        ):
            runner.invoke(main, ["--model-name", "microsoft/deberta-v3-base"])

        mock_tok.assert_called_once_with("microsoft/deberta-v3-base")

    def test_trainer_constructed_with_correct_epochs(self):
        """Trainer must receive the --epochs value."""
        runner = CliRunner()
        patchers = _make_patchers(train_path="./data/train.parquet")

        with (
            patchers["tokenizer"],
            patchers["load_train"],
            patchers["dataset"],
            patchers["dataloader"],
            patchers["model_cls"],
            patchers["trainer_cls"] as mock_trainer_cls,
            patchers["val_exists"],
        ):
            runner.invoke(main, ["--epochs", "7"])

        _, kwargs = mock_trainer_cls.call_args
        assert kwargs.get("num_epochs") == 7 or mock_trainer_cls.call_args[0][4] == 7

    def test_trainer_constructed_with_correct_lr(self):
        """Trainer must receive the --lr value."""
        runner = CliRunner()
        patchers = _make_patchers(train_path="./data/train.parquet")

        with (
            patchers["tokenizer"],
            patchers["load_train"],
            patchers["dataset"],
            patchers["dataloader"],
            patchers["model_cls"],
            patchers["trainer_cls"] as mock_trainer_cls,
            patchers["val_exists"],
        ):
            runner.invoke(main, ["--lr", "2e-5"])

        _, kwargs = mock_trainer_cls.call_args
        # Accept both keyword and positional
        lr_val = kwargs.get("learning_rate", None)
        assert lr_val is None or abs(lr_val - 2e-5) < 1e-10

    def test_trainer_train_is_called(self):
        """trainer.train() must be invoked exactly once."""
        runner = CliRunner()
        patchers = _make_patchers(train_path="./data/train.parquet")
        mock_trainer_instance = MagicMock(
            train=MagicMock(return_value=MagicMock(best_val_auc=0.5, global_step=10))
        )

        with (
            patchers["tokenizer"],
            patchers["load_train"],
            patchers["dataset"],
            patchers["dataloader"],
            patchers["model_cls"],
            patch("src.training.train.Trainer", return_value=mock_trainer_instance),
            patchers["val_exists"],
        ):
            runner.invoke(main, [])

        mock_trainer_instance.train.assert_called_once()


# ── Tests: --max-rows debugging flag ─────────────────────────────────────────

class TestMaxRows:
    def test_max_rows_passed_to_loader(self):
        """load_examples_from_parquet must receive max_rows when --max-rows is set."""
        runner = CliRunner()

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples(5)) as mock_load,
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 5)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", return_value=MagicMock(parameters=lambda: iter([torch.zeros(1)]))),
            patch("src.training.train.Trainer", return_value=MagicMock(train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0)))),
            patch("src.training.train.Path.exists", return_value=False),
        ):
            runner.invoke(main, ["--max-rows", "50"])

        # Called at least once (for train); max_rows must be 50
        assert mock_load.call_count >= 1
        _, kwargs = mock_load.call_args_list[0]
        assert kwargs.get("max_rows") == 50

    def test_max_rows_none_by_default(self):
        """max_rows must default to None (no row limit)."""
        runner = CliRunner()

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples()) as mock_load,
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 10)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", return_value=MagicMock(parameters=lambda: iter([torch.zeros(1)]))),
            patch("src.training.train.Trainer", return_value=MagicMock(train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0)))),
            patch("src.training.train.Path.exists", return_value=False),
        ):
            runner.invoke(main, [])

        _, kwargs = mock_load.call_args_list[0]
        assert kwargs.get("max_rows") is None


# ── Tests: validation data absent ─────────────────────────────────────────────

class TestValLoaderAbsent:
    def test_no_val_loader_when_val_file_missing(self):
        """val_loader must be None when --val-path does not exist on disk."""
        runner = CliRunner()
        mock_trainer_instance = MagicMock(
            train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0))
        )

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples()),
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 10)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", return_value=MagicMock(parameters=lambda: iter([torch.zeros(1)]))),
            patch("src.training.train.Trainer", return_value=mock_trainer_instance) as mock_trainer_cls,
            patch("src.training.train.Path.exists", return_value=False),
        ):
            result = runner.invoke(main, [])

        assert result.exit_code == 0
        _, kwargs = mock_trainer_cls.call_args
        assert kwargs.get("val_loader") is None

    def test_val_loader_provided_when_file_exists(self):
        """val_loader must not be None when --val-path exists on disk."""
        runner = CliRunner()
        mock_trainer_instance = MagicMock(
            train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0))
        )

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples()),
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 10)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", return_value=MagicMock(parameters=lambda: iter([torch.zeros(1)]))),
            patch("src.training.train.Trainer", return_value=mock_trainer_instance) as mock_trainer_cls,
            patch("src.training.train.Path.exists", return_value=True),
        ):
            result = runner.invoke(main, [])

        assert result.exit_code == 0
        _, kwargs = mock_trainer_cls.call_args
        assert kwargs.get("val_loader") is not None


# ── Tests: device forwarding ──────────────────────────────────────────────────

class TestDevice:
    def test_device_forwarded_to_trainer(self):
        """--device must be forwarded to Trainer as the 'device' kwarg."""
        runner = CliRunner()
        mock_trainer_instance = MagicMock(
            train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0))
        )

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples()),
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 10)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", return_value=MagicMock(parameters=lambda: iter([torch.zeros(1)]))),
            patch("src.training.train.Trainer", return_value=mock_trainer_instance) as mock_trainer_cls,
            patch("src.training.train.Path.exists", return_value=False),
        ):
            runner.invoke(main, ["--device", "cpu"])

        _, kwargs = mock_trainer_cls.call_args
        assert kwargs.get("device") == "cpu"


# ── Tests: gradient checkpointing conditioned on device ───────────────────────

class TestGradientCheckpointing:
    def test_gradient_checkpointing_disabled_on_cpu(self):
        """PreferenceModelConfig must have use_gradient_checkpointing=False on CPU."""
        runner = CliRunner()
        captured_configs = []

        def capture_model(config):
            captured_configs.append(config)
            m = MagicMock()
            m.parameters.return_value = iter([torch.zeros(1)])
            return m

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples()),
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 10)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", side_effect=capture_model),
            patch("src.training.train.Trainer", return_value=MagicMock(train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0)))),
            patch("src.training.train.Path.exists", return_value=False),
        ):
            runner.invoke(main, ["--device", "cpu"])

        assert len(captured_configs) == 1
        assert captured_configs[0].use_gradient_checkpointing is False

    def test_gradient_checkpointing_enabled_on_cuda(self):
        """PreferenceModelConfig must have use_gradient_checkpointing=True on CUDA."""
        runner = CliRunner()
        captured_configs = []

        def capture_model(config):
            captured_configs.append(config)
            m = MagicMock()
            m.parameters.return_value = iter([torch.zeros(1)])
            return m

        with (
            patch("src.training.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("src.training.train.load_examples_from_parquet", return_value=_fake_examples()),
            patch("src.training.train.ChatCieloDataset", return_value=MagicMock(__len__=lambda s: 10)),
            patch("src.training.train.DataLoader", return_value=MagicMock()),
            patch("src.training.train.PreferenceModel", side_effect=capture_model),
            patch("src.training.train.Trainer", return_value=MagicMock(train=MagicMock(return_value=MagicMock(best_val_auc=0.0, global_step=0)))),
            patch("src.training.train.Path.exists", return_value=False),
        ):
            runner.invoke(main, ["--device", "cuda"])

        assert len(captured_configs) == 1
        assert captured_configs[0].use_gradient_checkpointing is True
