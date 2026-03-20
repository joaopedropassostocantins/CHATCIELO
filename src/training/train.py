"""
CLI training script for the CHATCIELO preference model.

Usage:
    python src/training/train.py [OPTIONS]

Options:
    --train-path    Path to training Parquet file
    --val-path      Path to validation Parquet file
    --output-dir    Directory to save checkpoints
    --model-name    HuggingFace model name
    --epochs        Number of training epochs
    --lr            Learning rate
    --batch-size    Per-device batch size
    --grad-accum    Gradient accumulation steps
    --max-length    Maximum sequence length
    --device        Device: cpu / cuda / mps
    --max-rows      Limit dataset rows (debugging)
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Allow running from repo root: python src/training/train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.dataset import ChatCieloDataset, load_examples_from_parquet
from src.models.preference_model import PreferenceModel, PreferenceModelConfig
from src.training.trainer import Trainer


@click.command()
@click.option("--train-path", default="./data/train.parquet", show_default=True)
@click.option("--val-path", default="./data/val.parquet", show_default=True)
@click.option("--output-dir", default="./artifacts/models", show_default=True)
@click.option("--model-name", default="microsoft/deberta-v3-large", show_default=True)
@click.option("--epochs", default=3, show_default=True, type=int)
@click.option("--lr", default=1e-5, show_default=True, type=float)
@click.option("--batch-size", default=8, show_default=True, type=int)
@click.option("--grad-accum", default=4, show_default=True, type=int)
@click.option("--max-length", default=1024, show_default=True, type=int)
@click.option("--device", default="cpu", show_default=True)
@click.option("--max-rows", default=None, type=int, help="Limit rows for debugging")
def main(
    train_path: str,
    val_path: str,
    output_dir: str,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    device: str,
    max_rows: int | None,
) -> None:
    """Train the CHATCIELO preference model."""
    click.echo(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    click.echo(f"Loading training data: {train_path}")
    train_examples = load_examples_from_parquet(train_path, split="train", max_rows=max_rows)
    click.echo(f"  → {len(train_examples)} training examples")

    val_examples = []
    if Path(val_path).exists():
        val_examples = load_examples_from_parquet(val_path, split="val", max_rows=max_rows)
        click.echo(f"  → {len(val_examples)} validation examples")

    train_ds = ChatCieloDataset(train_examples, tokenizer, max_length=max_length)
    val_ds = ChatCieloDataset(val_examples, tokenizer, max_length=max_length) if val_examples else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False) if val_ds else None

    click.echo(f"Building model: {model_name}")
    model_cfg = PreferenceModelConfig(
        model_name=model_name,
        use_gradient_checkpointing=(device == "cuda"),
    )
    model = PreferenceModel(model_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"  Total params: {total_params:,} | Trainable: {trainable:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        num_epochs=epochs,
        learning_rate=lr,
        gradient_accumulation_steps=grad_accum,
        device=device,
    )

    click.echo("Starting training...")
    state = trainer.train()

    click.echo(f"\nTraining complete.")
    click.echo(f"  Best val AUC : {state.best_val_auc:.4f}")
    click.echo(f"  Total steps  : {state.global_step}")
    click.echo(f"  Checkpoint   : {output_dir}/best.pt")


if __name__ == "__main__":
    main()
