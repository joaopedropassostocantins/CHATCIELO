"""
CLI evaluation script — runs full metrics suite on a val/test set.

Usage:
    python src/evaluation/evaluate.py \\
        --data-path ./data/val.parquet \\
        --model-path ./artifacts/models \\
        --output ./artifacts/eval_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.dataset import ChatCieloDataset, load_examples_from_parquet
from src.evaluation.metrics import compute_ece, compute_metrics
from src.models.preference_model import PreferenceModel, PreferenceModelConfig


@click.command()
@click.option("--data-path", required=True, help="Path to eval Parquet file")
@click.option("--model-path", default="./artifacts/models", show_default=True)
@click.option("--model-name", default="microsoft/deberta-v3-large", show_default=True)
@click.option("--max-length", default=1024, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--device", default="cpu")
@click.option("--output", default=None, help="JSON output file")
@click.option("--max-rows", default=None, type=int)
def main(
    data_path: str,
    model_path: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str,
    output: str | None,
    max_rows: int | None,
) -> None:
    """Evaluate model on a labeled dataset and report metrics."""
    # Load model
    model_cfg = PreferenceModelConfig(model_name=model_name)
    model = PreferenceModel(model_cfg)

    checkpoint = Path(model_path) / "best.pt"
    if checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state, strict=False)
        click.echo(f"Loaded: {checkpoint}")
    else:
        click.echo(f"Warning: No checkpoint at {checkpoint}. Using random weights.")

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    examples = load_examples_from_parquet(data_path, split="eval", max_rows=max_rows)
    click.echo(f"Evaluating on {len(examples)} examples...")

    ds = ChatCieloDataset(examples, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_probs.append(out["probabilities"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(labels, probs)
    metrics["ece"] = compute_ece(labels, probs)

    # Print results
    click.echo("\n" + "="*50)
    click.echo("EVALUATION RESULTS")
    click.echo("="*50)
    for k, v in metrics.items():
        flag = ""
        if k == "ece" and v >= 0.05:
            flag = " ⚠️  ABOVE THRESHOLD (0.05)"
        click.echo(f"  {k:<30}: {v:.4f}{flag}")
    click.echo("="*50)

    # Acceptance criteria
    if metrics["auc"] < 0.70:
        click.echo("FAIL: AUC below 0.70 threshold.", err=True)
        sys.exit(1)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f"Results written to: {output}")


if __name__ == "__main__":
    main()
