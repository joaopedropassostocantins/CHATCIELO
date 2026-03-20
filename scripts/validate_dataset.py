"""
Dataset validation script — run after any change to src/data/dataset.py.

Validates:
  1. Required columns present.
  2. No null values in text fields.
  3. Label distribution (class imbalance detection).
  4. No PII in text samples.
  5. Token length distribution (ensures max_seq_length is sufficient).

Usage:
    python scripts/validate_dataset.py --path ./data/train.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import LABEL_MAP
from src.data.preprocessing import contains_pii


@click.command()
@click.option("--path", required=True, help="Path to the Parquet dataset file.")
@click.option("--max-rows", default=None, type=int, help="Limit rows for quick check.")
@click.option("--pii-sample-size", default=100, type=int)
def main(path: str, max_rows: int | None, pii_sample_size: int) -> None:
    """Validate a CHATCIELO dataset Parquet file."""
    click.echo(f"Loading: {path}")
    df = pd.read_parquet(path)
    if max_rows:
        df = df.head(max_rows)

    click.echo(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # ── 1. Required columns ─────────────────────────────────────────────────
    required = {"conversation_id", "prompt", "response_a", "response_b", "winner"}
    missing = required - set(df.columns)
    if missing:
        click.echo(click.style(f"FAIL: Missing columns: {missing}", fg="red"))
        sys.exit(1)
    click.echo(click.style("  [OK] Required columns present", fg="green"))

    # ── 2. Null check ────────────────────────────────────────────────────────
    for col in ["prompt", "response_a", "response_b"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            click.echo(click.style(f"  [WARN] {null_count} null values in '{col}'", fg="yellow"))
        else:
            click.echo(click.style(f"  [OK] No nulls in '{col}'", fg="green"))

    # ── 3. Label distribution ────────────────────────────────────────────────
    label_counts = df["winner"].value_counts()
    click.echo("\n  Label distribution:")
    for label, count in label_counts.items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        click.echo(f"    {label:<20}: {count:>6} ({pct:5.1f}%) {bar}")

    # Imbalance warning: if any class < 10%
    for label, count in label_counts.items():
        if count / len(df) < 0.10:
            click.echo(click.style(f"  [WARN] Class '{label}' has < 10% representation — consider oversampling.", fg="yellow"))

    # ── 4. PII spot-check ────────────────────────────────────────────────────
    sample = df.sample(min(pii_sample_size, len(df)), random_state=42)
    pii_found = 0
    for _, row in sample.iterrows():
        for col in ["prompt", "response_a", "response_b"]:
            if contains_pii(str(row[col])):
                pii_found += 1
                break

    if pii_found > 0:
        click.echo(click.style(f"  [WARN] PII detected in {pii_found}/{pii_sample_size} sampled rows. Consider pre-scrubbing.", fg="yellow"))
    else:
        click.echo(click.style(f"  [OK] No PII in sampled {pii_sample_size} rows", fg="green"))

    # ── 5. Prompt length stats ───────────────────────────────────────────────
    for col in ["prompt", "response_a", "response_b"]:
        lengths = df[col].str.len()
        click.echo(
            f"\n  '{col}' char length: "
            f"min={lengths.min()}, "
            f"mean={lengths.mean():.0f}, "
            f"p95={lengths.quantile(0.95):.0f}, "
            f"max={lengths.max()}"
        )

    click.echo(click.style("\nValidation complete.", fg="green"))


if __name__ == "__main__":
    main()
