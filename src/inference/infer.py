"""
CLI inference script for the CHATCIELO preference model.

Usage:
    python src/inference/infer.py \\
        --prompt "Como funciona o parcelamento?" \\
        --response-a "O parcelamento funciona em até 12x." \\
        --response-b "Você pode parcelar em até 18x com juros." \\
        --segment MEI
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config.settings import MerchantSegment
from src.inference.predictor import Predictor
from src.models.preference_model import PreferenceModel, PreferenceModelConfig


@click.command()
@click.option("--prompt", required=True, help="User query")
@click.option("--response-a", required=True, help="First candidate response")
@click.option("--response-b", required=True, help="Second candidate response")
@click.option("--segment", default="VAREJO", type=click.Choice(["MEI", "VAREJO", "CORPORATE"]))
@click.option("--model-path", default="./artifacts/models", show_default=True)
@click.option("--model-name", default="microsoft/deberta-v3-large", show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--output-format", default="pretty", type=click.Choice(["pretty", "json"]))
def main(
    prompt: str,
    response_a: str,
    response_b: str,
    segment: str,
    model_path: str,
    model_name: str,
    device: str,
    output_format: str,
) -> None:
    """Run single-example preference inference."""
    model_cfg = PreferenceModelConfig(model_name=model_name)
    model = PreferenceModel(model_cfg)

    checkpoint = Path(model_path) / "best.pt"
    if checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu")
        model.load_state_dict(state, strict=False)
        click.echo(f"Loaded checkpoint: {checkpoint}", err=True)
    else:
        click.echo(f"Warning: No checkpoint at {checkpoint}, using random weights.", err=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    predictor = Predictor(model=model, tokenizer=tokenizer, device=device)

    result = predictor.predict(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        segment=MerchantSegment(segment),
    )

    if output_format == "json":
        click.echo(json.dumps({
            "prob_a_wins": result.prob_a_wins,
            "prob_b_wins": result.prob_b_wins,
            "prob_tie": result.prob_tie,
            "winner": result.winner,
            "latency_ms": result.latency_ms,
        }, indent=2))
    else:
        click.echo(f"\n{'='*50}")
        click.echo(f"  Winner     : {result.winner}")
        click.echo(f"  P(A wins)  : {result.prob_a_wins:.4f}")
        click.echo(f"  P(B wins)  : {result.prob_b_wins:.4f}")
        click.echo(f"  P(tie)     : {result.prob_tie:.4f}")
        click.echo(f"  Latency    : {result.latency_ms:.1f}ms")
        click.echo(f"{'='*50}\n")


if __name__ == "__main__":
    main()
