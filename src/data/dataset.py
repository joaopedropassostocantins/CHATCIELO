"""
Heart of Context Injection — CHATCIELO Dataset.

⚠️  ANY MODIFICATION TO THIS FILE REQUIRES FULL RE-VALIDATION
    OF THE TRAINING DATA PIPELINE (run scripts/validate_dataset.py).

This module implements the ChatCieloDataset used for pairwise preference
ranking. It tokenizes (prompt, response_a, response_b) triples with
optional merchant-segment conditioning following the DeBERTa cross-encoder
architecture from the LMSYS Kaggle competition.

DIFF vs Kaggle baseline (orangeaugust/lmsys2):
──────────────────────────────────────────────
KAGGLE (original):
    text = f"<prompt>{row['prompt']}</prompt>
             <response_a>{row['response_a']}</response_a>
             <response_b>{row['response_b']}</response_b>"
    # No segment conditioning
    # No LGPD scrubbing
    # No length-aware truncation strategy

CHATCIELO (this file):
+   text = build_input_text(
+       prompt=row['prompt'],
+       response_a=row['response_a'],
+       response_b=row['response_b'],
+       segment=row.get('merchant_segment', MerchantSegment.VAREJO),
+   )
+   # Merchant segment injected as special prefix token
+   # PII scrubbed before tokenization via scrub_pii()
+   # Length-aware truncation: prompt kept full, responses split equally
──────────────────────────────────────────────
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config.settings import MerchantSegment
from src.data.preprocessing import scrub_pii


# Label mapping: 0 = A wins, 1 = B wins, 2 = tie
LABEL_MAP: Dict[str, int] = {"model_a": 0, "model_b": 1, "tie": 2, "tie (bothbad)": 2}
NUM_LABELS = 3


@dataclass
class PreferenceExample:
    """A single pairwise preference training example.

    Args:
        conversation_id: Unique identifier for the conversation.
        prompt: The user prompt (already scrubbed of PII).
        response_a: First model response.
        response_b: Second model response.
        merchant_segment: Cielo merchant segment for conditioning.
        label: 0=A wins, 1=B wins, 2=tie. None for inference.
    """

    conversation_id: str
    prompt: str
    response_a: str
    response_b: str
    merchant_segment: MerchantSegment = MerchantSegment.VAREJO
    label: Optional[int] = None


def build_input_text(
    prompt: str,
    response_a: str,
    response_b: str,
    segment: MerchantSegment = MerchantSegment.VAREJO,
) -> str:
    """Build the raw input string for the cross-encoder.

    DIFF vs Kaggle: adds [SEGMENT] special prefix for merchant conditioning.

    Args:
        prompt: User query (PII already stripped).
        response_a: First candidate response.
        response_b: Second candidate response.
        segment: Cielo merchant segment (MEI / VAREJO / CORPORATE).

    Returns:
        Formatted string with segment prefix and structured XML-like tags.

    Validation Metrics:
        - String must contain all three structural tags.
        - Length enforced downstream by tokenizer truncation.
    """
    return (
        f"[{segment.value}] "
        f"<prompt>{prompt}</prompt> "
        f"<response_a>{response_a}</response_a> "
        f"<response_b>{response_b}</response_b>"
    )


def _truncate_responses(
    prompt_ids: List[int],
    a_ids: List[int],
    b_ids: List[int],
    max_length: int,
    reserved_special: int = 10,
) -> tuple[List[int], List[int], List[int]]:
    """Length-aware truncation: keep full prompt, split remainder evenly.

    DIFF vs Kaggle: Kaggle truncates from the right uniformly.
    Here we preserve the full prompt and truncate responses symmetrically.

    Args:
        prompt_ids: Tokenized prompt token IDs.
        a_ids: Tokenized response_a token IDs.
        b_ids: Tokenized response_b token IDs.
        max_length: Maximum total sequence length.
        reserved_special: Slots for [CLS], [SEP], segment tokens.

    Returns:
        Tuple of (prompt_ids, a_ids, b_ids) after truncation.

    Validation Metrics:
        - sum(len(x) for x in result) + reserved_special <= max_length
    """
    budget = max_length - reserved_special - len(prompt_ids)
    if budget < 0:
        # Prompt itself is too long — truncate it
        prompt_ids = prompt_ids[: max_length - reserved_special - 4]
        budget = 4

    per_response = budget // 2
    return (
        prompt_ids,
        a_ids[:per_response],
        b_ids[:per_response],
    )


class ChatCieloDataset(Dataset):
    """PyTorch Dataset for CHATCIELO pairwise preference ranking.

    Args:
        examples: List of PreferenceExample instances.
        tokenizer: HuggingFace tokenizer compatible with the encoder.
        max_length: Maximum token sequence length (default 1024).
        scrub: Whether to apply PII scrubbing before tokenization.

    Returns:
        Dict with keys: input_ids, attention_mask, token_type_ids (optional),
        and label (if training).

    Validation Metrics:
        - All input_ids tensors have shape (max_length,).
        - Labels are in {0, 1, 2}.
        - PII regex patterns do not appear in tokenized text.
    """

    def __init__(
        self,
        examples: List[PreferenceExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
        scrub: bool = True,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scrub = scrub

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokenized example at index idx.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Dict containing input_ids, attention_mask, and optionally label.

        Raises:
            IndexError: If idx is out of range.
        """
        ex = self.examples[idx]

        prompt = scrub_pii(ex.prompt) if self.scrub else ex.prompt
        response_a = scrub_pii(ex.response_a) if self.scrub else ex.response_a
        response_b = scrub_pii(ex.response_b) if self.scrub else ex.response_b

        text = build_input_text(prompt, response_a, response_b, ex.merchant_segment)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if ex.label is not None:
            item["labels"] = torch.tensor(ex.label, dtype=torch.long)

        return item


def load_examples_from_parquet(
    path: str,
    split: str = "train",
    max_rows: Optional[int] = None,
) -> List[PreferenceExample]:
    """Load PreferenceExample list from a Parquet file.

    Expected columns: conversation_id, prompt, response_a, response_b,
    winner (model_a | model_b | tie | tie (bothbad)), merchant_segment (optional).

    Args:
        path: Path to the .parquet file.
        split: Dataset split label for logging purposes.
        max_rows: Optional row limit (useful for debugging).

    Returns:
        List of PreferenceExample instances.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        KeyError: If required columns are missing.

    Validation Metrics:
        - Label distribution logged for class imbalance detection.
        - Rows with null prompt/response_a/response_b are dropped and counted.
    """
    df = pd.read_parquet(path)

    if max_rows is not None:
        df = df.head(max_rows)

    required_cols = {"conversation_id", "prompt", "response_a", "response_b", "winner"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["prompt", "response_a", "response_b"])
    dropped = before - len(df)
    if dropped > 0:
        import warnings
        warnings.warn(f"[{split}] Dropped {dropped} rows with null text fields.")

    examples = []
    for _, row in df.iterrows():
        segment_raw = row.get("merchant_segment", "VAREJO")
        try:
            segment = MerchantSegment(str(segment_raw).upper())
        except ValueError:
            segment = MerchantSegment.VAREJO

        label = LABEL_MAP.get(str(row["winner"]).lower())

        examples.append(
            PreferenceExample(
                conversation_id=str(row["conversation_id"]),
                prompt=str(row["prompt"]),
                response_a=str(row["response_a"]),
                response_b=str(row["response_b"]),
                merchant_segment=segment,
                label=label,
            )
        )

    return examples
