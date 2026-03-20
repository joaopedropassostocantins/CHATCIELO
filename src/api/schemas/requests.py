"""Request schemas for the CHATCIELO API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PreferenceRequest(BaseModel):
    """Request body for the /preference endpoint.

    Args:
        prompt: User query. Must not be empty.
        response_a: First candidate response.
        response_b: Second candidate response.
        merchant_segment: Cielo segment: MEI, VAREJO, or CORPORATE.
        conversation_id: Optional trace ID for audit logging.
    """

    prompt: str = Field(..., min_length=1, max_length=4096)
    response_a: str = Field(..., min_length=1, max_length=4096)
    response_b: str = Field(..., min_length=1, max_length=4096)
    merchant_segment: Literal["MEI", "VAREJO", "CORPORATE"] = "VAREJO"
    conversation_id: Optional[str] = Field(None, max_length=128)

    @field_validator("prompt", "response_a", "response_b")
    @classmethod
    def no_empty_after_strip(cls, v: str) -> str:
        """Reject strings that are only whitespace."""
        if not v.strip():
            raise ValueError("Field must contain non-whitespace characters.")
        return v


class FeedbackRequest(BaseModel):
    """User feedback on a prediction — used for the feedback loop.

    Args:
        conversation_id: Must match a prior PreferenceRequest conversation_id.
        actual_winner: The winner the user confirmed: 'A', 'B', or 'tie'.
        satisfaction_score: Optional CSAT score 1-5.
        channel: The channel through which feedback was submitted.
    """

    conversation_id: str = Field(..., min_length=1, max_length=128)
    actual_winner: Literal["A", "B", "tie"]
    satisfaction_score: Optional[int] = Field(None, ge=1, le=5)
    channel: Literal["webchat", "api", "whatsapp", "mobile"] = "api"


class BatchPreferenceRequest(BaseModel):
    """Batch inference request (up to 32 triples per call).

    Args:
        items: List of preference requests.
    """

    items: List[PreferenceRequest] = Field(..., min_length=1, max_length=32)
