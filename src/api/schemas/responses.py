"""Response schemas for the CHATCIELO API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PreferenceResponse(BaseModel):
    """Response from the /preference endpoint.

    Args:
        prob_a_wins: Probability in [0, 1].
        prob_b_wins: Probability in [0, 1].
        prob_tie: Probability in [0, 1].
        winner: Predicted winner: 'A', 'B', or 'tie'.
        latency_ms: Server-side inference time.
        from_cache: True if served from Redis cache.
        conversation_id: Echo of the request conversation_id.
    """

    prob_a_wins: float = Field(..., ge=0.0, le=1.0)
    prob_b_wins: float = Field(..., ge=0.0, le=1.0)
    prob_tie: float = Field(..., ge=0.0, le=1.0)
    winner: Literal["A", "B", "tie"]
    latency_ms: float
    from_cache: bool = False
    conversation_id: Optional[str] = None


class BatchPreferenceResponse(BaseModel):
    """Response from the /preference/batch endpoint."""

    results: List[PreferenceResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Response from /health."""

    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    cache_available: bool
    version: str


class FeedbackResponse(BaseModel):
    """Acknowledgment of a feedback submission."""

    accepted: bool
    message: str
