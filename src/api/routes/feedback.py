"""Feedback collection route — feeds the retraining loop."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from src.api.schemas.requests import FeedbackRequest
from src.api.schemas.responses import FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
) -> FeedbackResponse:
    """Accept user feedback and enqueue it for the retraining loop.

    Feedback is persisted to Redis stream 'chatcielo:feedback' for
    async consumption by the retraining pipeline.

    Args:
        body: FeedbackRequest with conversation_id and actual_winner.
        request: FastAPI request for app state access.

    Returns:
        FeedbackResponse acknowledging receipt.
    """
    redis = getattr(request.app.state, "redis", None)

    record = {
        "conversation_id": body.conversation_id,
        "actual_winner": body.actual_winner,
        "satisfaction_score": body.satisfaction_score,
        "channel": body.channel,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if redis is not None:
        try:
            redis.xadd("chatcielo:feedback", {"data": json.dumps(record)})
            return FeedbackResponse(accepted=True, message="Feedback enqueued.")
        except Exception as e:
            return FeedbackResponse(accepted=False, message=f"Cache write failed: {e}")

    # Graceful degradation: log to stdout if Redis unavailable
    import structlog
    log = structlog.get_logger()
    log.info("feedback_received", **record)
    return FeedbackResponse(accepted=True, message="Feedback logged (no cache).")
