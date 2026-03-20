"""Preference prediction routes."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas.requests import BatchPreferenceRequest, PreferenceRequest
from src.api.schemas.responses import BatchPreferenceResponse, PreferenceResponse
from src.config.settings import MerchantSegment

router = APIRouter(prefix="/preference", tags=["preference"])


def get_predictor(request: Request) -> Any:
    """Dependency: retrieve the Predictor from app state."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return predictor


@router.post("/", response_model=PreferenceResponse, status_code=200)
async def predict_preference(
    body: PreferenceRequest,
    predictor: Any = Depends(get_predictor),
) -> PreferenceResponse:
    """Predict which response is preferred by the user.

    Args:
        body: PreferenceRequest containing prompt and two candidate responses.
        predictor: Injected Predictor instance.

    Returns:
        PreferenceResponse with probabilities and predicted winner.

    Raises:
        HTTPException 422: If input validation fails.
        HTTPException 503: If model is not loaded.
        HTTPException 500: On unexpected inference error.
    """
    try:
        segment = MerchantSegment(body.merchant_segment)
        result = predictor.predict(
            prompt=body.prompt,
            response_a=body.response_a,
            response_b=body.response_b,
            segment=segment,
        )
        return PreferenceResponse(
            prob_a_wins=result.prob_a_wins,
            prob_b_wins=result.prob_b_wins,
            prob_tie=result.prob_tie,
            winner=result.winner,
            latency_ms=result.latency_ms,
            from_cache=result.from_cache,
            conversation_id=body.conversation_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference error.") from e


@router.post("/batch", response_model=BatchPreferenceResponse, status_code=200)
async def predict_batch(
    body: BatchPreferenceRequest,
    predictor: Any = Depends(get_predictor),
) -> BatchPreferenceResponse:
    """Batch preference prediction for up to 32 triples.

    Args:
        body: BatchPreferenceRequest with list of triples.
        predictor: Injected Predictor instance.

    Returns:
        BatchPreferenceResponse with one result per input.
    """
    t0 = time.perf_counter()
    results = []
    for item in body.items:
        segment = MerchantSegment(item.merchant_segment)
        r = predictor.predict(
            prompt=item.prompt,
            response_a=item.response_a,
            response_b=item.response_b,
            segment=segment,
        )
        results.append(
            PreferenceResponse(
                prob_a_wins=r.prob_a_wins,
                prob_b_wins=r.prob_b_wins,
                prob_tie=r.prob_tie,
                winner=r.winner,
                latency_ms=r.latency_ms,
                from_cache=r.from_cache,
                conversation_id=item.conversation_id,
            )
        )

    return BatchPreferenceResponse(
        results=results,
        total_latency_ms=(time.perf_counter() - t0) * 1000,
    )
