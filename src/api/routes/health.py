"""Health check route."""
from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas.responses import HealthResponse
from src.__init__ import __version__

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health status.

    Returns:
        HealthResponse with model_loaded, cache_available, and version.
    """
    predictor = getattr(request.app.state, "predictor", None)
    redis = getattr(request.app.state, "redis", None)

    cache_ok = False
    if redis is not None:
        try:
            redis.ping()
            cache_ok = True
        except Exception:
            pass

    model_loaded = predictor is not None
    status = "ok" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        cache_available=cache_ok,
        version=__version__,
    )
