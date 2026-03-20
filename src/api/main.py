"""
CHATCIELO FastAPI Application Entry Point.

Startup sequence:
  1. Load settings
  2. Initialize Redis client
  3. Load PreferenceModel + Tokenizer
  4. Mount routes
  5. Start Prometheus metrics endpoint

Shutdown sequence:
  1. Close Redis connection
"""
from __future__ import annotations

import structlog
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.__init__ import __version__
from src.api.routes import feedback_router, health_router, preference_router
from src.config.settings import get_settings

log = structlog.get_logger()


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routes and middleware mounted.
    """
    cfg = get_settings()

    app = FastAPI(
        title="CHATCIELO — Preference Ranking API",
        description=(
            "Pairwise preference ranking for LLM response selection. "
            "Merchant-aware, LGPD-compliant, latency target P99 < 300ms."
        ),
        version=__version__,
        docs_url="/docs" if cfg.app_env.value != "production" else None,
        redoc_url="/redoc" if cfg.app_env.value != "production" else None,
    )

    # ── Startup/Shutdown ────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup() -> None:
        log.info("startup", version=__version__, env=cfg.app_env.value)

        # Redis
        try:
            import redis as redis_lib
            app.state.redis = redis_lib.from_url(cfg.redis_url, decode_responses=True)
            app.state.redis.ping()
            log.info("redis_connected", url=cfg.redis_url)
        except Exception as e:
            log.warning("redis_unavailable", error=str(e))
            app.state.redis = None

        # Model
        try:
            from src.inference.predictor import Predictor
            app.state.predictor = Predictor.from_config()
            log.info("model_loaded", model=cfg.model_name)
        except Exception as e:
            log.error("model_load_failed", error=str(e))
            app.state.predictor = None

    @app.on_event("shutdown")
    async def shutdown() -> None:
        redis = getattr(app.state, "redis", None)
        if redis is not None:
            redis.close()
        log.info("shutdown")

    # ── Routes ──────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(preference_router)
    app.include_router(feedback_router)

    # ── Prometheus metrics ───────────────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()
