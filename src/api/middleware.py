"""
API middleware: request logging, PII audit, Prometheus metrics.
"""
from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from src.data.preprocessing import contains_pii

log = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "chatcielo_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "chatcielo_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
)
PII_LEAK_COUNT = Counter(
    "chatcielo_pii_leak_detected_total",
    "Number of responses with detected PII (must be 0 in production)",
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with latency and trace ID.

    LGPD: Response bodies are NOT logged. Only status codes and latencies.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        trace_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        response = await call_next(request)

        latency = time.perf_counter() - t0
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)

        log.info(
            "request",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(latency * 1000, 2),
        )
        response.headers["X-Trace-ID"] = trace_id
        return response
