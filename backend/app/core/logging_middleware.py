"""
TrustLoop Observability, Tracing & Error Handling Middleware.

Provides request correlation IDs (X-Request-ID), structured audit logging,
request body size limits, and uniform exception formatting.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging
import json

from backend.app.core.config import settings

logger = logging.getLogger("trustloop.access")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches correlation IDs, enforces payload size limits,
    and records latency metrics for every incoming HTTP request.
    """
    async def dispatch(self, request: Request, call_next):
        # 1. Extract or generate Correlation ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # 2. Enforce request body size limit
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.MAX_REQUEST_SIZE_BYTES:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "error": "PAYLOAD_TOO_LARGE",
                            "detail": f"Request body exceeds maximum allowed size of {settings.MAX_REQUEST_SIZE_BYTES} bytes.",
                            "request_id": request_id,
                            "status_code": status.HTTP_413_CONTENT_TOO_LARGE,
                        },
                        headers={"X-Request-ID": request_id},
                    )
            except ValueError:
                pass

        # 3. Process request with high-resolution latency timer
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = str(latency_ms)

            # Structured access log
            if not request.url.path.startswith("/health"):
                logger.info(
                    json.dumps({
                        "event": "HTTP_REQUEST",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "latency_ms": latency_ms,
                        "client_ip": request.client.host if request.client else "unknown",
                    })
                )

            return response

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                json.dumps({
                    "event": "UNHANDLED_EXCEPTION",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": latency_ms,
                    "error": str(exc),
                })
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "INTERNAL_SERVER_ERROR",
                    "detail": "An unexpected error occurred during request processing.",
                    "request_id": request_id,
                    "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                },
                headers={"X-Request-ID": request_id},
            )
