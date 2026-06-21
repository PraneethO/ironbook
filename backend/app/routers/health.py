"""Health endpoint — reports the active reconstruction backend."""
from __future__ import annotations

import sentry_sdk
from fastapi import APIRouter

from ..models import HealthResponse
from ..services import jobs as job_svc

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", reconstruction_backend=job_svc.backend_name())


@router.get("/debug/error")
def debug_error() -> dict:
    """Deliberately raises an exception so you can verify Sentry error capture."""
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("debug", "true")
        scope.set_extra("triggered_by", "smoke-test")
    raise RuntimeError("Ironbook Sentry smoke test — intentional error")
