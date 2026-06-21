"""Health endpoint — reports the active reconstruction backend."""
from __future__ import annotations

from fastapi import APIRouter

from ..models import HealthResponse
from ..services import jobs as job_svc

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", reconstruction_backend=job_svc.backend_name())
