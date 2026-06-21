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
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("debug", "true")
    raise RuntimeError("Ironbook Sentry smoke test — intentional error")


@router.get("/debug/value-error")
def debug_value_error() -> dict:
    data = {"reconstruction_quality": "ultra"}
    allowed = {"low", "medium", "high"}
    if data["reconstruction_quality"] not in allowed:
        raise ValueError(
            f"Invalid reconstruction_quality {data['reconstruction_quality']!r}; "
            f"must be one of {allowed}"
        )
    return data


@router.get("/debug/key-error")
def debug_key_error() -> dict:
    project_metadata: dict = {}
    sentry_sdk.add_breadcrumb(category="debug", message="looking up missing key", level="info")
    _ = project_metadata["colmap_poses"]  # KeyError: colmap_poses not found
    return project_metadata


@router.get("/debug/type-error")
def debug_type_error() -> dict:
    splat_count = "41000"  # accidentally a string
    threshold = 10000
    if splat_count > threshold:  # TypeError: '>' not supported between str and int
        return {"needs_lod": True}
    return {"needs_lod": False}


@router.get("/debug/zero-division")
def debug_zero_division() -> dict:
    total_frames = 0
    sentry_sdk.add_breadcrumb(category="debug", message="computing coverage ratio", level="info")
    coverage = 1 / total_frames  # ZeroDivisionError
    return {"coverage": coverage}
