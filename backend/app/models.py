"""Pydantic schemas matching CONTRACT.md §1 exactly."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class Project(BaseModel):
    id: str
    name: str
    status: str  # draft | uploading | queued | processing | ready | failed
    created_at: str
    updated_at: str
    photo_count: int = 0
    thumbnail_url: Optional[str] = None
    has_asset: bool = False


class RejectedFile(BaseModel):
    filename: str
    reason: str


class ValidationReport(BaseModel):
    accepted: int = 0
    rejected: List[RejectedFile] = Field(default_factory=list)
    photo_count: int = 0
    coverage_score: float = 0.0
    quality_score: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    ready_to_reconstruct: bool = False


class UploadItem(BaseModel):
    filename: str
    thumbnail_url: str
    width: int
    height: int
    sharpness: float


class LogEntry(BaseModel):
    ts: str
    level: str  # info | warn | error
    stage: str
    message: str


class StageState(BaseModel):
    key: str
    label: str
    status: str  # pending | active | done | failed
    progress: float = 0.0


class Job(BaseModel):
    project_id: str
    status: str  # queued | processing | ready | failed
    progress: float = 0.0
    current_stage: Optional[str] = None
    stages: List[StageState] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    error: Optional[str] = None


class ShareResponse(BaseModel):
    url: str


class AssetInfo(BaseModel):
    splat_count: int
    bytes: int
    bounds: dict
    format: str = "splat"


class HealthResponse(BaseModel):
    status: str
    reconstruction_backend: str
