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


# --- Reasoning navigation agent (CONTRACT extension) ---------------------


class CameraSnapshot(BaseModel):
    mode: str = "orbit"
    fov: float = 1.0
    eye: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    target: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds: dict = Field(default_factory=lambda: {"min": [0, 0, 0], "max": [0, 0, 0]})


class AgentTurn(BaseModel):
    role: str  # "user" | "assistant"
    text: str


class AgentActRequest(BaseModel):
    message: str
    screenshot_b64: Optional[str] = None  # PNG base64, no data: prefix
    camera: CameraSnapshot = Field(default_factory=CameraSnapshot)
    history: List[AgentTurn] = Field(default_factory=list)


class AgentAction(BaseModel):
    type: str
    direction: Optional[str] = None
    amount: Optional[float] = None
    target_2d: Optional[List[float]] = None
    label: Optional[str] = None


class AgentActResponse(BaseModel):
    answer: str
    diagram: str = ""
    actions: List[AgentAction] = Field(default_factory=list)
