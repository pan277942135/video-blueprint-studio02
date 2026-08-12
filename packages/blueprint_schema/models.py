from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"

class VideoRecord(BaseModel):
    video_id: UUID
    file_name: str
    sha256: str
    status: str = "uploaded"

class CreateAnalysisRequest(BaseModel):
    video_id: UUID
    modules: list[str] | None = Field(
        default_factory=lambda: [
            "shots", "people", "pose", "face", "hands",
            "masks", "camera", "flow", "micro_motion", "environment", "overlays"
        ]
    )
    config_overrides: dict[str, Any] | None = None

class RetryAnalysisRequest(BaseModel):
    stages: list[str] | None = None
    invalidate_downstream: bool = True

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None

class AnalysisJob(BaseModel):
    analysis_id: UUID
    video_id: UUID
    status: JobStatus
    progress: float = 0.0
    stages: list[dict[str, Any]] = Field(default_factory=list)
    error: ErrorDetail | None = None

class ValidationReport(BaseModel):
    valid: bool
    schema_version: str = "Draft 2020-12"
    validated_at: str
    errors: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

class PatchOperation(BaseModel):
    op: str # add, replace, remove
    path: str
    value: Any | None = None
