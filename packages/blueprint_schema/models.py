from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"

class VideoRecord(BaseModel):
    video_id: str
    file_name: str
    sha256: str
    status: str = "uploaded"

class CreateAnalysisRequest(BaseModel):
    video_id: str
    modules: Optional[List[str]] = Field(
        default_factory=lambda: [
            "shots", "people", "pose", "face", "hands",
            "masks", "camera", "flow", "micro_motion", "environment", "overlays"
        ]
    )
    config_overrides: Optional[Dict[str, Any]] = None

class RetryAnalysisRequest(BaseModel):
    stages: Optional[List[str]] = None
    invalidate_downstream: bool = True

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class AnalysisJob(BaseModel):
    analysis_id: str
    video_id: str
    status: JobStatus
    progress: float = 0.0
    stages: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[ErrorDetail] = None

class ValidationReport(BaseModel):
    valid: bool
    schema_version: str = "Draft 2020-12"
    validated_at: str
    errors: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

class PatchOperation(BaseModel):
    op: str # add, replace, remove
    path: str
    value: Optional[Any] = None
