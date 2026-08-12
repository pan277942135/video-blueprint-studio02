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

class CreateJobRequest(BaseModel):
    video_id: str
    attestation_accepted: bool = Field(
        ...,
        description="Must accept source video authorization and compliance attestations."
    )

class JobResponse(BaseModel):
    job_id: str
    video_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    blueprint_id: Optional[str] = None
    quality_score: Optional[float] = None
    error_message: Optional[str] = None

class VideoUploadResponse(BaseModel):
    video_id: str
    file_name: str
    file_size: int
    sha256_hash: str
    uploaded_at: str

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
