import datetime
import hashlib
import os
import tempfile
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from packages.blueprint_schema.models import (
    JobStatus, CreateJobRequest, JobResponse, VideoUploadResponse, ValidationReport
)
from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from apps.api.job_store import job_store

app = FastAPI(
    title="Video Blueprint Studio API",
    version="0.1.0",
    description="Video Blueprint Studio M1 - Epic E0 Deterministic Mock REST Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

validator = BlueprintValidator()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "epic": "E0",
        "service": "Video Blueprint Studio API",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

@app.post("/api/v1/videos/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    sha256_hash = hashlib.sha256(contents).hexdigest()
    
    video_record = job_store.store_video(
        file_name=file.filename or "uploaded_video.mp4",
        file_size=len(contents),
        sha256_hash=sha256_hash
    )
    
    return VideoUploadResponse(**video_record)

@app.post("/api/v1/jobs", response_model=JobResponse)
def create_job(req: CreateJobRequest):
    if not req.attestation_accepted:
        raise HTTPException(
            status_code=400,
            detail="Mandatory attestation: You must explicitly confirm authorization and legal compliance before processing video."
        )
    
    if req.video_id not in job_store.videos:
        # Create a mock video entry if created without explicit upload first
        job_store.store_video("mock_video.mp4", 1024*1024, "00000000000000000000000000000000")
        
    job_record = job_store.create_job(req.video_id)
    return JobResponse(**job_record)

@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Auto-advance queued job via deterministic mock pipeline
    if job["status"] in ["queued", "running"]:
        video = job_store.videos.get(job["video_id"], {})
        video_name = video.get("file_name", "sample_video.mp4")
        video_sha256 = video.get("sha256_hash", "00000000000000000000000000000000")
        
        # Run Mock Pipeline
        blueprint, sidecars = run_deterministic_mock_pipeline(job_id, video_name, video_sha256)
        
        # Validate against Draft 2020-12 Schema
        is_valid, val_errors = validator.validate(blueprint)
        val_report = {
            "valid": is_valid,
            "schema_version": "Draft 2020-12",
            "validated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "errors": val_errors,
            "summary": {"passed_rules": 25 - len(val_errors), "failed_rules": len(val_errors)}
        }
        
        blueprint_id = blueprint["blueprint_id"]
        job_store.store_blueprint(blueprint_id, blueprint, sidecars)
        
        # Export ZIP Bundle
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, f"bundle_{blueprint_id}.zip")
        create_bundle_zip(blueprint, sidecars, val_report, zip_path)
        job_store.store_bundle_path(blueprint_id, zip_path)
        
        # Update job status
        job = job_store.update_job(
            job_id,
            status="succeeded",
            blueprint_id=blueprint_id,
            quality_score=blueprint["quality"]["overall_score"]
        )

    return JobResponse(**job)

@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    updated = job_store.update_job(job_id, status="cancelled")
    return JobResponse(**updated)

@app.get("/api/v1/blueprints/{blueprint_id}")
def get_blueprint(blueprint_id: str):
    bp = job_store.get_blueprint(blueprint_id)
    if not bp:
        raise HTTPException(status_code=404, detail=f"Blueprint {blueprint_id} not found")
    return bp

@app.post("/api/v1/blueprints/validate", response_model=ValidationReport)
def validate_blueprint(blueprint: Dict[str, Any]):
    is_valid, errors = validator.validate(blueprint)
    return ValidationReport(
        valid=is_valid,
        validated_at=datetime.datetime.utcnow().isoformat() + "Z",
        errors=errors,
        summary={"passed_rules": 25 - len(errors), "failed_rules": len(errors)}
    )

@app.get("/api/v1/bundles/{blueprint_id}")
def download_bundle(blueprint_id: str):
    zip_path = job_store.get_bundle_path(blueprint_id)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"Bundle for blueprint {blueprint_id} not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"bundle_{blueprint_id}.zip"
    )
