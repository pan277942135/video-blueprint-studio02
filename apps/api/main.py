import datetime
import hashlib
import json
import os
import tempfile
from typing import Any
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from apps.api.job_store import job_store
from packages.blueprint_schema.models import (
    AnalysisJob,
    CreateAnalysisRequest,
    RetryAnalysisRequest,
    ValidationReport,
    VideoRecord,
)
from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline

app = FastAPI(
    title="Video Blueprint Studio API",
    version="1.0.0",
    description="Local/offline API for video analysis and blueprint export."
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
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    }

@app.post("/api/v1/videos", response_model=VideoRecord, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    authorization_attested: bool = Form(...),
    adult_subject_attested: bool = Form(...)
):
    if not authorization_attested or not adult_subject_attested:
        raise HTTPException(
            status_code=400,
            detail="Both authorization_attested and adult_subject_attested must be true."
        )

    contents = await file.read()
    sha256_hash = hashlib.sha256(contents).hexdigest()
    
    video_record = job_store.store_video(
        file_name=file.filename or "uploaded_video.mp4",
        sha256_hash=sha256_hash
    )
    
    return VideoRecord.model_validate(video_record)

@app.post("/api/v1/analyses", response_model=AnalysisJob, status_code=status.HTTP_202_ACCEPTED)
def create_analysis(req: CreateAnalysisRequest):
    video_id_str = str(req.video_id)
    if video_id_str not in job_store.videos:
        # Register mock video record for video_id if not created via upload endpoint first
        job_store.store_video(
            file_name="source_video.mp4",
            sha256_hash="0000000000000000000000000000000000000000000000000000000000000000",
            custom_video_id=video_id_str
        )
        
    job_record = job_store.create_analysis(
        video_id=video_id_str,
        modules=req.modules,
        config_overrides=req.config_overrides
    )
    return AnalysisJob(**job_record)

def _execute_mock_analysis(analysis_id: str):
    job = job_store.get_analysis(analysis_id)
    if not job:
        return
    
    video = job_store.videos.get(job["video_id"], {})
    video_name = video.get("file_name", "sample_video.mp4")
    video_sha256 = video.get("sha256", "0000000000000000000000000000000000000000000000000000000000000000")
    
    blueprint, sidecars = run_deterministic_mock_pipeline(analysis_id, video_name, video_sha256)
    
    is_valid, val_errors = validator.validate(blueprint)
    val_report = {
        "valid": is_valid,
        "schema_version": "Draft 2020-12",
        "validated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "errors": val_errors,
        "summary": {"passed_rules": 25 - len(val_errors), "failed_rules": len(val_errors)}
    }
    
    job_store.store_blueprint(analysis_id, blueprint, sidecars)
    
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, f"bundle_{analysis_id}.zip")
    create_bundle_zip(blueprint, sidecars, val_report, zip_path)
    job_store.store_bundle_path(analysis_id, zip_path)
    
    updated_stages = [
        {"stage": "shots", "status": "succeeded", "progress": 1.0},
        {"stage": "people", "status": "succeeded", "progress": 1.0},
        {"stage": "pose", "status": "succeeded", "progress": 1.0},
        {"stage": "camera", "status": "succeeded", "progress": 1.0},
        {"stage": "micro_motion", "status": "succeeded", "progress": 1.0}
    ]
    
    job_store.update_analysis(
        analysis_id,
        status="succeeded",
        progress=1.0,
        stages=updated_stages
    )

@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisJob)
def get_analysis(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")
    
    if job["status"] in ["queued", "running"]:
        _execute_mock_analysis(analysis_id_str)
        job = job_store.get_analysis(analysis_id_str)

    assert job is not None
    return AnalysisJob.model_validate(job)

@app.post("/api/v1/analyses/{analysis_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
def cancel_analysis(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")
    
    updated = job_store.update_analysis(analysis_id_str, status="cancelled")
    return AnalysisJob(**updated)

@app.post("/api/v1/analyses/{analysis_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_analysis(analysis_id: UUID, req: RetryAnalysisRequest | None = None):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")
    
    reset_stages = [
        {"stage": "shots", "status": "pending", "progress": 0.0},
        {"stage": "people", "status": "pending", "progress": 0.0},
        {"stage": "pose", "status": "pending", "progress": 0.0},
        {"stage": "camera", "status": "pending", "progress": 0.0},
        {"stage": "micro_motion", "status": "pending", "progress": 0.0}
    ]
    
    updated = job_store.update_analysis(
        analysis_id_str,
        status="queued",
        progress=0.0,
        stages=reset_stages,
        error=None
    )
    return AnalysisJob(**updated)

@app.get("/api/v1/analyses/{analysis_id}/blueprint")
def get_blueprint(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    bp = job_store.get_blueprint(analysis_id_str)
    if not bp:
        job = job_store.get_analysis(analysis_id_str)
        if job:
            _execute_mock_analysis(analysis_id_str)
            bp = job_store.get_blueprint(analysis_id_str)
            
    if not bp:
        raise HTTPException(status_code=404, detail=f"Blueprint for analysis {analysis_id_str} not found")
    return bp

@app.get("/api/v1/analyses/{analysis_id}/bundle")
def download_bundle(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    zip_path = job_store.get_bundle_path(analysis_id_str)
    if not zip_path or not os.path.exists(zip_path):
        job = job_store.get_analysis(analysis_id_str)
        if job:
            _execute_mock_analysis(analysis_id_str)
            zip_path = job_store.get_bundle_path(analysis_id_str)
            
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"Bundle for analysis {analysis_id_str} not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"bundle_{analysis_id_str}.zip"
    )

@app.post("/api/v1/analyses/{analysis_id}/validate", response_model=ValidationReport)
def validate_analysis(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    bp = job_store.get_blueprint(analysis_id_str)
    if not bp:
        job = job_store.get_analysis(analysis_id_str)
        if job:
            _execute_mock_analysis(analysis_id_str)
            bp = job_store.get_blueprint(analysis_id_str)
            
    if not bp:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")
        
    is_valid, errors = validator.validate(bp)
    return ValidationReport(
        valid=is_valid,
        validated_at=datetime.datetime.now(datetime.UTC).isoformat(),
        errors=errors,
        summary={"passed_rules": 25 - len(errors), "failed_rules": len(errors)}
    )

@app.get("/api/v1/analyses/{analysis_id}/artifacts")
def list_artifacts(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")
    return {"analysis_id": analysis_id_str, "artifacts": []}

@app.patch("/api/v1/analyses/{analysis_id}/annotations")
def patch_annotations(analysis_id: UUID, body: dict[str, Any]):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")
    return {"analysis_id": analysis_id_str, "status": "revision_created"}

@app.get("/api/v1/analyses/{analysis_id}/events")
def stream_progress(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")

    def event_generator():
        data_json = json.dumps({"status": job["status"], "progress": job.get("progress", 0.0)})
        yield f"event: status\ndata: {data_json}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

