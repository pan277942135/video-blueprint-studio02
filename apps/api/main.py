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
from packages.pipeline_core.media_normalizer import MediaNormalizationError
from packages.pipeline_core.media_probe import MediaProbeError
from packages.pipeline_core.production_media_pipeline import run_production_media_pipeline

app = FastAPI(
    title="Video Blueprint Studio API",
    version="1.0.0",
    description="Local/offline API for video analysis and blueprint export.",
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
        "epic": "E10",
        "service": "Video Blueprint Studio API",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


@app.post("/api/v1/videos", response_model=VideoRecord, status_code=status.HTTP_201_CREATED)
def upload_video(
    file: UploadFile = File(...),
    authorization_attested: bool = Form(...),
    adult_subject_attested: bool = Form(...),
):
    if not authorization_attested or not adult_subject_attested:
        raise HTTPException(
            status_code=400,
            detail="Both authorization_attested and adult_subject_attested must be true.",
        )

    contents = file.file.read()
    sha256_hash = hashlib.sha256(contents).hexdigest()
    file_name = file.filename or "uploaded_video.mp4"
    temp_dir = tempfile.gettempdir()
    saved_path = os.path.join(temp_dir, f"vbs_upload_{sha256_hash[:12]}_{file_name}")
    with open(saved_path, "wb") as f:
        f.write(contents)

    video_record = job_store.store_video(
        file_name=file_name,
        sha256_hash=sha256_hash,
        file_path=saved_path,
    )
    return VideoRecord.model_validate(video_record)


@app.post("/api/v1/analyses", response_model=AnalysisJob, status_code=status.HTTP_202_ACCEPTED)
def create_analysis(req: CreateAnalysisRequest):
    video_id_str = str(req.video_id)
    video = job_store.videos.get(video_id_str)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video {video_id_str} not found")

    video_path = video.get("file_path")
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(
            status_code=400,
            detail="Analysis requires a persisted video uploaded through /api/v1/videos.",
        )

    job_record = job_store.create_analysis(
        video_id=video_id_str,
        modules=req.modules,
        config_overrides=req.config_overrides,
    )
    return AnalysisJob(**job_record)


def _execute_real_analysis(analysis_id: str):
    job = job_store.get_analysis(analysis_id)
    if not job:
        return

    video = job_store.videos.get(job["video_id"])
    if not video:
        job_store.update_analysis(
            analysis_id,
            status="failed",
            progress=0.0,
            error={"code": "REAL_MEDIA_REQUIRED", "message": "Uploaded video record not found."},
        )
        return

    video_name = video.get("file_name", "uploaded_video.mp4")
    video_sha256 = video.get("sha256", "")
    video_path = video.get("file_path")
    if not video_path or not os.path.isfile(video_path):
        job_store.update_analysis(
            analysis_id,
            status="failed",
            progress=0.0,
            error={"code": "REAL_MEDIA_REQUIRED", "message": "Persisted uploaded video file is missing."},
        )
        return

    try:
        blueprint, sidecars = run_production_media_pipeline(
            job_id=analysis_id,
            video_file_name=video_name,
            video_sha256=video_sha256,
            video_path=video_path,
        )
    except Exception as e:  # noqa: BLE001
        if isinstance(e, MediaNormalizationError):
            err_code = "MEDIA_NORMALIZATION_FAILED"
        elif isinstance(e, MediaProbeError):
            err_code = "MEDIA_PROBE_FAILED"
        else:
            err_code = "ANALYSIS_FAILED"

        failed_stages = [
            {"stage": "shots", "status": "failed", "progress": 0.0},
            {"stage": "people", "status": "failed", "progress": 0.0},
            {"stage": "pose", "status": "failed", "progress": 0.0},
            {"stage": "camera", "status": "failed", "progress": 0.0},
            {"stage": "micro_motion", "status": "failed", "progress": 0.0},
            {"stage": "environment", "status": "failed", "progress": 0.0},
        ]
        job_store.update_analysis(
            analysis_id,
            status="failed",
            progress=0.0,
            stages=failed_stages,
            error={"code": err_code, "message": str(e)},
        )
        return

    is_valid, val_errors = validator.validate(blueprint)
    val_report = {
        "valid": is_valid,
        "schema_version": "Draft 2020-12",
        "validated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "errors": val_errors,
        "summary": {"passed_rules": max(0, 25 - len(val_errors)), "failed_rules": len(val_errors)},
    }

    if not is_valid:
        job_store.update_analysis(
            analysis_id,
            status="failed",
            progress=0.0,
            error={"code": "BLUEPRINT_VALIDATION_FAILED", "message": "; ".join(val_errors)},
        )
        return

    job_store.store_blueprint(analysis_id, blueprint, sidecars)

    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, f"bundle_{analysis_id}.zip")
    create_bundle_zip(blueprint, sidecars, val_report, zip_path)
    job_store.store_bundle_path(analysis_id, zip_path)

    processing_stages = blueprint.get("processing", {}).get("stages", [])
    updated_stages = [
        {
            "stage": str(stage.get("name")),
            "status": str(stage.get("status", "succeeded")),
            "progress": float(stage.get("progress", 1.0)),
        }
        for stage in processing_stages
        if isinstance(stage, dict) and isinstance(stage.get("name"), str)
    ]

    job_store.update_analysis(
        analysis_id,
        status="succeeded",
        progress=1.0,
        stages=updated_stages,
        error=None,
    )


@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisJob)
def get_analysis(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")

    if job["status"] in ["queued", "running"]:
        _execute_real_analysis(analysis_id_str)
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

    video = job_store.videos.get(job["video_id"])
    if not video or not video.get("file_path") or not os.path.isfile(video["file_path"]):
        raise HTTPException(status_code=400, detail="Retry requires the original persisted uploaded video.")

    reset_stages = [
        {"stage": "shots", "status": "pending", "progress": 0.0},
        {"stage": "people", "status": "pending", "progress": 0.0},
        {"stage": "pose", "status": "pending", "progress": 0.0},
        {"stage": "camera", "status": "pending", "progress": 0.0},
        {"stage": "micro_motion", "status": "pending", "progress": 0.0},
        {"stage": "environment", "status": "pending", "progress": 0.0},
    ]

    updated = job_store.update_analysis(
        analysis_id_str,
        status="queued",
        progress=0.0,
        stages=reset_stages,
        error=None,
    )
    return AnalysisJob(**updated)


def _require_completed_job(analysis_id_str: str) -> dict[str, Any]:
    job = job_store.get_analysis(analysis_id_str)
    if not job:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")

    if job["status"] in ["queued", "running"]:
        _execute_real_analysis(analysis_id_str)
        job = job_store.get_analysis(analysis_id_str)

    assert job is not None
    if job.get("status") == "failed":
        err_obj = job.get("error")
        err_msg = err_obj.get("message") if isinstance(err_obj, dict) else "Unknown error"
        raise HTTPException(status_code=400, detail=f"Analysis job failed: {err_msg}")
    return job


@app.get("/api/v1/analyses/{analysis_id}/blueprint")
def get_blueprint(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    _require_completed_job(analysis_id_str)

    bp = job_store.get_blueprint(analysis_id_str)
    if not bp:
        raise HTTPException(status_code=404, detail=f"Blueprint for analysis {analysis_id_str} not found")
    return bp


@app.get("/api/v1/analyses/{analysis_id}/bundle")
def download_bundle(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    _require_completed_job(analysis_id_str)

    zip_path = job_store.get_bundle_path(analysis_id_str)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"Bundle for analysis {analysis_id_str} not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"bundle_{analysis_id_str}.zip",
    )


@app.post("/api/v1/analyses/{analysis_id}/validate", response_model=ValidationReport)
def validate_analysis(analysis_id: UUID):
    analysis_id_str = str(analysis_id)
    _require_completed_job(analysis_id_str)

    bp = job_store.get_blueprint(analysis_id_str)
    if not bp:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id_str} not found")

    is_valid, errors = validator.validate(bp)
    return ValidationReport(
        valid=is_valid,
        validated_at=datetime.datetime.now(datetime.UTC).isoformat(),
        errors=errors,
        summary={"passed_rules": max(0, 25 - len(errors)), "failed_rules": len(errors)},
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
