import datetime
from typing import Dict, Any, Optional
import uuid

class InMemoryJobStore:
    """
    In-memory E0 job and artifact store.
    No external database dependencies (Postgres/Redis) for E0.
    """
    def __init__(self):
        self.videos: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.blueprints: Dict[str, Dict[str, Any]] = {}
        self.sidecars: Dict[str, Dict[str, Any]] = {}
        self.bundle_paths: Dict[str, str] = {}

    def store_video(self, file_name: str, file_size: int, sha256_hash: str) -> Dict[str, Any]:
        video_id = f"video_{uuid.uuid4().hex[:12]}"
        video_record = {
            "video_id": video_id,
            "file_name": file_name,
            "file_size": file_size,
            "sha256_hash": sha256_hash,
            "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z"
        }
        self.videos[video_id] = video_record
        return video_record

    def create_job(self, video_id: str) -> Dict[str, Any]:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.utcnow().isoformat() + "Z"
        job_record = {
            "job_id": job_id,
            "video_id": video_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "blueprint_id": None,
            "quality_score": None,
            "error_message": None
        }
        self.jobs[job_id] = job_record
        return job_record

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> Dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found.")
        for k, v in kwargs.items():
            job[k] = v
        job["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        return job

    def store_blueprint(self, blueprint_id: str, blueprint_data: Dict[str, Any], sidecars_data: Dict[str, Any]):
        self.blueprints[blueprint_id] = blueprint_data
        self.sidecars[blueprint_id] = sidecars_data

    def get_blueprint(self, blueprint_id: str) -> Optional[Dict[str, Any]]:
        return self.blueprints.get(blueprint_id)

    def store_bundle_path(self, blueprint_id: str, path: str):
        self.bundle_paths[blueprint_id] = path

    def get_bundle_path(self, blueprint_id: str) -> Optional[str]:
        return self.bundle_paths.get(blueprint_id)

job_store = InMemoryJobStore()
