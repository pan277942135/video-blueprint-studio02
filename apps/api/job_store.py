import datetime
from typing import Dict, Any, Optional, List
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

    def store_video(self, file_name: str, sha256_hash: str, custom_video_id: Optional[str] = None) -> Dict[str, Any]:
        video_id = custom_video_id or str(uuid.uuid4())
        video_record = {
            "video_id": video_id,
            "file_name": file_name,
            "sha256": sha256_hash,
            "status": "uploaded"
        }
        self.videos[video_id] = video_record
        return video_record

    def create_analysis(self, video_id: str, modules: Optional[List[str]] = None, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        analysis_id = str(uuid.uuid4())
        job_record = {
            "analysis_id": analysis_id,
            "video_id": video_id,
            "status": "queued",
            "progress": 0.0,
            "stages": [
                {"stage": "shots", "status": "pending", "progress": 0.0},
                {"stage": "people", "status": "pending", "progress": 0.0},
                {"stage": "pose", "status": "pending", "progress": 0.0},
                {"stage": "camera", "status": "pending", "progress": 0.0},
                {"stage": "micro_motion", "status": "pending", "progress": 0.0}
            ],
            "requested_modules": modules or [
                "shots", "people", "pose", "face", "hands",
                "masks", "camera", "flow", "micro_motion", "environment", "overlays"
            ],
            "config_overrides": config_overrides or {},
            "error": None
        }
        self.jobs[analysis_id] = job_record
        return job_record

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(analysis_id)

    def update_analysis(self, analysis_id: str, **kwargs) -> Dict[str, Any]:
        job = self.jobs.get(analysis_id)
        if not job:
            raise KeyError(f"Analysis {analysis_id} not found.")
        for k, v in kwargs.items():
            job[k] = v
        return job

    def store_blueprint(self, analysis_id: str, blueprint_data: Dict[str, Any], sidecars_data: Dict[str, Any]):
        self.blueprints[analysis_id] = blueprint_data
        self.sidecars[analysis_id] = sidecars_data

    def get_blueprint(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        return self.blueprints.get(analysis_id)

    def store_bundle_path(self, analysis_id: str, path: str):
        self.bundle_paths[analysis_id] = path

    def get_bundle_path(self, analysis_id: str) -> Optional[str]:
        return self.bundle_paths.get(analysis_id)

job_store = InMemoryJobStore()
