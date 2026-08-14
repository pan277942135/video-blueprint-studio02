from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2

from packages.pipeline_core.camera_motion import CameraMotionConfig, CameraMotionError, run_camera_motion
from packages.pipeline_core.dense_flow import DenseFlowConfig
from packages.pipeline_core.e4_media_pipeline import run_e4_media_pipeline
from packages.pipeline_core.person_mask import PersonMaskSegmenter
from packages.pipeline_core.sparse_motion import SparseMotionConfig


def _camera_hash(current: str, config: CameraMotionConfig) -> str:
    return hashlib.sha256(f"{current}|camera-motion:{config.token()}".encode("utf-8")).hexdigest()


def run_e5_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
    *,
    person_mask_segmenter: PersonMaskSegmenter | None = None,
    point_track_config: SparseMotionConfig | None = None,
    dense_flow_config: DenseFlowConfig | None = None,
    camera_motion_config: CameraMotionConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blueprint, sidecars = run_e4_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
        person_mask_segmenter=person_mask_segmenter,
        point_track_config=point_track_config,
        dense_flow_config=dense_flow_config,
    )
    if camera_motion_config is None:
        camera_motion_config = CameraMotionConfig.from_environment()
    if camera_motion_config is None:
        return blueprint, sidecars

    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise CameraMotionError("E5 requires the normalized CFR analysis video")
    normalized = Path(str(normalized_path))
    try:
        artifact_root = str(normalized.parents[2])
    except IndexError as exc:
        raise CameraMotionError("Could not resolve E5 artifact root") from exc

    camera, camera_sidecars, report_ref, extension, quality = run_camera_motion(
        str(normalized),
        shots=blueprint.get("shots", []),
        characters=blueprint.get("characters", []),
        frame_count=int(blueprint["timebase"]["frame_count"]),
        output_dir=artifact_root,
        sidecars=sidecars,
        config=camera_motion_config,
    )
    sidecars.update(camera_sidecars)
    blueprint["camera"] = camera
    blueprint["artifacts"]["reports"].append(report_ref)
    blueprint["extensions"]["e5_camera_motion"] = extension

    processing = blueprint["processing"]
    processing["pipeline_version"] = "0.5.0-e5.1"
    processing["config_hash"] = _camera_hash(str(processing["config_hash"]), camera_motion_config)
    processing["stages"].append(
        {
            "name": "camera_motion",
            "status": "succeeded",
            "progress": 1.0,
            "message": "E5.1 emitted background RANSAC 2D camera-motion evidence per shot",
        }
    )
    camera_score = float(quality["score"])
    blueprint["quality"]["module_scores"]["camera"] = camera_score
    if camera.get("per_shot"):
        blueprint["quality"]["overall_score"] = min(float(blueprint["quality"]["overall_score"]), camera_score)
    blueprint["provenance"]["tools"].append(
        {
            "module": "camera_motion",
            "tool": "OpenCV Shi-Tomasi + pyramidal LK + estimateAffinePartial2D RANSAC",
            "version": cv2.__version__,
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": str(processing["config_hash"]),
            "license": "Apache-2.0",
        }
    )
    return blueprint, sidecars


__all__ = ["run_e5_media_pipeline"]
