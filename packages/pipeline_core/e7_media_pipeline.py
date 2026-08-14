from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from packages.pipeline_core.body_local_frame import BodyLocalFrameConfig
from packages.pipeline_core.camera_motion import CameraMotionConfig
from packages.pipeline_core.dense_flow import DenseFlowConfig
from packages.pipeline_core.e6_media_pipeline import run_e6_media_pipeline
from packages.pipeline_core.person_mask import PersonMaskSegmenter
from packages.pipeline_core.sparse_motion import SparseMotionConfig
from packages.pipeline_core.surface_motion import SurfaceMotionConfig, SurfaceMotionError, run_surface_motion


def _surface_hash(current: str, config: SurfaceMotionConfig) -> str:
    return hashlib.sha256(f"{current}|surface-motion:{config.token()}".encode()).hexdigest()


def run_e7_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
    *,
    person_mask_segmenter: PersonMaskSegmenter | None = None,
    point_track_config: SparseMotionConfig | None = None,
    dense_flow_config: DenseFlowConfig | None = None,
    camera_motion_config: CameraMotionConfig | None = None,
    body_local_frame_config: BodyLocalFrameConfig | None = None,
    surface_motion_config: SurfaceMotionConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blueprint, sidecars = run_e6_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
        person_mask_segmenter=person_mask_segmenter,
        point_track_config=point_track_config,
        dense_flow_config=dense_flow_config,
        camera_motion_config=camera_motion_config,
        body_local_frame_config=body_local_frame_config,
    )
    if surface_motion_config is None:
        surface_motion_config = SurfaceMotionConfig.from_environment()
    if surface_motion_config is None:
        return blueprint, sidecars

    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise SurfaceMotionError("E7 requires the normalized CFR analysis video")
    normalized = Path(str(normalized_path))
    try:
        artifact_root = str(normalized.parents[2])
    except IndexError as exc:
        raise SurfaceMotionError("Could not resolve E7 artifact root") from exc

    surface_sidecars, report_ref, extension, quality = run_surface_motion(
        blueprint,
        output_dir=artifact_root,
        sidecars=sidecars,
        config=surface_motion_config,
    )
    sidecars.update(surface_sidecars)
    blueprint["artifacts"]["reports"].append(report_ref)
    blueprint["extensions"]["e7_surface_motion"] = extension

    processing = blueprint["processing"]
    processing["pipeline_version"] = "0.7.0-e7.1"
    processing["config_hash"] = _surface_hash(str(processing["config_hash"]), surface_motion_config)
    processing["stages"].append(
        {
            "name": "surface_motion",
            "status": "succeeded",
            "progress": 1.0,
            "message": "E7.1 emitted sparse body-local residual surface motion",
        }
    )
    blueprint["quality"]["module_scores"]["surface_motion"] = float(quality["score"])
    blueprint["provenance"]["tools"].append(
        {
            "module": "surface_motion",
            "tool": "E4.2 sparse point tracks + E6 body-local transforms",
            "version": "1",
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": str(processing["config_hash"]),
            "license": "PROJECT-CODE",
        }
    )
    return blueprint, sidecars


__all__ = ["run_e7_media_pipeline"]
