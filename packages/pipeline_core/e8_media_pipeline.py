from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from packages.pipeline_core.body_local_frame import BodyLocalFrameConfig
from packages.pipeline_core.camera_motion import CameraMotionConfig
from packages.pipeline_core.dense_flow import DenseFlowConfig
from packages.pipeline_core.e7_media_pipeline import run_e7_media_pipeline
from packages.pipeline_core.micro_motion import MicroMotionConfig, MicroMotionError
from packages.pipeline_core.micro_motion_v2 import run_micro_motion_v2
from packages.pipeline_core.person_mask import PersonMaskSegmenter
from packages.pipeline_core.sparse_motion import SparseMotionConfig
from packages.pipeline_core.surface_motion import SurfaceMotionConfig


def _micro_hash(current: str, config: MicroMotionConfig) -> str:
    return hashlib.sha256(f"{current}|micro-motion:{config.token()}".encode()).hexdigest()


def run_e8_media_pipeline(
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
    micro_motion_config: MicroMotionConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blueprint, sidecars = run_e7_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
        person_mask_segmenter=person_mask_segmenter,
        point_track_config=point_track_config,
        dense_flow_config=dense_flow_config,
        camera_motion_config=camera_motion_config,
        body_local_frame_config=body_local_frame_config,
        surface_motion_config=surface_motion_config,
    )
    if micro_motion_config is None:
        micro_motion_config = MicroMotionConfig.from_environment()
    if micro_motion_config is None:
        return blueprint, sidecars

    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise MicroMotionError("E8 requires the normalized CFR analysis video artifact root")
    normalized = Path(str(normalized_path))
    try:
        artifact_root = str(normalized.parents[2])
    except IndexError as exc:
        raise MicroMotionError("Could not resolve E8 artifact root") from exc

    micro_sidecars, report_ref, extension, quality = run_micro_motion_v2(
        blueprint,
        output_dir=artifact_root,
        sidecars=sidecars,
        config=micro_motion_config,
    )
    sidecars.update(micro_sidecars)
    canonical_report_ref = {key: report_ref[key] for key in ("kind", "uri", "sha256")}
    blueprint["artifacts"]["reports"].append(canonical_report_ref)
    blueprint["extensions"]["e8_micro_motion"] = extension

    processing = blueprint["processing"]
    processing["pipeline_version"] = "0.8.0-e8.1"
    processing["config_hash"] = _micro_hash(str(processing["config_hash"]), micro_motion_config)
    processing["stages"].append(
        {
            "name": "micro_motion",
            "status": "succeeded",
            "progress": 1.0,
            "message": "E8.1 emitted geometry-only per-track-consensus micro-motion evidence",
        }
    )
    blueprint["quality"]["module_scores"]["micro_motion"] = float(quality["score"])
    blueprint["provenance"]["tools"].append(
        {
            "module": "micro_motion",
            "tool": "E7 sparse body-local per-track frequency consensus and local coherence analysis",
            "version": "2",
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": str(processing["config_hash"]),
            "license": "PROJECT-CODE",
        }
    )
    return blueprint, sidecars


__all__ = ["run_e8_media_pipeline"]
