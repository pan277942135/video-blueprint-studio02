from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from packages.pipeline_core.body_local_frame import BodyLocalFrameConfig, BodyLocalFrameError, run_body_local_frame
from packages.pipeline_core.camera_motion import CameraMotionConfig
from packages.pipeline_core.dense_flow import DenseFlowConfig
from packages.pipeline_core.e5_media_pipeline import run_e5_media_pipeline
from packages.pipeline_core.person_mask import PersonMaskSegmenter
from packages.pipeline_core.sparse_motion import SparseMotionConfig


def _body_local_hash(current: str, config: BodyLocalFrameConfig) -> str:
    return hashlib.sha256(f"{current}|body-local-frame:{config.token()}".encode()).hexdigest()


def run_e6_media_pipeline(
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    blueprint, sidecars = run_e5_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
        person_mask_segmenter=person_mask_segmenter,
        point_track_config=point_track_config,
        dense_flow_config=dense_flow_config,
        camera_motion_config=camera_motion_config,
    )
    if body_local_frame_config is None:
        body_local_frame_config = BodyLocalFrameConfig.from_environment()
    if body_local_frame_config is None:
        return blueprint, sidecars

    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise BodyLocalFrameError("E6 requires the normalized CFR analysis video")
    normalized = Path(str(normalized_path))
    try:
        artifact_root = str(normalized.parents[2])
    except IndexError as exc:
        raise BodyLocalFrameError("Could not resolve E6 artifact root") from exc

    body_sidecars, report_ref, extension, quality = run_body_local_frame(
        blueprint,
        output_dir=artifact_root,
        sidecars=sidecars,
        config=body_local_frame_config,
    )
    sidecars.update(body_sidecars)
    blueprint["artifacts"]["reports"].append(report_ref)
    blueprint["extensions"]["e6_body_local_frame"] = extension

    processing = blueprint["processing"]
    processing["pipeline_version"] = "0.6.0-e6.1"
    processing["config_hash"] = _body_local_hash(str(processing["config_hash"]), body_local_frame_config)
    processing["stages"].append(
        {
            "name": "body_local_frame",
            "status": "succeeded",
            "progress": 1.0,
            "message": "E6.1 emitted camera-compensated COCO-17 torso body-local 2D frames",
        }
    )
    blueprint["quality"]["module_scores"]["body_local_frame"] = float(quality["score"])
    if extension["character_count"] > 0:
        blueprint["quality"]["overall_score"] = min(
            float(blueprint["quality"]["overall_score"]),
            float(quality["score"]),
        )
    blueprint["provenance"]["tools"].append(
        {
            "module": "body_local_frame",
            "tool": "COCO-17 torso similarity frame + E5 camera stabilization",
            "version": "1",
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": str(processing["config_hash"]),
            "license": "PROJECT-CODE",
        }
    )
    return blueprint, sidecars


__all__ = ["run_e6_media_pipeline"]
