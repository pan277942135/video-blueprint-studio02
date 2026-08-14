from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2

from packages.pipeline_core.person_mask import (
    PersonMaskError,
    PersonMaskSegmenter,
    run_person_mask_refinement,
)
from packages.pipeline_core.point_tracks import PointTrackError, run_point_track_refinement
from packages.pipeline_core.real_media_pipeline import run_real_media_pipeline
from packages.pipeline_core.rtmdet_ins_mask_backend import RTMDetInsPersonMaskSegmenter
from packages.pipeline_core.sparse_motion import SparseMotionConfig


def _extended_hash(current: str, segmenter: PersonMaskSegmenter) -> str:
    token = getattr(segmenter, "config_sha256", type(segmenter).__name__)
    return hashlib.sha256(f"{current}|person-mask:{token}".encode()).hexdigest()


def _point_hash(current: str, config: SparseMotionConfig) -> str:
    return hashlib.sha256(f"{current}|point-tracks:{config.token()}".encode()).hexdigest()


def run_e4_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
    *,
    person_mask_segmenter: PersonMaskSegmenter | None = None,
    point_track_config: SparseMotionConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run E4.1 masks and optional E4.2 image-space sparse motion."""
    blueprint, sidecars = run_real_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
    )
    if person_mask_segmenter is None:
        person_mask_segmenter = RTMDetInsPersonMaskSegmenter.from_environment()
    if point_track_config is None:
        point_track_config = SparseMotionConfig.from_environment()
    if person_mask_segmenter is None and point_track_config is None:
        return blueprint, sidecars

    characters = blueprint.get("characters")
    if not isinstance(characters, list) or not characters:
        raise PersonMaskError("E4 requires existing anonymous person tracks")
    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise PersonMaskError("E4 requires the normalized CFR analysis video")

    normalized = Path(str(normalized_path))
    try:
        artifact_root = str(normalized.parents[2])
    except IndexError as exc:
        raise PersonMaskError("Could not resolve E4 artifact root") from exc

    frame_count = int(blueprint["timebase"]["frame_count"])
    processing = blueprint["processing"]

    if person_mask_segmenter is not None:
        characters, mask_sidecars, report_ref, mask_quality = run_person_mask_refinement(
            str(normalized_path),
            characters=characters,
            frame_count=frame_count,
            segmenter=person_mask_segmenter,
            output_dir=artifact_root,
            sidecars=sidecars,
        )
        sidecars.update(mask_sidecars)
        blueprint["characters"] = characters
        blueprint["artifacts"]["reports"].append(report_ref)
        processing["pipeline_version"] = "0.4.0-e4.1"
        processing["config_hash"] = _extended_hash(str(processing["config_hash"]), person_mask_segmenter)
        processing["stages"].append(
            {
                "name": "person_mask",
                "status": "succeeded",
                "progress": 1.0,
                "message": f"E4.1 emitted frame-aligned masks for {len(characters)} anonymous track(s)",
            }
        )
        module_scores = blueprint["quality"]["module_scores"]
        module_scores["person_mask"] = float(mask_quality["score"])
        if mask_quality["confidence_available"] and characters:
            blueprint["quality"]["overall_score"] = min(
                float(blueprint["quality"]["overall_score"]),
                float(mask_quality["score"]),
            )
        blueprint["extensions"]["e4_person_mask"] = {
            "enabled": True,
            "encoding": "row_major_binary_rle_v1",
            "character_count": len(characters),
            "coverage": float(mask_quality["coverage"]),
            "confidence_available": bool(mask_quality["confidence_available"]),
        }
        provenance = getattr(person_mask_segmenter, "provenance", None)
        if callable(provenance):
            blueprint["provenance"]["tools"].append(
                provenance(
                    code_commit=os.environ.get("GITHUB_SHA"),
                    config_hash=str(processing["config_hash"]),
                )
            )

    if point_track_config is not None:
        if not all(isinstance(character.get("person_mask_ref"), dict) for character in characters):
            raise PointTrackError("E4.2 requires E4.1 person masks for every anonymous character")
        extension, motion_sidecars, motion_report, motion_quality = run_point_track_refinement(
            str(normalized_path),
            characters=characters,
            shots=blueprint.get("shots", []),
            frame_count=frame_count,
            output_dir=artifact_root,
            sidecars=sidecars,
            config=point_track_config,
        )
        sidecars.update(motion_sidecars)
        blueprint["artifacts"]["reports"].append(motion_report)
        blueprint["extensions"]["e4_point_tracks"] = extension
        processing["pipeline_version"] = "0.4.0-e4.2"
        processing["config_hash"] = _point_hash(str(processing["config_hash"]), point_track_config)
        processing["stages"].append(
            {
                "name": "point_tracks",
                "status": "succeeded",
                "progress": 1.0,
                "message": "E4.2 emitted mask-constrained sparse motion evidence",
            }
        )
        blueprint["quality"]["module_scores"]["point_tracks"] = float(motion_quality["score"])
        blueprint["provenance"]["tools"].append(
            {
                "module": "point_tracks",
                "tool": "OpenCV Shi-Tomasi + pyramidal Lucas-Kanade",
                "version": cv2.__version__,
                "code_commit": os.environ.get("GITHUB_SHA"),
                "weights_sha256": None,
                "config_hash": str(processing["config_hash"]),
                "license": "Apache-2.0",
            }
        )

    return blueprint, sidecars
