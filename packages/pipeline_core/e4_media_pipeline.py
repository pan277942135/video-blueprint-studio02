from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from packages.pipeline_core.person_mask import (
    PersonMaskError,
    PersonMaskSegmenter,
    run_person_mask_refinement,
)
from packages.pipeline_core.real_media_pipeline import run_real_media_pipeline
from packages.pipeline_core.rtmdet_ins_mask_backend import RTMDetInsPersonMaskSegmenter


def _extended_hash(current: str, segmenter: PersonMaskSegmenter) -> str:
    token = getattr(segmenter, "config_sha256", type(segmenter).__name__)
    return hashlib.sha256(f"{current}|person-mask:{token}".encode("utf-8")).hexdigest()


def run_e4_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
    *,
    person_mask_segmenter: PersonMaskSegmenter | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the certified base pipeline and optionally append E4.1 person masks.

    An explicitly injected segmenter is used by deterministic tests/certification.
    Otherwise the approved RTMDet-Ins adapter is resolved from the environment.
    A completely absent E4.1 configuration leaves the certified E3 pipeline
    unchanged; a partial/unapproved/mismatched configuration fails closed.
    """
    blueprint, sidecars = run_real_media_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
    )
    if person_mask_segmenter is None:
        person_mask_segmenter = RTMDetInsPersonMaskSegmenter.from_environment()
    if person_mask_segmenter is None:
        return blueprint, sidecars

    characters = blueprint.get("characters")
    if not isinstance(characters, list) or not characters:
        raise PersonMaskError("E4.1 requires existing anonymous person tracks")
    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise PersonMaskError("E4.1 requires the normalized CFR analysis video")

    normalized = Path(str(normalized_path))
    try:
        artifact_root = str(normalized.parents[2])
    except IndexError as exc:
        raise PersonMaskError("Could not resolve E4 artifact root") from exc

    frame_count = int(blueprint["timebase"]["frame_count"])
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

    processing = blueprint["processing"]
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
    return blueprint, sidecars
