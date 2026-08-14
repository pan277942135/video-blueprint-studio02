from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import uuid

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.pipeline_core.e4_media_pipeline import run_e4_media_pipeline
from packages.pipeline_core.media_probe import compute_sha256

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E4_MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E4.2 mask-constrained sparse motion")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e4_2_sparse_motion_certification.json")
    return parser.parse_args()


def _adjacent_motion(positions: np.ndarray, valid: np.ndarray, track_ids: np.ndarray) -> list[float]:
    distances: list[float] = []
    for frame_idx in range(1, positions.shape[0]):
        previous_ids = track_ids[frame_idx - 1]
        current_ids = track_ids[frame_idx]
        previous_valid = valid[frame_idx - 1]
        current_valid = valid[frame_idx]
        for slot in range(positions.shape[1]):
            track_id = int(current_ids[slot])
            if track_id < 0 or not current_valid[slot]:
                continue
            matches = np.flatnonzero(previous_valid & (previous_ids == track_id))
            if matches.size != 1:
                continue
            before = positions[frame_idx - 1, int(matches[0])]
            after = positions[frame_idx, slot]
            distance = float(np.linalg.norm(after - before))
            if math.isfinite(distance):
                distances.append(distance)
    return distances


def main() -> int:
    args = _args()
    video = pathlib.Path(args.video).resolve()
    det_config = pathlib.Path(args.det_config).resolve()
    det_checkpoint = pathlib.Path(args.det_checkpoint).resolve()
    mask_config = pathlib.Path(args.mask_config).resolve()
    mask_checkpoint = pathlib.Path(args.mask_checkpoint).resolve()
    output = pathlib.Path(args.output).resolve()
    for path in (video, det_config, det_checkpoint, mask_config, mask_checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing certification input: {path}")

    detector_sha = _sha256(det_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    if detector_sha != E2_SHA256:
        raise SystemExit(f"E2 detector SHA256 mismatch: {detector_sha}")
    if mask_sha != E4_MASK_SHA256:
        raise SystemExit(f"E4.1 mask SHA256 mismatch: {mask_sha}")

    os.environ.update(
        {
            "VBS_RTMDET_CONFIG": str(det_config),
            "VBS_RTMDET_CHECKPOINT": str(det_checkpoint),
            "VBS_RTMDET_WEIGHTS_LICENSE": "approved project RTMDet precedent",
            "VBS_RTMDET_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_DEVICE": "cpu",
            "VBS_RTMDET_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_CONFIG": str(mask_config),
            "VBS_RTMDET_INS_CHECKPOINT": str(mask_checkpoint),
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal CI certification; redistribution not approved",
            "VBS_RTMDET_INS_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_INS_DEVICE": "cpu",
            "VBS_RTMDET_INS_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_IOU_THRESHOLD": "0.20",
            "VBS_E4_POINT_TRACKS_ENABLED": "true",
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e4_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )

    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    if blueprint["processing"]["pipeline_version"] != "0.4.0-e4.2":
        raise SystemExit("E4.2 pipeline version was not emitted")

    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    for stage_name in ("person_mask", "point_tracks"):
        if stages.get(stage_name, {}).get("status") != "succeeded":
            raise SystemExit(f"{stage_name} stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e4_point_tracks")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E4.2 extension did not enable")
    expected = {
        "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
        "coordinate_space": "pixel_xy",
        "mask_constrained": True,
        "shot_boundary_reset": True,
        "interpolation": False,
    }
    for key, value in expected.items():
        if extension.get(key) != value:
            raise SystemExit(f"E4.2 extension semantic mismatch: {key}")

    frame_count = int(blueprint["timebase"]["frame_count"])
    max_points = int(extension["max_points"])
    refs = []
    total_samples = 0
    all_distances: list[float] = []
    for character in blueprint.get("characters", []):
        character_id = str(character["character_id"])
        row = extension.get("characters", {}).get(character_id, {})
        ref = row.get("track_points_ref") if isinstance(row, dict) else None
        if not isinstance(ref, dict):
            continue
        refs.append(ref)
        if ref.get("shape") != [frame_count, max_points, 2]:
            raise SystemExit("point track shape mismatch")
        if ref.get("coordinate_space") != "pixel_xy" or ref.get("interpolation_policy") != "none":
            raise SystemExit("point track coordinate/interpolation semantics mismatch")
        path_value = sidecars.get(ref["uri"])
        if not isinstance(path_value, (str, os.PathLike)):
            raise SystemExit(f"point-track URI did not resolve: {ref['uri']}")
        path = pathlib.Path(path_value)
        if not path.is_file() or _sha256(path) != ref["checksum_sha256"]:
            raise SystemExit(f"point-track sidecar checksum mismatch: {ref['uri']}")
        with np.load(path, allow_pickle=False) as arrays:
            positions = arrays["positions_xy"]
            valid = arrays["valid"]
            errors = arrays["tracking_error"]
            track_ids = arrays["track_id"]
        if positions.shape != (frame_count, max_points, 2):
            raise SystemExit("positions array shape mismatch")
        if valid.shape != (frame_count, max_points) or errors.shape != valid.shape or track_ids.shape != valid.shape:
            raise SystemExit("companion point-track array shape mismatch")
        if not np.all(np.isnan(positions[~valid])):
            raise SystemExit("invalid point positions must remain NaN")
        if not np.all(track_ids[~valid] == -1) or not np.all(track_ids[valid] >= 0):
            raise SystemExit("point track ID missing-value semantics mismatch")
        total_samples += int(np.count_nonzero(valid))
        all_distances.extend(_adjacent_motion(positions, valid, track_ids))

        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False:
            raise SystemExit("identity inference privacy invariant failed")
        if privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("biometric export privacy invariant failed")

    if not refs or total_samples <= 0:
        raise SystemExit("real-media E4.2 emitted no sparse motion samples")
    if not all_distances:
        raise SystemExit("no adjacent same-feature displacement was measurable")
    median_motion = float(np.median(np.asarray(all_distances, dtype=np.float32)))
    p95_motion = float(np.percentile(np.asarray(all_distances, dtype=np.float32), 95))
    if median_motion <= 0.25:
        raise SystemExit(f"sparse motion signal is too small: median={median_motion:.6f}")
    if p95_motion >= 15.0:
        raise SystemExit(f"sparse motion signal is implausibly large: p95={p95_motion:.6f}")

    provenance = next(
        (tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "point_tracks"),
        None,
    )
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("point-track provenance must exist and carry no model weights")

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "mask_checkpoint_sha256": mask_sha,
        "point_track_algorithm": extension["algorithm"],
        "character_count": len(blueprint.get("characters", [])),
        "point_track_ref_count": len(refs),
        "valid_point_samples": total_samples,
        "adjacent_motion_samples": len(all_distances),
        "median_motion_px": round(median_motion, 6),
        "p95_motion_px": round(p95_motion, 6),
        "coverage": blueprint["quality"]["module_scores"].get("point_tracks"),
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
