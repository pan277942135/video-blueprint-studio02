from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import uuid

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core.e7_media_pipeline import run_e7_media_pipeline
from packages.pipeline_core.media_probe import compute_sha256

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E3_SHA256 = "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"
E4_MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E7.1 sparse body-local surface residual motion")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e7_surface_motion_certification.json")
    return parser.parse_args()


def _load(ref: dict, sidecars: dict, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    path_value = sidecars.get(ref["uri"])
    if not isinstance(path_value, (str, os.PathLike)):
        raise SystemExit(f"physical sidecar unavailable: {ref['uri']}")
    path = pathlib.Path(path_value)
    if not path.is_file() or _sha256(path) != ref["checksum_sha256"]:
        raise SystemExit(f"sidecar checksum mismatch: {ref['uri']}")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in keys:
            if key not in arrays:
                raise SystemExit(f"array {key!r} missing: {ref['uri']}")
            result[key] = np.asarray(arrays[key])
    return result


def main() -> int:
    args = _args()
    video = pathlib.Path(args.video).resolve()
    det_config = pathlib.Path(args.det_config).resolve()
    det_checkpoint = pathlib.Path(args.det_checkpoint).resolve()
    pose_config = pathlib.Path(args.pose_config).resolve()
    pose_checkpoint = pathlib.Path(args.pose_checkpoint).resolve()
    mask_config = pathlib.Path(args.mask_config).resolve()
    mask_checkpoint = pathlib.Path(args.mask_checkpoint).resolve()
    output = pathlib.Path(args.output).resolve()
    for path in (video, det_config, det_checkpoint, pose_config, pose_checkpoint, mask_config, mask_checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing certification input: {path}")
    detector_sha = _sha256(det_checkpoint)
    pose_sha = _sha256(pose_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    if detector_sha != E2_SHA256 or pose_sha != E3_SHA256 or mask_sha != E4_MASK_SHA256:
        raise SystemExit("certified checkpoint SHA256 mismatch")

    os.environ.update(
        {
            "VBS_RTMDET_CONFIG": str(det_config),
            "VBS_RTMDET_CHECKPOINT": str(det_checkpoint),
            "VBS_RTMDET_WEIGHTS_LICENSE": "approved project RTMDet precedent",
            "VBS_RTMDET_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_DEVICE": "cpu",
            "VBS_RTMDET_SCORE_THRESHOLD": "0.30",
            "VBS_RTMPOSE_CONFIG": str(pose_config),
            "VBS_RTMPOSE_CHECKPOINT": str(pose_checkpoint),
            "VBS_RTMPOSE_EXPECTED_SHA256": E3_SHA256,
            "VBS_RTMPOSE_WEIGHTS_LICENSE": "approved project RTMPose precedent",
            "VBS_RTMPOSE_WEIGHTS_APPROVED": "true",
            "VBS_RTMPOSE_DEVICE": "cpu",
            "VBS_RTMDET_INS_CONFIG": str(mask_config),
            "VBS_RTMDET_INS_CHECKPOINT": str(mask_checkpoint),
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal CI certification; redistribution not approved",
            "VBS_RTMDET_INS_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_INS_DEVICE": "cpu",
            "VBS_RTMDET_INS_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_IOU_THRESHOLD": "0.20",
            "VBS_E4_POINT_TRACKS_ENABLED": "true",
            "VBS_E4_DENSE_FLOW_ENABLED": "true",
            "VBS_E5_CAMERA_MOTION_ENABLED": "true",
            "VBS_E6_BODY_LOCAL_FRAME_ENABLED": "true",
            "VBS_E7_SURFACE_MOTION_ENABLED": "true",
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e7_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )
    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    valid, errors = BlueprintValidator(schema).validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E7 output: " + " | ".join(errors))
    if blueprint["processing"]["pipeline_version"] != "0.7.0-e7.1":
        raise SystemExit("E7.1 pipeline version was not emitted")
    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    for stage_name in ("point_tracks", "camera_motion", "body_local_frame", "surface_motion"):
        if stages.get(stage_name, {}).get("status") != "succeeded":
            raise SystemExit(f"{stage_name} stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e7_surface_motion")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E7 extension did not enable")
    if extension.get("coordinate_frame") != "body_local_2d" or extension.get("dense_residual_emitted") is not False:
        raise SystemExit("E7 sparse-only coordinate contract drifted")
    if extension.get("micro_motion_analyzed") is not False:
        raise SystemExit("E7 must not claim E8 micro-motion analysis")

    point_rows = blueprint["extensions"]["e4_point_tracks"]["characters"]
    residual_samples = 0
    eligible_samples = 0
    raw_magnitudes: list[np.ndarray] = []
    residual_pixel_equivalent: list[np.ndarray] = []
    residual_body_units: list[np.ndarray] = []
    region_count = 0

    for character in blueprint.get("characters", []):
        character_id = character["character_id"]
        surface = character["surface_motion"]
        regions = surface.get("regions", [])
        track_ref = point_rows[character_id]["track_points_ref"]
        if track_ref is None:
            if regions:
                raise SystemExit("E7 emitted a region without E4.2 point evidence")
            continue
        if len(regions) != 1:
            raise SystemExit("E7 real-media character must expose one sparse surface region")
        region_count += 1
        region = regions[0]
        micro = region["micro_motion"]
        if micro.get("confidence") != 0.0 or micro.get("usable_for_generation") is not False:
            raise SystemExit("E7 placeholder micro-motion became usable before E8")
        residual_ref = region["residual_flow_ref"]
        local_ref = region["track_points_ref"]
        if residual_ref["uri"] != local_ref["uri"] or residual_ref["checksum_sha256"] != local_ref["checksum_sha256"]:
            raise SystemExit("E7 region refs do not share one physical sidecar")
        surface_arrays = _load(
            residual_ref,
            sidecars,
            ("residual_displacement", "residual_valid", "track_id"),
        )
        point_arrays = _load(track_ref, sidecars, ("positions_xy", "valid", "track_id"))
        body_ref = surface["body_frame_transform_ref"]
        body_arrays = _load(body_ref, sidecars, ("torso_scale_px", "valid_frame"))
        residual = surface_arrays["residual_displacement"]
        residual_valid = surface_arrays["residual_valid"].astype(bool)
        positions = point_arrays["positions_xy"]
        point_valid = point_arrays["valid"].astype(bool)
        track_id = point_arrays["track_id"]
        torso_scale = body_arrays["torso_scale_px"]
        body_valid = body_arrays["valid_frame"].astype(bool)
        shot_starts = {
            int(shot["frame_start"])
            for shot in blueprint["shots"]
            if int(shot["frame_start"]) > 0
        }
        for frame_idx in range(1, len(positions)):
            if frame_idx in shot_starts:
                if np.any(residual_valid[frame_idx]):
                    raise SystemExit("E7 residual crossed a shot boundary")
                continue
            same_track = (
                point_valid[frame_idx - 1]
                & point_valid[frame_idx]
                & (track_id[frame_idx - 1] >= 0)
                & (track_id[frame_idx - 1] == track_id[frame_idx])
            )
            eligible_samples += int(np.count_nonzero(same_track))
            if not (body_valid[frame_idx - 1] and body_valid[frame_idx]):
                if np.any(residual_valid[frame_idx]):
                    raise SystemExit("E7 residual exists without consecutive valid body frames")
                continue
            slots = np.flatnonzero(residual_valid[frame_idx])
            if slots.size:
                if not np.all(same_track[slots]):
                    raise SystemExit("E7 residual reused a point slot after track-ID change")
                raw = positions[frame_idx, slots] - positions[frame_idx - 1, slots]
                raw_magnitudes.append(np.linalg.norm(raw, axis=1))
                body_mag = np.linalg.norm(residual[frame_idx, slots], axis=1)
                residual_body_units.append(body_mag)
                residual_pixel_equivalent.append(body_mag * float(torso_scale[frame_idx]))
                residual_samples += int(slots.size)

    if region_count <= 0 or residual_samples <= 0 or eligible_samples <= 0:
        raise SystemExit("E7 certification produced no sparse residual evidence")
    coverage = residual_samples / eligible_samples
    raw_values = np.concatenate(raw_magnitudes)
    body_values = np.concatenate(residual_body_units)
    equivalent_values = np.concatenate(residual_pixel_equivalent)
    raw_median = float(np.median(raw_values))
    residual_median = float(np.median(body_values))
    equivalent_median = float(np.median(equivalent_values))
    reduction_ratio = equivalent_median / raw_median if raw_median > 0 else float("inf")
    if coverage < 0.50:
        raise SystemExit(f"E7 residual coverage too low: {coverage:.6f}")
    if raw_median < 0.50:
        raise SystemExit(f"certification video did not contain enough raw motion: {raw_median:.6f}")
    if reduction_ratio >= 0.80:
        raise SystemExit(
            f"E7 body-local residual did not sufficiently suppress rigid camera/body motion: {reduction_ratio:.6f}"
        )

    provenance = next((tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "surface_motion"), None)
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("E7 provenance must exist and introduce no model weights")
    for character in blueprint.get("characters", []):
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False or privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("identity/biometric privacy invariant failed")

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "pose_sha256": pose_sha,
        "mask_checkpoint_sha256": mask_sha,
        "surface_algorithm": extension["algorithm"],
        "region_count": region_count,
        "eligible_residual_samples": eligible_samples,
        "valid_residual_samples": residual_samples,
        "coverage": round(coverage, 6),
        "median_raw_point_motion_px": round(raw_median, 6),
        "median_body_local_residual_units": round(residual_median, 6),
        "median_residual_pixel_equivalent": round(equivalent_median, 6),
        "rigid_motion_residual_ratio": round(reduction_ratio, 6),
        "dense_residual_emitted": False,
        "micro_motion_analyzed": False,
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
