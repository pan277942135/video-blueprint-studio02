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
from packages.pipeline_core.e6_media_pipeline import run_e6_media_pipeline
from packages.pipeline_core.media_probe import compute_sha256

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E3_SHA256 = "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"
E4_MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"
ANCHOR_INDICES = np.asarray([5, 6, 11, 12], dtype=np.int64)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E6.1 camera-compensated 2D body-local frames")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e6_body_local_certification.json")
    return parser.parse_args()


def _load_ref_array(ref: dict, sidecars: dict, key: str) -> np.ndarray:
    path_value = sidecars.get(ref["uri"])
    if not isinstance(path_value, (str, os.PathLike)):
        raise SystemExit(f"physical sidecar unavailable for {ref['uri']}")
    path = pathlib.Path(path_value)
    if not path.is_file() or _sha256(path) != ref["checksum_sha256"]:
        raise SystemExit(f"sidecar checksum mismatch for {ref['uri']}")
    with np.load(path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise SystemExit(f"array {key!r} missing from {ref['uri']}")
        return np.asarray(arrays[key])


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points.astype(np.float64), np.ones((len(points), 1))), axis=1)
    return (matrix.astype(np.float64) @ homogeneous.T).T[:, :2]


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
        raise SystemExit("certified E2/E3/E4 checkpoint SHA256 mismatch")

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
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e6_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )

    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    valid, validation_errors = BlueprintValidator(schema).validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E6 output: " + " | ".join(validation_errors))
    if blueprint["processing"]["pipeline_version"] != "0.6.0-e6.1":
        raise SystemExit("E6.1 pipeline version was not emitted")

    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    for stage_name in ("pose_2d", "person_mask", "point_tracks", "dense_flow", "camera_motion", "body_local_frame"):
        if stages.get(stage_name, {}).get("status") != "succeeded":
            raise SystemExit(f"{stage_name} stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e6_body_local_frame")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E6 extension did not enable")
    if extension.get("coordinate_frame") != "body_local_2d" or extension.get("source_coordinate_space") != "pixel_xy":
        raise SystemExit("E6 coordinate semantics drifted")
    if extension.get("camera_stabilization_applied") is not True or extension.get("camera_backend") != "opencv_ransac_2d":
        raise SystemExit("E6 did not use certified E5 stabilization")
    if extension.get("identity_inference_performed") is not False or extension.get("biometric_embedding_exported") is not False:
        raise SystemExit("E6 privacy boundary changed")

    frame_count = int(blueprint["timebase"]["frame_count"])
    valid_frames_total = 0
    present_frames_total = 0
    anchor_error_values: list[float] = []
    roundtrip_error_values: list[float] = []
    confidence_values: list[float] = []
    torso_scales: list[float] = []

    for character in blueprint.get("characters", []):
        pose = character.get("pose", {})
        if pose.get("enabled") is not True or pose.get("skeleton_name") != "coco17":
            raise SystemExit("E6 character lacks certified COCO-17 pose")
        surface = character.get("surface_motion", {})
        ref = surface.get("body_frame_transform_ref")
        if not isinstance(ref, dict):
            raise SystemExit("E6 character missing body_frame_transform_ref")
        if ref.get("coordinate_space") != "body_local_2d" or ref.get("interpolation_policy") != "none":
            raise SystemExit("E6 body frame ref semantic mismatch")
        source_to_body = _load_ref_array(ref, sidecars, "source_pixel_to_body_local")
        inverse = _load_ref_array(ref, sidecars, "body_local_to_source_pixel")
        valid_frame = _load_ref_array(ref, sidecars, "valid_frame").astype(bool)
        anchor_confidence = _load_ref_array(ref, sidecars, "anchor_confidence")
        torso_scale = _load_ref_array(ref, sidecars, "torso_scale_px")
        if source_to_body.shape != (frame_count, 2, 3) or inverse.shape != (frame_count, 2, 3):
            raise SystemExit("E6 body transform physical array shape mismatch")
        if valid_frame.shape != (frame_count,):
            raise SystemExit("E6 valid-frame physical array shape mismatch")
        if np.any(np.isfinite(source_to_body[~valid_frame])):
            raise SystemExit("E6 invalid frames must remain explicit NaN")

        keypoints_ref = pose.get("keypoints_2d_ref")
        confidence_ref = pose.get("confidence_ref")
        bbox_ref = character.get("bbox_ref")
        if not isinstance(keypoints_ref, dict) or not isinstance(confidence_ref, dict) or not isinstance(bbox_ref, dict):
            raise SystemExit("E6 certification could not resolve pose/bbox refs")
        keypoints = _load_ref_array(keypoints_ref, sidecars, "keypoints_xy")
        pose_confidence = _load_ref_array(confidence_ref, sidecars, "confidence")
        bboxes = _load_ref_array(bbox_ref, sidecars, "bbox_xyxy")
        present = np.all(np.isfinite(bboxes), axis=1)
        present_frames_total += int(np.count_nonzero(present))

        for frame_idx in np.where(valid_frame)[0]:
            anchors = keypoints[frame_idx, ANCHOR_INDICES]
            scores = pose_confidence[frame_idx, ANCHOR_INDICES]
            if not np.all(np.isfinite(anchors)) or float(np.min(scores)) < 0.30:
                raise SystemExit("E6 marked a frame valid without sufficient anchor evidence")
            local = _apply(source_to_body[frame_idx], anchors)
            shoulder_center = (local[0] + local[1]) * 0.5
            hip_center = (local[2] + local[3]) * 0.5
            error = max(
                float(np.linalg.norm(hip_center - np.asarray([0.0, 0.0]))),
                float(np.linalg.norm(shoulder_center - np.asarray([0.0, 1.0]))),
            )
            anchor_error_values.append(error)
            recovered = _apply(inverse[frame_idx], local)
            roundtrip_error_values.append(float(np.max(np.linalg.norm(recovered - anchors, axis=1))))
            confidence_values.append(float(anchor_confidence[frame_idx]))
            torso_scales.append(float(torso_scale[frame_idx]))
            if float((local[0, 0] + local[2, 0]) - (local[1, 0] + local[3, 0])) <= 0.0:
                raise SystemExit("E6 anatomical +X orientation is not toward COCO left anchors")
        valid_frames_total += int(np.count_nonzero(valid_frame))

    if present_frames_total <= 0 or valid_frames_total <= 0:
        raise SystemExit("E6 real-media certification produced no valid character body frames")
    coverage = valid_frames_total / present_frames_total
    if coverage < 0.70:
        raise SystemExit(f"E6 body-frame coverage too low: {coverage:.6f}")
    median_anchor_error = float(np.median(anchor_error_values))
    max_roundtrip_error = float(np.max(roundtrip_error_values))
    min_anchor_confidence = float(np.min(confidence_values))
    median_torso_scale = float(np.median(torso_scales))
    if median_anchor_error > 0.03:
        raise SystemExit(f"E6 canonical anchor error too high: {median_anchor_error:.6f}")
    if max_roundtrip_error > 0.01:
        raise SystemExit(f"E6 affine inverse roundtrip error too high: {max_roundtrip_error:.6f}")
    if min_anchor_confidence < 0.30 or median_torso_scale < 8.0:
        raise SystemExit("E6 valid body frame violated confidence/torso-scale threshold")

    provenance = next((tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "body_local_frame"), None)
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("E6 provenance must exist and introduce no model weights")
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
        "body_local_algorithm": extension["algorithm"],
        "character_count": len(blueprint.get("characters", [])),
        "valid_body_frames": valid_frames_total,
        "present_character_frames": present_frames_total,
        "coverage": round(coverage, 6),
        "median_canonical_anchor_error": round(median_anchor_error, 6),
        "max_inverse_roundtrip_error_px": round(max_roundtrip_error, 6),
        "min_anchor_confidence": round(min_anchor_confidence, 6),
        "median_torso_scale_px": round(median_torso_scale, 6),
        "camera_stabilization_applied": True,
        "coordinate_frame": "body_local_2d",
        "interpolation": False,
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
