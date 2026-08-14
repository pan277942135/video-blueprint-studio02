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

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core.e5_media_pipeline import run_e5_media_pipeline
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
    parser = argparse.ArgumentParser(description="Certify E5.1 background RANSAC camera motion")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--expected-tx", type=float, default=1.0)
    parser.add_argument("--expected-ty", type=float, default=0.5)
    parser.add_argument("--output", default=".cert/e5_camera_motion_certification.json")
    return parser.parse_args()


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
    if detector_sha != E2_SHA256 or mask_sha != E4_MASK_SHA256:
        raise SystemExit("certified E2/E4.1 checkpoint SHA256 mismatch")

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
            "VBS_E4_DENSE_FLOW_ENABLED": "true",
            "VBS_E5_CAMERA_MOTION_ENABLED": "true",
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e5_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )

    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    valid, validation_errors = BlueprintValidator(schema).validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E5 output: " + " | ".join(validation_errors))
    if blueprint["processing"]["pipeline_version"] != "0.5.0-e5.1":
        raise SystemExit("E5.1 pipeline version was not emitted")

    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    for stage_name in ("person_mask", "point_tracks", "dense_flow", "camera_motion"):
        if stages.get(stage_name, {}).get("status") != "succeeded":
            raise SystemExit(f"{stage_name} stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e5_camera_motion")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E5 extension did not enable")
    if extension.get("foreground_exclusion") is not True:
        raise SystemExit("E5 certification did not use E4.1 person-mask foreground exclusion")
    if extension.get("identity_inference_performed") is not False or extension.get("biometric_embedding_exported") is not False:
        raise SystemExit("E5 privacy boundary changed")

    camera = blueprint.get("camera")
    rows = camera.get("per_shot") if isinstance(camera, dict) else None
    if not isinstance(rows, list) or not rows:
        raise SystemExit("E5 emitted no camera rows")
    allowed = {"static", "pan", "tilt", "roll", "zoom", "compound", "unknown"}
    unsupported = {"dolly", "truck", "pedestal"}
    all_tx: list[float] = []
    all_ty: list[float] = []
    all_inliers: list[float] = []
    valid_pairs = 0
    expected_pairs = 0
    background_samples = 0

    for row in rows:
        if row.get("classification") not in allowed or row.get("classification") in unsupported:
            raise SystemExit(f"unsupported E5 2D classification: {row.get('classification')}")
        if row.get("reconstruction_backend") != "opencv_ransac_2d":
            raise SystemExit("E5 backend drifted from opencv_ransac_2d")
        if row.get("intrinsics") is not None or row.get("extrinsics_ref") is not None:
            raise SystemExit("uncalibrated E5 backend claimed 3D calibration")
        ref = row.get("affine_ref")
        tracks_ref = row.get("background_tracks_ref")
        if not isinstance(ref, dict) or not isinstance(tracks_ref, dict):
            raise SystemExit("E5 physical evidence refs are missing")
        path_value = sidecars.get(ref["uri"])
        if not isinstance(path_value, (str, os.PathLike)):
            raise SystemExit(f"camera URI did not resolve: {ref['uri']}")
        path = pathlib.Path(path_value)
        if not path.is_file() or _sha256(path) != ref["checksum_sha256"]:
            raise SystemExit("camera sidecar checksum mismatch")
        if tracks_ref.get("checksum_sha256") != ref.get("checksum_sha256"):
            raise SystemExit("camera refs disagree about physical sidecar checksum")
        with np.load(path, allow_pickle=False) as arrays:
            affine = arrays["frame_to_frame_affine"]
            stabilization = arrays["stabilization_affine"]
            inlier_ratio = arrays["ransac_inlier_ratio"]
            points = arrays["background_points_xy"]
            points_valid = arrays["background_valid"]
        if affine.dtype != np.float32 or affine.shape[1:] != (2, 3):
            raise SystemExit("E5 affine physical array shape/dtype mismatch")
        if not np.all(np.isnan(affine[0])):
            raise SystemExit("first shot frame must have no predecessor affine")
        if not np.allclose(stabilization[0], np.eye(3, dtype=np.float32)[:2], atol=1e-6):
            raise SystemExit("shot stabilization must start at identity")
        finite = np.all(np.isfinite(affine), axis=(1, 2))
        if np.any(finite):
            all_tx.extend(float(value) for value in affine[finite, 0, 2])
            all_ty.extend(float(value) for value in affine[finite, 1, 2])
            valid_pairs += int(np.count_nonzero(finite))
            ratios = inlier_ratio[finite]
            if not np.all(np.isfinite(ratios)):
                raise SystemExit("valid E5 transforms require finite RANSAC inlier ratios")
            all_inliers.extend(float(value) for value in ratios)
        expected_pairs += max(0, len(affine) - 1)
        background_samples += int(np.count_nonzero(points_valid))
        if points.shape[:2] != points_valid.shape or points.shape[-1] != 2:
            raise SystemExit("background-track sidecar shape mismatch")

    if not all_tx or not all_inliers:
        raise SystemExit("real-media E5 emitted no estimable camera transforms")
    coverage = valid_pairs / expected_pairs if expected_pairs else 1.0
    median_tx = float(np.median(np.asarray(all_tx)))
    median_ty = float(np.median(np.asarray(all_ty)))
    median_inlier = float(np.median(np.asarray(all_inliers)))
    if coverage < 0.90:
        raise SystemExit(f"camera transform coverage too low: {coverage:.6f}")
    if median_inlier < 0.60:
        raise SystemExit(f"RANSAC inlier ratio too low: {median_inlier:.6f}")
    if not math.isclose(median_tx, args.expected_tx, abs_tol=0.70):
        raise SystemExit(f"camera x translation mismatch: {median_tx:.6f} vs {args.expected_tx:.6f}")
    if not math.isclose(median_ty, args.expected_ty, abs_tol=0.70):
        raise SystemExit(f"camera y translation mismatch: {median_ty:.6f} vs {args.expected_ty:.6f}")
    if background_samples <= 0:
        raise SystemExit("E5 emitted no background track samples")

    provenance = next((tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "camera_motion"), None)
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("E5 camera provenance must exist and introduce no model weights")
    for character in blueprint.get("characters", []):
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False or privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("identity/biometric privacy invariant failed")

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "mask_checkpoint_sha256": mask_sha,
        "camera_algorithm": extension["algorithm"],
        "camera_rows": len(rows),
        "valid_camera_pairs": valid_pairs,
        "expected_camera_pairs": expected_pairs,
        "coverage": round(coverage, 6),
        "median_translation_x_px": round(median_tx, 6),
        "median_translation_y_px": round(median_ty, 6),
        "median_ransac_inlier_ratio": round(median_inlier, 6),
        "background_track_samples": background_samples,
        "foreground_exclusion": True,
        "unsupported_3d_labels_emitted": False,
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
