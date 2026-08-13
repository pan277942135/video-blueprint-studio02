from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import uuid
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator, FormatChecker

from packages.pipeline_core.media_probe import compute_sha256
from packages.pipeline_core.real_media_pipeline import run_real_media_pipeline


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify the approved E3.1 RTMPose Tiny runtime end to end")
    parser.add_argument("--video", required=True)
    parser.add_argument("--rtmdet-config", required=True)
    parser.add_argument("--rtmdet-checkpoint", required=True)
    parser.add_argument("--rtmpose-config", required=True)
    parser.add_argument("--rtmpose-checkpoint", required=True)
    parser.add_argument("--schema", default="contracts/video_blueprint.schema.json")
    parser.add_argument("--output", default=".cert/e3_rtmpose_certification.json")
    parser.add_argument("--rtmpose-sha-prefix", default="e613ba3f")
    parser.add_argument("--expected-rtmpose-sha256", default="")
    parser.add_argument(
        "--expected-rtmdet-sha256",
        default="78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb",
    )
    parser.add_argument("--min-mean-confidence", type=float, default=0.30)
    return parser.parse_args()


def _require_file(path: pathlib.Path) -> None:
    if not path.is_file():
        raise SystemExit(f"required certification input is missing: {path}")


def _validate_pose_sidecar(
    character: dict[str, Any],
    *,
    sidecars: dict[str, Any],
    frame_count: int,
) -> tuple[float, float]:
    character_id = str(character.get("character_id"))
    pose = character.get("pose")
    if not isinstance(pose, dict) or pose.get("enabled") is not True:
        raise SystemExit(f"E3.1 certification failed: pose is not enabled for {character_id}")
    if pose.get("skeleton_name") != "coco17" or pose.get("keypoint_count") != 17:
        raise SystemExit(f"E3.1 certification failed: unexpected skeleton for {character_id}")

    keypoints_ref = pose.get("keypoints_2d_ref")
    confidence_ref = pose.get("confidence_ref")
    if not isinstance(keypoints_ref, dict) or not isinstance(confidence_ref, dict):
        raise SystemExit(f"E3.1 certification failed: pose refs are missing for {character_id}")

    uri = keypoints_ref.get("uri")
    if not isinstance(uri, str) or confidence_ref.get("uri") != uri:
        raise SystemExit(f"E3.1 certification failed: pose refs do not share one sidecar for {character_id}")
    sidecar = sidecars.get(uri)
    if not isinstance(sidecar, (str, os.PathLike)):
        raise SystemExit(f"E3.1 certification failed: pose sidecar is unavailable for {character_id}")
    path = pathlib.Path(sidecar)
    _require_file(path)

    actual_sha256 = _sha256_file(path)
    if keypoints_ref.get("checksum_sha256") != actual_sha256:
        raise SystemExit(f"E3.1 certification failed: keypoint checksum mismatch for {character_id}")
    if confidence_ref.get("checksum_sha256") != actual_sha256:
        raise SystemExit(f"E3.1 certification failed: confidence checksum mismatch for {character_id}")

    with np.load(path, allow_pickle=False) as bundle:
        if "keypoints_xy" not in bundle or "confidence" not in bundle:
            raise SystemExit(f"E3.1 certification failed: pose arrays are missing for {character_id}")
        keypoints = np.asarray(bundle["keypoints_xy"], dtype=np.float32)
        confidence = np.asarray(bundle["confidence"], dtype=np.float32)

    if keypoints.shape != (frame_count, 17, 2):
        raise SystemExit(f"E3.1 certification failed: keypoint shape mismatch for {character_id}: {keypoints.shape}")
    if confidence.shape != (frame_count, 17):
        raise SystemExit(f"E3.1 certification failed: confidence shape mismatch for {character_id}: {confidence.shape}")
    if not np.all(np.isfinite(confidence)):
        raise SystemExit(f"E3.1 certification failed: non-finite confidence for {character_id}")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise SystemExit(f"E3.1 certification failed: confidence outside [0, 1] for {character_id}")

    finite_xy = np.isfinite(keypoints)
    if np.any(finite_xy[:, :, 0] != finite_xy[:, :, 1]):
        raise SystemExit(f"E3.1 certification failed: x/y finite masks differ for {character_id}")
    missing_frames = ~np.any(finite_xy, axis=(1, 2))
    if np.any(confidence[missing_frames] != 0.0):
        raise SystemExit(f"E3.1 certification failed: missing pose frames carry confidence for {character_id}")

    estimated_frames = ~missing_frames
    coverage = float(estimated_frames.sum() / frame_count)
    nonzero_confidence = confidence[confidence > 0]
    mean_confidence = float(nonzero_confidence.mean()) if nonzero_confidence.size else 0.0
    return coverage, mean_confidence


def main() -> int:
    args = _parse_args()
    video_path = pathlib.Path(args.video).resolve()
    rtmdet_config = pathlib.Path(args.rtmdet_config).resolve()
    rtmdet_checkpoint = pathlib.Path(args.rtmdet_checkpoint).resolve()
    rtmpose_config = pathlib.Path(args.rtmpose_config).resolve()
    rtmpose_checkpoint = pathlib.Path(args.rtmpose_checkpoint).resolve()
    schema_path = pathlib.Path(args.schema).resolve()
    output_path = pathlib.Path(args.output).resolve()

    for path in (
        video_path,
        rtmdet_config,
        rtmdet_checkpoint,
        rtmpose_config,
        rtmpose_checkpoint,
        schema_path,
    ):
        _require_file(path)

    rtmdet_sha256 = _sha256_file(rtmdet_checkpoint)
    if rtmdet_sha256 != args.expected_rtmdet_sha256.lower():
        raise SystemExit(
            f"RTMDet checkpoint SHA256 mismatch: expected {args.expected_rtmdet_sha256.lower()}, got {rtmdet_sha256}"
        )

    rtmpose_sha256 = _sha256_file(rtmpose_checkpoint)
    if not rtmpose_sha256.startswith(args.rtmpose_sha_prefix.lower()):
        raise SystemExit(
            "RTMPose checkpoint hash does not match the publisher filename hash prefix: "
            f"expected {args.rtmpose_sha_prefix.lower()}, got {rtmpose_sha256}"
        )
    if args.expected_rtmpose_sha256 and rtmpose_sha256 != args.expected_rtmpose_sha256.lower():
        raise SystemExit(
            "RTMPose checkpoint SHA256 mismatch: "
            f"expected {args.expected_rtmpose_sha256.lower()}, got {rtmpose_sha256}"
        )

    os.environ["VBS_RTMDET_CONFIG"] = str(rtmdet_config)
    os.environ["VBS_RTMDET_CHECKPOINT"] = str(rtmdet_checkpoint)
    os.environ["VBS_RTMDET_WEIGHTS_LICENSE"] = "OpenMMLab RTMDet: project-approved official checkpoint"
    os.environ["VBS_RTMDET_WEIGHTS_APPROVED"] = "true"
    os.environ["VBS_RTMDET_DEVICE"] = "cpu"
    os.environ["VBS_RTMDET_SCORE_THRESHOLD"] = "0.30"

    os.environ["VBS_RTMPOSE_CONFIG"] = str(rtmpose_config)
    os.environ["VBS_RTMPOSE_CHECKPOINT"] = str(rtmpose_checkpoint)
    os.environ["VBS_RTMPOSE_WEIGHTS_LICENSE"] = "OpenMMLab MMPose Apache-2.0; project-approved official checkpoint"
    os.environ["VBS_RTMPOSE_WEIGHTS_APPROVED"] = "true"
    os.environ["VBS_RTMPOSE_DEVICE"] = "cpu"

    job_id = str(uuid.uuid4())
    video_sha256 = compute_sha256(str(video_path))
    blueprint, sidecars = run_real_media_pipeline(
        job_id=job_id,
        video_file_name=video_path.name,
        video_sha256=video_sha256,
        video_path=str(video_path),
    )

    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)

    if blueprint.get("extensions", {}).get("e2_person_tracking", {}).get("enabled") is not True:
        raise SystemExit("E3.1 certification failed: E2.2 person tracking did not enable")
    if blueprint.get("extensions", {}).get("e3_pose_2d", {}).get("enabled") is not True:
        raise SystemExit("E3.1 certification failed: pose extension did not enable")

    stage = next(
        (item for item in blueprint["processing"]["stages"] if item["name"] == "pose_2d"),
        None,
    )
    if not stage or stage.get("status") != "succeeded":
        raise SystemExit("E3.1 certification failed: pose_2d stage did not succeed")

    characters = blueprint.get("characters", [])
    if not characters:
        raise SystemExit("E3.1 certification failed: no anonymous person tracks were available for pose")

    frame_count = int(blueprint["timebase"]["frame_count"])
    pose_metrics = [
        _validate_pose_sidecar(character, sidecars=sidecars, frame_count=frame_count)
        for character in characters
    ]
    max_coverage = max((coverage for coverage, _ in pose_metrics), default=0.0)
    max_mean_confidence = max((confidence for _, confidence in pose_metrics), default=0.0)
    if max_coverage <= 0.0:
        raise SystemExit("E3.1 certification failed: RTMPose produced no frame-aligned pose observations")
    if max_mean_confidence < args.min_mean_confidence:
        raise SystemExit(
            "E3.1 certification failed: strongest track mean nonzero confidence below threshold: "
            f"{max_mean_confidence:.6f} < {args.min_mean_confidence:.6f}"
        )

    report_kinds = {item["kind"] for item in blueprint.get("artifacts", {}).get("reports", [])}
    if "pose_2d" not in report_kinds:
        raise SystemExit("E3.1 certification failed: pose_2d report reference is missing")

    provenance = blueprint.get("provenance", {}).get("tools", [])
    pose_provenance = next((item for item in provenance if item.get("module") == "pose_2d"), None)
    if pose_provenance is None:
        raise SystemExit("E3.1 certification failed: RTMPose provenance is missing")
    if pose_provenance.get("weights_sha256") != rtmpose_sha256:
        raise SystemExit("E3.1 certification failed: recorded RTMPose hash differs from downloaded weights")

    for character in characters:
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False:
            raise SystemExit("privacy certification failed: identity inference was not explicitly false")
        if privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("privacy certification failed: biometric export was not explicitly false")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed",
        "job_id": job_id,
        "video_sha256": video_sha256,
        "rtmdet_weights_sha256": rtmdet_sha256,
        "rtmpose_weights_sha256": rtmpose_sha256,
        "rtmpose_weights_filename": rtmpose_checkpoint.name,
        "rtmpose_config_filename": rtmpose_config.name,
        "character_count": len(characters),
        "pose_character_count": len(pose_metrics),
        "max_pose_coverage": round(max_coverage, 6),
        "max_mean_nonzero_confidence": round(max_mean_confidence, 6),
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "pose_stage": stage,
        "privacy": {
            "identity_inference_performed": False,
            "biometric_embedding_exported": False,
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
