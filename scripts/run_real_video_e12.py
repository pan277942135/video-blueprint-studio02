from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import uuid

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core.artifact_integrity import require_blueprint_artifacts, verify_bundle_zip
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.pipeline_core.media_probe import compute_sha256
from packages.pipeline_core.mediapipe_face_hand_backend import (
    APPROVED_FACE_TASK_SHA256,
    APPROVED_HAND_TASK_SHA256,
)
from packages.pipeline_core.production_media_pipeline import run_production_media_pipeline
from packages.pipeline_core.real_video_acceptance import summarize_blueprint_acceptance

DETECTOR_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
POSE_SHA256 = "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"
MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the approved full Video Blueprint production stack and emit an E12 acceptance report"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--face-task", required=True)
    parser.add_argument("--hand-task", required=True)
    parser.add_argument("--output-dir", default=".e12")
    parser.add_argument("--low-score-threshold", type=float, default=0.5)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")


def _require_sha(path: pathlib.Path, *, label: str, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def _configure_runtime(
    *,
    det_config: pathlib.Path,
    det_checkpoint: pathlib.Path,
    pose_config: pathlib.Path,
    pose_checkpoint: pathlib.Path,
    mask_config: pathlib.Path,
    mask_checkpoint: pathlib.Path,
    face_task: pathlib.Path,
    hand_task: pathlib.Path,
) -> None:
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
            "VBS_RTMPOSE_EXPECTED_SHA256": POSE_SHA256,
            "VBS_RTMPOSE_WEIGHTS_LICENSE": "approved project RTMPose precedent",
            "VBS_RTMPOSE_WEIGHTS_APPROVED": "true",
            "VBS_RTMPOSE_DEVICE": "cpu",
            "VBS_RTMDET_INS_CONFIG": str(mask_config),
            "VBS_RTMDET_INS_CHECKPOINT": str(mask_checkpoint),
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal certification; redistribution not approved",
            "VBS_RTMDET_INS_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_INS_DEVICE": "cpu",
            "VBS_RTMDET_INS_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_IOU_THRESHOLD": "0.20",
            "VBS_MEDIAPIPE_FACE_TASK": str(face_task),
            "VBS_MEDIAPIPE_HAND_TASK": str(hand_task),
            "VBS_MEDIAPIPE_TASKS_APPROVED": "true",
            "VBS_E4_POINT_TRACKS_ENABLED": "true",
            "VBS_E4_DENSE_FLOW_ENABLED": "true",
            "VBS_E5_CAMERA_MOTION_ENABLED": "true",
            "VBS_E6_BODY_LOCAL_FRAME_ENABLED": "true",
            "VBS_E7_SURFACE_MOTION_ENABLED": "true",
            "VBS_E8_MICRO_MOTION_ENABLED": "true",
            "VBS_E9_ENVIRONMENT_ENABLED": "true",
        }
    )


def main() -> int:
    args = _args()
    if not 0.0 <= args.low_score_threshold <= 1.0:
        raise SystemExit("--low-score-threshold must be in [0,1]")

    video = pathlib.Path(args.video).resolve()
    det_config = pathlib.Path(args.det_config).resolve()
    det_checkpoint = pathlib.Path(args.det_checkpoint).resolve()
    pose_config = pathlib.Path(args.pose_config).resolve()
    pose_checkpoint = pathlib.Path(args.pose_checkpoint).resolve()
    mask_config = pathlib.Path(args.mask_config).resolve()
    mask_checkpoint = pathlib.Path(args.mask_checkpoint).resolve()
    face_task = pathlib.Path(args.face_task).resolve()
    hand_task = pathlib.Path(args.hand_task).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()

    for path, label in (
        (video, "video"),
        (det_config, "detector config"),
        (det_checkpoint, "detector checkpoint"),
        (pose_config, "pose config"),
        (pose_checkpoint, "pose checkpoint"),
        (mask_config, "mask config"),
        (mask_checkpoint, "mask checkpoint"),
        (face_task, "FaceLandmarker task"),
        (hand_task, "HandLandmarker task"),
    ):
        _require_file(path, label)

    runtime_hashes = {
        "detector": _require_sha(det_checkpoint, label="detector checkpoint", expected=DETECTOR_SHA256),
        "pose": _require_sha(pose_checkpoint, label="pose checkpoint", expected=POSE_SHA256),
        "mask": _require_sha(mask_checkpoint, label="mask checkpoint", expected=MASK_SHA256),
        "face_task": _require_sha(face_task, label="FaceLandmarker task", expected=APPROVED_FACE_TASK_SHA256),
        "hand_task": _require_sha(hand_task, label="HandLandmarker task", expected=APPROVED_HAND_TASK_SHA256),
    }

    _configure_runtime(
        det_config=det_config,
        det_checkpoint=det_checkpoint,
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        mask_config=mask_config,
        mask_checkpoint=mask_checkpoint,
        face_task=face_task,
        hand_task=hand_task,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    source_sha256 = compute_sha256(str(video))
    blueprint, sidecars = run_production_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=source_sha256,
        video_path=str(video),
    )

    valid, errors = BlueprintValidator().validate(blueprint)
    integrity = require_blueprint_artifacts(blueprint, sidecars) if valid else None
    validation_report = {
        "valid": valid,
        "schema_version": "Draft 2020-12",
        "validated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "errors": errors,
        "summary": {"passed_rules": max(0, 25 - len(errors)), "failed_rules": len(errors)},
        "artifact_integrity": integrity,
    }

    blueprint_path = output_dir / "blueprint.json"
    validation_path = output_dir / "validation_report.json"
    bundle_path = output_dir / "bundle.zip"
    acceptance_path = output_dir / "e12_real_video_acceptance.json"
    run_manifest_path = output_dir / "e12_run_manifest.json"

    blueprint_path.write_text(json.dumps(blueprint, indent=2) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")

    if not valid or integrity is None:
        raise SystemExit("production Blueprint failed canonical validation: " + " | ".join(errors))

    create_bundle_zip(blueprint, sidecars, validation_report, str(bundle_path))
    bundle_integrity = verify_bundle_zip(str(bundle_path))
    if not bundle_integrity["valid"]:
        raise SystemExit("post-write Bundle verification failed: " + " | ".join(bundle_integrity["errors"]))

    acceptance = summarize_blueprint_acceptance(
        blueprint,
        integrity=bundle_integrity,
        low_score_threshold=args.low_score_threshold,
    )
    acceptance["bundle"] = {
        "path": str(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "sha256": _sha256(bundle_path),
    }
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")

    run_manifest = {
        "job_id": job_id,
        "source": {"path": str(video), "sha256": source_sha256},
        "runtime_hashes": runtime_hashes,
        "outputs": {
            "blueprint": str(blueprint_path),
            "validation_report": str(validation_path),
            "bundle": str(bundle_path),
            "acceptance_report": str(acceptance_path),
        },
        "machine_gate": acceptance["machine_gate"],
        "final_acceptance": acceptance["final_acceptance"],
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))
    return 0 if acceptance["machine_gate"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
