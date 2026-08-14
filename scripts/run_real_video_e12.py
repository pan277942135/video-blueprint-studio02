from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import traceback
import uuid
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core.artifact_integrity import require_blueprint_artifacts, verify_bundle_zip
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.pipeline_core.e12_runtime import (
    DETECTOR_SHA256,
    MASK_SHA256,
    POSE_SHA256,
    E12RuntimePaths,
    bootstrap_e12_runtime,
    sha256_file,
)
from packages.pipeline_core.media_probe import compute_sha256
from packages.pipeline_core.mediapipe_face_hand_backend import (
    APPROVED_FACE_TASK_SHA256,
    APPROVED_HAND_TASK_SHA256,
)
from packages.pipeline_core.production_media_pipeline import run_production_media_pipeline
from packages.pipeline_core.real_video_acceptance import summarize_blueprint_acceptance

_RUNTIME_ARG_NAMES = (
    "det_config",
    "det_checkpoint",
    "pose_config",
    "pose_checkpoint",
    "mask_config",
    "mask_checkpoint",
    "face_task",
    "hand_task",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the approved full Video Blueprint production stack and emit an E12 acceptance report. "
            "By default, approved runtime assets are bootstrapped and hash-verified automatically."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--runtime-dir")
    parser.add_argument("--det-config")
    parser.add_argument("--det-checkpoint")
    parser.add_argument("--pose-config")
    parser.add_argument("--pose-checkpoint")
    parser.add_argument("--mask-config")
    parser.add_argument("--mask-checkpoint")
    parser.add_argument("--face-task")
    parser.add_argument("--hand-task")
    parser.add_argument("--output-dir", default=".e12")
    parser.add_argument("--low-score-threshold", type=float, default=0.5)
    return parser.parse_args()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")


def _require_sha(path: pathlib.Path, *, label: str, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def _explicit_runtime_paths(args: argparse.Namespace) -> E12RuntimePaths | None:
    supplied = {name: getattr(args, name) for name in _RUNTIME_ARG_NAMES}
    populated = [name for name, value in supplied.items() if value]
    if not populated:
        return None
    if len(populated) != len(_RUNTIME_ARG_NAMES):
        missing = sorted(name for name, value in supplied.items() if not value)
        raise RuntimeError(
            "partial explicit runtime configuration is not allowed; either omit all runtime path arguments "
            f"for verified bootstrap or provide every runtime path. Missing: {', '.join(missing)}"
        )
    return E12RuntimePaths(
        det_config=pathlib.Path(args.det_config).resolve(),
        det_checkpoint=pathlib.Path(args.det_checkpoint).resolve(),
        pose_config=pathlib.Path(args.pose_config).resolve(),
        pose_checkpoint=pathlib.Path(args.pose_checkpoint).resolve(),
        mask_config=pathlib.Path(args.mask_config).resolve(),
        mask_checkpoint=pathlib.Path(args.mask_checkpoint).resolve(),
        face_task=pathlib.Path(args.face_task).resolve(),
        hand_task=pathlib.Path(args.hand_task).resolve(),
        manifest=pathlib.Path(""),
    )


def _resolve_runtime(args: argparse.Namespace, output_dir: pathlib.Path) -> E12RuntimePaths:
    explicit = _explicit_runtime_paths(args)
    if explicit is not None:
        return explicit
    runtime_dir = pathlib.Path(args.runtime_dir).resolve() if args.runtime_dir else output_dir / "runtime"
    return bootstrap_e12_runtime(runtime_dir)


def _configure_runtime(paths: E12RuntimePaths) -> None:
    os.environ.update(
        {
            "VBS_RTMDET_CONFIG": str(paths.det_config),
            "VBS_RTMDET_CHECKPOINT": str(paths.det_checkpoint),
            "VBS_RTMDET_WEIGHTS_LICENSE": "approved project RTMDet precedent",
            "VBS_RTMDET_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_DEVICE": "cpu",
            "VBS_RTMDET_SCORE_THRESHOLD": "0.30",
            "VBS_RTMPOSE_CONFIG": str(paths.pose_config),
            "VBS_RTMPOSE_CHECKPOINT": str(paths.pose_checkpoint),
            "VBS_RTMPOSE_EXPECTED_SHA256": POSE_SHA256,
            "VBS_RTMPOSE_WEIGHTS_LICENSE": "approved project RTMPose precedent",
            "VBS_RTMPOSE_WEIGHTS_APPROVED": "true",
            "VBS_RTMPOSE_DEVICE": "cpu",
            "VBS_RTMDET_INS_CONFIG": str(paths.mask_config),
            "VBS_RTMDET_INS_CHECKPOINT": str(paths.mask_checkpoint),
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal certification; redistribution not approved",
            "VBS_RTMDET_INS_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_INS_DEVICE": "cpu",
            "VBS_RTMDET_INS_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_IOU_THRESHOLD": "0.20",
            "VBS_MEDIAPIPE_FACE_TASK": str(paths.face_task),
            "VBS_MEDIAPIPE_HAND_TASK": str(paths.hand_task),
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
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "e12_run_manifest.json"
    failure_path = output_dir / "e12_failure_report.json"
    job_id = str(uuid.uuid4())
    events: list[dict[str, Any]] = []
    phase = "initializing"
    source_sha256: str | None = None

    def event(status: str, detail: str | None = None) -> None:
        row: dict[str, Any] = {"at": _now(), "phase": phase, "status": status}
        if detail:
            row["detail"] = detail
        events.append(row)

    event("started")
    _write_json(
        run_manifest_path,
        {
            "status": "running",
            "job_id": job_id,
            "phase": phase,
            "events": events,
        },
    )

    try:
        if not 0.0 <= args.low_score_threshold <= 1.0:
            raise RuntimeError("--low-score-threshold must be in [0,1]")

        video = pathlib.Path(args.video).resolve()
        _require_file(video, "video")

        phase = "runtime_bootstrap_and_verification"
        event("started")
        runtime = _resolve_runtime(args, output_dir)
        for path, label in (
            (runtime.det_config, "detector config"),
            (runtime.det_checkpoint, "detector checkpoint"),
            (runtime.pose_config, "pose config"),
            (runtime.pose_checkpoint, "pose checkpoint"),
            (runtime.mask_config, "mask config"),
            (runtime.mask_checkpoint, "mask checkpoint"),
            (runtime.face_task, "FaceLandmarker task"),
            (runtime.hand_task, "HandLandmarker task"),
        ):
            _require_file(path, label)

        runtime_hashes = {
            "detector": _require_sha(
                runtime.det_checkpoint,
                label="detector checkpoint",
                expected=DETECTOR_SHA256,
            ),
            "pose": _require_sha(runtime.pose_checkpoint, label="pose checkpoint", expected=POSE_SHA256),
            "mask": _require_sha(runtime.mask_checkpoint, label="mask checkpoint", expected=MASK_SHA256),
            "face_task": _require_sha(
                runtime.face_task,
                label="FaceLandmarker task",
                expected=APPROVED_FACE_TASK_SHA256,
            ),
            "hand_task": _require_sha(
                runtime.hand_task,
                label="HandLandmarker task",
                expected=APPROVED_HAND_TASK_SHA256,
            ),
        }
        _configure_runtime(runtime)
        event("succeeded")

        phase = "source_hash"
        event("started")
        source_sha256 = compute_sha256(str(video))
        event("succeeded")

        phase = "production_pipeline"
        event("started")
        blueprint, sidecars = run_production_media_pipeline(
            job_id=job_id,
            video_file_name=video.name,
            video_sha256=source_sha256,
            video_path=str(video),
        )
        event("succeeded")

        phase = "canonical_validation"
        event("started")
        valid, errors = BlueprintValidator().validate(blueprint)
        integrity = require_blueprint_artifacts(blueprint, sidecars) if valid else None
        validation_report = {
            "valid": valid,
            "schema_version": "Draft 2020-12",
            "validated_at": _now(),
            "errors": errors,
            "summary": {"passed_rules": max(0, 25 - len(errors)), "failed_rules": len(errors)},
            "artifact_integrity": integrity,
        }

        blueprint_path = output_dir / "blueprint.json"
        validation_path = output_dir / "validation_report.json"
        bundle_path = output_dir / "bundle.zip"
        acceptance_path = output_dir / "e12_real_video_acceptance.json"

        _write_json(blueprint_path, blueprint)
        _write_json(validation_path, validation_report)
        if not valid or integrity is None:
            raise RuntimeError("production Blueprint failed canonical validation: " + " | ".join(errors))
        event("succeeded")

        phase = "bundle_write_and_physical_verification"
        event("started")
        create_bundle_zip(blueprint, sidecars, validation_report, str(bundle_path))
        bundle_integrity = verify_bundle_zip(str(bundle_path))
        if not bundle_integrity["valid"]:
            raise RuntimeError(
                "post-write Bundle verification failed: " + " | ".join(bundle_integrity["errors"])
            )
        event("succeeded")

        phase = "e12_diagnostics"
        event("started")
        acceptance = summarize_blueprint_acceptance(
            blueprint,
            integrity=bundle_integrity,
            low_score_threshold=args.low_score_threshold,
        )
        acceptance["bundle"] = {
            "path": str(bundle_path),
            "size_bytes": bundle_path.stat().st_size,
            "sha256": sha256_file(bundle_path),
        }
        _write_json(acceptance_path, acceptance)
        event("succeeded")

        phase = "completed"
        event("succeeded")
        run_manifest = {
            "status": "completed",
            "job_id": job_id,
            "source": {"path": str(video), "sha256": source_sha256},
            "runtime_hashes": runtime_hashes,
            "runtime_manifest": str(runtime.manifest) if runtime.manifest.name else None,
            "outputs": {
                "blueprint": str(blueprint_path),
                "validation_report": str(validation_path),
                "bundle": str(bundle_path),
                "acceptance_report": str(acceptance_path),
                "failure_report": None,
            },
            "machine_gate": acceptance["machine_gate"],
            "final_acceptance": acceptance["final_acceptance"],
            "events": events,
        }
        _write_json(run_manifest_path, run_manifest)
        print(json.dumps(run_manifest, indent=2))
        return 0 if acceptance["machine_gate"] != "failed" else 2
    except BaseException as exc:
        event("failed", str(exc))
        failure = {
            "status": "failed",
            "job_id": job_id,
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "source_sha256": source_sha256,
            "events": events,
        }
        _write_json(failure_path, failure)
        _write_json(
            run_manifest_path,
            {
                "status": "failed",
                "job_id": job_id,
                "phase": phase,
                "source_sha256": source_sha256,
                "failure_report": str(failure_path),
                "events": events,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
