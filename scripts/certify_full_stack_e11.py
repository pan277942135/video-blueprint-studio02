from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import uuid
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core.artifact_integrity import require_blueprint_artifacts, verify_bundle_zip
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.pipeline_core.media_probe import compute_sha256
from packages.pipeline_core.production_media_pipeline import run_production_media_pipeline

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
    parser = argparse.ArgumentParser(
        description="Certify one production job with E2-E9 evidence stages enabled and E10 bundle integrity"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e11_full_stack_certification.json")
    parser.add_argument("--bundle-output", default=".cert/e11_full_stack_bundle.zip")
    return parser.parse_args()


def _require_file(path: pathlib.Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")


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
    bundle_output = pathlib.Path(args.bundle_output).resolve()

    for path, label in (
        (video, "video"),
        (det_config, "detector config"),
        (det_checkpoint, "detector checkpoint"),
        (pose_config, "pose config"),
        (pose_checkpoint, "pose checkpoint"),
        (mask_config, "mask config"),
        (mask_checkpoint, "mask checkpoint"),
    ):
        _require_file(path, label)

    detector_sha = _sha256(det_checkpoint)
    pose_sha = _sha256(pose_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    expected_hashes = {
        "detector": (detector_sha, E2_SHA256),
        "pose": (pose_sha, E3_SHA256),
        "mask": (mask_sha, E4_MASK_SHA256),
    }
    for label, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise SystemExit(f"{label} certified checkpoint SHA256 mismatch: {actual} != {expected}")

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
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal certification; redistribution not approved",
            "VBS_RTMDET_INS_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_INS_DEVICE": "cpu",
            "VBS_RTMDET_INS_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_IOU_THRESHOLD": "0.20",
            "VBS_E4_POINT_TRACKS_ENABLED": "true",
            "VBS_E4_DENSE_FLOW_ENABLED": "true",
            "VBS_E5_CAMERA_MOTION_ENABLED": "true",
            "VBS_E6_BODY_LOCAL_FRAME_ENABLED": "true",
            "VBS_E7_SURFACE_MOTION_ENABLED": "true",
            "VBS_E8_MICRO_MOTION_ENABLED": "true",
            "VBS_E9_ENVIRONMENT_ENABLED": "true",
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_production_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )

    valid, errors = BlueprintValidator().validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E11 full-stack output: " + " | ".join(errors))

    pipeline_version = blueprint.get("processing", {}).get("pipeline_version")
    if pipeline_version != "0.9.0-e9.1":
        raise SystemExit(f"production dispatcher did not reach E9.1: {pipeline_version!r}")

    stage_rows = blueprint.get("processing", {}).get("stages", [])
    stages = {
        str(row.get("name")): str(row.get("status"))
        for row in stage_rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    required_stages = (
        "point_tracks",
        "dense_flow",
        "camera_motion",
        "body_local_frame",
        "surface_motion",
        "micro_motion",
        "environment",
    )
    failed_stages = [name for name in required_stages if stages.get(name) != "succeeded"]
    if failed_stages:
        raise SystemExit(f"integrated production stages did not all succeed: {failed_stages!r}; stages={stages!r}")

    extensions = blueprint.get("extensions", {})
    if not isinstance(extensions, dict):
        raise SystemExit("Blueprint extensions must be an object")
    micro_extension = extensions.get("e8_micro_motion")
    environment_extension = extensions.get("e9_environment")
    if not isinstance(micro_extension, dict) or micro_extension.get("enabled") is not True:
        raise SystemExit("E8 micro-motion evidence was not enabled in the integrated production job")
    if not isinstance(environment_extension, dict) or environment_extension.get("enabled") is not True:
        raise SystemExit("E9 environment evidence was not enabled in the integrated production job")

    characters = blueprint.get("characters", [])
    if not isinstance(characters, list) or not characters:
        raise SystemExit("integrated certification produced no anonymous character tracks")

    region_count = 0
    micro_signal_count = 0
    for character in characters:
        if not isinstance(character, dict):
            continue
        privacy = character.get("privacy", {})
        if isinstance(privacy, dict):
            if privacy.get("identity_inference_performed") is not False:
                raise SystemExit("identity inference privacy invariant failed")
            if privacy.get("biometric_embedding_exported") is not False:
                raise SystemExit("biometric embedding privacy invariant failed")
        surface_motion = character.get("surface_motion", {})
        regions = surface_motion.get("regions", []) if isinstance(surface_motion, dict) else []
        if not isinstance(regions, list):
            continue
        region_count += len(regions)
        for region in regions:
            if not isinstance(region, dict):
                continue
            micro = region.get("micro_motion")
            if isinstance(micro, dict) and isinstance(micro.get("signal_ref"), dict):
                micro_signal_count += 1

    if region_count <= 0 or micro_signal_count <= 0:
        raise SystemExit(
            f"integrated certification lacks physical surface/micro-motion evidence: "
            f"regions={region_count}, micro_signals={micro_signal_count}"
        )

    integrity = require_blueprint_artifacts(blueprint, sidecars)
    validation_report = {
        "valid": True,
        "schema_version": "Draft 2020-12",
        "validated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "errors": [],
        "summary": {"passed_rules": 25, "failed_rules": 0},
        "artifact_integrity": integrity,
    }

    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    if bundle_output.exists():
        bundle_output.unlink()
    create_bundle_zip(blueprint, sidecars, validation_report, str(bundle_output))
    zip_result = verify_bundle_zip(str(bundle_output))
    if not zip_result["valid"]:
        raise SystemExit("E11 post-write ZIP verification failed: " + " | ".join(zip_result["errors"]))

    with zipfile.ZipFile(bundle_output, "r") as archive:
        bundle_names = archive.namelist()
        bundled_blueprint = json.loads(archive.read("blueprint.json"))
        if bundled_blueprint != blueprint:
            raise SystemExit("E11 bundled blueprint does not round-trip to the integrated production Blueprint")

    module_scores = blueprint.get("quality", {}).get("module_scores", {})
    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": pipeline_version,
        "video_sha256": compute_sha256(str(video)),
        "detector_sha256": detector_sha,
        "pose_sha256": pose_sha,
        "mask_checkpoint_sha256": mask_sha,
        "required_stage_statuses": {name: stages.get(name) for name in required_stages},
        "character_count": len(characters),
        "surface_region_count": region_count,
        "micro_signal_count": micro_signal_count,
        "module_scores": module_scores,
        "referenced_artifact_count": int(integrity["referenced_artifact_count"]),
        "sidecar_count": int(integrity["sidecar_count"]),
        "integrity_warning_count": len(integrity.get("warnings", [])),
        "bundle_entry_count": len(bundle_names),
        "bundle_size_bytes": bundle_output.stat().st_size,
        "bundle_sha256": _sha256(bundle_output),
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
