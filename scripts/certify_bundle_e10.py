from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import uuid
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core.artifact_integrity import verify_bundle_zip
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.pipeline_core.media_probe import compute_sha256
from packages.pipeline_core.production_media_pipeline import run_production_media_pipeline

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E4_MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E10 production dispatch and physical Bundle integrity")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e10_bundle_certification.json")
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
            raise SystemExit(f"missing E10 certification input: {path}")

    detector_sha = _sha256(det_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    if detector_sha != E2_SHA256 or mask_sha != E4_MASK_SHA256:
        raise SystemExit("E10 certified checkpoint SHA256 mismatch")

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
            "VBS_E4_POINT_TRACKS_ENABLED": "false",
            "VBS_E4_DENSE_FLOW_ENABLED": "false",
            "VBS_E5_CAMERA_MOTION_ENABLED": "false",
            "VBS_E6_BODY_LOCAL_FRAME_ENABLED": "false",
            "VBS_E7_SURFACE_MOTION_ENABLED": "false",
            "VBS_E8_MICRO_MOTION_ENABLED": "false",
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
        raise SystemExit("BlueprintValidator rejected E10 production output: " + " | ".join(errors))
    if blueprint.get("processing", {}).get("pipeline_version") != "0.9.0-e9.1":
        raise SystemExit("production dispatcher did not reach E9.1")

    extension = blueprint.get("extensions", {}).get("e9_environment")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("production dispatcher did not emit E9 environment evidence")
    privacy_flags = {
        "semantic_scene_inference_performed": False,
        "weather_inference_performed": False,
        "material_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    for key, expected in privacy_flags.items():
        if extension.get(key) != expected:
            raise SystemExit(f"E10 privacy/evidence boundary drifted: {key}")

    validation_report = {
        "valid": True,
        "schema_version": "Draft 2020-12",
        "validated_at": "2026-08-14T00:00:00Z",
        "errors": [],
        "summary": {"passed_rules": 25, "failed_rules": 0},
    }
    with tempfile.TemporaryDirectory(prefix="vbs_e10_cert_") as temp_dir:
        bundle_path = pathlib.Path(temp_dir) / "bundle.zip"
        create_bundle_zip(blueprint, sidecars, validation_report, str(bundle_path))
        zip_result = verify_bundle_zip(str(bundle_path))
        if not zip_result["valid"]:
            raise SystemExit("E10 post-write ZIP verification failed: " + " | ".join(zip_result["errors"]))

        with zipfile.ZipFile(bundle_path, "r") as archive:
            names = archive.namelist()
            bundled_blueprint = json.loads(archive.read("blueprint.json"))
            bundled_validation = json.loads(archive.read("validation_report.json"))
            manifest = json.loads(archive.read("bundle_manifest.json"))
            if bundled_blueprint != blueprint:
                raise SystemExit("E10 bundle blueprint bytes do not round-trip to production Blueprint")
            integrity = bundled_validation.get("artifact_integrity")
            if not isinstance(integrity, dict) or integrity.get("valid") is not True:
                raise SystemExit("E10 validation_report does not contain passing artifact integrity evidence")
            if manifest.get("file_count") != len(names):
                raise SystemExit("E10 bundle manifest file_count drifted from physical ZIP")
            if "artifacts/reports/environment_photometry.json" not in names:
                raise SystemExit("E10 physical environment report is missing from final ZIP")
            if "artifacts/timeseries/environment_photometry.npz" not in names:
                raise SystemExit("E10 physical environment photometry NPZ is missing from final ZIP")
            if "artifacts/timeseries/environment_background_mask.rle.json" not in names:
                raise SystemExit("E10 physical environment background mask is missing from final ZIP")
            bundle_sha = _sha256(bundle_path)
            bundle_size = bundle_path.stat().st_size

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "mask_checkpoint_sha256": mask_sha,
        "bundle_sha256": bundle_sha,
        "bundle_size_bytes": bundle_size,
        "bundle_entry_count": int(manifest["file_count"]),
        "referenced_artifact_count": int(integrity["referenced_artifact_count"]),
        "sidecar_count": int(integrity["sidecar_count"]),
        "integrity_warning_count": len(integrity.get("warnings", [])),
        **privacy_flags,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
