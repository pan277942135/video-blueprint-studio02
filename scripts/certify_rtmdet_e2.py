from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import uuid

# `python scripts/certify_rtmdet_e2.py` places only the scripts directory at
# the front of sys.path. Add the repository root explicitly so the in-repo
# `packages.*` namespace is available both in CI and for local certification.
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
    parser = argparse.ArgumentParser(description="Certify the approved E2.2 RTMDet Tiny runtime end to end")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--schema", default="contracts/video_blueprint.schema.json")
    parser.add_argument("--output", default=".cert/e2_rtmdet_certification.json")
    parser.add_argument("--sha-prefix", default="78e30dcc")
    parser.add_argument("--expected-sha256", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    video_path = pathlib.Path(args.video).resolve()
    config_path = pathlib.Path(args.config).resolve()
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    schema_path = pathlib.Path(args.schema).resolve()
    output_path = pathlib.Path(args.output).resolve()

    for path in (video_path, config_path, checkpoint_path, schema_path):
        if not path.is_file():
            raise SystemExit(f"required certification input is missing: {path}")

    weights_sha256 = _sha256_file(checkpoint_path)
    if not weights_sha256.startswith(args.sha_prefix.lower()):
        raise SystemExit(
            "checkpoint hash does not match the publisher filename hash prefix: "
            f"expected prefix {args.sha_prefix}, got {weights_sha256}"
        )
    if args.expected_sha256 and weights_sha256 != args.expected_sha256.lower():
        raise SystemExit(
            f"checkpoint SHA256 mismatch: expected {args.expected_sha256.lower()}, got {weights_sha256}"
        )

    os.environ["VBS_RTMDET_CONFIG"] = str(config_path)
    os.environ["VBS_RTMDET_CHECKPOINT"] = str(checkpoint_path)
    os.environ["VBS_RTMDET_WEIGHTS_LICENSE"] = (
        "OpenMMLab RTMDet deployment: Apache-2.0; checkpoint model-zoo licensing clarification pending"
    )
    os.environ["VBS_RTMDET_WEIGHTS_APPROVED"] = "true"
    os.environ["VBS_RTMDET_DEVICE"] = "cpu"
    os.environ["VBS_RTMDET_SCORE_THRESHOLD"] = "0.30"

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

    extension = blueprint.get("extensions", {}).get("e2_person_tracking", {})
    if extension.get("enabled") is not True:
        raise SystemExit("E2.2 certification failed: person tracking did not enable")
    characters = blueprint.get("characters", [])
    if not characters:
        raise SystemExit("E2.2 certification failed: approved RTMDet produced no person track")

    stage = next(
        (item for item in blueprint["processing"]["stages"] if item["name"] == "person_detection_tracking"),
        None,
    )
    if not stage or stage.get("status") != "succeeded":
        raise SystemExit("E2.2 certification failed: person_detection_tracking stage did not succeed")

    for character in characters:
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False:
            raise SystemExit("privacy certification failed: identity inference was not explicitly false")
        if privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("privacy certification failed: biometric export was not explicitly false")

    overlay_refs = blueprint.get("artifacts", {}).get("overlays", [])
    if not overlay_refs:
        raise SystemExit("E2.2 certification failed: no real person-tracking overlay artifact was emitted")
    for artifact in overlay_refs:
        uri = artifact["uri"]
        sidecar = sidecars.get(uri)
        if not isinstance(sidecar, (str, os.PathLike)) or not pathlib.Path(sidecar).is_file():
            raise SystemExit(f"E2.2 certification failed: overlay sidecar missing for {uri}")

    report_kinds = {item["kind"] for item in blueprint.get("artifacts", {}).get("reports", [])}
    if "person_tracking" not in report_kinds:
        raise SystemExit("E2.2 certification failed: person_tracking report reference is missing")

    provenance = blueprint.get("provenance", {}).get("tools", [])
    detector_provenance = next((item for item in provenance if item.get("module") == "person_detection"), None)
    if detector_provenance is None:
        raise SystemExit("E2.2 certification failed: detector provenance is missing")
    if detector_provenance.get("weights_sha256") != weights_sha256:
        raise SystemExit("E2.2 certification failed: recorded checkpoint hash differs from downloaded weights")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed",
        "job_id": job_id,
        "video_sha256": video_sha256,
        "weights_sha256": weights_sha256,
        "weights_filename": checkpoint_path.name,
        "config_filename": config_path.name,
        "character_count": len(characters),
        "shot_count": len(blueprint.get("shots", [])),
        "overlay_count": len(overlay_refs),
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "person_tracking_stage": stage,
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
