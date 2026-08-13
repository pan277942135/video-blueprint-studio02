from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import uuid

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.binary_rle import decode_rle
from packages.pipeline_core.e4_media_pipeline import run_e4_media_pipeline
from packages.pipeline_core.media_probe import compute_sha256

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E4_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E4.1 RTMDet-Ins person masks")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e4_1_rtmdet_ins_certification.json")
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

    det_sha = _sha256(det_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    if det_sha != E2_SHA256:
        raise SystemExit(f"E2 detector SHA256 mismatch: {det_sha}")
    if mask_sha != E4_SHA256:
        raise SystemExit(f"E4 mask checkpoint SHA256 mismatch: {mask_sha}")

    os.environ.update(
        {
            "VBS_RTMDET_CONFIG": str(det_config),
            "VBS_RTMDET_CHECKPOINT": str(det_checkpoint),
            "VBS_RTMDET_WEIGHTS_LICENSE": "OpenMMLab RTMDet deployment / Apache-2.0 project precedent",
            "VBS_RTMDET_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_DEVICE": "cpu",
            "VBS_RTMDET_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_CONFIG": str(mask_config),
            "VBS_RTMDET_INS_CHECKPOINT": str(mask_checkpoint),
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal CI certification; Model Zoo checkpoint clarification pending",
            "VBS_RTMDET_INS_WEIGHTS_APPROVED": "true",
            "VBS_RTMDET_INS_DEVICE": "cpu",
            "VBS_RTMDET_INS_SCORE_THRESHOLD": "0.30",
            "VBS_RTMDET_INS_IOU_THRESHOLD": "0.20",
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e4_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )
    valid, errors = BlueprintValidator().validate(blueprint)
    if not valid:
        raise SystemExit(f"Blueprint validation failed: {errors}")
    if blueprint["processing"]["pipeline_version"] != "0.4.0-e4.1":
        raise SystemExit("E4.1 pipeline version was not emitted")

    extension = blueprint.get("extensions", {}).get("e4_person_mask", {})
    if extension.get("enabled") is not True:
        raise SystemExit("E4.1 extension did not enable")
    stage = next(
        (stage for stage in blueprint["processing"]["stages"] if stage["name"] == "person_mask"),
        None,
    )
    if not stage or stage.get("status") != "succeeded":
        raise SystemExit("person_mask stage did not succeed")

    characters = blueprint.get("characters", [])
    refs = [character.get("person_mask_ref") for character in characters]
    refs = [ref for ref in refs if isinstance(ref, dict)]
    if not refs:
        raise SystemExit("real RTMDet-Ins produced no character mask refs")

    observed_frames = 0
    for ref in refs:
        path = sidecars.get(ref["uri"])
        if not isinstance(path, (str, os.PathLike)) or not pathlib.Path(path).is_file():
            raise SystemExit(f"mask sidecar missing: {ref['uri']}")
        if _sha256(pathlib.Path(path)) != ref["checksum_sha256"]:
            raise SystemExit(f"mask sidecar checksum mismatch: {ref['uri']}")
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if len(payload["frames"]) != blueprint["timebase"]["frame_count"]:
            raise SystemExit("mask sidecar frame count mismatch")
        for item in payload["frames"]:
            if item is None:
                continue
            mask = decode_rle(item["counts"], payload["height"], payload["width"])
            if not mask.any():
                raise SystemExit("observed RLE mask decoded empty")
            observed_frames += 1

    if observed_frames == 0:
        raise SystemExit("no observed segmentation frames were certified")
    provenance = next(
        (tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "person_mask"),
        None,
    )
    if not provenance or provenance.get("weights_sha256") != E4_SHA256:
        raise SystemExit("person-mask provenance is missing or has the wrong checkpoint hash")
    for character in characters:
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False:
            raise SystemExit("identity inference privacy invariant failed")
        if privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("biometric export privacy invariant failed")

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": det_sha,
        "mask_checkpoint_sha256": mask_sha,
        "character_count": len(characters),
        "mask_ref_count": len(refs),
        "observed_mask_frames": observed_frames,
        "coverage": extension.get("coverage"),
        "person_mask_stage": stage,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "checkpoint_redistributed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
