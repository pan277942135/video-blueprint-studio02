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
from packages.pipeline_core.e9_media_pipeline import run_e9_media_pipeline
from packages.pipeline_core.media_probe import compute_sha256
from packages.pipeline_core.person_mask import decode_binary_rle

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E4_MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E9.1 evidence-backed environment photometry")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--stimulus", required=True)
    parser.add_argument("--output", default=".cert/e9_environment_certification.json")
    return parser.parse_args()


def main() -> int:
    args = _args()
    video = pathlib.Path(args.video).resolve()
    det_config = pathlib.Path(args.det_config).resolve()
    det_checkpoint = pathlib.Path(args.det_checkpoint).resolve()
    mask_config = pathlib.Path(args.mask_config).resolve()
    mask_checkpoint = pathlib.Path(args.mask_checkpoint).resolve()
    stimulus_path = pathlib.Path(args.stimulus).resolve()
    output = pathlib.Path(args.output).resolve()
    for path in (video, det_config, det_checkpoint, mask_config, mask_checkpoint, stimulus_path):
        if not path.is_file():
            raise SystemExit(f"missing E9 certification input: {path}")

    detector_sha = _sha256(det_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    if detector_sha != E2_SHA256 or mask_sha != E4_MASK_SHA256:
        raise SystemExit("E9 certified checkpoint SHA256 mismatch")
    stimulus = json.loads(stimulus_path.read_text(encoding="utf-8"))
    gains = np.asarray(stimulus.get("background_gain"), dtype=np.float64)
    if gains.ndim != 1 or len(gains) < 4 or not np.all(np.diff(gains) > 0.0):
        raise SystemExit("E9 stimulus gain sequence must be strictly increasing")

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
    blueprint, sidecars = run_e9_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )
    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    valid, errors = BlueprintValidator(schema).validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E9 output: " + " | ".join(errors))
    if blueprint["processing"]["pipeline_version"] != "0.9.0-e9.1":
        raise SystemExit("E9.1 pipeline version was not emitted")
    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    if stages.get("person_mask", {}).get("status") != "succeeded":
        raise SystemExit("E4 person-mask stage did not succeed before E9")
    if stages.get("environment", {}).get("status") != "succeeded":
        raise SystemExit("E9 environment stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e9_environment")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E9 extension did not enable")
    expected_boundaries = {
        "depth_emitted": False,
        "occluder_tracks_emitted": False,
        "semantic_scene_inference_performed": False,
        "weather_inference_performed": False,
        "material_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
    }
    for key, expected in expected_boundaries.items():
        if extension.get(key) != expected:
            raise SystemExit(f"E9 evidence/privacy boundary drifted: {key}")
    environment = blueprint.get("environment")
    if not isinstance(environment, dict) or environment.get("depth_ref") is not None or environment.get("occluder_tracks") != []:
        raise SystemExit("E9 canonical environment overclaimed depth or occluders")

    metric_ref = environment.get("luminance_ref")
    if not isinstance(metric_ref, dict):
        raise SystemExit("E9 luminance ref missing")
    metric_path_value = sidecars.get(metric_ref["uri"])
    if not isinstance(metric_path_value, (str, os.PathLike)):
        raise SystemExit("E9 physical metric NPZ missing")
    metric_path = pathlib.Path(metric_path_value)
    if _sha256(metric_path) != metric_ref.get("checksum_sha256"):
        raise SystemExit("E9 metric NPZ checksum mismatch")
    with np.load(metric_path, allow_pickle=False) as arrays:
        luminance = np.asarray(arrays["luminance"], dtype=np.float64)
        exposure = np.asarray(arrays["exposure_change_ev_proxy"], dtype=np.float64)
        chroma = np.asarray(arrays["white_balance_chromaticity_rb"], dtype=np.float64)
        sharpness = np.asarray(arrays["blur_laplacian_variance"], dtype=np.float64)
        valid_frame = np.asarray(arrays["valid_frame"]).astype(bool)
        exposure_valid = np.asarray(arrays["exposure_valid"]).astype(bool)
        background_fraction = np.asarray(arrays["background_fraction"], dtype=np.float64)
    if len(luminance) != len(gains):
        raise SystemExit("E9 metric timeline does not match certification stimulus")
    valid_indices = np.flatnonzero(valid_frame)
    if len(valid_indices) < max(4, math.ceil(0.75 * len(gains))):
        raise SystemExit("E9 real-runtime background photometry coverage is below 75%")
    if not np.all(np.isfinite(chroma[valid_indices])) or not np.all(np.isfinite(sharpness[valid_indices])):
        raise SystemExit("E9 valid frames contain non-finite chromaticity/sharpness evidence")
    if not np.all((background_fraction[valid_indices] > 0.10) & (background_fraction[valid_indices] < 0.995)):
        raise SystemExit("E9 background mask did not isolate a plausible non-human background region")
    gain_valid = gains[valid_indices]
    luma_valid = luminance[valid_indices]
    correlation = float(np.corrcoef(gain_valid, luma_valid)[0, 1])
    if not math.isfinite(correlation) or correlation < 0.90:
        raise SystemExit(f"E9 background luminance failed known gain tracking: correlation={correlation}")
    exposure_values = exposure[exposure_valid]
    if len(exposure_values) < 2 or float(np.median(exposure_values)) <= 0.0:
        raise SystemExit("E9 exposure-change proxy did not preserve positive brightness-ramp direction")

    mask_ref = environment.get("background_mask_ref")
    if not isinstance(mask_ref, dict):
        raise SystemExit("E9 background mask ref missing")
    mask_path_value = sidecars.get(mask_ref["uri"])
    if not isinstance(mask_path_value, (str, os.PathLike)):
        raise SystemExit("E9 physical background mask missing")
    mask_path = pathlib.Path(mask_path_value)
    if _sha256(mask_path) != mask_ref.get("checksum_sha256"):
        raise SystemExit("E9 background mask checksum mismatch")
    mask_payload = json.loads(mask_path.read_text(encoding="utf-8"))
    decoded_background_pixels: list[int] = []
    for frame_idx in valid_indices:
        encoded = mask_payload["frames"][int(frame_idx)]
        if encoded is None:
            raise SystemExit("E9 valid frame unexpectedly has null physical background mask")
        mask = decode_binary_rle(encoded)
        decoded_background_pixels.append(int(np.count_nonzero(mask)))
    if not decoded_background_pixels:
        raise SystemExit("E9 did not emit any decodable background mask")

    provenance = next((tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "environment"), None)
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("E9 provenance must exist and introduce no model weights")

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "mask_checkpoint_sha256": mask_sha,
        "environment_algorithm": extension["algorithm"],
        "frame_count": len(gains),
        "valid_frame_count": int(np.count_nonzero(valid_frame)),
        "coverage": extension["coverage"],
        "background_luminance_gain_correlation": round(correlation, 6),
        "median_positive_exposure_change_ev_proxy": round(float(np.median(exposure_values)), 6),
        "background_fraction_p50": round(float(np.median(background_fraction[valid_indices])), 6),
        "chroma_r_p50": round(float(np.median(chroma[valid_indices, 0])), 6),
        "chroma_b_p50": round(float(np.median(chroma[valid_indices, 1])), 6),
        "sharpness_proxy_p50": round(float(np.median(sharpness[valid_indices])), 8),
        **expected_boundaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
