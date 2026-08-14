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

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.e4_media_pipeline import run_e4_media_pipeline
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
    parser = argparse.ArgumentParser(description="Certify E4.3 global dense optical-flow evidence")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e4_3_dense_flow_certification.json")
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
    if detector_sha != E2_SHA256:
        raise SystemExit(f"E2 detector SHA256 mismatch: {detector_sha}")
    if mask_sha != E4_MASK_SHA256:
        raise SystemExit(f"E4.1 mask SHA256 mismatch: {mask_sha}")

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
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e4_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )

    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    valid, validation_errors = BlueprintValidator(schema).validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E4.3 output: " + " | ".join(validation_errors))
    if blueprint["processing"]["pipeline_version"] != "0.4.0-e4.3":
        raise SystemExit("E4.3 pipeline version was not emitted")

    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    for stage_name in ("person_mask", "point_tracks", "dense_flow"):
        if stages.get(stage_name, {}).get("status") != "succeeded":
            raise SystemExit(f"{stage_name} stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e4_dense_flow")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E4.3 extension did not enable")
    expected = {
        "algorithm": "opencv_farneback_v1",
        "coordinate_space": "pixel_xy",
        "vector_unit": "px",
        "shot_boundary_reset": True,
        "interpolation": False,
    }
    for key, value in expected.items():
        if extension.get(key) != value:
            raise SystemExit(f"E4.3 extension semantic mismatch: {key}")

    ref = extension.get("flow_ref")
    if not isinstance(ref, dict):
        raise SystemExit("E4.3 flow_ref missing")
    frame_count = int(blueprint["timebase"]["frame_count"])
    shape = ref.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or shape[0] != frame_count or shape[-1] != 2:
        raise SystemExit("dense flow manifest shape mismatch")
    if ref.get("coordinate_space") != "pixel_xy" or ref.get("unit") != "px":
        raise SystemExit("dense flow coordinate semantics mismatch")
    if ref.get("interpolation_policy") != "none":
        raise SystemExit("dense flow interpolation must remain disabled")

    path_value = sidecars.get(ref["uri"])
    if not isinstance(path_value, (str, os.PathLike)):
        raise SystemExit(f"dense-flow URI did not resolve: {ref['uri']}")
    path = pathlib.Path(path_value)
    if not path.is_file() or _sha256(path) != ref["checksum_sha256"]:
        raise SystemExit("dense-flow sidecar checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        flow = arrays["flow_xy"]
        valid_frame = arrays["valid_frame"]
    if flow.shape != tuple(shape) or flow.dtype != np.float32:
        raise SystemExit("dense-flow physical array shape/dtype mismatch")
    if valid_frame.shape != (frame_count,) or valid_frame.dtype != np.bool_:
        raise SystemExit("dense-flow valid_frame shape/dtype mismatch")
    if bool(valid_frame[0]) or not np.all(np.isnan(flow[0])):
        raise SystemExit("frame 0 must be explicit missing dense flow")
    if not np.all(np.isnan(flow[~valid_frame])):
        raise SystemExit("invalid dense-flow frames must remain NaN")
    if not np.all(np.isfinite(flow[valid_frame])):
        raise SystemExit("valid dense-flow frames must be finite")

    shot_starts = {
        int(shot["frame_start"])
        for shot in blueprint.get("shots", [])
        if isinstance(shot, dict)
        and isinstance(shot.get("frame_start"), int)
        and 0 < int(shot["frame_start"]) < frame_count
    }
    for frame_idx in shot_starts:
        if bool(valid_frame[frame_idx]) or not np.all(np.isnan(flow[frame_idx])):
            raise SystemExit(f"hard-shot frame {frame_idx} crossed dense-flow continuity")

    valid_vectors = flow[valid_frame].reshape(-1, 2)
    if valid_vectors.size == 0:
        raise SystemExit("real-media E4.3 emitted no dense-flow vectors")
    magnitudes = np.linalg.norm(valid_vectors, axis=1)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    if finite_magnitudes.size == 0:
        raise SystemExit("real-media E4.3 emitted no finite motion magnitudes")
    median_motion = float(np.median(finite_magnitudes))
    p95_motion = float(np.percentile(finite_magnitudes, 95))
    moving_fraction = float(np.mean(finite_magnitudes > 0.20))
    if p95_motion <= 0.30 or moving_fraction <= 0.02:
        raise SystemExit(
            f"dense motion signal is too weak: median={median_motion:.6f}, "
            f"p95={p95_motion:.6f}, moving_fraction={moving_fraction:.6f}"
        )
    if p95_motion >= 10.0:
        raise SystemExit(f"dense motion signal is implausibly large: p95={p95_motion:.6f}")

    provenance = next(
        (tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "dense_flow"),
        None,
    )
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("dense-flow provenance must exist and carry no model weights")

    for character in blueprint.get("characters", []):
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False:
            raise SystemExit("identity inference privacy invariant failed")
        if privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("biometric export privacy invariant failed")

    quality = extension.get("quality", {})
    expected_pairs = max(0, frame_count - 1 - len(shot_starts))
    if quality.get("valid_pairs") != int(np.count_nonzero(valid_frame)):
        raise SystemExit("dense-flow valid-pair report mismatch")
    if quality.get("expected_pairs") != expected_pairs:
        raise SystemExit("dense-flow expected-pair report mismatch")
    coverage = float(quality.get("coverage", -1.0))
    if not math.isclose(coverage, 1.0, abs_tol=1e-6):
        raise SystemExit(f"dense-flow coverage was not complete: {coverage}")

    result = {
        "status": "passed",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "mask_checkpoint_sha256": mask_sha,
        "dense_flow_algorithm": extension["algorithm"],
        "frame_count": frame_count,
        "grid_height": shape[1],
        "grid_width": shape[2],
        "valid_flow_pairs": int(np.count_nonzero(valid_frame)),
        "dense_vector_samples": int(valid_vectors.shape[0]),
        "median_motion_px": round(median_motion, 6),
        "p95_motion_px": round(p95_motion, 6),
        "moving_fraction_gt_0_20px": round(moving_fraction, 6),
        "coverage": coverage,
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
