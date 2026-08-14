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
from packages.pipeline_core.e8_media_pipeline import run_e8_media_pipeline
from packages.pipeline_core.media_probe import compute_sha256

E2_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
E3_SHA256 = "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"
E4_MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"
TARGET_FREQUENCY_HZ = 1.0


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certify E8.1 geometry-only micro-motion analysis")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--mask-config", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--output", default=".cert/e8_micro_motion_certification.json")
    return parser.parse_args()


def _load(ref: dict, sidecars: dict, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    path_value = sidecars.get(ref["uri"])
    if not isinstance(path_value, (str, os.PathLike)):
        raise SystemExit(f"physical sidecar unavailable: {ref['uri']}")
    path = pathlib.Path(path_value)
    if not path.is_file() or _sha256(path) != ref["checksum_sha256"]:
        raise SystemExit(f"sidecar checksum mismatch: {ref['uri']}")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in keys:
            if key not in arrays:
                raise SystemExit(f"array {key!r} missing: {ref['uri']}")
            result[key] = np.asarray(arrays[key])
    return result


def _write_result(output: pathlib.Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _longest_valid_run(valid: np.ndarray) -> np.ndarray:
    best = np.asarray([], dtype=np.int64)
    start: int | None = None
    for idx, is_valid in enumerate(np.asarray(valid, dtype=bool)):
        if is_valid:
            if start is None:
                start = idx
        elif start is not None:
            candidate = np.arange(start, idx, dtype=np.int64)
            if len(candidate) > len(best):
                best = candidate
            start = None
    if start is not None:
        candidate = np.arange(start, len(valid), dtype=np.int64)
        if len(candidate) > len(best):
            best = candidate
    return best


def _spectral_diagnostic(values: np.ndarray, valid: np.ndarray, *, fps: float) -> dict[str, float | int | None]:
    indices = _longest_valid_run(valid)
    if len(indices) < 12:
        return {
            "window_frames": len(indices),
            "dominant_frequency_hz": None,
            "periodicity_score": 0.0,
            "target_1hz_power_fraction": 0.0,
        }
    signal = np.asarray(values[indices], dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    if len(signal) < 12:
        return {
            "window_frames": len(signal),
            "dominant_frequency_hz": None,
            "periodicity_score": 0.0,
            "target_1hz_power_fraction": 0.0,
        }
    signal -= float(np.mean(signal))
    if float(np.sqrt(np.mean(signal * signal))) < 1e-12:
        return {
            "window_frames": len(signal),
            "dominant_frequency_hz": None,
            "periodicity_score": 0.0,
            "target_1hz_power_fraction": 0.0,
        }
    spectrum = np.fft.rfft(signal * np.hanning(len(signal)))
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    power = np.abs(spectrum) ** 2
    band = (frequencies >= 0.2) & (frequencies <= min(3.0, fps * 0.45)) & (frequencies > 0.0)
    band_indices = np.flatnonzero(band)
    if not len(band_indices):
        return {
            "window_frames": len(signal),
            "dominant_frequency_hz": None,
            "periodicity_score": 0.0,
            "target_1hz_power_fraction": 0.0,
        }
    band_power = float(np.sum(power[band_indices]))
    if not math.isfinite(band_power) or band_power <= 1e-18:
        return {
            "window_frames": len(signal),
            "dominant_frequency_hz": None,
            "periodicity_score": 0.0,
            "target_1hz_power_fraction": 0.0,
        }
    peak_index = int(band_indices[int(np.argmax(power[band_indices]))])
    target_index = int(band_indices[int(np.argmin(np.abs(frequencies[band_indices] - TARGET_FREQUENCY_HZ)))])
    return {
        "window_frames": len(signal),
        "dominant_frequency_hz": round(float(frequencies[peak_index]), 6),
        "periodicity_score": round(float(power[peak_index] / band_power), 6),
        "target_1hz_power_fraction": round(float(power[target_index] / band_power), 6),
    }


def _signed_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(valid)) < 4:
        return None
    x = np.asarray(left[valid], dtype=np.float64)
    y = np.asarray(right[valid], dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return round(value, 6) if math.isfinite(value) else None


def _pose_component_diagnostics(character: dict, sidecars: dict, velocity: np.ndarray) -> dict[str, float | None]:
    body_ref = character.get("surface_motion", {}).get("body_frame_transform_ref")
    if not isinstance(body_ref, dict):
        return {}
    arrays = _load(
        body_ref,
        sidecars,
        ("source_pixel_to_body_local", "body_origin_stabilized_xy", "torso_scale_px", "valid_frame"),
    )
    transforms = arrays["source_pixel_to_body_local"].astype(np.float64, copy=False)
    origins = arrays["body_origin_stabilized_xy"].astype(np.float64, copy=False)
    scales = arrays["torso_scale_px"].astype(np.float64, copy=False)
    body_valid = arrays["valid_frame"].astype(bool, copy=False)
    frame_count = len(velocity)
    origin_dx = np.full(frame_count, np.nan, dtype=np.float64)
    origin_dy = np.full(frame_count, np.nan, dtype=np.float64)
    log_scale = np.full(frame_count, np.nan, dtype=np.float64)
    angle_delta = np.full(frame_count, np.nan, dtype=np.float64)
    for frame_idx in range(1, frame_count):
        if not body_valid[frame_idx - 1] or not body_valid[frame_idx]:
            continue
        previous_scale = float(scales[frame_idx - 1])
        current_scale = float(scales[frame_idx])
        if previous_scale <= 1e-12 or current_scale <= 1e-12:
            continue
        mean_scale = 0.5 * (previous_scale + current_scale)
        origin_dx[frame_idx] = float(origins[frame_idx, 0] - origins[frame_idx - 1, 0]) / mean_scale
        origin_dy[frame_idx] = float(origins[frame_idx, 1] - origins[frame_idx - 1, 1]) / mean_scale
        log_scale[frame_idx] = math.log(current_scale / previous_scale)
        previous_angle = math.atan2(float(transforms[frame_idx - 1, 0, 1]), float(transforms[frame_idx - 1, 0, 0]))
        current_angle = math.atan2(float(transforms[frame_idx, 0, 1]), float(transforms[frame_idx, 0, 0]))
        angle_delta[frame_idx] = (current_angle - previous_angle + math.pi) % (2.0 * math.pi) - math.pi
    return {
        "velocity_vs_origin_dx_corr": _signed_corr(velocity, origin_dx),
        "velocity_vs_origin_dy_corr": _signed_corr(velocity, origin_dy),
        "velocity_vs_log_scale_corr": _signed_corr(velocity, log_scale),
        "velocity_vs_angle_delta_corr": _signed_corr(velocity, angle_delta),
    }


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
    for path in (video, det_config, det_checkpoint, pose_config, pose_checkpoint, mask_config, mask_checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing certification input: {path}")

    detector_sha = _sha256(det_checkpoint)
    pose_sha = _sha256(pose_checkpoint)
    mask_sha = _sha256(mask_checkpoint)
    if detector_sha != E2_SHA256 or pose_sha != E3_SHA256 or mask_sha != E4_MASK_SHA256:
        raise SystemExit("certified checkpoint SHA256 mismatch")

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
            "VBS_RTMDET_INS_WEIGHTS_LICENSE": "internal CI certification; redistribution not approved",
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
        }
    )

    job_id = str(uuid.uuid4())
    blueprint, sidecars = run_e8_media_pipeline(
        job_id=job_id,
        video_file_name=video.name,
        video_sha256=compute_sha256(str(video)),
        video_path=str(video),
    )
    schema = json.loads((REPO_ROOT / "contracts" / "video_blueprint.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(blueprint)
    valid, errors = BlueprintValidator(schema).validate(blueprint)
    if not valid:
        raise SystemExit("BlueprintValidator rejected E8 output: " + " | ".join(errors))
    if blueprint["processing"]["pipeline_version"] != "0.8.0-e8.1":
        raise SystemExit("E8.1 pipeline version was not emitted")
    stages = {stage["name"]: stage for stage in blueprint["processing"]["stages"]}
    for stage_name in ("point_tracks", "camera_motion", "body_local_frame", "surface_motion", "micro_motion"):
        if stages.get(stage_name, {}).get("status") != "succeeded":
            raise SystemExit(f"{stage_name} stage did not succeed")

    extension = blueprint.get("extensions", {}).get("e8_micro_motion")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise SystemExit("E8 extension did not enable")
    expected_privacy = {
        "interpretation_scope": "geometry_only",
        "physiological_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "radial_expansion_emitted": False,
        "area_change_emitted": False,
    }
    for key, expected in expected_privacy.items():
        if extension.get(key) != expected:
            raise SystemExit(f"E8 evidence/privacy boundary drifted: {key}")

    periodic_rows: list[dict] = []
    region_rows: list[dict] = []
    analyzed_regions = 0
    physical_signal_rows = 0
    acceleration_valid_rows = 0
    for character in blueprint.get("characters", []):
        privacy = character.get("privacy", {})
        if privacy.get("identity_inference_performed") is not False:
            raise SystemExit("identity inference privacy invariant failed")
        if privacy.get("biometric_embedding_exported") is not False:
            raise SystemExit("biometric embedding privacy invariant failed")
        regions = character.get("surface_motion", {}).get("regions", [])
        for region in regions:
            micro = region.get("micro_motion", {})
            if not isinstance(micro, dict):
                raise SystemExit("E8 region lacks MicroMotion object")
            analyzed_regions += 1
            limitations = micro.get("limitations", [])
            if not any(isinstance(item, str) and "no breathing" in item and "physiological" in item for item in limitations):
                raise SystemExit("E8 output lost explicit physiological inference prohibition")
            if micro.get("radial_expansion_ratio_ref") is not None or micro.get("area_change_ratio_ref") is not None:
                raise SystemExit("E8.1 unexpectedly emitted radial/area deformation modes")
            signal_ref = micro.get("signal_ref")
            acceleration_ref = micro.get("acceleration_ref")
            spectral_velocity: dict[str, float | int | None] = {}
            pose_components: dict[str, float | None] = {}
            if isinstance(signal_ref, dict):
                arrays = _load(
                    signal_ref,
                    sidecars,
                    (
                        "signal",
                        "signal_valid",
                        "detrended_signal",
                        "vertical_displacement",
                        "vertical_valid",
                        "velocity",
                        "velocity_valid",
                        "acceleration",
                        "acceleration_valid",
                    ),
                )
                physical_signal_rows += 1
                acceleration = arrays["acceleration"]
                acceleration_valid = arrays["acceleration_valid"].astype(bool)
                if np.any(acceleration_valid & ~np.isfinite(acceleration)):
                    raise SystemExit("E8 acceleration_valid marks NaN acceleration as valid")
                if np.any(np.isfinite(acceleration) & ~acceleration_valid):
                    raise SystemExit("E8 finite acceleration lacks explicit validity evidence")
                acceleration_valid_rows += 1
                if not isinstance(acceleration_ref, dict) or acceleration_ref.get("metadata", {}).get("valid_array_key") != "acceleration_valid":
                    raise SystemExit("E8 acceleration ref does not own its validity mask")
                spectral_velocity = _spectral_diagnostic(
                    arrays["velocity"],
                    arrays["velocity_valid"].astype(bool),
                    fps=float(blueprint["timebase"]["fps_num"]) / float(blueprint["timebase"]["fps_den"]),
                )
                pose_components = _pose_component_diagnostics(character, sidecars, arrays["velocity"])

            row = {
                "character_id": character.get("character_id"),
                "kind": micro.get("kind"),
                "dominant_frequency_hz": micro.get("dominant_frequency_hz"),
                "amplitude_norm_p50": micro.get("amplitude_norm_p50"),
                "amplitude_px_p50": micro.get("amplitude_px_p50"),
                "periodicity_score": micro.get("periodicity_score"),
                "spatial_coherence": micro.get("spatial_coherence"),
                "camera_leakage_score": micro.get("camera_leakage_score"),
                "pose_leakage_score": micro.get("pose_leakage_score"),
                "observation_dropout_ratio": micro.get("occlusion_ratio"),
                "confidence": micro.get("confidence"),
                "usable_for_generation": micro.get("usable_for_generation"),
                "velocity_spectrum_diagnostic": spectral_velocity,
                "pose_component_diagnostics": pose_components,
                "limitations": limitations,
            }
            region_rows.append(row)
            if micro.get("kind") == "periodic_micro_motion":
                periodic_rows.append(row)

    diagnostics = {
        "status": "diagnostic",
        "job_id": job_id,
        "pipeline_version": blueprint["processing"]["pipeline_version"],
        "detector_sha256": detector_sha,
        "pose_sha256": pose_sha,
        "mask_checkpoint_sha256": mask_sha,
        "micro_motion_algorithm": extension["algorithm"],
        "target_frequency_hz": TARGET_FREQUENCY_HZ,
        "thresholds": extension.get("thresholds"),
        "analyzed_regions": analyzed_regions,
        "physical_signal_rows": physical_signal_rows,
        "region_rows": region_rows,
        "periodic_rows": periodic_rows,
        "physiological_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "radial_expansion_emitted": False,
        "area_change_emitted": False,
    }
    _write_result(output, diagnostics)

    if analyzed_regions <= 0 or physical_signal_rows <= 0 or acceleration_valid_rows != physical_signal_rows:
        raise SystemExit("E8 certification produced incomplete physical MicroMotion evidence")
    usable_periodic = [row for row in periodic_rows if row["usable_for_generation"] is True]
    if not usable_periodic:
        raise SystemExit("E8 real-runtime certification found no usable periodic geometry")
    best = min(
        usable_periodic,
        key=lambda row: abs(float(row["dominant_frequency_hz"]) - TARGET_FREQUENCY_HZ),
    )
    frequency_error = abs(float(best["dominant_frequency_hz"]) - TARGET_FREQUENCY_HZ)
    if frequency_error > 0.35:
        raise SystemExit(f"E8 dominant frequency missed injected 1 Hz geometry: error={frequency_error:.6f}")
    if float(best["camera_leakage_score"]) > float(extension["thresholds"]["max_camera_leakage"]):
        raise SystemExit("E8 usable periodic row exceeded camera leakage gate")
    if float(best["pose_leakage_score"]) > float(extension["thresholds"]["max_pose_leakage"]):
        raise SystemExit("E8 usable periodic row exceeded pose leakage gate")

    provenance = next((tool for tool in blueprint["provenance"]["tools"] if tool.get("module") == "micro_motion"), None)
    if not provenance or provenance.get("weights_sha256") is not None:
        raise SystemExit("E8 provenance must exist and introduce no model weights")

    result = {
        **diagnostics,
        "status": "passed",
        "selected_periodic_row": best,
        "frequency_error_hz": round(frequency_error, 6),
    }
    _write_result(output, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
