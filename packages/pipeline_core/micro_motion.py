from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file


class MicroMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MicroMotionConfig:
    min_valid_points_per_frame: int = 4
    min_periodic_frames: int = 12
    min_frequency_hz: float = 0.20
    max_frequency_hz: float = 3.00
    min_cycles: float = 2.0
    min_amplitude_norm: float = 0.001
    periodicity_threshold: float = 0.50
    coherence_threshold: float = 0.35
    max_camera_leakage: float = 0.60
    max_pose_leakage: float = 0.60
    max_observation_dropout: float = 0.50
    usable_confidence_threshold: float = 0.40

    def validate(self) -> None:
        if self.min_valid_points_per_frame < 1:
            raise MicroMotionError("min_valid_points_per_frame must be positive")
        if self.min_periodic_frames < 4:
            raise MicroMotionError("min_periodic_frames must be at least 4")
        if not 0.0 < self.min_frequency_hz < self.max_frequency_hz:
            raise MicroMotionError("frequency bounds are invalid")
        if self.min_cycles <= 0.0 or self.min_amplitude_norm < 0.0:
            raise MicroMotionError("min_cycles must be positive and min_amplitude_norm non-negative")
        for name in (
            "periodicity_threshold",
            "coherence_threshold",
            "max_camera_leakage",
            "max_pose_leakage",
            "max_observation_dropout",
            "usable_confidence_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise MicroMotionError(f"{name} must be within [0, 1]")

    def token(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_environment(cls) -> MicroMotionConfig | None:
        raw = os.environ.get("VBS_E8_MICRO_MOTION_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise MicroMotionError("VBS_E8_MICRO_MOTION_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, indent=2, sort_keys=True).encode()).hexdigest()


def _load_npz(
    ref: dict[str, Any],
    sidecars: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    uri = ref.get("uri")
    if not isinstance(uri, str):
        raise MicroMotionError("E8 TimeSeriesRef URI is missing")
    path_value = sidecars.get(uri)
    if not isinstance(path_value, (str, os.PathLike)) or not os.path.isfile(str(path_value)):
        raise MicroMotionError(f"E8 physical sidecar unavailable: {uri}")
    if ref.get("checksum_sha256") != sha256_file(str(path_value)):
        raise MicroMotionError(f"E8 input checksum mismatch: {uri}")
    result: dict[str, np.ndarray] = {}
    with np.load(str(path_value), allow_pickle=False) as arrays:
        for key in keys:
            if key not in arrays:
                raise MicroMotionError(f"E8 input array {key!r} missing from {uri}")
            result[key] = np.asarray(arrays[key])
    return result


def _shot_ranges(blueprint: dict[str, Any], frame_count: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for shot in blueprint.get("shots", []):
        if not isinstance(shot, dict):
            continue
        frame_start = shot.get("frame_start")
        frame_end = shot.get("frame_end")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise MicroMotionError("E8 shot frame bounds must be integers")
        if frame_start < 0 or frame_end < frame_start or frame_end >= frame_count:
            raise MicroMotionError("E8 shot frame bounds are invalid")
        ranges.append((frame_start, frame_end))
    if not ranges:
        raise MicroMotionError("E8 requires at least one shot")
    return ranges


def estimate_principal_axis(residual: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if values.ndim != 3 or values.shape[-1] != 2 or mask.shape != values.shape[:2]:
        raise MicroMotionError("E8 residual arrays have invalid shape")
    samples = values[mask]
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 2:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm < 1e-12:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    axis = axis / norm
    dominant_index = int(np.argmax(np.abs(axis)))
    if axis[dominant_index] < 0:
        axis *= -1.0
    return axis.astype(np.float32)


def _winsorized_mean(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or not len(vectors):
        return np.asarray([math.nan, math.nan], dtype=np.float64)
    magnitudes = np.linalg.norm(vectors, axis=1)
    finite = np.isfinite(magnitudes) & np.all(np.isfinite(vectors), axis=1)
    vectors = vectors[finite]
    magnitudes = magnitudes[finite]
    if not len(vectors):
        return np.asarray([math.nan, math.nan], dtype=np.float64)
    clip = float(np.percentile(magnitudes, 95)) if len(magnitudes) > 1 else float(magnitudes[0])
    if math.isfinite(clip) and clip > 0.0:
        scales = np.minimum(1.0, clip / np.maximum(magnitudes, 1e-12))
        vectors = vectors * scales[:, None]
    return np.mean(vectors, axis=0)


def aggregate_residual_velocity(
    residual: np.ndarray,
    valid: np.ndarray,
    axis: np.ndarray,
    *,
    min_valid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = residual.shape[0]
    velocity = np.full(frame_count, np.nan, dtype=np.float32)
    vertical = np.full(frame_count, np.nan, dtype=np.float32)
    coherence = np.full(frame_count, np.nan, dtype=np.float32)
    signal_valid = np.zeros(frame_count, dtype=np.bool_)
    axis64 = np.asarray(axis, dtype=np.float64)
    for frame_idx in range(frame_count):
        slots = np.flatnonzero(valid[frame_idx])
        if len(slots) < min_valid_points:
            continue
        values = np.asarray(residual[frame_idx, slots], dtype=np.float64)
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values) < min_valid_points:
            continue
        mean_vector = _winsorized_mean(values)
        if not np.all(np.isfinite(mean_vector)):
            continue
        velocity[frame_idx] = float(np.dot(mean_vector, axis64))
        vertical[frame_idx] = float(mean_vector[1])
        mean_magnitude = float(np.mean(np.linalg.norm(values, axis=1)))
        ratio = float(np.linalg.norm(mean_vector) / mean_magnitude) if mean_magnitude > 1e-12 else 1.0
        coherence[frame_idx] = max(0.0, min(1.0, ratio))
        signal_valid[frame_idx] = True
    return velocity, vertical, coherence, signal_valid


def _integrate(
    velocity: np.ndarray,
    valid: np.ndarray,
    shot_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full_like(velocity, np.nan, dtype=np.float32)
    result_valid = np.zeros_like(valid, dtype=np.bool_)
    for frame_start, frame_end in shot_ranges:
        result[frame_start] = 0.0
        result_valid[frame_start] = True
        cumulative = 0.0
        previous_valid = True
        for frame_idx in range(frame_start + 1, frame_end + 1):
            if not valid[frame_idx] or not math.isfinite(float(velocity[frame_idx])):
                previous_valid = False
                continue
            if previous_valid:
                cumulative += float(velocity[frame_idx])
            else:
                cumulative = float(velocity[frame_idx])
            result[frame_idx] = cumulative
            result_valid[frame_idx] = True
            previous_valid = True
    return result, result_valid


def _finite_runs(valid: np.ndarray, frame_start: int, frame_end: int) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for frame_idx in range(frame_start, frame_end + 1):
        if bool(valid[frame_idx]):
            if start is None:
                start = frame_idx
        elif start is not None:
            runs.append(np.arange(start, frame_idx, dtype=np.int64))
            start = None
    if start is not None:
        runs.append(np.arange(start, frame_end + 1, dtype=np.int64))
    return runs


def _linear_detrend(
    signal: np.ndarray,
    valid: np.ndarray,
    shot_ranges: list[tuple[int, int]],
) -> np.ndarray:
    result = np.full_like(signal, np.nan, dtype=np.float32)
    for frame_start, frame_end in shot_ranges:
        for indices in _finite_runs(valid, frame_start, frame_end):
            values = signal[indices].astype(np.float64)
            if len(indices) == 1:
                result[indices[0]] = 0.0
                continue
            x = np.arange(len(indices), dtype=np.float64)
            design = np.column_stack((x, np.ones(len(indices), dtype=np.float64)))
            slope, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
            trend = slope * x + intercept
            result[indices] = (values - trend).astype(np.float32)
    return result


def _acceleration(
    velocity: np.ndarray,
    valid: np.ndarray,
    shot_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full_like(velocity, np.nan, dtype=np.float32)
    result_valid = np.zeros_like(valid, dtype=np.bool_)
    for frame_start, frame_end in shot_ranges:
        for frame_idx in range(frame_start + 1, frame_end + 1):
            if valid[frame_idx] and valid[frame_idx - 1]:
                result[frame_idx] = float(velocity[frame_idx] - velocity[frame_idx - 1])
                result_valid[frame_idx] = True
    return result, result_valid


def analyze_frequency(
    signal: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    shot_ranges: list[tuple[int, int]],
    config: MicroMotionConfig,
) -> tuple[float | None, float | None, float, float, int]:
    best_indices: np.ndarray | None = None
    for frame_start, frame_end in shot_ranges:
        for indices in _finite_runs(valid, frame_start, frame_end):
            if best_indices is None or len(indices) > len(best_indices):
                best_indices = indices
    if best_indices is None or len(best_indices) < config.min_periodic_frames:
        return None, None, 0.0, 0.0, 0
    values = signal[best_indices].astype(np.float64)
    values -= float(np.mean(values))
    if float(np.sqrt(np.mean(values * values))) < 1e-12:
        return None, None, 0.0, 0.0, len(best_indices)
    window = np.hanning(len(values))
    spectrum = np.fft.rfft(values * window)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / fps)
    power = np.abs(spectrum) ** 2
    max_frequency = min(config.max_frequency_hz, fps * 0.45)
    band = (frequencies >= config.min_frequency_hz) & (frequencies <= max_frequency) & (frequencies > 0.0)
    indices = np.flatnonzero(band)
    if not len(indices):
        return None, None, 0.0, 0.0, len(best_indices)
    peak_index = int(indices[int(np.argmax(power[indices]))])
    band_power = float(np.sum(power[indices]))
    if not math.isfinite(band_power) or band_power <= 1e-18:
        return None, None, 0.0, 0.0, len(best_indices)
    dominant_frequency = float(frequencies[peak_index])
    phase = float(np.angle(spectrum[peak_index]))
    periodicity = max(0.0, min(1.0, float(power[peak_index] / band_power)))
    duration_s = max(0.0, (len(best_indices) - 1) / fps)
    cycles = dominant_frequency * duration_s
    return dominant_frequency, phase, periodicity, cycles, len(best_indices)


def _correlation_leakage(signal: np.ndarray, driver: np.ndarray) -> tuple[float, bool]:
    valid = np.isfinite(signal) & np.isfinite(driver)
    if int(np.count_nonzero(valid)) < 4:
        return 1.0, False
    left = np.abs(signal[valid].astype(np.float64))
    right = np.abs(driver[valid].astype(np.float64))
    if float(np.std(right)) < 1e-12 or float(np.std(left)) < 1e-12:
        return 0.0, True
    correlation = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(correlation):
        return 1.0, False
    return max(0.0, min(1.0, abs(correlation))), True


def _camera_energy(
    blueprint: dict[str, Any],
    sidecars: dict[str, Any],
    frame_count: int,
) -> tuple[np.ndarray, bool]:
    camera = blueprint.get("camera")
    rows = camera.get("per_shot") if isinstance(camera, dict) else None
    if not isinstance(rows, list):
        return np.full(frame_count, np.nan, dtype=np.float32), False
    width = float(blueprint.get("source_video", {}).get("width", 0) or 0)
    height = float(blueprint.get("source_video", {}).get("height", 0) or 0)
    diagonal = math.hypot(width, height)
    if diagonal <= 0.0:
        return np.full(frame_count, np.nan, dtype=np.float32), False
    energy = np.full(frame_count, np.nan, dtype=np.float32)
    for row in rows:
        if not isinstance(row, dict):
            continue
        ref = row.get("affine_ref")
        if not isinstance(ref, dict):
            continue
        metadata = ref.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("array_key") != "frame_to_frame_affine":
            continue
        arrays = _load_npz(ref, sidecars, ("frame_to_frame_affine",))
        affine = arrays["frame_to_frame_affine"].astype(np.float64, copy=False)
        frame_start = ref.get("frame_start")
        frame_end = ref.get("frame_end")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise MicroMotionError("E8 camera ref frame bounds are invalid")
        if affine.shape != (frame_end - frame_start + 1, 2, 3):
            raise MicroMotionError("E8 camera affine physical shape mismatch")
        energy[frame_start] = 0.0
        for local_idx in range(1, len(affine)):
            matrix = affine[local_idx]
            if not np.all(np.isfinite(matrix)):
                continue
            translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2])) / diagonal
            scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
            rotation = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
            scale_delta = abs(math.log(scale)) if scale > 1e-12 else 1.0
            energy[frame_start + local_idx] = math.sqrt(translation * translation + rotation * rotation + scale_delta * scale_delta)
    return energy, bool(np.count_nonzero(np.isfinite(energy)))


def _pose_energy(body_arrays: dict[str, np.ndarray], frame_count: int) -> np.ndarray:
    transforms = body_arrays["source_pixel_to_body_local"].astype(np.float64, copy=False)
    origins = body_arrays["body_origin_stabilized_xy"].astype(np.float64, copy=False)
    scales = body_arrays["torso_scale_px"].astype(np.float64, copy=False)
    valid = body_arrays["valid_frame"].astype(np.bool_, copy=False)
    if transforms.shape != (frame_count, 2, 3) or origins.shape != (frame_count, 2):
        raise MicroMotionError("E8 body-frame physical shape mismatch")
    if scales.shape != (frame_count,) or valid.shape != (frame_count,):
        raise MicroMotionError("E8 body-frame support arrays have invalid shape")
    energy = np.full(frame_count, np.nan, dtype=np.float32)
    for frame_idx in range(1, frame_count):
        if not valid[frame_idx - 1] or not valid[frame_idx]:
            continue
        previous_scale = float(scales[frame_idx - 1])
        current_scale = float(scales[frame_idx])
        if previous_scale <= 1e-12 or current_scale <= 1e-12:
            continue
        mean_scale = 0.5 * (previous_scale + current_scale)
        origin_delta = float(np.linalg.norm(origins[frame_idx] - origins[frame_idx - 1])) / mean_scale
        scale_delta = abs(math.log(current_scale / previous_scale))
        previous_angle = math.atan2(float(transforms[frame_idx - 1, 0, 1]), float(transforms[frame_idx - 1, 0, 0]))
        current_angle = math.atan2(float(transforms[frame_idx, 0, 1]), float(transforms[frame_idx, 0, 0]))
        angle_delta = (current_angle - previous_angle + math.pi) % (2.0 * math.pi) - math.pi
        energy[frame_idx] = math.sqrt(origin_delta * origin_delta + scale_delta * scale_delta + angle_delta * angle_delta)
    return energy


def _series_ref(
    *,
    uri: str,
    checksum: str,
    frame_count: int,
    array_key: str,
    valid_key: str,
    unit: str,
    config: MicroMotionConfig,
    source_surface_uri: str,
    body_frame_uri: str,
) -> dict[str, Any]:
    return {
        "uri": uri,
        "format": "npz",
        "dtype": "float32",
        "shape": [frame_count],
        "axes": ["frame"],
        "unit": unit,
        "coordinate_space": "body_local_2d",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": {
            "array_key": array_key,
            "valid_array_key": valid_key,
            "algorithm": "body_local_sparse_periodicity_v1",
            "source_surface_uri": source_surface_uri,
            "body_frame_transform_uri": body_frame_uri,
            "aggregation": "winsorized_mean_residual_projected_to_principal_axis",
            "shot_boundary_reset": True,
            "linear_detrend_only": True,
            "interpolation": False,
            "physiological_interpretation": False,
            "config_sha256": config.token(),
        },
    }


def run_micro_motion(
    blueprint: dict[str, Any],
    *,
    output_dir: str,
    sidecars: dict[str, Any],
    config: MicroMotionConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config.validate()
    frame_count = int(blueprint["timebase"]["frame_count"])
    fps_num = int(blueprint["timebase"]["fps_num"])
    fps_den = int(blueprint["timebase"]["fps_den"])
    fps = fps_num / fps_den
    if fps <= 0.0:
        raise MicroMotionError("E8 requires a positive normalized FPS")
    extension_e7 = blueprint.get("extensions", {}).get("e7_surface_motion")
    if not isinstance(extension_e7, dict) or extension_e7.get("enabled") is not True:
        raise MicroMotionError("E8 requires E7 sparse surface motion")
    shot_ranges = _shot_ranges(blueprint, frame_count)
    shot_start_mask = np.zeros(frame_count, dtype=np.bool_)
    for frame_start, _ in shot_ranges:
        shot_start_mask[frame_start] = True
    camera_energy, camera_available = _camera_energy(blueprint, sidecars, frame_count)
    characters = blueprint.get("characters")
    if not isinstance(characters, list) or not characters:
        raise MicroMotionError("E8 requires existing anonymous characters")

    emitted: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    extension_rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    coverages: list[float] = []

    for character in characters:
        if not isinstance(character, dict):
            raise MicroMotionError("E8 character row must be an object")
        character_id = str(character["character_id"])
        surface = character.get("surface_motion")
        if not isinstance(surface, dict) or surface.get("coordinate_frame") != "body_local_2d":
            raise MicroMotionError(f"{character_id} requires body_local_2d surface motion")
        body_ref = surface.get("body_frame_transform_ref")
        if not isinstance(body_ref, dict):
            raise MicroMotionError(f"{character_id} body-frame ref is missing")
        body_arrays = _load_npz(
            body_ref,
            sidecars,
            (
                "source_pixel_to_body_local",
                "body_origin_stabilized_xy",
                "torso_scale_px",
                "valid_frame",
            ),
        )
        body_valid = body_arrays["valid_frame"].astype(np.bool_, copy=False)
        torso_scale_px = body_arrays["torso_scale_px"].astype(np.float32, copy=False)
        pose_energy = _pose_energy(body_arrays, frame_count)
        regions = surface.get("regions")
        if not isinstance(regions, list):
            raise MicroMotionError(f"{character_id} surface regions must be an array")
        if not regions:
            extension_rows.append(
                {
                    "character_id": character_id,
                    "kind": "not_detected",
                    "signal_ref": None,
                    "cycles_observed": 0.0,
                    "frequency_window_frames": 0,
                }
            )
            report_rows.append(
                {
                    "character_id": character_id,
                    "kind": "not_detected",
                    "reason": "no_e7_surface_region",
                    "coverage": 0.0,
                    "confidence": 0.0,
                }
            )
            confidences.append(0.0)
            coverages.append(0.0)
            continue
        if len(regions) != 1 or not isinstance(regions[0], dict):
            raise MicroMotionError(f"{character_id} E8 v1 requires exactly one E7 sparse surface region")
        region = regions[0]
        residual_ref = region.get("residual_flow_ref")
        local_ref = region.get("track_points_ref")
        if not isinstance(residual_ref, dict) or not isinstance(local_ref, dict):
            raise MicroMotionError(f"{character_id} E7 residual refs are missing")
        if residual_ref.get("uri") != local_ref.get("uri"):
            raise MicroMotionError(f"{character_id} E7 sparse surface refs must share one NPZ")
        surface_arrays = _load_npz(
            residual_ref,
            sidecars,
            (
                "body_local_positions",
                "body_local_valid",
                "residual_displacement",
                "residual_valid",
                "track_id",
            ),
        )
        residual = surface_arrays["residual_displacement"].astype(np.float32, copy=False)
        residual_valid = surface_arrays["residual_valid"].astype(np.bool_, copy=False)
        if residual.ndim != 3 or residual.shape[0] != frame_count or residual.shape[2] != 2:
            raise MicroMotionError(f"{character_id} E7 residual shape mismatch")
        if residual_valid.shape != residual.shape[:2]:
            raise MicroMotionError(f"{character_id} E7 residual validity shape mismatch")

        axis = estimate_principal_axis(residual, residual_valid)
        velocity, vertical_velocity, coherence_frame, signal_valid = aggregate_residual_velocity(
            residual,
            residual_valid,
            axis,
            min_valid_points=config.min_valid_points_per_frame,
        )
        displacement, displacement_valid = _integrate(velocity, signal_valid, shot_ranges)
        vertical_displacement, vertical_valid = _integrate(vertical_velocity, signal_valid, shot_ranges)
        detrended = _linear_detrend(displacement, displacement_valid, shot_ranges)
        acceleration, acceleration_valid = _acceleration(velocity, signal_valid, shot_ranges)

        dominant_frequency, phase, periodicity, cycles_observed, window_frames = analyze_frequency(
            detrended,
            displacement_valid,
            fps=fps,
            shot_ranges=shot_ranges,
            config=config,
        )
        coherent_values = coherence_frame[np.isfinite(coherence_frame)]
        spatial_coherence = float(np.median(coherent_values)) if len(coherent_values) else 0.0
        camera_leakage, camera_assessed = _correlation_leakage(velocity, camera_energy)
        pose_leakage, pose_assessed = _correlation_leakage(velocity, pose_energy)
        eligible_mask = body_valid & ~shot_start_mask
        eligible_frames = int(np.count_nonzero(eligible_mask))
        valid_frames = int(np.count_nonzero(signal_valid & eligible_mask))
        coverage = valid_frames / eligible_frames if eligible_frames else 0.0
        observation_dropout = 1.0 - coverage if eligible_frames else 1.0
        coverages.append(coverage)

        finite_detrended = np.isfinite(detrended) & displacement_valid
        amplitude_norm = float(np.median(np.abs(detrended[finite_detrended]))) if np.any(finite_detrended) else 0.0
        amplitude_px_values = np.abs(detrended.astype(np.float64) * torso_scale_px.astype(np.float64))
        amplitude_px_mask = finite_detrended & np.isfinite(torso_scale_px)
        amplitude_px = float(np.median(amplitude_px_values[amplitude_px_mask])) if np.any(amplitude_px_mask) else 0.0

        has_periodic_evidence = (
            dominant_frequency is not None
            and cycles_observed >= config.min_cycles
            and periodicity >= config.periodicity_threshold
            and amplitude_norm >= config.min_amplitude_norm
        )
        if has_periodic_evidence:
            kind = "periodic_micro_motion"
        elif amplitude_norm >= config.min_amplitude_norm and valid_frames > 0:
            kind = "unclassified"
        else:
            kind = "not_detected"

        signal_quality = min(
            spatial_coherence,
            1.0 - camera_leakage,
            1.0 - pose_leakage,
            1.0 - observation_dropout,
        )
        if kind == "periodic_micro_motion":
            confidence = min(signal_quality, periodicity)
        elif kind == "unclassified":
            confidence = 0.5 * signal_quality
        else:
            confidence = max(0.0, min(1.0, signal_quality))
        confidence = max(0.0, min(1.0, confidence))
        confidences.append(confidence)
        usable = (
            kind == "periodic_micro_motion"
            and confidence >= config.usable_confidence_threshold
            and spatial_coherence >= config.coherence_threshold
            and camera_leakage <= config.max_camera_leakage
            and pose_leakage <= config.max_pose_leakage
            and observation_dropout <= config.max_observation_dropout
        )

        limitations = [
            "geometry-only classification; no breathing, heart-rate, emotion, health, or other physiological inference",
            "occlusion_ratio is an observation-dropout proxy, not semantic occlusion classification",
            "radial expansion and area-change modes are not emitted in E8.1",
        ]
        if not camera_available or not camera_assessed:
            limitations.append("camera leakage could not be fully assessed; fail-closed leakage score applied")
        if not pose_assessed:
            limitations.append("pose leakage could not be fully assessed; fail-closed leakage score applied")
        if dominant_frequency is None:
            limitations.append("insufficient contiguous evidence for a dominant frequency")
        elif cycles_observed < config.min_cycles:
            limitations.append("dominant frequency observed for fewer than the configured minimum cycles")

        uri = f"artifacts/timeseries/{character_id}_micro_motion_geometry.npz"
        path = os.path.join(output_dir, uri.replace("/", os.sep))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(
            path,
            signal=displacement,
            signal_valid=displacement_valid,
            detrended_signal=detrended,
            vertical_displacement=vertical_displacement,
            vertical_valid=vertical_valid,
            velocity=velocity,
            velocity_valid=signal_valid,
            acceleration=acceleration,
            acceleration_valid=acceleration_valid,
        )
        checksum = sha256_file(path)
        source_surface_uri = str(residual_ref["uri"])
        body_frame_uri = str(body_ref["uri"])
        signal_ref = _series_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            array_key="signal",
            valid_key="signal_valid",
            unit="body_local_unit",
            config=config,
            source_surface_uri=source_surface_uri,
            body_frame_uri=body_frame_uri,
        )
        detrended_ref = _series_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            array_key="detrended_signal",
            valid_key="signal_valid",
            unit="body_local_unit",
            config=config,
            source_surface_uri=source_surface_uri,
            body_frame_uri=body_frame_uri,
        )
        vertical_ref = _series_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            array_key="vertical_displacement",
            valid_key="vertical_valid",
            unit="body_local_unit",
            config=config,
            source_surface_uri=source_surface_uri,
            body_frame_uri=body_frame_uri,
        )
        velocity_ref = _series_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            array_key="velocity",
            valid_key="velocity_valid",
            unit="body_local_unit_per_frame",
            config=config,
            source_surface_uri=source_surface_uri,
            body_frame_uri=body_frame_uri,
        )
        acceleration_ref = _series_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            array_key="acceleration",
            valid_key="acceleration_valid",
            unit="body_local_unit_per_frame2",
            config=config,
            source_surface_uri=source_surface_uri,
            body_frame_uri=body_frame_uri,
        )
        micro_motion = {
            "kind": kind,
            "signal_ref": signal_ref,
            "detrended_signal_ref": detrended_ref,
            "vertical_displacement_ref": vertical_ref,
            "radial_expansion_ratio_ref": None,
            "area_change_ratio_ref": None,
            "velocity_ref": velocity_ref,
            "acceleration_ref": acceleration_ref,
            "axis_2d": [round(float(axis[0]), 8), round(float(axis[1]), 8)],
            "amplitude_px_p50": round(max(0.0, amplitude_px), 8),
            "amplitude_norm_p50": round(max(0.0, amplitude_norm), 8),
            "dominant_frequency_hz": round(dominant_frequency, 8) if dominant_frequency is not None else None,
            "phase_rad": round(phase, 8) if phase is not None else None,
            "periodicity_score": round(periodicity, 6),
            "spatial_coherence": round(spatial_coherence, 6),
            "camera_leakage_score": round(camera_leakage, 6),
            "pose_leakage_score": round(pose_leakage, 6),
            "occlusion_ratio": round(observation_dropout, 6),
            "confidence": round(confidence, 6),
            "usable_for_generation": bool(usable),
            "limitations": limitations,
            "smoothing": {"method": "none", "parameters": {}},
        }
        region["micro_motion"] = micro_motion
        region_quality = region.get("quality")
        if isinstance(region_quality, dict):
            region_quality["score"] = round(confidence, 6)
            region_quality["coverage"] = round(coverage, 6)
            warnings = region_quality.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            region_quality["warnings"] = [
                warning for warning in warnings if warning != "micro-motion analysis deferred to E8"
            ] + ["E8 micro-motion is geometry-only; no physiological interpretation"]
        emitted[uri] = path
        extension_rows.append(
            {
                "character_id": character_id,
                "kind": kind,
                "signal_ref": signal_ref,
                "cycles_observed": round(cycles_observed, 6),
                "frequency_window_frames": window_frames,
                "camera_leakage_assessed": bool(camera_available and camera_assessed),
                "pose_leakage_assessed": bool(pose_assessed),
            }
        )
        report_rows.append(
            {
                "character_id": character_id,
                "kind": kind,
                "coverage": round(coverage, 6),
                "axis_2d": micro_motion["axis_2d"],
                "amplitude_norm_p50": micro_motion["amplitude_norm_p50"],
                "amplitude_px_p50": micro_motion["amplitude_px_p50"],
                "dominant_frequency_hz": micro_motion["dominant_frequency_hz"],
                "cycles_observed": round(cycles_observed, 6),
                "periodicity_score": micro_motion["periodicity_score"],
                "spatial_coherence": micro_motion["spatial_coherence"],
                "camera_leakage_score": micro_motion["camera_leakage_score"],
                "pose_leakage_score": micro_motion["pose_leakage_score"],
                "observation_dropout_ratio": micro_motion["occlusion_ratio"],
                "confidence": micro_motion["confidence"],
                "usable_for_generation": micro_motion["usable_for_generation"],
            }
        )

    overall_score = min(confidences, default=0.0)
    overall_coverage = min(coverages, default=0.0)
    report_uri = "artifacts/reports/micro_motion_geometry.json"
    report = {
        "stage": "micro_motion",
        "algorithm": "body_local_sparse_periodicity_v1",
        "interpretation_scope": "geometry_only",
        "physiological_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "characters": report_rows,
    }
    emitted[report_uri] = report
    report_ref = {
        "kind": "micro_motion_geometry",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "metadata": {"algorithm": "body_local_sparse_periodicity_v1", "config_sha256": config.token()},
    }
    extension = {
        "enabled": True,
        "algorithm": "body_local_sparse_periodicity_v1",
        "interpretation_scope": "geometry_only",
        "source_surface_motion": "e7_surface_motion",
        "body_frame_source": "e6_body_local_frame",
        "camera_source": "e5_camera_motion",
        "class_scope": ["periodic_micro_motion", "unclassified", "not_detected"],
        "radial_expansion_emitted": False,
        "area_change_emitted": False,
        "physiological_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "occlusion_metric_semantics": "observation_dropout_proxy_not_semantic_occlusion",
        "config_sha256": config.token(),
        "thresholds": {
            "min_valid_points_per_frame": config.min_valid_points_per_frame,
            "min_periodic_frames": config.min_periodic_frames,
            "min_frequency_hz": config.min_frequency_hz,
            "max_frequency_hz": config.max_frequency_hz,
            "min_cycles": config.min_cycles,
            "min_amplitude_norm": config.min_amplitude_norm,
            "periodicity_threshold": config.periodicity_threshold,
            "coherence_threshold": config.coherence_threshold,
            "max_camera_leakage": config.max_camera_leakage,
            "max_pose_leakage": config.max_pose_leakage,
            "max_observation_dropout": config.max_observation_dropout,
            "usable_confidence_threshold": config.usable_confidence_threshold,
        },
        "characters": extension_rows,
    }
    quality = {
        "score": round(overall_score, 6),
        "coverage": round(overall_coverage, 6),
        "warnings": ["geometry-only micro-motion evidence; no physiological interpretation"],
        "errors": [],
    }
    return emitted, report_ref, extension, quality


__all__ = [
    "MicroMotionConfig",
    "MicroMotionError",
    "aggregate_residual_velocity",
    "analyze_frequency",
    "estimate_principal_axis",
    "run_micro_motion",
]
