from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from packages.pipeline_core.micro_motion import (
    MicroMotionConfig,
    MicroMotionError,
    _acceleration,
    _camera_energy,
    _correlation_leakage,
    _integrate,
    _json_sha256,
    _linear_detrend,
    _load_npz,
    _pose_energy,
    _shot_ranges,
    aggregate_residual_velocity,
    analyze_frequency,
    estimate_principal_axis,
)
from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file

ALGORITHM = "body_local_sparse_periodicity_v2"
SELECTION_METHOD = "per_track_frequency_locked_consensus_v2"
TRACK_PERIODICITY_FLOOR = 0.25
MIN_CONSENSUS_TRACKS = 4


def _longest_valid_indices(valid: np.ndarray, shot_ranges: list[tuple[int, int]]) -> np.ndarray:
    best = np.asarray([], dtype=np.int64)
    mask = np.asarray(valid, dtype=np.bool_)
    for frame_start, frame_end in shot_ranges:
        start: int | None = None
        for frame_idx in range(frame_start, frame_end + 1):
            if bool(mask[frame_idx]):
                if start is None:
                    start = frame_idx
            elif start is not None:
                candidate = np.arange(start, frame_idx, dtype=np.int64)
                if len(candidate) > len(best):
                    best = candidate
                start = None
        if start is not None:
            candidate = np.arange(start, frame_end + 1, dtype=np.int64)
            if len(candidate) > len(best):
                best = candidate
    return best


def _spectral_profile(
    signal: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    shot_ranges: list[tuple[int, int]],
    config: MicroMotionConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    indices = _longest_valid_indices(valid, shot_ranges)
    duration_s = max(0.0, (len(indices) - 1) / fps) if len(indices) else 0.0
    resolution = fps / max(len(indices), 1) if len(indices) else 0.0
    if len(indices) < config.min_periodic_frames:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), resolution, duration_s
    values = np.asarray(signal[indices], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), resolution, duration_s
    values -= float(np.mean(values))
    rms = float(np.sqrt(np.mean(values * values)))
    if not math.isfinite(rms) or rms < 1e-12:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), resolution, duration_s
    spectrum = np.fft.rfft(values * np.hanning(len(values)))
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / fps)
    power = np.abs(spectrum) ** 2
    max_frequency = min(config.max_frequency_hz, fps * 0.45)
    band = (frequencies >= config.min_frequency_hz) & (frequencies <= max_frequency) & (frequencies > 0.0)
    band_indices = np.flatnonzero(band)
    if not len(band_indices):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), resolution, duration_s
    band_power = float(np.sum(power[band_indices]))
    if not math.isfinite(band_power) or band_power <= 1e-18:
        return frequencies[band_indices].astype(np.float64), np.zeros(len(band_indices), dtype=np.float64), resolution, duration_s
    return (
        frequencies[band_indices].astype(np.float64),
        (power[band_indices] / band_power).astype(np.float64),
        resolution,
        duration_s,
    )


def _track_signal(
    residual: np.ndarray,
    residual_valid: np.ndarray,
    track_id: np.ndarray,
    candidate_id: int,
    *,
    shot_ranges: list[tuple[int, int]],
    fps: float,
    config: MicroMotionConfig,
) -> dict[str, Any] | None:
    frame_count = residual.shape[0]
    vectors = np.full((frame_count, 1, 2), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count, 1), dtype=np.bool_)
    for frame_idx in range(frame_count):
        slots = np.flatnonzero(residual_valid[frame_idx] & (track_id[frame_idx] == candidate_id))
        if len(slots) > 1:
            raise MicroMotionError(f"E8 v2 found duplicate physical samples for track_id={candidate_id}")
        if len(slots) == 1:
            slot = int(slots[0])
            vectors[frame_idx, 0] = residual[frame_idx, slot]
            valid[frame_idx, 0] = True
    if int(np.count_nonzero(valid)) < config.min_periodic_frames:
        return None
    axis = estimate_principal_axis(vectors, valid)
    if float(np.linalg.norm(axis)) < 1e-8:
        return None
    velocity = np.full(frame_count, np.nan, dtype=np.float32)
    velocity_valid = valid[:, 0].copy()
    finite_vectors = vectors[:, 0].astype(np.float64, copy=False)
    velocity[velocity_valid] = (finite_vectors[velocity_valid] @ axis.astype(np.float64)).astype(np.float32)
    displacement, displacement_valid = _integrate(velocity, velocity_valid, shot_ranges)
    detrended = _linear_detrend(displacement, displacement_valid, shot_ranges)
    frequency, phase, periodicity, cycles, window_frames = analyze_frequency(
        detrended,
        displacement_valid,
        fps=fps,
        shot_ranges=shot_ranges,
        config=config,
    )
    spectral_frequencies, spectral_power_fraction, frequency_resolution, duration_s = _spectral_profile(
        detrended,
        displacement_valid,
        fps=fps,
        shot_ranges=shot_ranges,
        config=config,
    )
    finite = displacement_valid & np.isfinite(detrended)
    amplitude = float(np.median(np.abs(detrended[finite]))) if np.any(finite) else 0.0
    return {
        "track_id": candidate_id,
        "frequency_hz": frequency,
        "phase_rad": phase,
        "periodicity_score": periodicity,
        "cycles": cycles,
        "window_frames": window_frames,
        "amplitude_norm": amplitude,
        "spectral_frequencies": spectral_frequencies,
        "spectral_power_fraction": spectral_power_fraction,
        "frequency_resolution_hz": frequency_resolution,
        "duration_s": duration_s,
    }


def _locked_power(row: dict[str, Any], candidate_frequency: float) -> float:
    frequencies = row.get("spectral_frequencies")
    power_fraction = row.get("spectral_power_fraction")
    resolution = row.get("frequency_resolution_hz")
    if not isinstance(frequencies, np.ndarray) or not isinstance(power_fraction, np.ndarray):
        return 0.0
    if frequencies.ndim != 1 or power_fraction.shape != frequencies.shape or not len(frequencies):
        return 0.0
    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool) or float(resolution) <= 0.0:
        return 0.0
    nearest = int(np.argmin(np.abs(frequencies - candidate_frequency)))
    if abs(float(frequencies[nearest]) - candidate_frequency) > 0.55 * float(resolution) + 1e-12:
        return 0.0
    value = float(power_fraction[nearest])
    return value if math.isfinite(value) and value >= 0.0 else 0.0


def _consensus_tracks(
    evidence: list[dict[str, Any]],
    *,
    fps: float,
    config: MicroMotionConfig,
) -> tuple[list[int], float | None, float, dict[str, Any]]:
    del fps  # individual spectral grids already retain their physical resolution
    analyzable = [
        row
        for row in evidence
        if row.get("frequency_hz") is not None
        and float(row.get("amplitude_norm", 0.0)) >= config.min_amplitude_norm
        and float(row.get("duration_s", 0.0)) > 0.0
    ]
    if not analyzable:
        return [], None, 0.0, {
            "candidate_track_count": 0,
            "qualified_track_count": 0,
            "frequency_tolerance_hz": None,
            "consensus_periodicity_mean": 0.0,
        }

    seed_frequencies = sorted(
        {
            round(float(row["frequency_hz"]), 9)
            for row in analyzable
            if float(row["frequency_hz"]) * float(row["duration_s"]) >= config.min_cycles
        }
    )
    required_consensus = max(MIN_CONSENSUS_TRACKS, config.min_valid_points_per_frame)
    best_selected: list[tuple[dict[str, Any], float]] = []
    best_score = -1.0
    best_frequency: float | None = None
    for candidate_frequency in seed_frequencies:
        selected: list[tuple[dict[str, Any], float]] = []
        for row in analyzable:
            if candidate_frequency * float(row["duration_s"]) < config.min_cycles:
                continue
            locked_power = _locked_power(row, candidate_frequency)
            if locked_power < TRACK_PERIODICITY_FLOOR:
                continue
            selected.append((row, locked_power))
        if len(selected) < required_consensus:
            continue
        score = sum(
            locked_power * math.sqrt(max(float(row["amplitude_norm"]), 1e-12))
            for row, locked_power in selected
        )
        if score > best_score or (
            math.isclose(score, best_score)
            and (
                len(selected) > len(best_selected)
                or (
                    len(selected) == len(best_selected)
                    and (best_frequency is None or candidate_frequency < best_frequency)
                )
            )
        ):
            best_score = score
            best_selected = selected
            best_frequency = candidate_frequency

    resolutions = [
        float(row["frequency_resolution_hz"])
        for row in analyzable
        if isinstance(row.get("frequency_resolution_hz"), (int, float))
        and not isinstance(row.get("frequency_resolution_hz"), bool)
        and float(row["frequency_resolution_hz"]) > 0.0
    ]
    tolerance = 0.55 * max(resolutions, default=0.0)
    if best_frequency is None or len(best_selected) < required_consensus:
        return [], None, 0.0, {
            "candidate_track_count": len(analyzable),
            "qualified_track_count": 0,
            "selected_track_count": 0,
            "frequency_tolerance_hz": round(tolerance, 6) if tolerance > 0.0 else None,
            "consensus_periodicity_mean": 0.0,
        }

    locked_values = np.asarray([locked for _, locked in best_selected], dtype=np.float64)
    consensus_periodicity = float(np.mean(locked_values))
    support_fraction = len(best_selected) / max(len(analyzable), 1)
    track_ids = sorted(int(row["track_id"]) for row, _ in best_selected)
    return track_ids, best_frequency, support_fraction, {
        "candidate_track_count": len(analyzable),
        "qualified_track_count": len(best_selected),
        "selected_track_count": len(track_ids),
        "frequency_tolerance_hz": round(tolerance, 6) if tolerance > 0.0 else None,
        "consensus_periodicity_mean": round(consensus_periodicity, 6),
    }


def _series_ref_v2(
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
            "algorithm": ALGORITHM,
            "selection_method": SELECTION_METHOD,
            "selected_track_id_array_key": "selected_track_id",
            "source_surface_uri": source_surface_uri,
            "body_frame_transform_uri": body_frame_uri,
            "aggregation": "winsorized_mean_selected_residual_projected_to_principal_axis",
            "shot_boundary_reset": True,
            "linear_detrend_only": True,
            "interpolation": False,
            "physiological_interpretation": False,
            "config_sha256": config.token(),
        },
    }


def run_micro_motion_v2(
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
        raise MicroMotionError("E8 v2 requires a positive normalized FPS")
    extension_e7 = blueprint.get("extensions", {}).get("e7_surface_motion")
    if not isinstance(extension_e7, dict) or extension_e7.get("enabled") is not True:
        raise MicroMotionError("E8 v2 requires E7 sparse surface motion")
    shot_ranges = _shot_ranges(blueprint, frame_count)
    shot_start_mask = np.zeros(frame_count, dtype=np.bool_)
    for frame_start, _ in shot_ranges:
        shot_start_mask[frame_start] = True
    camera_energy, camera_available = _camera_energy(blueprint, sidecars, frame_count)
    characters = blueprint.get("characters")
    if not isinstance(characters, list) or not characters:
        raise MicroMotionError("E8 v2 requires existing anonymous characters")

    emitted: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    extension_rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    coverages: list[float] = []

    for character in characters:
        if not isinstance(character, dict):
            raise MicroMotionError("E8 v2 character row must be an object")
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
            ("source_pixel_to_body_local", "body_origin_stabilized_xy", "torso_scale_px", "valid_frame"),
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
                    "selected_track_count": 0,
                    "candidate_track_count": 0,
                    "consensus_frequency_hz": None,
                    "consensus_support_fraction": 0.0,
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
            raise MicroMotionError(f"{character_id} E8 v2 requires exactly one E7 sparse surface region")
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
            ("body_local_positions", "body_local_valid", "residual_displacement", "residual_valid", "track_id"),
        )
        residual = surface_arrays["residual_displacement"].astype(np.float32, copy=False)
        residual_valid = surface_arrays["residual_valid"].astype(np.bool_, copy=False)
        track_id = surface_arrays["track_id"].astype(np.int64, copy=False)
        if residual.ndim != 3 or residual.shape[0] != frame_count or residual.shape[2] != 2:
            raise MicroMotionError(f"{character_id} E7 residual shape mismatch")
        if residual_valid.shape != residual.shape[:2] or track_id.shape != residual.shape[:2]:
            raise MicroMotionError(f"{character_id} E7 residual support shape mismatch")

        unique_track_ids = sorted(int(value) for value in np.unique(track_id[residual_valid]) if int(value) >= 0)
        evidence: list[dict[str, Any]] = []
        for candidate_id in unique_track_ids:
            row = _track_signal(
                residual,
                residual_valid,
                track_id,
                candidate_id,
                shot_ranges=shot_ranges,
                fps=fps,
                config=config,
            )
            if row is not None:
                evidence.append(row)
        selected_track_ids, consensus_frequency, support_fraction, consensus_meta = _consensus_tracks(
            evidence,
            fps=fps,
            config=config,
        )
        selected_id_array = np.asarray(selected_track_ids, dtype=np.int64)
        if selected_track_ids:
            selected_valid = residual_valid & np.isin(track_id, selected_id_array)
        else:
            selected_valid = residual_valid.copy()

        axis = estimate_principal_axis(residual, selected_valid)
        velocity, vertical_velocity, coherence_frame, signal_valid = aggregate_residual_velocity(
            residual,
            selected_valid,
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
        raw_pose_leakage, pose_assessed = _correlation_leakage(velocity, pose_energy)
        pose_leakage = min(1.0, raw_pose_leakage * support_fraction) if selected_track_ids else raw_pose_leakage
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

        required_consensus = max(MIN_CONSENSUS_TRACKS, config.min_valid_points_per_frame)
        has_consensus = len(selected_track_ids) >= required_consensus
        has_periodic_evidence = (
            has_consensus
            and dominant_frequency is not None
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
            "pose leakage is body-frame energy correlation weighted by frequency-consensus spatial support",
        ]
        if not selected_track_ids:
            limitations.append("no per-track frequency-locked periodic consensus; whole-region signal retained only as unclassified evidence")
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
            selected_track_id=selected_id_array,
        )
        checksum = sha256_file(path)
        source_surface_uri = str(residual_ref["uri"])
        body_frame_uri = str(body_ref["uri"])
        signal_ref = _series_ref_v2(
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
        detrended_ref = _series_ref_v2(
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
        vertical_ref = _series_ref_v2(
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
        velocity_ref = _series_ref_v2(
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
        acceleration_ref = _series_ref_v2(
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
                "candidate_track_count": int(consensus_meta.get("candidate_track_count", 0)),
                "qualified_track_count": int(consensus_meta.get("qualified_track_count", 0)),
                "selected_track_count": len(selected_track_ids),
                "consensus_frequency_hz": round(consensus_frequency, 8) if consensus_frequency is not None else None,
                "consensus_support_fraction": round(support_fraction, 6),
                "consensus_periodicity_mean": float(consensus_meta.get("consensus_periodicity_mean", 0.0)),
                "frequency_tolerance_hz": consensus_meta.get("frequency_tolerance_hz"),
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
                "consensus_frequency_hz": round(consensus_frequency, 8) if consensus_frequency is not None else None,
                "selected_track_count": len(selected_track_ids),
                "candidate_track_count": int(consensus_meta.get("candidate_track_count", 0)),
                "consensus_support_fraction": round(support_fraction, 6),
                "cycles_observed": round(cycles_observed, 6),
                "periodicity_score": micro_motion["periodicity_score"],
                "spatial_coherence": micro_motion["spatial_coherence"],
                "camera_leakage_score": micro_motion["camera_leakage_score"],
                "raw_pose_leakage_score": round(raw_pose_leakage, 6),
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
        "algorithm": ALGORITHM,
        "selection_method": SELECTION_METHOD,
        "interpretation_scope": "geometry_only",
        "physiological_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "characters": report_rows,
    }
    emitted[report_uri] = report
    report_ref = {"kind": "micro_motion_geometry", "uri": report_uri, "sha256": _json_sha256(report)}
    extension = {
        "enabled": True,
        "algorithm": ALGORITHM,
        "selection_method": SELECTION_METHOD,
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
        "pose_leakage_semantics": "body_frame_energy_correlation_weighted_by_frequency_consensus_support",
        "track_periodicity_floor": TRACK_PERIODICITY_FLOOR,
        "minimum_consensus_tracks": MIN_CONSENSUS_TRACKS,
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


__all__ = ["ALGORITHM", "SELECTION_METHOD", "run_micro_motion_v2"]
