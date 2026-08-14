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


class SurfaceMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SurfaceMotionConfig:
    minimum_body_frame_coverage: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.minimum_body_frame_coverage <= 1.0:
            raise SurfaceMotionError("minimum_body_frame_coverage must be within [0, 1]")

    def token(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_environment(cls) -> SurfaceMotionConfig | None:
        raw = os.environ.get("VBS_E7_SURFACE_MOTION_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise SurfaceMotionError("VBS_E7_SURFACE_MOTION_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, indent=2, sort_keys=True).encode()).hexdigest()


def _load_npz(ref: dict[str, Any], sidecars: dict[str, Any], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    uri = ref.get("uri")
    if not isinstance(uri, str):
        raise SurfaceMotionError("E7 TimeSeriesRef URI is missing")
    path_value = sidecars.get(uri)
    if not isinstance(path_value, (str, os.PathLike)) or not os.path.isfile(str(path_value)):
        raise SurfaceMotionError(f"E7 physical sidecar unavailable: {uri}")
    if ref.get("checksum_sha256") != sha256_file(str(path_value)):
        raise SurfaceMotionError(f"E7 input checksum mismatch: {uri}")
    result: dict[str, np.ndarray] = {}
    with np.load(str(path_value), allow_pickle=False) as arrays:
        for key in keys:
            if key not in arrays:
                raise SurfaceMotionError(f"E7 input array {key!r} missing from {uri}")
            result[key] = np.asarray(arrays[key])
    return result


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points.astype(np.float64), np.ones((len(points), 1))), axis=1)
    return (matrix.astype(np.float64) @ homogeneous.T).T[:, :2].astype(np.float32)


def _micro_motion_pending() -> dict[str, Any]:
    return {
        "kind": "not_detected",
        "signal_ref": None,
        "detrended_signal_ref": None,
        "vertical_displacement_ref": None,
        "radial_expansion_ratio_ref": None,
        "area_change_ratio_ref": None,
        "velocity_ref": None,
        "acceleration_ref": None,
        "axis_2d": [0.0, 0.0],
        "amplitude_px_p50": 0.0,
        "amplitude_norm_p50": 0.0,
        "dominant_frequency_hz": None,
        "phase_rad": None,
        "periodicity_score": 0.0,
        "spatial_coherence": 0.0,
        "camera_leakage_score": 0.0,
        "pose_leakage_score": 0.0,
        "occlusion_ratio": 0.0,
        "confidence": 0.0,
        "usable_for_generation": False,
        "limitations": ["micro-motion analysis not run in E7.1; E8 owns this field"],
        "smoothing": {"method": "none", "parameters": {}},
    }


def _surface_ref(
    *,
    uri: str,
    checksum: str,
    frame_count: int,
    point_slots: int,
    array_key: str,
    valid_key: str,
    unit: str,
    config: SurfaceMotionConfig,
    source_track_uri: str,
    body_frame_uri: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "array_key": array_key,
        "valid_array_key": valid_key,
        "track_id_array_key": "track_id",
        "algorithm": "body_local_sparse_residual_v1",
        "source_point_track_uri": source_track_uri,
        "body_frame_transform_uri": body_frame_uri,
        "same_track_id_required": True,
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config.token(),
    }
    if array_key == "body_local_positions":
        metadata["semantics"] = "point_position_after_per_frame_source_pixel_to_body_local_transform"
    else:
        metadata["semantics"] = "current_body_local_position_minus_previous_body_local_position"
    return {
        "uri": uri,
        "format": "npz",
        "dtype": "float32",
        "shape": [frame_count, point_slots, 2],
        "axes": ["frame", "point_slot", "xy"],
        "unit": unit,
        "coordinate_space": "body_local_2d",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": metadata,
    }


def run_surface_motion(
    blueprint: dict[str, Any],
    *,
    output_dir: str,
    sidecars: dict[str, Any],
    config: SurfaceMotionConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config.validate()
    frame_count = int(blueprint["timebase"]["frame_count"])
    characters = blueprint.get("characters")
    if not isinstance(characters, list) or not characters:
        raise SurfaceMotionError("E7 requires existing anonymous characters")
    point_extension = blueprint.get("extensions", {}).get("e4_point_tracks")
    body_extension = blueprint.get("extensions", {}).get("e6_body_local_frame")
    if not isinstance(point_extension, dict) or point_extension.get("enabled") is not True:
        raise SurfaceMotionError("E7 requires E4.2 sparse point tracks")
    if not isinstance(body_extension, dict) or body_extension.get("enabled") is not True:
        raise SurfaceMotionError("E7 requires E6 body-local frames")
    point_rows = point_extension.get("characters")
    if not isinstance(point_rows, dict):
        raise SurfaceMotionError("E7 point-track extension has invalid character rows")
    shot_starts = {
        int(shot["frame_start"])
        for shot in blueprint.get("shots", [])
        if isinstance(shot, dict) and isinstance(shot.get("frame_start"), int) and int(shot["frame_start"]) > 0
    }

    emitted: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    extension_rows: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    residual_magnitudes: list[np.ndarray] = []

    for character in characters:
        if not isinstance(character, dict):
            raise SurfaceMotionError("E7 character row must be an object")
        character_id = str(character["character_id"])
        surface = character.get("surface_motion")
        if not isinstance(surface, dict) or surface.get("coordinate_frame") != "body_local_2d":
            raise SurfaceMotionError(f"{character_id} requires E6 body_local_2d surface contract")
        body_ref = surface.get("body_frame_transform_ref")
        if not isinstance(body_ref, dict):
            raise SurfaceMotionError(f"{character_id} E6 body-frame ref is missing")
        body_arrays = _load_npz(body_ref, sidecars, ("source_pixel_to_body_local", "valid_frame"))
        transforms = body_arrays["source_pixel_to_body_local"].astype(np.float32, copy=False)
        body_valid = body_arrays["valid_frame"].astype(np.bool_, copy=False)
        if transforms.shape != (frame_count, 2, 3) or body_valid.shape != (frame_count,):
            raise SurfaceMotionError(f"{character_id} body-frame physical shape mismatch")
        body_coverage = float(np.mean(body_valid))
        if body_coverage < config.minimum_body_frame_coverage:
            raise SurfaceMotionError(f"{character_id} body-frame coverage below configured minimum")

        point_row = point_rows.get(character_id)
        track_ref = point_row.get("track_points_ref") if isinstance(point_row, dict) else None
        if not isinstance(track_ref, dict):
            surface["regions"] = []
            surface["events"] = []
            surface["global_postural_sway_ref"] = None
            surface["quality"] = {
                "score": 0.0,
                "coverage": 0.0,
                "warnings": ["E4.2 emitted no sparse point-track samples for this character"],
                "errors": [],
            }
            extension_rows.append({"character_id": character_id, "track_points_ref": None, "residual_flow_ref": None})
            report_rows.append(
                {
                    "character_id": character_id,
                    "eligible_residual_samples": 0,
                    "valid_residual_samples": 0,
                    "coverage": 0.0,
                    "median_residual_body_local_units": None,
                    "p95_residual_body_local_units": None,
                }
            )
            coverage_values.append(0.0)
            continue

        point_arrays = _load_npz(track_ref, sidecars, ("positions_xy", "valid", "track_id"))
        positions = point_arrays["positions_xy"].astype(np.float32, copy=False)
        valid = point_arrays["valid"].astype(np.bool_, copy=False)
        track_id = point_arrays["track_id"].astype(np.int64, copy=False)
        if positions.ndim != 3 or positions.shape[0] != frame_count or positions.shape[2] != 2:
            raise SurfaceMotionError(f"{character_id} point-track positions shape mismatch")
        if valid.shape != positions.shape[:2] or track_id.shape != positions.shape[:2]:
            raise SurfaceMotionError(f"{character_id} point-track validity/id shape mismatch")
        point_slots = positions.shape[1]
        local_positions = np.full_like(positions, np.nan, dtype=np.float32)
        local_valid = valid & body_valid[:, None]
        for frame_idx in range(frame_count):
            slots = np.flatnonzero(local_valid[frame_idx])
            if slots.size:
                local_positions[frame_idx, slots] = _apply_affine(transforms[frame_idx], positions[frame_idx, slots])

        residual = np.full_like(positions, np.nan, dtype=np.float32)
        residual_valid = np.zeros_like(valid, dtype=np.bool_)
        eligible = 0
        for frame_idx in range(1, frame_count):
            if frame_idx in shot_starts:
                continue
            same_track = (
                valid[frame_idx - 1]
                & valid[frame_idx]
                & (track_id[frame_idx - 1] >= 0)
                & (track_id[frame_idx - 1] == track_id[frame_idx])
            )
            eligible += int(np.count_nonzero(same_track))
            usable = same_track & body_valid[frame_idx - 1] & body_valid[frame_idx]
            slots = np.flatnonzero(usable)
            if slots.size:
                values = local_positions[frame_idx, slots] - local_positions[frame_idx - 1, slots]
                finite = np.all(np.isfinite(values), axis=1)
                good_slots = slots[finite]
                residual[frame_idx, good_slots] = values[finite]
                residual_valid[frame_idx, good_slots] = True

        valid_samples = int(np.count_nonzero(residual_valid))
        coverage = valid_samples / eligible if eligible else 0.0
        coverage_values.append(coverage)
        magnitudes = np.linalg.norm(residual[residual_valid], axis=1) if valid_samples else np.empty(0, dtype=np.float32)
        if magnitudes.size:
            residual_magnitudes.append(magnitudes)
            median_magnitude = float(np.median(magnitudes))
            p95_magnitude = float(np.percentile(magnitudes, 95))
        else:
            median_magnitude = math.nan
            p95_magnitude = math.nan

        uri = f"artifacts/timeseries/{character_id}_surface_sparse_body_local.npz"
        path = os.path.join(output_dir, uri.replace("/", os.sep))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(
            path,
            body_local_positions=local_positions,
            body_local_valid=local_valid,
            residual_displacement=residual,
            residual_valid=residual_valid,
            track_id=track_id,
        )
        checksum = sha256_file(path)
        source_uri = str(track_ref["uri"])
        body_uri = str(body_ref["uri"])
        local_ref = _surface_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            point_slots=point_slots,
            array_key="body_local_positions",
            valid_key="body_local_valid",
            unit="body_local_unit",
            config=config,
            source_track_uri=source_uri,
            body_frame_uri=body_uri,
        )
        residual_ref = _surface_ref(
            uri=uri,
            checksum=checksum,
            frame_count=frame_count,
            point_slots=point_slots,
            array_key="residual_displacement",
            valid_key="residual_valid",
            unit="body_local_unit_per_frame",
            config=config,
            source_track_uri=source_uri,
            body_frame_uri=body_uri,
        )
        score = max(0.0, min(1.0, coverage))
        region = {
            "region_id": "custom",
            "display_name": "whole_body_sparse_surface",
            "roi_definition": {
                "type": "mask_intersection",
                "anchor_keypoints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
                "expansion_ratio": 1.0,
            },
            "mask_ref": character.get("person_mask_ref"),
            "track_points_ref": local_ref,
            "residual_flow_ref": residual_ref,
            "deformation_modes_ref": None,
            "micro_motion": _micro_motion_pending(),
            "quality": {
                "score": round(score, 6),
                "coverage": round(coverage, 6),
                "warnings": ["micro-motion analysis deferred to E8"],
                "errors": [],
            },
        }
        surface["global_postural_sway_ref"] = None
        surface["regions"] = [region]
        surface["events"] = []
        surface["quality"] = {
            "score": round(score, 6),
            "coverage": round(coverage, 6),
            "warnings": ["E7.1 provides sparse residual surface motion only; dense residual deferred"],
            "errors": [],
        }
        emitted[uri] = path
        extension_rows.append(
            {
                "character_id": character_id,
                "track_points_ref": local_ref,
                "residual_flow_ref": residual_ref,
            }
        )
        report_rows.append(
            {
                "character_id": character_id,
                "eligible_residual_samples": eligible,
                "valid_residual_samples": valid_samples,
                "coverage": round(coverage, 6),
                "median_residual_body_local_units": round(median_magnitude, 6) if math.isfinite(median_magnitude) else None,
                "p95_residual_body_local_units": round(p95_magnitude, 6) if math.isfinite(p95_magnitude) else None,
            }
        )

    overall_coverage = min(coverage_values) if coverage_values else 0.0
    if residual_magnitudes:
        all_magnitudes = np.concatenate(residual_magnitudes)
        overall_median = float(np.median(all_magnitudes))
        overall_p95 = float(np.percentile(all_magnitudes, 95))
    else:
        overall_median = math.nan
        overall_p95 = math.nan
    report_uri = "artifacts/reports/surface_motion_sparse.json"
    report = {
        "stage": "surface_motion",
        "algorithm": "body_local_sparse_residual_v1",
        "coordinate_frame": "body_local_2d",
        "source_point_tracks": "e4_point_tracks",
        "body_frame_source": "e6_body_local_frame",
        "shot_boundary_reset": True,
        "same_track_id_required": True,
        "interpolation": False,
        "dense_residual_emitted": False,
        "micro_motion_analyzed": False,
        "characters": report_rows,
        "overall_median_residual_body_local_units": round(overall_median, 6) if math.isfinite(overall_median) else None,
        "overall_p95_residual_body_local_units": round(overall_p95, 6) if math.isfinite(overall_p95) else None,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    emitted[report_uri] = report
    report_ref = {
        "kind": "surface_motion_sparse",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    extension = {
        "enabled": True,
        "algorithm": "body_local_sparse_residual_v1",
        "coordinate_frame": "body_local_2d",
        "source_point_tracks": "e4_point_tracks",
        "body_frame_source": "e6_body_local_frame",
        "shot_boundary_reset": True,
        "same_track_id_required": True,
        "interpolation": False,
        "dense_residual_emitted": False,
        "micro_motion_analyzed": False,
        "config_sha256": config.token(),
        "characters": extension_rows,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    quality = {
        "coverage": round(overall_coverage, 6),
        "score": round(overall_coverage, 6),
        "confidence_available": False,
    }
    return emitted, report_ref, extension, quality
