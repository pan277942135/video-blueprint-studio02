from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import cv2
import numpy as np

from packages.pipeline_core.sparse_motion import (
    SparseMotionConfig,
    SparseMotionError,
    atomic_npz,
    lk_step,
    load_mask_frames,
    seed_features,
    sha256_file,
)

PointTrackError = SparseMotionError


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point_ref(
    uri: str,
    checksum: str,
    frame_count: int,
    max_points: int,
    coverage: float,
    config: SparseMotionConfig,
) -> dict[str, Any]:
    return {
        "uri": uri,
        "format": "npz",
        "dtype": "float32",
        "shape": [frame_count, max_points, 2],
        "axes": ["frame", "point_slot", "xy"],
        "unit": "px",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": {
            "array_key": "positions_xy",
            "valid_array_key": "valid",
            "error_array_key": "tracking_error",
            "track_id_array_key": "track_id",
            "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
            "mask_constraint": "character.person_mask_ref",
            "point_slot_semantics": "reusable_slot_track_id_disambiguates_generation",
            "shot_boundary_reset": True,
            "coverage": round(coverage, 6),
            "config_sha256": config.token(),
        },
    }


def _write_tracks(
    character_id: str,
    output_dir: str,
    positions: np.ndarray,
    valid: np.ndarray,
    errors: np.ndarray,
    track_ids: np.ndarray,
    frame_count: int,
    coverage: float,
    config: SparseMotionConfig,
) -> tuple[dict[str, Any], str, str]:
    uri = f"artifacts/timeseries/{character_id}_point_tracks.npz"
    path = os.path.join(output_dir, uri.replace("/", os.sep))
    atomic_npz(path, positions_xy=positions, valid=valid, tracking_error=errors, track_id=track_ids)
    checksum = sha256_file(path)
    return _point_ref(uri, checksum, frame_count, config.max_points, coverage, config), uri, path


def _reseed(
    gray: np.ndarray,
    mask: np.ndarray,
    xy: np.ndarray,
    tids: np.ndarray,
    next_id: int,
    config: SparseMotionConfig,
) -> int:
    free = np.flatnonzero(tids < 0)
    if free.size == 0:
        return next_id
    seed_mask_u8 = mask.astype(np.uint8) * 255
    exclusion_radius = max(1, round(config.min_distance_px))
    for slot in np.flatnonzero(tids >= 0):
        x, y = xy[slot]
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(seed_mask_u8, (round(float(x)), round(float(y))), exclusion_radius, 0, -1)
    seeds = seed_features(gray, seed_mask_u8.astype(np.bool_), int(free.size), config)
    for slot, point in zip(free, seeds, strict=False):
        xy[slot] = point
        tids[slot] = next_id
        next_id += 1
    return next_id


def run_point_track_refinement(
    video_path: str,
    *,
    characters: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    frame_count: int,
    output_dir: str,
    sidecars: dict[str, Any],
    config: SparseMotionConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Track frame-local visual features inside E4.1 person masks."""
    config.validate()
    if not os.path.isfile(video_path) or frame_count <= 0 or not characters:
        raise PointTrackError("E4.2 requires a normalized video, positive frame_count, and characters")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise PointTrackError("normalized video could not be opened")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise PointTrackError("normalized video dimensions are invalid")

    ids = [str(character["character_id"]) for character in characters]
    masks = {
        character_id: load_mask_frames(character, frame_count, height, width, sidecars)
        for character_id, character in zip(ids, characters, strict=True)
    }
    shot_starts = {
        int(shot["frame_start"])
        for shot in shots
        if isinstance(shot, dict) and isinstance(shot.get("frame_start"), int) and int(shot["frame_start"]) > 0
    }

    positions = {key: np.full((frame_count, config.max_points, 2), np.nan, np.float32) for key in ids}
    valid = {key: np.zeros((frame_count, config.max_points), np.bool_) for key in ids}
    errors = {key: np.full((frame_count, config.max_points), np.nan, np.float32) for key in ids}
    track_ids = {key: np.full((frame_count, config.max_points), -1, np.int64) for key in ids}
    active_xy = {key: np.full((config.max_points, 2), np.nan, np.float32) for key in ids}
    active_id = {key: np.full(config.max_points, -1, np.int64) for key in ids}
    next_id = {key: 0 for key in ids}
    measured_errors: dict[str, list[float]] = {key: [] for key in ids}

    previous_gray: np.ndarray | None = None
    decoded = 0
    try:
        while decoded < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (height, width):
                raise PointTrackError("normalized frame dimensions changed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shot_reset = decoded in shot_starts
            for character_id in ids:
                mask = masks[character_id][decoded]
                xy = active_xy[character_id]
                tids = active_id[character_id]
                if mask is None:
                    xy[:] = np.nan
                    tids[:] = -1
                    continue
                if previous_gray is None or shot_reset:
                    xy[:] = np.nan
                    tids[:] = -1
                else:
                    slots = np.flatnonzero(tids >= 0)
                    if slots.size:
                        moved, keep, lk_error = lk_step(previous_gray, gray, xy[slots], mask, config)
                        for local, slot in enumerate(slots):
                            if keep[local]:
                                xy[slot] = moved[local]
                                positions[character_id][decoded, slot] = moved[local]
                                valid[character_id][decoded, slot] = True
                                track_ids[character_id][decoded, slot] = tids[slot]
                                value = float(lk_error[local])
                                if math.isfinite(value):
                                    errors[character_id][decoded, slot] = value
                                    measured_errors[character_id].append(value)
                            else:
                                xy[slot] = np.nan
                                tids[slot] = -1

                minimum = math.ceil(config.max_points * config.reseed_below_ratio)
                if int(np.count_nonzero(tids >= 0)) < minimum:
                    next_id[character_id] = _reseed(gray, mask, xy, tids, next_id[character_id], config)
                for slot in np.flatnonzero(tids >= 0):
                    if not valid[character_id][decoded, slot]:
                        positions[character_id][decoded, slot] = xy[slot]
                        valid[character_id][decoded, slot] = True
                        track_ids[character_id][decoded, slot] = tids[slot]
            previous_gray = gray
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count:
        raise PointTrackError(f"decoded {decoded} frames, expected {frame_count}")

    emitted: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    for character_id in ids:
        mask_frames = sum(mask is not None for mask in masks[character_id])
        observed_frames = int(np.count_nonzero(np.any(valid[character_id], axis=1)))
        coverage = observed_frames / mask_frames if mask_frames else 0.0
        coverage_values.append(coverage)
        samples = int(np.count_nonzero(valid[character_id]))
        unique_ids = int(np.unique(track_ids[character_id][track_ids[character_id] >= 0]).size)
        median_error = float(np.median(measured_errors[character_id])) if measured_errors[character_id] else None
        ref = None
        if samples:
            ref, uri, path = _write_tracks(
                character_id,
                output_dir,
                positions[character_id],
                valid[character_id],
                errors[character_id],
                track_ids[character_id],
                frame_count,
                coverage,
                config,
            )
            emitted[uri] = path
        quality = {
            "coverage": round(coverage, 6),
            "valid_point_samples": samples,
            "unique_track_ids": unique_ids,
            "median_lk_error": round(median_error, 6) if median_error is not None else None,
        }
        rows[character_id] = {"track_points_ref": ref, "quality": quality}
        reports.append({"character_id": character_id, **quality})

    report_uri = "artifacts/reports/point_tracks.json"
    report = {
        "stage": "point_tracks",
        "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
        "coordinate_space": "pixel_xy",
        "frame_count": frame_count,
        "max_points": config.max_points,
        "mask_constrained": True,
        "shot_boundary_reset": True,
        "interpolation": False,
        "characters": reports,
    }
    emitted[report_uri] = report
    report_ref = {
        "kind": "point_tracks",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    extension = {
        "enabled": True,
        "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
        "coordinate_space": "pixel_xy",
        "max_points": config.max_points,
        "mask_constrained": True,
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config.token(),
        "characters": rows,
    }
    minimum_coverage = min(coverage_values) if coverage_values else 0.0
    quality = {
        "coverage": round(minimum_coverage, 6),
        "score": round(minimum_coverage, 6),
        "confidence_available": False,
    }
    return extension, emitted, report_ref, quality
