from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from packages.pipeline_core.sparse_motion import atomic_npz, load_mask_frames, sha256_file


class CameraMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraMotionConfig:
    max_points: int = 400
    quality_level: float = 0.01
    min_distance_px: float = 7.0
    lk_window_px: int = 21
    lk_max_level: int = 3
    ransac_reproj_threshold_px: float = 2.5
    min_matches: int = 12
    foreground_dilate_px: int = 7
    static_translation_diag_ratio: float = 0.001
    roll_threshold_deg: float = 0.10
    zoom_threshold_ratio: float = 0.002

    def validate(self) -> None:
        if self.max_points < self.min_matches or self.min_matches < 3:
            raise CameraMotionError("camera tracking requires max_points >= min_matches >= 3")
        if not 0.0 < self.quality_level <= 1.0:
            raise CameraMotionError("quality_level must be within (0, 1]")
        if self.min_distance_px <= 0 or self.lk_window_px < 3 or self.lk_window_px % 2 == 0:
            raise CameraMotionError("invalid camera feature-tracking geometry")
        if self.lk_max_level < 0 or self.ransac_reproj_threshold_px <= 0:
            raise CameraMotionError("invalid camera LK/RANSAC configuration")
        if self.foreground_dilate_px < 0:
            raise CameraMotionError("foreground_dilate_px must be non-negative")

    def token(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_environment(cls) -> CameraMotionConfig | None:
        raw = os.environ.get("VBS_E5_CAMERA_MOTION_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise CameraMotionError("VBS_E5_CAMERA_MOTION_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, indent=2, sort_keys=True).encode("utf-8")).hexdigest()


def _background_masks(
    characters: list[dict[str, Any]],
    *,
    frame_count: int,
    height: int,
    width: int,
    sidecars: dict[str, Any],
    dilate_px: int,
) -> tuple[list[np.ndarray], bool]:
    foreground = [np.zeros((height, width), dtype=np.bool_) for _ in range(frame_count)]
    used = False
    for character in characters:
        if not isinstance(character.get("person_mask_ref"), dict):
            continue
        rows = load_mask_frames(character, frame_count, height, width, sidecars)
        for frame_idx, mask in enumerate(rows):
            if mask is not None:
                foreground[frame_idx] |= mask
                used = True
    kernel = None
    if dilate_px > 0:
        size = dilate_px * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
    result: list[np.ndarray] = []
    for mask in foreground:
        expanded = mask
        if kernel is not None and np.any(mask):
            expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
        result.append(~expanded)
    return result, used


def _seed(gray: np.ndarray, background: np.ndarray, limit: int, config: CameraMotionConfig) -> np.ndarray:
    if limit <= 0:
        return np.empty((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=limit,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance_px,
        mask=background.astype(np.uint8) * 255,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return points[np.lexsort((points[:, 0], points[:, 1]))]


def _lk(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray,
    background: np.ndarray,
    config: CameraMotionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points.copy(), np.empty(0, dtype=np.bool_)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        previous_gray,
        gray,
        points.reshape(-1, 1, 2),
        None,
        winSize=(config.lk_window_px, config.lk_window_px),
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if moved is None or status is None:
        return np.full_like(points, np.nan), np.zeros(len(points), dtype=np.bool_)
    moved = np.asarray(moved, dtype=np.float32).reshape(-1, 2)
    ok = np.asarray(status).reshape(-1).astype(np.bool_)
    height, width = background.shape
    for index, (x, y) in enumerate(moved):
        xi, yi = round(float(x)), round(float(y))
        ok[index] = bool(
            ok[index]
            and math.isfinite(float(x))
            and math.isfinite(float(y))
            and 0 <= xi < width
            and 0 <= yi < height
            and background[yi, xi]
        )
    return moved, ok


def estimate_similarity_ransac(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    config: CameraMotionConfig,
) -> tuple[np.ndarray | None, float | None]:
    config.validate()
    if previous_points.shape != current_points.shape or previous_points.ndim != 2 or previous_points.shape[1:] != (2,):
        raise CameraMotionError("camera correspondences must be equal [N,2] arrays")
    if len(previous_points) < config.min_matches:
        return None, None
    matrix, inliers = cv2.estimateAffinePartial2D(
        previous_points.astype(np.float32),
        current_points.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold_px,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return None, None
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise CameraMotionError("RANSAC returned an invalid affine matrix")
    return matrix, float(np.mean(np.asarray(inliers).reshape(-1) != 0))


def classify_camera_motion(
    matrices: np.ndarray,
    *,
    width: int,
    height: int,
    config: CameraMotionConfig,
) -> str:
    valid = matrices[np.all(np.isfinite(matrices), axis=(1, 2))]
    if len(valid) == 0:
        return "unknown"
    tx = float(np.median(valid[:, 0, 2]))
    ty = float(np.median(valid[:, 1, 2]))
    scales = np.sqrt(valid[:, 0, 0] ** 2 + valid[:, 1, 0] ** 2)
    scale_delta = abs(float(np.median(scales)) - 1.0)
    rotations = np.degrees(np.arctan2(valid[:, 1, 0], valid[:, 0, 0]))
    rotation = abs(float(np.median(rotations)))
    translation_threshold = math.hypot(width, height) * config.static_translation_diag_ratio
    horizontal = abs(tx) > translation_threshold
    vertical = abs(ty) > translation_threshold
    roll = rotation > config.roll_threshold_deg
    zoom = scale_delta > config.zoom_threshold_ratio
    if not any((horizontal, vertical, roll, zoom)):
        return "static"
    active = sum((horizontal, vertical, roll, zoom))
    if active > 1:
        if horizontal and vertical and not roll and not zoom:
            if abs(tx) >= 1.5 * abs(ty):
                return "pan"
            if abs(ty) >= 1.5 * abs(tx):
                return "tilt"
        return "compound"
    if horizontal:
        return "pan"
    if vertical:
        return "tilt"
    if roll:
        return "roll"
    return "zoom"


def _ref(
    *,
    uri: str,
    checksum: str,
    shape: list[int],
    axes: list[str],
    unit: str,
    coordinate_space: str,
    frame_start: int,
    frame_end: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "uri": uri,
        "format": "npz",
        "dtype": "float32",
        "shape": shape,
        "axes": axes,
        "unit": unit,
        "coordinate_space": coordinate_space,
        "sampling": "per_frame",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": metadata,
    }


def run_camera_motion(
    video_path: str,
    *,
    shots: list[dict[str, Any]],
    characters: list[dict[str, Any]],
    frame_count: int,
    output_dir: str,
    sidecars: dict[str, Any],
    config: CameraMotionConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config.validate()
    if not os.path.isfile(video_path) or frame_count <= 0:
        raise CameraMotionError("E5 requires a normalized video and positive frame_count")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise CameraMotionError("normalized video could not be opened")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise CameraMotionError("normalized video dimensions are invalid")
    background, foreground_exclusion = _background_masks(
        characters,
        frame_count=frame_count,
        height=height,
        width=width,
        sidecars=sidecars,
        dilate_px=config.foreground_dilate_px,
    )

    emitted: dict[str, Any] = {}
    camera_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    total_expected = 0
    total_valid = 0
    confidence_values: list[float] = []

    for shot in shots:
        if not isinstance(shot, dict):
            continue
        frame_start = int(shot["frame_start"])
        frame_end = int(shot["frame_end"])
        shot_id = str(shot["shot_id"])
        if frame_start < 0 or frame_end < frame_start or frame_end >= frame_count:
            raise CameraMotionError(f"shot {shot_id} has invalid frame bounds")
        length = frame_end - frame_start + 1
        expected_pairs = max(0, length - 1)
        total_expected += expected_pairs
        points_xy = np.full((length, config.max_points, 2), np.nan, dtype=np.float32)
        valid_points = np.zeros((length, config.max_points), dtype=np.bool_)
        affine = np.full((length, 2, 3), np.nan, dtype=np.float32)
        stabilization = np.full((length, 2, 3), np.nan, dtype=np.float32)
        zoom_proxy = np.full(length, np.nan, dtype=np.float32)
        inlier_ratio = np.full(length, np.nan, dtype=np.float32)
        stabilization[0] = np.eye(3, dtype=np.float32)[:2]
        zoom_proxy[0] = 1.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise CameraMotionError(f"could not open normalized video for shot {shot_id}")
        if frame_start > 0 and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start):
            cap.release()
            raise CameraMotionError(f"could not seek normalized video to shot {shot_id}")
        previous_gray: np.ndarray | None = None
        active_slots = np.empty(0, dtype=np.int32)
        active_points = np.empty((0, 2), dtype=np.float32)
        cumulative = np.eye(3, dtype=np.float64)
        valid_pairs = 0
        inliers_for_quality: list[float] = []
        try:
            for local_idx in range(length):
                ok, frame = cap.read()
                if not ok or frame is None or frame.shape[:2] != (height, width):
                    raise CameraMotionError(f"shot {shot_id} frame decode failed")
                global_idx = frame_start + local_idx
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if previous_gray is None:
                    seeds = _seed(gray, background[global_idx], config.max_points, config)
                    active_slots = np.arange(len(seeds), dtype=np.int32)
                    active_points = seeds
                    if len(seeds):
                        points_xy[local_idx, active_slots] = seeds
                        valid_points[local_idx, active_slots] = True
                    previous_gray = gray
                    continue

                previous_points = active_points.copy()
                previous_slots = active_slots.copy()
                moved, ok_mask = _lk(previous_gray, gray, previous_points, background[global_idx], config)
                good_previous = previous_points[ok_mask]
                good_current = moved[ok_mask]
                good_slots = previous_slots[ok_mask]
                matrix, ratio = estimate_similarity_ransac(good_previous, good_current, config)
                if matrix is not None and ratio is not None:
                    affine[local_idx] = matrix
                    inlier_ratio[local_idx] = ratio
                    valid_pairs += 1
                    total_valid += 1
                    inliers_for_quality.append(ratio)
                    step = np.eye(3, dtype=np.float64)
                    step[:2] = matrix
                    cumulative = step @ cumulative
                    inverse = np.linalg.inv(cumulative)
                    stabilization[local_idx] = inverse[:2].astype(np.float32)
                    zoom_proxy[local_idx] = float(math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
                if len(good_slots):
                    points_xy[local_idx, good_slots] = good_current
                    valid_points[local_idx, good_slots] = True

                active_slots = good_slots
                active_points = good_current
                free_slots = np.setdiff1d(np.arange(config.max_points, dtype=np.int32), active_slots, assume_unique=True)
                if len(free_slots):
                    seed_mask_u8 = background[global_idx].astype(np.uint8) * 255
                    for x, y in active_points:
                        cv2.circle(
                            seed_mask_u8,
                            (round(float(x)), round(float(y))),
                            round(config.min_distance_px),
                            0,
                            -1,
                        )
                    seeds = _seed(gray, seed_mask_u8 != 0, len(free_slots), config)
                    if len(seeds):
                        new_slots = free_slots[: len(seeds)]
                        active_slots = np.concatenate((active_slots, new_slots))
                        active_points = np.concatenate((active_points, seeds), axis=0)
                        points_xy[local_idx, new_slots] = seeds
                        valid_points[local_idx, new_slots] = True
                previous_gray = gray
        finally:
            cap.release()

        coverage = valid_pairs / expected_pairs if expected_pairs else 1.0
        median_inlier = float(np.median(inliers_for_quality)) if inliers_for_quality else 0.0
        confidence = float(max(0.0, min(1.0, coverage * median_inlier)))
        confidence_values.append(confidence)
        classification = classify_camera_motion(affine, width=width, height=height, config=config)
        failure_reason = None
        if expected_pairs and valid_pairs == 0:
            failure_reason = "insufficient_background_correspondences"
            classification = "unknown"

        camera_motion_id = f"camera-{shot_id}"
        uri = f"artifacts/timeseries/{camera_motion_id}_camera_2d.npz"
        path = os.path.join(output_dir, uri.replace("/", os.sep))
        atomic_npz(
            path,
            background_points_xy=points_xy,
            background_valid=valid_points,
            frame_to_frame_affine=affine,
            stabilization_affine=stabilization,
            zoom_proxy=zoom_proxy,
            ransac_inlier_ratio=inlier_ratio,
        )
        checksum = sha256_file(path)
        common = {
            "algorithm": "opencv_background_lk_ransac_affine_v1",
            "config_sha256": config.token(),
            "shot_boundary_reset": True,
            "foreground_exclusion": foreground_exclusion,
        }
        affine_ref = _ref(
            uri=uri,
            checksum=checksum,
            shape=[length, 2, 3],
            axes=["frame", "matrix_row", "matrix_col"],
            unit="mixed_pixel_affine",
            coordinate_space="pixel_xy",
            frame_start=frame_start,
            frame_end=frame_end,
            metadata={
                **common,
                "array_key": "frame_to_frame_affine",
                "stabilization_array_key": "stabilization_affine",
                "matrix_semantics": "previous_frame_to_current_frame_affine",
            },
        )
        background_ref = _ref(
            uri=uri,
            checksum=checksum,
            shape=[length, config.max_points, 2],
            axes=["frame", "background_track", "xy"],
            unit="px",
            coordinate_space="pixel_xy",
            frame_start=frame_start,
            frame_end=frame_end,
            metadata={
                **common,
                "array_key": "background_points_xy",
                "valid_array_key": "background_valid",
                "point_slot_semantics": "persistent_within_shot_reseedable_slot",
            },
        )
        zoom_ref = _ref(
            uri=uri,
            checksum=checksum,
            shape=[length],
            axes=["frame"],
            unit="ratio",
            coordinate_space="none",
            frame_start=frame_start,
            frame_end=frame_end,
            metadata={
                **common,
                "array_key": "zoom_proxy",
                "inlier_ratio_array_key": "ransac_inlier_ratio",
            },
        )
        shot["camera_motion_id"] = camera_motion_id
        camera_rows.append(
            {
                "camera_motion_id": camera_motion_id,
                "shot_id": shot_id,
                "classification": classification,
                "affine_ref": affine_ref,
                "homography_ref": None,
                "crop_ref": None,
                "zoom_proxy_ref": zoom_ref,
                "shake_ref": None,
                "background_tracks_ref": background_ref,
                "intrinsics": None,
                "extrinsics_ref": None,
                "reconstruction_backend": "opencv_ransac_2d",
                "confidence": round(confidence, 6),
                "failure_reason": failure_reason,
            }
        )
        emitted[uri] = path
        report_rows.append(
            {
                "shot_id": shot_id,
                "camera_motion_id": camera_motion_id,
                "classification": classification,
                "expected_pairs": expected_pairs,
                "valid_pairs": valid_pairs,
                "coverage": round(coverage, 6),
                "median_ransac_inlier_ratio": round(median_inlier, 6),
                "confidence": round(confidence, 6),
                "failure_reason": failure_reason,
            }
        )

    overall_coverage = total_valid / total_expected if total_expected else 1.0
    score = min(confidence_values, default=0.0)
    camera = {
        "per_shot": camera_rows,
        "quality": {
            "score": round(score, 6),
            "coverage": round(overall_coverage, 6),
            "warnings": [] if foreground_exclusion else ["person-mask foreground exclusion unavailable"],
            "errors": [],
        },
    }
    report_uri = "artifacts/reports/camera_motion.json"
    report = {
        "stage": "camera_motion",
        "algorithm": "opencv_background_lk_ransac_affine_v1",
        "reconstruction_backend": "opencv_ransac_2d",
        "frame_count": frame_count,
        "foreground_exclusion": foreground_exclusion,
        "classification_scope": ["static", "pan", "tilt", "roll", "zoom", "compound", "unknown"],
        "unsupported_3d_labels": ["dolly", "truck", "pedestal"],
        "shots": report_rows,
    }
    emitted[report_uri] = report
    report_ref = {
        "kind": "camera_motion",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    extension = {
        "enabled": True,
        "algorithm": "opencv_background_lk_ransac_affine_v1",
        "reconstruction_backend": "opencv_ransac_2d",
        "foreground_exclusion": foreground_exclusion,
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config.token(),
        "shot_count": len(camera_rows),
        "supported_classifications": ["static", "pan", "tilt", "roll", "zoom", "compound", "unknown"],
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    quality = {
        "coverage": round(overall_coverage, 6),
        "score": round(score, 6),
        "confidence_available": True,
    }
    return camera, emitted, report_ref, extension, quality
