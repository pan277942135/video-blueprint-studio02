from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file


class DenseFlowError(RuntimeError):
    pass


@dataclass(frozen=True)
class DenseFlowConfig:
    max_side_px: int = 320
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    def validate(self) -> None:
        if self.max_side_px < 32:
            raise DenseFlowError("max_side_px must be at least 32")
        if not 0.0 < self.pyr_scale < 1.0:
            raise DenseFlowError("pyr_scale must be within (0, 1)")
        if self.levels <= 0 or self.iterations <= 0:
            raise DenseFlowError("levels and iterations must be positive")
        if self.winsize < 3 or self.winsize % 2 == 0:
            raise DenseFlowError("winsize must be an odd integer >= 3")
        if self.poly_n not in {5, 7}:
            raise DenseFlowError("poly_n must be 5 or 7")
        if self.poly_sigma <= 0:
            raise DenseFlowError("poly_sigma must be positive")
        if self.flags < 0:
            raise DenseFlowError("flags must be non-negative")

    def token(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_environment(cls) -> DenseFlowConfig | None:
        raw = os.environ.get("VBS_E4_DENSE_FLOW_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise DenseFlowError("VBS_E4_DENSE_FLOW_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def _grid_size(width: int, height: int, max_side_px: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise DenseFlowError("source dimensions must be positive")
    scale = min(1.0, max_side_px / max(width, height))
    grid_width = max(1, round(width * scale))
    grid_height = max(1, round(height * scale))
    return grid_width, grid_height


def _resize_gray(frame: np.ndarray, grid_width: int, grid_height: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape == (grid_height, grid_width):
        return gray
    interpolation = cv2.INTER_AREA if grid_width < gray.shape[1] or grid_height < gray.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(gray, (grid_width, grid_height), interpolation=interpolation)


def farneback_step(previous_gray: np.ndarray, gray: np.ndarray, config: DenseFlowConfig) -> np.ndarray:
    config.validate()
    if previous_gray.ndim != 2 or gray.ndim != 2 or previous_gray.shape != gray.shape:
        raise DenseFlowError("Farneback input frames must be equal-size grayscale images")
    initial_flow = np.zeros((*gray.shape, 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        initial_flow,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )
    if flow is None:
        raise DenseFlowError("OpenCV Farneback returned no flow")
    result = np.asarray(flow, dtype=np.float32)
    if result.shape != (*gray.shape, 2) or not np.all(np.isfinite(result)):
        raise DenseFlowError("OpenCV Farneback returned invalid flow")
    return result


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_dense_flow(
    video_path: str,
    *,
    shots: list[dict[str, Any]],
    frame_count: int,
    output_dir: str,
    config: DenseFlowConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Emit global frame-to-frame image-space dense flow as a physical NPZ sidecar."""
    config.validate()
    if not os.path.isfile(video_path) or frame_count <= 0:
        raise DenseFlowError("E4.3 requires a normalized video and positive frame_count")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise DenseFlowError("normalized video could not be opened")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise DenseFlowError("normalized video dimensions are invalid")
    grid_width, grid_height = _grid_size(width, height, config.max_side_px)
    scale_x = width / grid_width
    scale_y = height / grid_height

    shot_starts = {
        int(shot["frame_start"])
        for shot in shots
        if isinstance(shot, dict)
        and isinstance(shot.get("frame_start"), int)
        and 0 < int(shot["frame_start"]) < frame_count
    }
    flow_xy = np.full((frame_count, grid_height, grid_width, 2), np.nan, dtype=np.float32)
    valid_frame = np.zeros(frame_count, dtype=np.bool_)
    previous_gray: np.ndarray | None = None
    decoded = 0
    magnitudes: list[np.ndarray] = []
    try:
        while decoded < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (height, width):
                raise DenseFlowError("normalized frame dimensions changed")
            gray = _resize_gray(frame, grid_width, grid_height)
            if previous_gray is not None and decoded not in shot_starts:
                flow = farneback_step(previous_gray, gray, config)
                flow[..., 0] *= scale_x
                flow[..., 1] *= scale_y
                flow_xy[decoded] = flow
                valid_frame[decoded] = True
                magnitudes.append(np.linalg.norm(flow, axis=2).reshape(-1))
            previous_gray = gray
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count:
        raise DenseFlowError(f"decoded {decoded} frames, expected {frame_count}")

    expected_pairs = max(0, frame_count - 1 - len(shot_starts))
    valid_pairs = int(np.count_nonzero(valid_frame))
    coverage = valid_pairs / expected_pairs if expected_pairs else 0.0
    if magnitudes:
        magnitude_values = np.concatenate(magnitudes)
        median_motion = float(np.median(magnitude_values))
        p95_motion = float(np.percentile(magnitude_values, 95))
    else:
        median_motion = math.nan
        p95_motion = math.nan

    uri = "artifacts/timeseries/source_dense_flow.npz"
    path = os.path.join(output_dir, uri.replace("/", os.sep))
    atomic_npz(path, flow_xy=flow_xy, valid_frame=valid_frame)
    checksum = sha256_file(path)
    ref = {
        "uri": uri,
        "format": "npz",
        "dtype": "float32",
        "shape": [frame_count, grid_height, grid_width, 2],
        "axes": ["frame", "grid_y", "grid_x", "xy"],
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
            "array_key": "flow_xy",
            "valid_frame_array_key": "valid_frame",
            "algorithm": "opencv_farneback_v1",
            "vector_semantics": "previous_frame_to_current_frame_displacement",
            "source_width": width,
            "source_height": height,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "grid_to_source_scale_x": scale_x,
            "grid_to_source_scale_y": scale_y,
            "vectors_scaled_to_source_pixels": True,
            "shot_boundary_reset": True,
            "config_sha256": config.token(),
        },
    }
    report_uri = "artifacts/reports/dense_flow.json"
    report = {
        "stage": "dense_flow",
        "algorithm": "opencv_farneback_v1",
        "coordinate_space": "pixel_xy",
        "vector_unit": "px",
        "frame_count": frame_count,
        "source_width": width,
        "source_height": height,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "valid_pairs": valid_pairs,
        "expected_pairs": expected_pairs,
        "coverage": round(coverage, 6),
        "median_motion_px": round(median_motion, 6) if math.isfinite(median_motion) else None,
        "p95_motion_px": round(p95_motion, 6) if math.isfinite(p95_motion) else None,
        "shot_boundary_reset": True,
        "interpolation": False,
    }
    report_ref = {
        "kind": "dense_flow",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    extension = {
        "enabled": True,
        "algorithm": "opencv_farneback_v1",
        "coordinate_space": "pixel_xy",
        "vector_unit": "px",
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config.token(),
        "flow_ref": ref,
        "quality": {
            "coverage": round(coverage, 6),
            "valid_pairs": valid_pairs,
            "expected_pairs": expected_pairs,
            "median_motion_px": round(median_motion, 6) if math.isfinite(median_motion) else None,
            "p95_motion_px": round(p95_motion, 6) if math.isfinite(p95_motion) else None,
        },
    }
    emitted: dict[str, Any] = {uri: path, report_uri: report}
    quality = {
        "coverage": round(coverage, 6),
        "score": round(coverage, 6),
        "confidence_available": False,
    }
    return extension, emitted, report_ref, quality
