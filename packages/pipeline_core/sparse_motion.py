from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from packages.pipeline_core.binary_rle import decode_rle


class SparseMotionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseMotionConfig:
    max_points: int = 96
    quality_level: float = 0.01
    min_distance_px: float = 5.0
    lk_window_px: int = 21
    lk_max_level: int = 3
    reseed_below_ratio: float = 0.5

    def validate(self) -> None:
        if self.max_points <= 0:
            raise SparseMotionError("max_points must be positive")
        if not 0 < self.quality_level <= 1:
            raise SparseMotionError("quality_level must be within (0, 1]")
        if self.min_distance_px <= 0 or self.lk_window_px < 3 or self.lk_window_px % 2 == 0:
            raise SparseMotionError("invalid feature-tracking geometry")
        if self.lk_max_level < 0 or not 0 <= self.reseed_below_ratio <= 1:
            raise SparseMotionError("invalid LK/reseed configuration")

    def token(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_environment(cls) -> SparseMotionConfig | None:
        raw = os.environ.get("VBS_E4_POINT_TRACKS_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise SparseMotionError("VBS_E4_POINT_TRACKS_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path: str, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_mask_frames(
    character: dict[str, Any], frame_count: int, height: int, width: int, sidecars: dict[str, Any]
) -> list[np.ndarray | None]:
    character_id = str(character["character_id"])
    ref = character.get("person_mask_ref")
    uri = ref.get("uri") if isinstance(ref, dict) else None
    path = sidecars.get(uri) if isinstance(uri, str) else None
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(str(path)):
        raise SparseMotionError(f"{character_id} requires a physical E4.1 person-mask sidecar")
    with open(str(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if [payload.get("frame_count"), payload.get("height"), payload.get("width")] != [frame_count, height, width]:
        raise SparseMotionError(f"{character_id} person-mask dimensions mismatch")
    rows = payload.get("frames")
    if not isinstance(rows, list) or len(rows) != frame_count:
        raise SparseMotionError(f"{character_id} person-mask timeline mismatch")
    result: list[np.ndarray | None] = []
    for frame_idx, row in enumerate(rows):
        if row is None:
            result.append(None)
            continue
        if not isinstance(row, dict) or row.get("size") != [height, width]:
            raise SparseMotionError(f"{character_id} frame {frame_idx} mask metadata mismatch")
        counts = row.get("counts")
        if not isinstance(counts, list) or not all(isinstance(v, int) for v in counts):
            raise SparseMotionError(f"{character_id} frame {frame_idx} mask RLE is invalid")
        mask = decode_rle(counts, height, width)
        if not np.any(mask):
            raise SparseMotionError(f"{character_id} frame {frame_idx} mask is empty")
        result.append(mask)
    return result


def seed_features(gray: np.ndarray, mask: np.ndarray, limit: int, config: SparseMotionConfig) -> np.ndarray:
    if limit <= 0:
        return np.empty((0, 2), dtype=np.float32)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=limit,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance_px,
        mask=mask.astype(np.uint8) * 255,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return points[np.lexsort((points[:, 0], points[:, 1]))]


def lk_step(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray,
    current_mask: np.ndarray,
    config: SparseMotionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        return points.copy(), np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.float32)
    moved, status, errors = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        previous_gray,
        gray,
        points.reshape(-1, 1, 2),
        None,
        winSize=(config.lk_window_px, config.lk_window_px),
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if moved is None or status is None:
        return np.full_like(points, np.nan), np.zeros(len(points), dtype=np.bool_), np.full(len(points), np.nan)
    moved = np.asarray(moved, dtype=np.float32).reshape(-1, 2)
    ok = np.asarray(status).reshape(-1).astype(np.bool_)
    err = np.asarray(errors, dtype=np.float32).reshape(-1) if errors is not None else np.full(len(points), np.nan)
    height, width = current_mask.shape
    for index, (x, y) in enumerate(moved):
        xi, yi = round(float(x)), round(float(y))
        ok[index] = bool(
            ok[index]
            and math.isfinite(float(x))
            and math.isfinite(float(y))
            and 0 <= xi < width
            and 0 <= yi < height
            and current_mask[yi, xi]
        )
    return moved, ok, err
