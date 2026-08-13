from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import cv2
import numpy as np

from packages.pipeline_core.face_hand_refinement import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    FaceHandObservation,
    FaceObservation,
    HandObservation,
)


APPROVED_FACE_TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
APPROVED_HAND_TASK_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
APPROVED_MEDIAPIPE_VERSION = "0.10.35"


class MediaPipeFaceHandConfigurationError(RuntimeError):
    """Raised when E3.2 MediaPipe runtime/artifacts are absent, altered, or invalid."""


@dataclass(frozen=True)
class MediaPipeFaceHandConfig:
    face_task_path: str
    hand_task_path: str
    fps_num: int
    fps_den: int
    crop_padding_ratio: float = 0.08
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_face_tracking_confidence: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5


@dataclass
class _CharacterTasks:
    face_landmarker: Any
    hand_landmarker: Any
    last_timestamp_ms: int = -1


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise MediaPipeFaceHandConfigurationError("Expected SHA256 must contain exactly 64 hexadecimal characters")
    return normalized


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _config_hash(config: MediaPipeFaceHandConfig, face_sha256: str, hand_sha256: str) -> str:
    payload = "|".join(
        [
            f"face:{face_sha256}",
            f"hand:{hand_sha256}",
            f"fps:{config.fps_num}/{config.fps_den}",
            f"padding:{config.crop_padding_ratio}",
            f"face-thresholds:{config.min_face_detection_confidence}:{config.min_face_presence_confidence}:{config.min_face_tracking_confidence}",
            f"hand-thresholds:{config.min_hand_detection_confidence}:{config.min_hand_presence_confidence}:{config.min_hand_tracking_confidence}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_config(config: MediaPipeFaceHandConfig) -> None:
    if config.fps_num <= 0 or config.fps_den <= 0:
        raise MediaPipeFaceHandConfigurationError("MediaPipe E3.2 requires a positive rational FPS")
    if not 0.0 <= config.crop_padding_ratio <= 0.5:
        raise MediaPipeFaceHandConfigurationError("crop_padding_ratio must be within [0, 0.5]")
    for label, value in (
        ("min_face_detection_confidence", config.min_face_detection_confidence),
        ("min_face_presence_confidence", config.min_face_presence_confidence),
        ("min_face_tracking_confidence", config.min_face_tracking_confidence),
        ("min_hand_detection_confidence", config.min_hand_detection_confidence),
        ("min_hand_presence_confidence", config.min_hand_presence_confidence),
        ("min_hand_tracking_confidence", config.min_hand_tracking_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise MediaPipeFaceHandConfigurationError(f"{label} must be within [0, 1]")


def _clip_crop(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    padding_ratio: float,
) -> tuple[np.ndarray, int, int]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        raise MediaPipeFaceHandConfigurationError("Anonymous person bbox is invalid for MediaPipe refinement")
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise MediaPipeFaceHandConfigurationError("Anonymous person bbox produced an empty MediaPipe crop")
    return frame[top:bottom, left:right], left, top


def _pixel_landmarks(
    landmarks: list[Any],
    *,
    expected_count: int,
    crop_width: int,
    crop_height: int,
    offset_x: int,
    offset_y: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    if len(landmarks) != expected_count:
        raise MediaPipeFaceHandConfigurationError(
            f"MediaPipe returned {len(landmarks)} landmarks; expected {expected_count}"
        )
    points = np.empty((expected_count, 2), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        x = float(getattr(landmark, "x", np.nan))
        y = float(getattr(landmark, "y", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            raise MediaPipeFaceHandConfigurationError("MediaPipe returned a non-finite landmark coordinate")
        points[index, 0] = np.clip(offset_x + x * crop_width, 0.0, max(0.0, frame_width - 1.0))
        points[index, 1] = np.clip(offset_y + y * crop_height, 0.0, max(0.0, frame_height - 1.0))
    return points


def _bbox_from_points(points: np.ndarray) -> tuple[float, float, float, float]:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    x1, y1 = float(minimum[0]), float(minimum[1])
    x2, y2 = float(maximum[0]), float(maximum[1])
    if x2 <= x1 or y2 <= y1:
        raise MediaPipeFaceHandConfigurationError("MediaPipe landmarks produced a degenerate geometry bbox")
    return x1, y1, x2, y2


def _conservative_presence_score(landmarks: list[Any], threshold_lower_bound: float) -> float:
    """Return measured landmark presence when exposed, otherwise a conservative lower bound.

    MediaPipe Tasks does not expose a dedicated frame-level face/hand detection
    score in its result object. A result is emitted only after the configured
    presence gate succeeds, so the configured threshold is recorded as a lower
    bound rather than fabricating an unobserved 1.0 score.
    """
    values: list[float] = []
    for landmark in landmarks:
        raw = getattr(landmark, "presence", None)
        if raw is None:
            continue
        value = float(raw)
        if np.isfinite(value):
            values.append(float(np.clip(value, 0.0, 1.0)))
    if values:
        return float(np.mean(values))
    return float(threshold_lower_bound)


class MediaPipeFaceHandRefiner:
    """Approved MediaPipe Tasks adapter for anonymous 2D face/hand geometry only.

    One VIDEO-mode task pair is maintained per anonymous character track so
    MediaPipe sees strictly increasing timestamps even when multiple people are
    processed in the same source frame. No identity, embedding, blendshape,
    transformation-matrix, world-landmark, or sensitive-attribute output is
    requested or exported.
    """

    def __init__(
        self,
        config: MediaPipeFaceHandConfig,
        *,
        artifacts_approved: bool,
        mediapipe_module: Any | None = None,
    ) -> None:
        if not artifacts_approved:
            raise MediaPipeFaceHandConfigurationError(
                "Pinned MediaPipe Face/Hand task artifacts require explicit project-owner approval before inference"
            )
        _validate_config(config)
        if not os.path.isfile(config.face_task_path):
            raise MediaPipeFaceHandConfigurationError(f"FaceLandmarker task does not exist: {config.face_task_path}")
        if not os.path.isfile(config.hand_task_path):
            raise MediaPipeFaceHandConfigurationError(f"HandLandmarker task does not exist: {config.hand_task_path}")

        face_sha256 = _normalize_sha256(_sha256_file(config.face_task_path))
        hand_sha256 = _normalize_sha256(_sha256_file(config.hand_task_path))
        if face_sha256 != APPROVED_FACE_TASK_SHA256:
            raise MediaPipeFaceHandConfigurationError(
                f"FaceLandmarker task SHA256 mismatch: expected {APPROVED_FACE_TASK_SHA256}, got {face_sha256}"
            )
        if hand_sha256 != APPROVED_HAND_TASK_SHA256:
            raise MediaPipeFaceHandConfigurationError(
                f"HandLandmarker task SHA256 mismatch: expected {APPROVED_HAND_TASK_SHA256}, got {hand_sha256}"
            )

        if mediapipe_module is None:
            try:
                import mediapipe as mediapipe_module
            except ImportError as exc:
                raise MediaPipeFaceHandConfigurationError(
                    f"MediaPipe {APPROVED_MEDIAPIPE_VERSION} runtime is not installed; E3.2 fails closed"
                ) from exc
        runtime_version = str(getattr(mediapipe_module, "__version__", "unknown"))
        if runtime_version != APPROVED_MEDIAPIPE_VERSION:
            raise MediaPipeFaceHandConfigurationError(
                f"MediaPipe runtime version must be {APPROVED_MEDIAPIPE_VERSION}, got {runtime_version}"
            )

        self.config = config
        self.face_task_sha256 = face_sha256
        self.hand_task_sha256 = hand_sha256
        self.config_sha256 = _config_hash(config, face_sha256, hand_sha256)
        self.weights_sha256 = hashlib.sha256(f"{face_sha256}:{hand_sha256}".encode("utf-8")).hexdigest()
        self._mp = mediapipe_module
        self._states: dict[str, _CharacterTasks] = {}

    @classmethod
    def from_environment(
        cls,
        *,
        fps_num: int,
        fps_den: int,
    ) -> MediaPipeFaceHandRefiner | None:
        face_task_path = os.environ.get("VBS_MEDIAPIPE_FACE_TASK")
        hand_task_path = os.environ.get("VBS_MEDIAPIPE_HAND_TASK")
        approval_raw = os.environ.get("VBS_MEDIAPIPE_TASKS_APPROVED")
        configured_values = [face_task_path, hand_task_path, approval_raw]
        if all(value is None for value in configured_values):
            return None
        if any(value is None for value in configured_values):
            raise MediaPipeFaceHandConfigurationError(
                "Incomplete E3.2 MediaPipe configuration: VBS_MEDIAPIPE_FACE_TASK, VBS_MEDIAPIPE_HAND_TASK, "
                "and VBS_MEDIAPIPE_TASKS_APPROVED are all required"
            )
        approved = str(approval_raw).strip().lower() in {"1", "true", "yes"}
        return cls(
            MediaPipeFaceHandConfig(
                face_task_path=str(face_task_path),
                hand_task_path=str(hand_task_path),
                fps_num=fps_num,
                fps_den=fps_den,
            ),
            artifacts_approved=approved,
        )

    def _new_state(self) -> _CharacterTasks:
        vision = self._mp.tasks.vision
        base_options = self._mp.tasks.BaseOptions
        face_options = vision.FaceLandmarkerOptions(
            base_options=base_options(model_asset_path=self.config.face_task_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.min_face_detection_confidence,
            min_face_presence_confidence=self.config.min_face_presence_confidence,
            min_tracking_confidence=self.config.min_face_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=self.config.hand_task_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_hand_tracking_confidence,
        )
        return _CharacterTasks(
            face_landmarker=vision.FaceLandmarker.create_from_options(face_options),
            hand_landmarker=vision.HandLandmarker.create_from_options(hand_options),
        )

    def _timestamp_ms(self, frame_idx: int, state: _CharacterTasks) -> int:
        timestamp = int(round(frame_idx * 1000.0 * self.config.fps_den / self.config.fps_num))
        if timestamp <= state.last_timestamp_ms:
            timestamp = state.last_timestamp_ms + 1
        state.last_timestamp_ms = timestamp
        return timestamp

    def refine(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        frame_idx: int,
        character_id: str,
    ) -> FaceHandObservation | None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise MediaPipeFaceHandConfigurationError("MediaPipe E3.2 expects BGR uint8 video frames")
        crop, offset_x, offset_y = _clip_crop(frame, bbox_xyxy, self.config.crop_padding_ratio)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        state = self._states.get(character_id)
        if state is None:
            state = self._new_state()
            self._states[character_id] = state
        timestamp_ms = self._timestamp_ms(frame_idx, state)
        face_result = state.face_landmarker.detect_for_video(image, timestamp_ms)
        hand_result = state.hand_landmarker.detect_for_video(image, timestamp_ms)

        frame_height, frame_width = frame.shape[:2]
        crop_height, crop_width = crop.shape[:2]
        face: FaceObservation | None = None
        face_landmarks = list(getattr(face_result, "face_landmarks", []) or [])
        if len(face_landmarks) > 1:
            raise MediaPipeFaceHandConfigurationError("E3.2 FaceLandmarker returned more than one face for one anonymous track")
        if face_landmarks:
            landmarks = list(face_landmarks[0])
            points = _pixel_landmarks(
                landmarks,
                expected_count=FACE_LANDMARK_COUNT,
                crop_width=crop_width,
                crop_height=crop_height,
                offset_x=offset_x,
                offset_y=offset_y,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            face = FaceObservation(
                landmarks_xy=points,
                bbox_xyxy=_bbox_from_points(points),
                confidence=_conservative_presence_score(landmarks, self.config.min_face_presence_confidence),
            )

        hand_landmarks = list(getattr(hand_result, "hand_landmarks", []) or [])
        handedness = list(getattr(hand_result, "handedness", []) or [])
        if len(hand_landmarks) != len(handedness):
            raise MediaPipeFaceHandConfigurationError("MediaPipe hand landmarks and handedness counts do not match")
        best_by_side: dict[str, HandObservation] = {}
        for landmarks_raw, categories_raw in zip(hand_landmarks, handedness, strict=True):
            landmarks = list(landmarks_raw)
            categories = list(categories_raw)
            if not categories:
                raise MediaPipeFaceHandConfigurationError("MediaPipe hand result is missing handedness classification")
            category = max(categories, key=lambda item: float(getattr(item, "score", 0.0)))
            category_name = str(getattr(category, "category_name", "")).strip().lower()
            if category_name not in {"left", "right"}:
                raise MediaPipeFaceHandConfigurationError(f"Unexpected MediaPipe handedness category: {category_name!r}")
            handedness_score = float(getattr(category, "score", np.nan))
            if not np.isfinite(handedness_score):
                raise MediaPipeFaceHandConfigurationError("MediaPipe handedness score is non-finite")
            handedness_score = float(np.clip(handedness_score, 0.0, 1.0))
            points = _pixel_landmarks(
                landmarks,
                expected_count=HAND_LANDMARK_COUNT,
                crop_width=crop_width,
                crop_height=crop_height,
                offset_x=offset_x,
                offset_y=offset_y,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            observation = HandObservation(
                side=category_name,
                landmarks_xy=points,
                bbox_xyxy=_bbox_from_points(points),
                confidence=_conservative_presence_score(landmarks, self.config.min_hand_presence_confidence),
                handedness_confidence=handedness_score,
            )
            previous = best_by_side.get(category_name)
            if previous is None or observation.handedness_confidence > previous.handedness_confidence:
                best_by_side[category_name] = observation

        if face is None and not best_by_side:
            return None
        return FaceHandObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            face=face,
            hands=tuple(best_by_side[side] for side in ("left", "right") if side in best_by_side),
        )

    def close(self) -> None:
        states = list(self._states.values())
        self._states.clear()
        errors: list[Exception] = []
        for state in states:
            for task in (state.face_landmarker, state.hand_landmarker):
                try:
                    task.close()
                except Exception as exc:  # pragma: no cover - defensive cleanup path
                    errors.append(exc)
        if errors:
            raise MediaPipeFaceHandConfigurationError(f"MediaPipe E3.2 task cleanup failed: {errors[0]}")

    def provenance(self, *, code_commit: str | None, config_hash: str) -> dict[str, Any]:
        return {
            "module": "face_hands_2d",
            "tool": "MediaPipe FaceLandmarker + HandLandmarker",
            "version": _package_version("mediapipe"),
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": "Apache-2.0 / project-approved pinned model artifacts",
        }
