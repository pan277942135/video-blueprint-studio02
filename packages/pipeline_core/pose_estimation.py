# ruff: noqa: I001
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


COCO17_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class PoseEstimationError(RuntimeError):
    """Raised when E3.1 cannot produce trustworthy frame-aligned pose output."""


@dataclass(frozen=True)
class PoseObservation:
    frame_idx: int
    character_id: str
    keypoints_xy: np.ndarray
    confidence: np.ndarray


class PoseEstimator(Protocol):
    def estimate(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        frame_idx: int,
        character_id: str,
    ) -> PoseObservation | None: ...


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, indent=2).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timeseries_ref(
    *,
    uri: str,
    checksum: str,
    shape: list[int],
    axes: list[str],
    unit: str,
    coordinate_space: str,
    frame_count: int,
    array_key: str,
    nan_policy: str,
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
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "compression": "zip",
        "nan_policy": nan_policy,
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
        "metadata": {"array_key": array_key},
    }


def _load_character_bboxes(
    character: dict[str, Any],
    *,
    frame_count: int,
    sidecars: dict[str, Any],
) -> np.ndarray:
    character_id = str(character["character_id"])
    bbox_ref = character.get("bbox_ref")
    if not isinstance(bbox_ref, dict):
        raise PoseEstimationError(f"Character {character_id} is missing bbox_ref")
    uri = bbox_ref.get("uri")
    array_key = bbox_ref.get("metadata", {}).get("array_key")
    if not isinstance(uri, str) or not isinstance(array_key, str):
        raise PoseEstimationError(f"Character {character_id} has an invalid bbox_ref")
    path = sidecars.get(uri)
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(str(path)):
        raise PoseEstimationError(f"Character {character_id} bbox sidecar is unavailable: {uri}")

    with np.load(str(path), allow_pickle=False) as bundle:
        if array_key not in bundle:
            raise PoseEstimationError(f"Character {character_id} bbox array is missing: {array_key}")
        bboxes = np.asarray(bundle[array_key], dtype=np.float32)
    if bboxes.shape != (frame_count, 4):
        raise PoseEstimationError(
            f"Character {character_id} bbox shape {bboxes.shape} does not match {(frame_count, 4)}"
        )
    return bboxes


def _validate_observation(
    observation: PoseObservation,
    *,
    frame_idx: int,
    character_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    if observation.frame_idx != frame_idx:
        raise PoseEstimationError(
            f"Pose frame mismatch for {character_id}: expected {frame_idx}, got {observation.frame_idx}"
        )
    if observation.character_id != character_id:
        raise PoseEstimationError(
            f"Pose character mismatch: expected {character_id}, got {observation.character_id}"
        )
    keypoints = np.asarray(observation.keypoints_xy, dtype=np.float32)
    confidence = np.asarray(observation.confidence, dtype=np.float32)
    if keypoints.shape != (17, 2):
        raise PoseEstimationError(f"Pose keypoints must have shape (17, 2), got {keypoints.shape}")
    if confidence.shape != (17,):
        raise PoseEstimationError(f"Pose confidence must have shape (17,), got {confidence.shape}")
    if not np.all(np.isfinite(confidence)):
        raise PoseEstimationError("Pose confidence contains non-finite values")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise PoseEstimationError("Pose confidence must be within [0, 1]")
    finite_xy = np.isfinite(keypoints)
    if np.any(finite_xy[:, 0] != finite_xy[:, 1]):
        raise PoseEstimationError("Pose keypoint x/y finite masks differ")
    return keypoints, confidence


def _write_pose_npz(
    character_id: str,
    *,
    keypoints_xy: np.ndarray,
    confidence: np.ndarray,
    output_dir: str,
) -> tuple[str, str, str]:
    uri = f"artifacts/timeseries/{character_id}_pose2d.npz"
    path = os.path.join(output_dir, uri.replace("/", os.sep))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keypoints_xy=keypoints_xy, confidence=confidence)
    return uri, path, _sha256_file(path)


def run_pose_estimation(
    video_path: str,
    *,
    characters: list[dict[str, Any]],
    frame_count: int,
    estimator: PoseEstimator,
    output_dir: str,
    sidecars: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Estimate COCO-17 pose for each anonymous tracked person on present frames only.

    The normalized timeline is preserved exactly. Frames without an E2.2 person
    bounding box, or frames where the estimator returns no usable observation,
    remain NaN for XY and zero for confidence. E3.1 performs no interpolation.
    """
    if not os.path.isfile(video_path):
        raise PoseEstimationError(f"Normalized video does not exist: {video_path}")
    if frame_count <= 0:
        raise PoseEstimationError("frame_count must be positive")

    bboxes_by_character: dict[str, np.ndarray] = {}
    pose_xy: dict[str, np.ndarray] = {}
    pose_confidence: dict[str, np.ndarray] = {}
    present_frames: dict[str, int] = {}
    estimated_frames: dict[str, int] = {}

    for character in characters:
        character_id = str(character["character_id"])
        if character_id in bboxes_by_character:
            raise PoseEstimationError(f"Duplicate character_id: {character_id}")
        bboxes = _load_character_bboxes(character, frame_count=frame_count, sidecars=sidecars)
        bboxes_by_character[character_id] = bboxes
        pose_xy[character_id] = np.full((frame_count, 17, 2), np.nan, dtype=np.float32)
        pose_confidence[character_id] = np.zeros((frame_count, 17), dtype=np.float32)
        present_frames[character_id] = int(np.all(np.isfinite(bboxes), axis=1).sum())
        estimated_frames[character_id] = 0

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise PoseEstimationError(f"OpenCV could not open normalized video: {video_path}")

    decoded_frames = 0
    try:
        while decoded_frames < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            for character in characters:
                character_id = str(character["character_id"])
                bbox = bboxes_by_character[character_id][decoded_frames]
                if not np.all(np.isfinite(bbox)):
                    continue
                x1, y1, x2, y2 = (float(value) for value in bbox)
                if x2 <= x1 or y2 <= y1:
                    raise PoseEstimationError(
                        f"Invalid bbox for {character_id} frame {decoded_frames}: {(x1, y1, x2, y2)}"
                    )
                observation = estimator.estimate(
                    frame,
                    (x1, y1, x2, y2),
                    decoded_frames,
                    character_id,
                )
                if observation is None:
                    continue
                keypoints, confidence = _validate_observation(
                    observation,
                    frame_idx=decoded_frames,
                    character_id=character_id,
                )
                pose_xy[character_id][decoded_frames] = keypoints
                pose_confidence[character_id][decoded_frames] = confidence
                estimated_frames[character_id] += 1
            decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames != frame_count:
        raise PoseEstimationError(
            f"Pose estimator decoded {decoded_frames} frames; normalized timeline requires {frame_count}"
        )

    pose_sidecars: dict[str, Any] = {}
    report_characters: list[dict[str, Any]] = []
    for character in characters:
        character_id = str(character["character_id"])
        xy = pose_xy[character_id]
        confidence = pose_confidence[character_id]
        uri, path, checksum = _write_pose_npz(
            character_id,
            keypoints_xy=xy,
            confidence=confidence,
            output_dir=output_dir,
        )
        pose_sidecars[uri] = path

        present_count = present_frames[character_id]
        estimated_count = estimated_frames[character_id]
        coverage = estimated_count / present_count if present_count else 0.0
        nonzero_confidence = confidence[confidence > 0]
        mean_confidence = float(nonzero_confidence.mean()) if nonzero_confidence.size else 0.0

        keypoints_ref = _timeseries_ref(
            uri=uri,
            checksum=checksum,
            shape=[frame_count, 17, 2],
            axes=["frame", "keypoint", "xy"],
            unit="px",
            coordinate_space="pixel_xy",
            frame_count=frame_count,
            array_key="keypoints_xy",
            nan_policy="preserve",
        )
        keypoints_ref["metadata"].update(
            {
                "skeleton": "coco17",
                "keypoint_names": COCO17_KEYPOINT_NAMES,
                "absent_value": "nan",
            }
        )
        confidence_ref = _timeseries_ref(
            uri=uri,
            checksum=checksum,
            shape=[frame_count, 17],
            axes=["frame", "keypoint"],
            unit="confidence",
            coordinate_space="none",
            frame_count=frame_count,
            array_key="confidence",
            nan_policy="zero_fill",
        )
        character["pose"] = {
            "enabled": True,
            "skeleton_name": "coco17",
            "keypoint_count": 17,
            "keypoint_names": COCO17_KEYPOINT_NAMES,
            "keypoints_2d_ref": keypoints_ref,
            "keypoints_3d_ref": None,
            "root_translation_ref": None,
            "root_rotation_ref": None,
            "joint_angles_ref": None,
            "scale_ref": None,
            "confidence_ref": confidence_ref,
            "smoothing": {"method": "none", "parameters": {}},
            "quality": {
                "score": round(max(0.0, min(1.0, mean_confidence)), 6),
                "coverage": round(max(0.0, min(1.0, coverage)), 6),
                "warnings": ["no temporal interpolation; missing frames remain explicit"],
                "errors": [],
            },
        }
        report_characters.append(
            {
                "character_id": character_id,
                "present_frames": present_count,
                "estimated_frames": estimated_count,
                "coverage": round(max(0.0, min(1.0, coverage)), 6),
                "mean_nonzero_confidence": round(max(0.0, min(1.0, mean_confidence)), 6),
            }
        )

    report_uri = "artifacts/reports/pose_2d.json"
    report = {
        "backend_contract": "topdown_coco17",
        "frame_count": frame_count,
        "keypoint_count": 17,
        "keypoint_names": COCO17_KEYPOINT_NAMES,
        "coordinate_space": "pixel_xy",
        "interpolation_policy": "none",
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "characters": report_characters,
    }
    pose_sidecars[report_uri] = report
    report_ref = {
        "kind": "pose_2d",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    return characters, pose_sidecars, report_ref
