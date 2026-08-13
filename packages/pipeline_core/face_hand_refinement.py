# ruff: noqa: I001
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np


FACE_LANDMARK_COUNT = 478
HAND_LANDMARK_COUNT = 21


class FaceHandRefinementError(RuntimeError):
    """Raised when E3.2 cannot produce trustworthy frame-aligned geometry."""


@dataclass(frozen=True)
class FaceObservation:
    landmarks_xy: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class HandObservation:
    side: Literal["left", "right"]
    landmarks_xy: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    handedness_confidence: float


@dataclass(frozen=True)
class FaceHandObservation:
    frame_idx: int
    character_id: str
    face: FaceObservation | None = None
    hands: tuple[HandObservation, ...] = ()


class FaceHandRefiner(Protocol):
    def refine(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        frame_idx: int,
        character_id: str,
    ) -> FaceHandObservation | None: ...


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, indent=2).encode("utf-8")).hexdigest()


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
        raise FaceHandRefinementError(f"Character {character_id} is missing bbox_ref")
    uri = bbox_ref.get("uri")
    array_key = bbox_ref.get("metadata", {}).get("array_key")
    if not isinstance(uri, str) or not isinstance(array_key, str):
        raise FaceHandRefinementError(f"Character {character_id} has an invalid bbox_ref")
    path = sidecars.get(uri)
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(str(path)):
        raise FaceHandRefinementError(f"Character {character_id} bbox sidecar is unavailable: {uri}")
    with np.load(str(path), allow_pickle=False) as bundle:
        if array_key not in bundle:
            raise FaceHandRefinementError(f"Character {character_id} bbox array is missing: {array_key}")
        bboxes = np.asarray(bundle[array_key], dtype=np.float32)
    if bboxes.shape != (frame_count, 4):
        raise FaceHandRefinementError(
            f"Character {character_id} bbox shape {bboxes.shape} does not match {(frame_count, 4)}"
        )
    return bboxes


def _validate_bbox(
    bbox: tuple[float, float, float, float],
    *,
    label: str,
) -> tuple[float, float, float, float]:
    values = np.asarray(bbox, dtype=np.float32)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise FaceHandRefinementError(f"{label} bbox must contain four finite values")
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        raise FaceHandRefinementError(f"{label} bbox must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def _validate_landmarks(landmarks: np.ndarray, *, count: int, label: str) -> np.ndarray:
    xy = np.asarray(landmarks, dtype=np.float32)
    if xy.shape != (count, 2):
        raise FaceHandRefinementError(f"{label} landmarks must have shape ({count}, 2), got {xy.shape}")
    if not np.all(np.isfinite(xy)):
        raise FaceHandRefinementError(f"{label} observation contains non-finite landmark coordinates")
    return xy


def _validate_confidence(value: float, *, label: str) -> float:
    confidence = float(value)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise FaceHandRefinementError(f"{label} confidence must be finite and within [0, 1]")
    return confidence


def _validate_observation(
    observation: FaceHandObservation,
    *,
    frame_idx: int,
    character_id: str,
) -> tuple[FaceObservation | None, dict[str, HandObservation]]:
    if observation.frame_idx != frame_idx:
        raise FaceHandRefinementError(
            f"Face/hands frame mismatch for {character_id}: expected {frame_idx}, got {observation.frame_idx}"
        )
    if observation.character_id != character_id:
        raise FaceHandRefinementError(
            f"Face/hands character mismatch: expected {character_id}, got {observation.character_id}"
        )

    face = observation.face
    if face is not None:
        _validate_landmarks(face.landmarks_xy, count=FACE_LANDMARK_COUNT, label="Face")
        _validate_bbox(face.bbox_xyxy, label="Face")
        _validate_confidence(face.confidence, label="Face")

    hands: dict[str, HandObservation] = {}
    for hand in observation.hands:
        if hand.side in hands:
            raise FaceHandRefinementError(f"Duplicate {hand.side} hand observation for {character_id}")
        _validate_landmarks(hand.landmarks_xy, count=HAND_LANDMARK_COUNT, label=f"{hand.side} hand")
        _validate_bbox(hand.bbox_xyxy, label=f"{hand.side} hand")
        _validate_confidence(hand.confidence, label=f"{hand.side} hand")
        _validate_confidence(hand.handedness_confidence, label=f"{hand.side} handedness")
        hands[hand.side] = hand
    return face, hands


def _atomic_write_npz(path: str, arrays: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "wb") as handle:
            # NumPy's runtime API accepts a binary file handle plus named arrays;
            # the current numpy typing stub misclassifies **arrays as keyword options.
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_character_sidecar(
    character_id: str,
    *,
    arrays: dict[str, np.ndarray],
    output_dir: str,
) -> tuple[str, str, str]:
    uri = f"artifacts/timeseries/{character_id}_face_hands.npz"
    path = os.path.join(output_dir, uri.replace("/", os.sep))
    _atomic_write_npz(path, arrays)
    return uri, path, _sha256_file(path)


def _quality(*, estimated: int, present: int, confidences: list[float], module: str) -> dict[str, Any]:
    coverage = estimated / present if present else 0.0
    score = float(np.mean(confidences)) if confidences else 0.0
    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "coverage": round(max(0.0, min(1.0, coverage)), 6),
        "warnings": [f"{module}: no temporal interpolation; missing frames remain explicit"],
        "errors": [],
    }


def _disabled_quality(module: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "coverage": 0.0,
        "warnings": [f"{module}: no observations; refs intentionally omitted"],
        "errors": [],
    }


def _person_bbox_from_row(row: np.ndarray, *, character_id: str, frame_idx: int) -> tuple[float, float, float, float]:
    return _validate_bbox(
        (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
        label=f"{character_id} person frame {frame_idx}",
    )


def _new_arrays(frame_count: int) -> dict[str, np.ndarray]:
    return {
        "face_landmarks_2d": np.full((frame_count, FACE_LANDMARK_COUNT, 2), np.nan, dtype=np.float32),
        "face_bbox_xyxy": np.full((frame_count, 4), np.nan, dtype=np.float32),
        "face_confidence": np.zeros((frame_count,), dtype=np.float32),
        "left_hand_landmarks_2d": np.full((frame_count, HAND_LANDMARK_COUNT, 2), np.nan, dtype=np.float32),
        "left_hand_bbox_xyxy": np.full((frame_count, 4), np.nan, dtype=np.float32),
        "left_hand_present": np.zeros((frame_count,), dtype=np.float32),
        "left_handedness_confidence": np.zeros((frame_count,), dtype=np.float32),
        "right_hand_landmarks_2d": np.full((frame_count, HAND_LANDMARK_COUNT, 2), np.nan, dtype=np.float32),
        "right_hand_bbox_xyxy": np.full((frame_count, 4), np.nan, dtype=np.float32),
        "right_hand_present": np.zeros((frame_count,), dtype=np.float32),
        "right_handedness_confidence": np.zeros((frame_count,), dtype=np.float32),
    }


def _face_manifest(
    *,
    frame_count: int,
    estimated: int,
    present: int,
    confidences: list[float],
    uri: str,
    checksum: str,
) -> dict[str, Any]:
    if estimated == 0:
        return {
            "enabled": False,
            "landmark_count": 0,
            "bbox_ref": None,
            "landmarks_2d_ref": None,
            "landmarks_3d_ref": None,
            "blendshapes_ref": None,
            "transform_ref": None,
            "head_pose_ref": None,
            "gaze_ref": None,
            "eye_openness_ref": None,
            "mouth_open_ref": None,
            "blink_events": [],
            "quality": _disabled_quality("face"),
        }

    landmarks_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        shape=[frame_count, FACE_LANDMARK_COUNT, 2],
        axes=["frame", "face_landmark", "xy"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key="face_landmarks_2d",
        nan_policy="preserve",
    )
    landmarks_ref["metadata"].update({"absent_value": "nan", "landmark_model_count": FACE_LANDMARK_COUNT})
    bbox_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        shape=[frame_count, 4],
        axes=["frame", "bbox_component"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key="face_bbox_xyxy",
        nan_policy="preserve",
    )
    bbox_ref["metadata"].update({"bbox_format": "xyxy", "absent_value": "nan"})
    return {
        "enabled": True,
        "landmark_count": FACE_LANDMARK_COUNT,
        "bbox_ref": bbox_ref,
        "landmarks_2d_ref": landmarks_ref,
        "landmarks_3d_ref": None,
        "blendshapes_ref": None,
        "transform_ref": None,
        "head_pose_ref": None,
        "gaze_ref": None,
        "eye_openness_ref": None,
        "mouth_open_ref": None,
        "blink_events": [],
        "quality": _quality(estimated=estimated, present=present, confidences=confidences, module="face"),
    }


def _hand_manifest(
    *,
    side: Literal["left", "right"],
    frame_count: int,
    estimated: int,
    present: int,
    confidences: list[float],
    uri: str,
    checksum: str,
) -> dict[str, Any]:
    if estimated == 0:
        return {
            "enabled": False,
            "landmark_count": 0,
            "present_ref": None,
            "bbox_ref": None,
            "landmarks_2d_ref": None,
            "landmarks_world_ref": None,
            "handedness_ref": None,
            "palm_normal_ref": None,
            "finger_curl_ref": None,
            "quality": _disabled_quality(f"{side} hand"),
        }

    landmarks_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        shape=[frame_count, HAND_LANDMARK_COUNT, 2],
        axes=["frame", "hand_landmark", "xy"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key=f"{side}_hand_landmarks_2d",
        nan_policy="preserve",
    )
    landmarks_ref["metadata"].update({"absent_value": "nan", "landmark_model_count": HAND_LANDMARK_COUNT})
    bbox_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        shape=[frame_count, 4],
        axes=["frame", "bbox_component"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key=f"{side}_hand_bbox_xyxy",
        nan_policy="preserve",
    )
    bbox_ref["metadata"].update({"bbox_format": "xyxy", "absent_value": "nan"})
    present_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        shape=[frame_count],
        axes=["frame"],
        unit="confidence",
        coordinate_space="none",
        frame_count=frame_count,
        array_key=f"{side}_hand_present",
        nan_policy="zero_fill",
    )
    handedness_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        shape=[frame_count],
        axes=["frame"],
        unit="confidence",
        coordinate_space="none",
        frame_count=frame_count,
        array_key=f"{side}_handedness_confidence",
        nan_policy="zero_fill",
    )
    return {
        "enabled": True,
        "landmark_count": HAND_LANDMARK_COUNT,
        "present_ref": present_ref,
        "bbox_ref": bbox_ref,
        "landmarks_2d_ref": landmarks_ref,
        "landmarks_world_ref": None,
        "handedness_ref": handedness_ref,
        "palm_normal_ref": None,
        "finger_curl_ref": None,
        "quality": _quality(
            estimated=estimated,
            present=present,
            confidences=confidences,
            module=f"{side} hand",
        ),
    }


def run_face_hand_refinement(
    video_path: str,
    *,
    characters: list[dict[str, Any]],
    frame_count: int,
    refiner: FaceHandRefiner,
    output_dir: str,
    sidecars: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run model-agnostic E3.2 face/hand geometry on existing anonymous tracks.

    Missing observations stay explicit. Geometry is never interpolated or
    synthesized, and a completely absent face/hand side exposes null refs.
    """
    if not os.path.isfile(video_path):
        raise FaceHandRefinementError(f"Normalized video does not exist: {video_path}")
    if frame_count <= 0:
        raise FaceHandRefinementError("frame_count must be positive")

    bboxes_by_character: dict[str, np.ndarray] = {}
    arrays_by_character: dict[str, dict[str, np.ndarray]] = {}
    present_frames: dict[str, int] = {}
    face_frames: dict[str, int] = {}
    hand_frames: dict[str, dict[str, int]] = {}
    face_confidences: dict[str, list[float]] = {}
    hand_confidences: dict[str, dict[str, list[float]]] = {}

    for character in characters:
        character_id = str(character["character_id"])
        if character_id in bboxes_by_character:
            raise FaceHandRefinementError(f"Duplicate character_id: {character_id}")
        bboxes = _load_character_bboxes(character, frame_count=frame_count, sidecars=sidecars)
        bboxes_by_character[character_id] = bboxes
        arrays_by_character[character_id] = _new_arrays(frame_count)
        present_frames[character_id] = int(np.all(np.isfinite(bboxes), axis=1).sum())
        face_frames[character_id] = 0
        hand_frames[character_id] = {"left": 0, "right": 0}
        face_confidences[character_id] = []
        hand_confidences[character_id] = {"left": [], "right": []}

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FaceHandRefinementError(f"OpenCV could not open normalized video: {video_path}")

    decoded_frames = 0
    try:
        while decoded_frames < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            for character in characters:
                character_id = str(character["character_id"])
                bbox_row = bboxes_by_character[character_id][decoded_frames]
                if not np.all(np.isfinite(bbox_row)):
                    continue
                person_bbox = _person_bbox_from_row(
                    bbox_row,
                    character_id=character_id,
                    frame_idx=decoded_frames,
                )
                observation = refiner.refine(frame, person_bbox, decoded_frames, character_id)
                if observation is None:
                    continue
                face, hands = _validate_observation(
                    observation,
                    frame_idx=decoded_frames,
                    character_id=character_id,
                )
                arrays = arrays_by_character[character_id]
                if face is not None:
                    confidence = _validate_confidence(face.confidence, label="Face")
                    arrays["face_landmarks_2d"][decoded_frames] = _validate_landmarks(
                        face.landmarks_xy, count=FACE_LANDMARK_COUNT, label="Face"
                    )
                    arrays["face_bbox_xyxy"][decoded_frames] = _validate_bbox(face.bbox_xyxy, label="Face")
                    arrays["face_confidence"][decoded_frames] = confidence
                    face_frames[character_id] += 1
                    face_confidences[character_id].append(confidence)
                for side, hand in hands.items():
                    confidence = _validate_confidence(hand.confidence, label=f"{side} hand")
                    handedness = _validate_confidence(
                        hand.handedness_confidence,
                        label=f"{side} handedness",
                    )
                    arrays[f"{side}_hand_landmarks_2d"][decoded_frames] = _validate_landmarks(
                        hand.landmarks_xy,
                        count=HAND_LANDMARK_COUNT,
                        label=f"{side} hand",
                    )
                    arrays[f"{side}_hand_bbox_xyxy"][decoded_frames] = _validate_bbox(
                        hand.bbox_xyxy,
                        label=f"{side} hand",
                    )
                    arrays[f"{side}_hand_present"][decoded_frames] = confidence
                    arrays[f"{side}_handedness_confidence"][decoded_frames] = handedness
                    hand_frames[character_id][side] += 1
                    hand_confidences[character_id][side].append(confidence)
            decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames != frame_count:
        raise FaceHandRefinementError(
            f"Face/hands refiner decoded {decoded_frames} frames; normalized timeline requires {frame_count}"
        )

    refinement_sidecars: dict[str, Any] = {}
    report_characters: list[dict[str, Any]] = []
    for character in characters:
        character_id = str(character["character_id"])
        uri, path, checksum = _write_character_sidecar(
            character_id,
            arrays=arrays_by_character[character_id],
            output_dir=output_dir,
        )
        refinement_sidecars[uri] = path
        present_count = present_frames[character_id]
        character["face"] = _face_manifest(
            frame_count=frame_count,
            estimated=face_frames[character_id],
            present=present_count,
            confidences=face_confidences[character_id],
            uri=uri,
            checksum=checksum,
        )
        character["hands"] = {
            side: _hand_manifest(
                side=side,
                frame_count=frame_count,
                estimated=hand_frames[character_id][side],
                present=present_count,
                confidences=hand_confidences[character_id][side],
                uri=uri,
                checksum=checksum,
            )
            for side in ("left", "right")
        }
        report_characters.append(
            {
                "character_id": character_id,
                "present_frames": present_count,
                "face_frames": face_frames[character_id],
                "left_hand_frames": hand_frames[character_id]["left"],
                "right_hand_frames": hand_frames[character_id]["right"],
            }
        )

    report_uri = "artifacts/reports/face_hands_refinement.json"
    report = {
        "backend_contract": "anonymous_face478_hands21_2d",
        "frame_count": frame_count,
        "face_landmark_count": FACE_LANDMARK_COUNT,
        "hand_landmark_count": HAND_LANDMARK_COUNT,
        "coordinate_space": "pixel_xy",
        "interpolation_policy": "none",
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "characters": report_characters,
    }
    refinement_sidecars[report_uri] = report
    report_ref = {
        "kind": "face_hands_refinement",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    return characters, refinement_sidecars, report_ref
