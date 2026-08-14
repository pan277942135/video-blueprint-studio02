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


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
ANCHOR_NAMES = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


class BodyLocalFrameError(RuntimeError):
    pass


@dataclass(frozen=True)
class BodyLocalFrameConfig:
    anchor_confidence_threshold: float = 0.30
    min_torso_scale_px: float = 8.0

    def validate(self) -> None:
        if not 0.0 <= self.anchor_confidence_threshold <= 1.0:
            raise BodyLocalFrameError("anchor_confidence_threshold must be within [0, 1]")
        if self.min_torso_scale_px <= 0:
            raise BodyLocalFrameError("min_torso_scale_px must be positive")

    def token(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_environment(cls) -> BodyLocalFrameConfig | None:
        raw = os.environ.get("VBS_E6_BODY_LOCAL_FRAME_ENABLED")
        if raw is None or raw.strip().lower() == "false":
            return None
        if raw.strip().lower() != "true":
            raise BodyLocalFrameError("VBS_E6_BODY_LOCAL_FRAME_ENABLED must be 'true' or 'false'")
        value = cls()
        value.validate()
        return value


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, indent=2, sort_keys=True).encode()).hexdigest()


def _load_npz_array(ref: dict[str, Any], sidecars: dict[str, Any], expected_key: str) -> np.ndarray:
    uri = ref.get("uri")
    metadata = ref.get("metadata")
    key = metadata.get("array_key") if isinstance(metadata, dict) else None
    if not isinstance(uri, str) or key != expected_key:
        raise BodyLocalFrameError(f"invalid TimeSeriesRef for {expected_key}")
    path = sidecars.get(uri)
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(str(path)):
        raise BodyLocalFrameError(f"physical sidecar unavailable for {uri}")
    with np.load(str(path), allow_pickle=False) as arrays:
        if expected_key not in arrays:
            raise BodyLocalFrameError(f"array {expected_key!r} missing from {uri}")
        return np.asarray(arrays[expected_key])


def _load_pose(character: dict[str, Any], frame_count: int, sidecars: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pose = character.get("pose")
    character_id = str(character.get("character_id"))
    if not isinstance(pose, dict) or pose.get("enabled") is not True or pose.get("skeleton_name") != "coco17":
        raise BodyLocalFrameError(f"character {character_id} requires enabled COCO-17 pose for body-local frame")
    keypoints_ref = pose.get("keypoints_2d_ref")
    confidence_ref = pose.get("confidence_ref")
    if not isinstance(keypoints_ref, dict) or not isinstance(confidence_ref, dict):
        raise BodyLocalFrameError(f"character {character_id} pose refs are missing")
    keypoints = _load_npz_array(keypoints_ref, sidecars, "keypoints_xy").astype(np.float32, copy=False)
    confidence = _load_npz_array(confidence_ref, sidecars, "confidence").astype(np.float32, copy=False)
    if keypoints.shape != (frame_count, 17, 2) or confidence.shape != (frame_count, 17):
        raise BodyLocalFrameError(f"character {character_id} pose sidecar shape mismatch")
    return keypoints, confidence


def _load_presence(character: dict[str, Any], frame_count: int, sidecars: dict[str, Any]) -> np.ndarray:
    bbox_ref = character.get("bbox_ref")
    if not isinstance(bbox_ref, dict):
        raise BodyLocalFrameError(f"character {character.get('character_id')} bbox_ref is missing")
    bboxes = _load_npz_array(bbox_ref, sidecars, "bbox_xyxy").astype(np.float32, copy=False)
    if bboxes.shape != (frame_count, 4):
        raise BodyLocalFrameError(f"character {character.get('character_id')} bbox sidecar shape mismatch")
    return np.all(np.isfinite(bboxes), axis=1)


def _load_stabilization(
    blueprint: dict[str, Any], frame_count: int, sidecars: dict[str, Any]
) -> np.ndarray:
    extension = blueprint.get("extensions", {}).get("e5_camera_motion")
    if not isinstance(extension, dict) or extension.get("enabled") is not True:
        raise BodyLocalFrameError("E6 body-local frame requires certified E5 camera motion")
    camera = blueprint.get("camera")
    rows = camera.get("per_shot") if isinstance(camera, dict) else None
    shots = blueprint.get("shots")
    if not isinstance(rows, list) or not isinstance(shots, list):
        raise BodyLocalFrameError("E6 could not resolve E5 camera rows/shots")
    shot_by_id = {str(shot.get("shot_id")): shot for shot in shots if isinstance(shot, dict)}
    result = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
    covered = np.zeros(frame_count, dtype=np.bool_)
    for row in rows:
        if not isinstance(row, dict) or row.get("reconstruction_backend") != "opencv_ransac_2d":
            raise BodyLocalFrameError("E6 supports only certified opencv_ransac_2d camera evidence")
        shot = shot_by_id.get(str(row.get("shot_id")))
        ref = row.get("affine_ref")
        if not isinstance(shot, dict) or not isinstance(ref, dict):
            raise BodyLocalFrameError("E6 camera row is missing shot/affine evidence")
        metadata = ref.get("metadata")
        uri = ref.get("uri")
        stabilization_key = metadata.get("stabilization_array_key") if isinstance(metadata, dict) else None
        if not isinstance(uri, str) or stabilization_key != "stabilization_affine":
            raise BodyLocalFrameError("E6 camera affine ref lacks stabilization array semantics")
        path = sidecars.get(uri)
        if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(str(path)):
            raise BodyLocalFrameError(f"E6 camera sidecar unavailable: {uri}")
        with np.load(str(path), allow_pickle=False) as arrays:
            if stabilization_key not in arrays:
                raise BodyLocalFrameError(f"E6 stabilization array missing from {uri}")
            values = np.asarray(arrays[stabilization_key], dtype=np.float32)
        frame_start = shot.get("frame_start")
        frame_end = shot.get("frame_end")
        if not isinstance(frame_start, int) or not isinstance(frame_end, int):
            raise BodyLocalFrameError("E6 shot frame bounds are invalid")
        length = frame_end - frame_start + 1
        if values.shape != (length, 2, 3):
            raise BodyLocalFrameError(f"E6 stabilization shape mismatch for shot {shot.get('shot_id')}")
        result[frame_start : frame_end + 1] = values
        covered[frame_start : frame_end + 1] = True
    if not np.all(covered):
        raise BodyLocalFrameError("E6 requires exactly one E5 stabilization row for every normalized frame")
    return result


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points.astype(np.float64), np.ones((len(points), 1))), axis=1)
    return (matrix.astype(np.float64) @ homogeneous.T).T[:, :2]


def derive_body_local_transform(
    anchors_xy: np.ndarray,
    *,
    stabilization_affine: np.ndarray,
    min_torso_scale_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    anchors = np.asarray(anchors_xy, dtype=np.float64)
    stabilization = np.asarray(stabilization_affine, dtype=np.float64)
    if anchors.shape != (4, 2) or stabilization.shape != (2, 3):
        raise BodyLocalFrameError("body-local anchors/stabilization have invalid shape")
    if not np.all(np.isfinite(anchors)) or not np.all(np.isfinite(stabilization)):
        return None
    stabilized = _apply_affine(stabilization, anchors)
    left_shoulder, right_shoulder, left_hip, right_hip = stabilized
    shoulder_center = (left_shoulder + right_shoulder) * 0.5
    hip_center = (left_hip + right_hip) * 0.5
    torso = shoulder_center - hip_center
    torso_scale = float(np.linalg.norm(torso))
    if not math.isfinite(torso_scale) or torso_scale < min_torso_scale_px:
        return None
    y_axis = torso / torso_scale
    anatomical_left = ((left_shoulder - right_shoulder) + (left_hip - right_hip)) * 0.5
    if float(np.linalg.norm(anatomical_left)) < 1e-6:
        return None
    x_axis = np.asarray([y_axis[1], -y_axis[0]], dtype=np.float64)
    if float(np.dot(x_axis, anatomical_left)) < 0:
        x_axis *= -1.0

    stabilized_to_local = np.asarray(
        [
            [x_axis[0] / torso_scale, x_axis[1] / torso_scale, -float(np.dot(x_axis, hip_center)) / torso_scale],
            [y_axis[0] / torso_scale, y_axis[1] / torso_scale, -float(np.dot(y_axis, hip_center)) / torso_scale],
        ],
        dtype=np.float64,
    )
    stabilization_h = np.eye(3, dtype=np.float64)
    stabilization_h[:2] = stabilization
    body_h = np.eye(3, dtype=np.float64)
    body_h[:2] = stabilized_to_local
    source_to_body_h = body_h @ stabilization_h
    try:
        body_to_source_h = np.linalg.inv(source_to_body_h)
    except np.linalg.LinAlgError:
        return None
    return (
        source_to_body_h[:2].astype(np.float32),
        body_to_source_h[:2].astype(np.float32),
        hip_center.astype(np.float32),
        torso_scale,
    )


def run_body_local_frame(
    blueprint: dict[str, Any],
    *,
    output_dir: str,
    sidecars: dict[str, Any],
    config: BodyLocalFrameConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config.validate()
    frame_count = int(blueprint["timebase"]["frame_count"])
    characters = blueprint.get("characters")
    if not isinstance(characters, list):
        raise BodyLocalFrameError("E6 characters must be an array")
    stabilization = _load_stabilization(blueprint, frame_count, sidecars)

    emitted: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []
    extension_refs: list[dict[str, Any]] = []
    present_total = 0
    valid_total = 0
    confidence_values: list[float] = []

    for character in characters:
        if not isinstance(character, dict):
            raise BodyLocalFrameError("E6 character row must be an object")
        character_id = str(character["character_id"])
        keypoints, confidence = _load_pose(character, frame_count, sidecars)
        present = _load_presence(character, frame_count, sidecars)
        present_count = int(np.count_nonzero(present))
        present_total += present_count
        source_to_body = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
        body_to_source = np.full((frame_count, 2, 3), np.nan, dtype=np.float32)
        origin_stabilized = np.full((frame_count, 2), np.nan, dtype=np.float32)
        torso_scale_px = np.full(frame_count, np.nan, dtype=np.float32)
        anchor_confidence = np.zeros(frame_count, dtype=np.float32)
        valid_frame = np.zeros(frame_count, dtype=np.bool_)
        anchor_indices = np.asarray([LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], dtype=np.int64)

        for frame_idx in range(frame_count):
            if not present[frame_idx]:
                continue
            anchors = keypoints[frame_idx, anchor_indices]
            scores = confidence[frame_idx, anchor_indices]
            if not np.all(np.isfinite(anchors)) or not np.all(np.isfinite(scores)):
                continue
            min_confidence = float(np.min(scores))
            if min_confidence < config.anchor_confidence_threshold:
                continue
            derived = derive_body_local_transform(
                anchors,
                stabilization_affine=stabilization[frame_idx],
                min_torso_scale_px=config.min_torso_scale_px,
            )
            if derived is None:
                continue
            source_matrix, inverse_matrix, origin, scale = derived
            source_to_body[frame_idx] = source_matrix
            body_to_source[frame_idx] = inverse_matrix
            origin_stabilized[frame_idx] = origin
            torso_scale_px[frame_idx] = scale
            anchor_confidence[frame_idx] = min_confidence
            valid_frame[frame_idx] = True

        valid_count = int(np.count_nonzero(valid_frame))
        valid_total += valid_count
        coverage = valid_count / present_count if present_count else 0.0
        nonzero_confidence = anchor_confidence[valid_frame]
        mean_confidence = float(np.mean(nonzero_confidence)) if nonzero_confidence.size else 0.0
        confidence_values.extend(float(value) for value in nonzero_confidence)

        uri = f"artifacts/timeseries/{character_id}_body_local_2d.npz"
        path = os.path.join(output_dir, uri.replace("/", os.sep))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(
            path,
            source_pixel_to_body_local=source_to_body,
            body_local_to_source_pixel=body_to_source,
            body_origin_stabilized_xy=origin_stabilized,
            torso_scale_px=torso_scale_px,
            anchor_confidence=anchor_confidence,
            valid_frame=valid_frame,
        )
        checksum = sha256_file(path)
        ref = {
            "uri": uri,
            "format": "npz",
            "dtype": "float32",
            "shape": [frame_count, 2, 3],
            "axes": ["frame", "matrix_row", "matrix_col"],
            "unit": "torso_length_normalized_affine",
            "coordinate_space": "body_local_2d",
            "sampling": "per_frame",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "compression": "zip",
            "nan_policy": "preserve",
            "interpolation_policy": "none",
            "checksum_sha256": checksum,
            "metadata": {
                "array_key": "source_pixel_to_body_local",
                "inverse_array_key": "body_local_to_source_pixel",
                "origin_array_key": "body_origin_stabilized_xy",
                "scale_array_key": "torso_scale_px",
                "confidence_array_key": "anchor_confidence",
                "valid_array_key": "valid_frame",
                "source_coordinate_space": "pixel_xy",
                "camera_stabilization_applied": True,
                "camera_backend": "opencv_ransac_2d",
                "pose_skeleton": "coco17",
                "anchor_names": ANCHOR_NAMES,
                "origin_semantics": "midpoint_of_left_and_right_hip_after_camera_stabilization",
                "positive_x_semantics": "toward_anatomical_left",
                "positive_y_semantics": "pelvis_to_shoulder_center",
                "scale_semantics": "stabilized_shoulder_center_to_hip_center_distance_equals_1_body_local_unit",
                "config_sha256": config.token(),
            },
        }
        surface_motion = character.get("surface_motion")
        if not isinstance(surface_motion, dict):
            raise BodyLocalFrameError(f"character {character_id} surface_motion contract is missing")
        surface_motion["coordinate_frame"] = "body_local_2d"
        surface_motion["body_frame_transform_ref"] = ref
        emitted[uri] = path
        extension_refs.append({"character_id": character_id, "body_frame_transform_ref": ref})
        report_rows.append(
            {
                "character_id": character_id,
                "present_frames": present_count,
                "valid_body_frame_frames": valid_count,
                "coverage": round(coverage, 6),
                "mean_anchor_confidence": round(mean_confidence, 6),
            }
        )

    overall_coverage = valid_total / present_total if present_total else 0.0
    score = float(np.mean(confidence_values)) * overall_coverage if confidence_values and present_total else 0.0
    report_uri = "artifacts/reports/body_local_frame_2d.json"
    report = {
        "stage": "body_local_frame",
        "algorithm": "coco17_torso_similarity_frame_v1",
        "coordinate_frame": "body_local_2d",
        "source_coordinate_space": "pixel_xy",
        "camera_stabilization_applied": True,
        "camera_backend": "opencv_ransac_2d",
        "pose_skeleton": "coco17",
        "anchor_names": ANCHOR_NAMES,
        "interpolation": False,
        "frame_count": frame_count,
        "characters": report_rows,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    emitted[report_uri] = report
    report_ref = {
        "kind": "body_local_frame_2d",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    extension = {
        "enabled": True,
        "algorithm": "coco17_torso_similarity_frame_v1",
        "coordinate_frame": "body_local_2d",
        "source_coordinate_space": "pixel_xy",
        "camera_stabilization_applied": True,
        "camera_backend": "opencv_ransac_2d",
        "pose_skeleton": "coco17",
        "anchor_names": ANCHOR_NAMES,
        "interpolation": False,
        "config_sha256": config.token(),
        "character_count": len(characters),
        "character_refs": extension_refs,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    quality = {
        "coverage": round(overall_coverage, 6),
        "score": round(max(0.0, min(1.0, score)), 6),
        "confidence_available": True,
    }
    return emitted, report_ref, extension, quality
