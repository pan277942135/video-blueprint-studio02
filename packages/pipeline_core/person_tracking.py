from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


class PersonTrackingError(RuntimeError):
    """Raised when anonymous person tracking cannot produce trustworthy output."""


@dataclass(frozen=True)
class PersonDetection:
    frame_idx: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class PersonTrackingConfig:
    iou_threshold: float = 0.30
    max_gap_frames: int = 2


@dataclass
class AnonymousTrack:
    character_id: str
    track_label: str
    shot_id: str
    detections: dict[int, PersonDetection] = field(default_factory=dict)

    @property
    def last_detection(self) -> PersonDetection:
        if not self.detections:
            raise PersonTrackingError("Anonymous track has no detections")
        return self.detections[max(self.detections)]


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PersonDetection]: ...


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _validate_tracking_config(config: PersonTrackingConfig) -> None:
    if not 0.0 <= config.iou_threshold <= 1.0:
        raise PersonTrackingError("iou_threshold must be within [0, 1]")
    if config.max_gap_frames < 0:
        raise PersonTrackingError("max_gap_frames must be >= 0")


def associate_anonymous_tracks(
    shots: list[dict[str, Any]],
    detections_by_frame: dict[int, list[PersonDetection]],
    *,
    config: PersonTrackingConfig | None = None,
) -> list[AnonymousTrack]:
    """Associate detections into anonymous per-shot tracks using deterministic IoU matching.

    Tracks never cross a shot boundary. No face embedding, re-identification model,
    or identity inference is used. Greedy matching is deterministic by IoU,
    track ID, and detection coordinates.
    """
    cfg = config or PersonTrackingConfig()
    _validate_tracking_config(cfg)

    tracks: list[AnonymousTrack] = []
    next_track_index = 0

    for shot in sorted(shots, key=lambda item: int(item["index"])):
        shot_id = str(shot["shot_id"])
        frame_start = int(shot["frame_start"])
        frame_end = int(shot["frame_end"])
        if frame_end < frame_start:
            raise PersonTrackingError(f"Invalid shot frame range: {frame_start}-{frame_end}")

        active: list[AnonymousTrack] = []
        for frame_idx in range(frame_start, frame_end + 1):
            detections = sorted(
                detections_by_frame.get(frame_idx, []),
                key=lambda item: (-item.score, item.bbox_xyxy),
            )
            for detection in detections:
                if detection.frame_idx != frame_idx:
                    raise PersonTrackingError(
                        f"Detection frame mismatch: map key {frame_idx}, detection {detection.frame_idx}"
                    )

            active = [
                track
                for track in active
                if frame_idx - track.last_detection.frame_idx <= cfg.max_gap_frames + 1
            ]

            candidates: list[tuple[float, str, int, int]] = []
            for track_idx, track in enumerate(active):
                last_bbox = track.last_detection.bbox_xyxy
                for det_idx, detection in enumerate(detections):
                    iou = bbox_iou(last_bbox, detection.bbox_xyxy)
                    if iou >= cfg.iou_threshold:
                        candidates.append((iou, track.character_id, track_idx, det_idx))

            candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()
            for _, _, track_idx, det_idx in candidates:
                if track_idx in matched_tracks or det_idx in matched_detections:
                    continue
                active[track_idx].detections[frame_idx] = detections[det_idx]
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)

            for det_idx, detection in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                character_id = f"char_{next_track_index:03d}"
                track = AnonymousTrack(
                    character_id=character_id,
                    track_label=f"anonymous_track_{next_track_index:03d}",
                    shot_id=shot_id,
                    detections={frame_idx: detection},
                )
                next_track_index += 1
                active.append(track)
                tracks.append(track)

    return tracks


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, indent=2).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _presence_intervals(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    intervals: list[dict[str, int]] = []
    start = frames[0]
    previous = frames[0]
    for frame_idx in frames[1:]:
        if frame_idx == previous + 1:
            previous = frame_idx
            continue
        intervals.append({"frame_start": start, "frame_end": previous})
        start = previous = frame_idx
    intervals.append({"frame_start": start, "frame_end": previous})
    return intervals


def _timeseries_ref(
    *,
    uri: str,
    checksum: str,
    dtype: str,
    shape: list[int],
    axes: list[str],
    unit: str,
    coordinate_space: str,
    frame_count: int,
    array_key: str,
    nan_policy: str = "preserve",
) -> dict[str, Any]:
    return {
        "uri": uri,
        "format": "npz",
        "dtype": dtype,
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


def _disabled_quality(module_name: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "coverage": 0.0,
        "warnings": [f"{module_name} not run before its dedicated epic"],
        "errors": [],
    }


def _write_track_npz(
    track: AnonymousTrack,
    *,
    frame_count: int,
    output_dir: str,
) -> tuple[str, str, dict[str, np.ndarray]]:
    bbox = np.full((frame_count, 4), np.nan, dtype=np.float32)
    center = np.full((frame_count, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((frame_count,), dtype=np.float32)
    pose_keypoints = np.empty((frame_count, 0, 3), dtype=np.float32)
    pose_confidence = np.empty((frame_count, 0), dtype=np.float32)

    for frame_idx, detection in track.detections.items():
        if not 0 <= frame_idx < frame_count:
            raise PersonTrackingError(f"Track frame {frame_idx} outside normalized timeline")
        x1, y1, x2, y2 = detection.bbox_xyxy
        bbox[frame_idx] = (x1, y1, x2, y2)
        center[frame_idx] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        visibility[frame_idx] = float(max(0.0, min(1.0, detection.score)))

    uri = f"artifacts/timeseries/{track.character_id}_tracking.npz"
    path = os.path.join(output_dir, uri.replace("/", os.sep))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bbox_xyxy=bbox,
        center_xy=center,
        visibility=visibility,
        pose_keypoints_2d=pose_keypoints,
        pose_confidence=pose_confidence,
    )
    checksum = _sha256_file(path)
    return uri, checksum, {
        "bbox_xyxy": bbox,
        "center_xy": center,
        "visibility": visibility,
        "pose_keypoints_2d": pose_keypoints,
        "pose_confidence": pose_confidence,
    }


def _write_bbox_preview(
    video_path: str,
    track: AnonymousTrack,
    *,
    output_dir: str,
) -> tuple[str, str]:
    best = max(track.detections.values(), key=lambda item: (item.score, -item.frame_idx))
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise PersonTrackingError(f"OpenCV could not open normalized video for overlay: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, best.frame_idx)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise PersonTrackingError(f"Could not decode overlay frame {best.frame_idx}")

    x1, y1, x2, y2 = (round(value) for value in best.bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    label_y = max(20, y1 - 8)
    cv2.putText(
        frame,
        track.track_label,
        (max(0, x1), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    uri = f"artifacts/overlays/{track.character_id}_bbox_preview.png"
    path = os.path.join(output_dir, uri.replace("/", os.sep))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(path, frame):
        raise PersonTrackingError(f"Could not write bbox overlay preview: {path}")
    return uri, path


def _character_manifest(
    track: AnonymousTrack,
    *,
    frame_count: int,
    uri: str,
    checksum: str,
) -> dict[str, Any]:
    frames = sorted(track.detections)
    scores = [track.detections[frame_idx].score for frame_idx in frames]
    span = frames[-1] - frames[0] + 1
    detection_coverage = len(frames) / span if span > 0 else 0.0
    average_score = sum(scores) / len(scores)
    appearance_frames = [
        detection.frame_idx
        for detection in sorted(
            track.detections.values(),
            key=lambda item: (-item.score, item.frame_idx),
        )[:3]
    ]

    bbox_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        dtype="float32",
        shape=[frame_count, 4],
        axes=["frame", "bbox_component"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key="bbox_xyxy",
    )
    bbox_ref["metadata"].update({"bbox_format": "xyxy", "absent_value": "nan"})
    center_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        dtype="float32",
        shape=[frame_count, 2],
        axes=["frame", "xy"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key="center_xy",
    )
    visibility_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        dtype="float32",
        shape=[frame_count],
        axes=["frame"],
        unit="confidence",
        coordinate_space="none",
        frame_count=frame_count,
        array_key="visibility",
        nan_policy="zero_fill",
    )
    pose_keypoints_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        dtype="float32",
        shape=[frame_count, 0, 3],
        axes=["frame", "keypoint", "xyc"],
        unit="px",
        coordinate_space="pixel_xy",
        frame_count=frame_count,
        array_key="pose_keypoints_2d",
    )
    pose_confidence_ref = _timeseries_ref(
        uri=uri,
        checksum=checksum,
        dtype="float32",
        shape=[frame_count, 0],
        axes=["frame", "keypoint"],
        unit="confidence",
        coordinate_space="none",
        frame_count=frame_count,
        array_key="pose_confidence",
    )

    return {
        "character_id": track.character_id,
        "track_label": track.track_label,
        "presence": _presence_intervals(frames),
        "bbox_ref": bbox_ref,
        "center_ref": center_ref,
        "person_mask_ref": None,
        "visibility_ref": visibility_ref,
        "occlusion_ref": None,
        "appearance_reference_frames": appearance_frames,
        "pose": {
            "enabled": False,
            "skeleton_name": "none",
            "keypoint_count": 0,
            "keypoint_names": [],
            "keypoints_2d_ref": pose_keypoints_ref,
            "keypoints_3d_ref": None,
            "root_translation_ref": None,
            "root_rotation_ref": None,
            "joint_angles_ref": None,
            "scale_ref": None,
            "confidence_ref": pose_confidence_ref,
            "smoothing": {"method": "none", "parameters": {}},
            "quality": _disabled_quality("pose"),
        },
        "face": {
            "enabled": False,
            "landmark_count": 0,
            "bbox_ref": None,
            "landmarks_3d_ref": None,
            "blendshapes_ref": None,
            "transform_ref": None,
            "head_pose_ref": None,
            "gaze_ref": None,
            "eye_openness_ref": None,
            "mouth_open_ref": None,
            "blink_events": [],
            "quality": _disabled_quality("face"),
        },
        "hands": {
            "left": {
                "enabled": False,
                "landmark_count": 0,
                "present_ref": None,
                "bbox_ref": None,
                "landmarks_2d_ref": None,
                "landmarks_world_ref": None,
                "handedness_ref": None,
                "palm_normal_ref": None,
                "finger_curl_ref": None,
                "quality": _disabled_quality("left hand"),
            },
            "right": {
                "enabled": False,
                "landmark_count": 0,
                "present_ref": None,
                "bbox_ref": None,
                "landmarks_2d_ref": None,
                "landmarks_world_ref": None,
                "handedness_ref": None,
                "palm_normal_ref": None,
                "finger_curl_ref": None,
                "quality": _disabled_quality("right hand"),
            },
        },
        "surface_motion": {
            "coordinate_frame": "body_local_2d",
            "body_frame_transform_ref": None,
            "global_postural_sway_ref": None,
            "regions": [],
            "events": [],
            "quality": _disabled_quality("surface motion"),
        },
        "quality": {
            "score": round(max(0.0, min(1.0, average_score)), 6),
            "coverage": round(max(0.0, min(1.0, detection_coverage)), 6),
            "warnings": ["anonymous IoU association; no identity inference performed"],
            "errors": [],
        },
        "privacy": {
            "biometric_embedding_exported": False,
            "identity_inference_performed": False,
        },
    }


def run_person_tracking(
    video_path: str,
    *,
    shots: list[dict[str, Any]],
    frame_count: int,
    detector: PersonDetector,
    output_dir: str,
    config: PersonTrackingConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run frame-level person detection, anonymous association, and sidecar export."""
    cfg = config or PersonTrackingConfig()
    _validate_tracking_config(cfg)
    if not os.path.isfile(video_path):
        raise PersonTrackingError(f"Normalized video does not exist: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise PersonTrackingError(f"OpenCV could not open normalized video: {video_path}")

    detections_by_frame: dict[int, list[PersonDetection]] = {}
    decoded_frames = 0
    try:
        while decoded_frames < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            detections_by_frame[decoded_frames] = detector.detect(frame, decoded_frames)
            decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames != frame_count:
        raise PersonTrackingError(
            f"Person detector decoded {decoded_frames} frames; normalized timeline requires {frame_count}"
        )

    tracks = associate_anonymous_tracks(shots, detections_by_frame, config=cfg)
    characters: list[dict[str, Any]] = []
    sidecars: dict[str, Any] = {}
    overlays: list[dict[str, Any]] = []

    for track in tracks:
        uri, checksum, _ = _write_track_npz(
            track,
            frame_count=frame_count,
            output_dir=output_dir,
        )
        sidecars[uri] = os.path.join(output_dir, uri.replace("/", os.sep))
        characters.append(
            _character_manifest(
                track,
                frame_count=frame_count,
                uri=uri,
                checksum=checksum,
            )
        )

        overlay_uri, overlay_path = _write_bbox_preview(
            video_path,
            track,
            output_dir=output_dir,
        )
        sidecars[overlay_uri] = overlay_path
        overlays.append(
            {
                "kind": "person_bbox_preview",
                "uri": overlay_uri,
                "sha256": _sha256_file(overlay_path),
                "mime_type": "image/png",
            }
        )

    for shot in shots:
        shot_id = str(shot["shot_id"])
        shot["dominant_character_ids"] = [
            track.character_id for track in tracks if track.shot_id == shot_id
        ]

    report_uri = "artifacts/reports/person_tracking.json"
    report = {
        "association_backend": "anonymous_iou",
        "iou_threshold": cfg.iou_threshold,
        "max_gap_frames": cfg.max_gap_frames,
        "frame_count": frame_count,
        "track_count": len(tracks),
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "tracks": [
            {
                "character_id": track.character_id,
                "track_label": track.track_label,
                "shot_id": track.shot_id,
                "frames": sorted(track.detections),
                "mean_detection_score": round(
                    sum(item.score for item in track.detections.values()) / len(track.detections),
                    6,
                ),
            }
            for track in tracks
        ],
    }
    sidecars[report_uri] = report
    report_ref = {
        "kind": "person_tracking",
        "uri": report_uri,
        "sha256": _json_sha256(report),
        "mime_type": "application/json",
    }
    return characters, sidecars, overlays, report_ref
