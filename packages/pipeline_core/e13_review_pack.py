from __future__ import annotations

import hashlib
import io
import json
import pathlib
import zipfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

COCO17_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass(frozen=True)
class ReviewFrame:
    frame_idx: int
    time_us: int | None
    shot_id: str | None
    kind: str


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_review_frames(blueprint: dict[str, Any], *, max_frames: int = 12) -> list[ReviewFrame]:
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    selected: dict[int, ReviewFrame] = {}
    for shot in blueprint.get("shots", []):
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("shot_id") if isinstance(shot.get("shot_id"), str) else None
        for keyframe in shot.get("keyframes", []):
            if not isinstance(keyframe, dict):
                continue
            frame_idx = keyframe.get("frame_idx")
            if not isinstance(frame_idx, int) or frame_idx < 0:
                continue
            time_us = keyframe.get("time_us") if isinstance(keyframe.get("time_us"), int) else None
            kind = keyframe.get("kind") if isinstance(keyframe.get("kind"), str) else "keyframe"
            selected.setdefault(
                frame_idx,
                ReviewFrame(frame_idx=frame_idx, time_us=time_us, shot_id=shot_id, kind=kind),
            )

    if not selected:
        source = blueprint.get("source_video") if isinstance(blueprint.get("source_video"), dict) else {}
        frame_count = source.get("source_frame_count")
        duration_us = source.get("duration_us")
        if isinstance(frame_count, int) and frame_count > 0:
            fallback = sorted({0, max(0, (frame_count - 1) // 2), frame_count - 1})
            for frame_idx in fallback:
                time_us = None
                if isinstance(duration_us, int) and frame_count > 1:
                    time_us = round(duration_us * frame_idx / (frame_count - 1))
                selected[frame_idx] = ReviewFrame(
                    frame_idx=frame_idx,
                    time_us=time_us,
                    shot_id=None,
                    kind="fallback",
                )

    rows = [selected[idx] for idx in sorted(selected)]
    if len(rows) <= max_frames:
        return rows

    positions = np.linspace(0, len(rows) - 1, num=max_frames)
    indices = sorted({round(float(position)) for position in positions})
    if len(indices) < max_frames:
        for idx in range(len(rows)):
            if idx not in indices:
                indices.append(idx)
                if len(indices) == max_frames:
                    break
        indices.sort()
    return [rows[idx] for idx in indices[:max_frames]]


def _npz_array(archive: zipfile.ZipFile, ref: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(ref, dict):
        return None
    uri = ref.get("uri")
    metadata = ref.get("metadata")
    if not isinstance(uri, str) or not isinstance(metadata, dict):
        return None
    array_key = metadata.get("array_key")
    if not isinstance(array_key, str):
        return None
    try:
        raw = archive.read(uri)
    except KeyError:
        return None
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        if array_key not in payload.files:
            return None
        return np.array(payload[array_key], copy=True)


def _draw_character_evidence(
    frame: np.ndarray,
    *,
    character_id: str,
    bbox: np.ndarray | None,
    keypoints: np.ndarray | None,
    confidence: np.ndarray | None,
    confidence_threshold: float,
) -> None:
    if bbox is not None and bbox.shape == (4,) and np.isfinite(bbox).all():
        x1, y1, x2, y2 = (round(float(value)) for value in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame,
            character_id,
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if keypoints is None or keypoints.ndim != 2 or keypoints.shape[1] != 2:
        return
    if confidence is None or confidence.ndim != 1 or len(confidence) != len(keypoints):
        confidence = np.ones((len(keypoints),), dtype=np.float32)

    for first, second in COCO17_EDGES:
        if first >= len(keypoints) or second >= len(keypoints):
            continue
        if confidence[first] < confidence_threshold or confidence[second] < confidence_threshold:
            continue
        p1 = keypoints[first]
        p2 = keypoints[second]
        if not np.isfinite(p1).all() or not np.isfinite(p2).all():
            continue
        cv2.line(
            frame,
            (round(float(p1[0])), round(float(p1[1]))),
            (round(float(p2[0])), round(float(p2[1]))),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for point, score in zip(keypoints, confidence, strict=False):
        if score < confidence_threshold or not np.isfinite(point).all():
            continue
        cv2.circle(
            frame,
            (round(float(point[0])), round(float(point[1]))),
            2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def _fit_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = target_height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)


def build_review_pack(
    *,
    video_path: pathlib.Path,
    bundle_path: pathlib.Path,
    output_dir: pathlib.Path,
    max_frames: int = 12,
    confidence_threshold: float = 0.3,
) -> dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    video_sha256 = sha256_file(video_path)
    bundle_sha256 = sha256_file(bundle_path)

    with zipfile.ZipFile(bundle_path, "r") as archive:
        blueprint = json.loads(archive.read("blueprint.json"))
        source = blueprint.get("source_video") if isinstance(blueprint.get("source_video"), dict) else {}
        expected_source_sha256 = source.get("sha256")
        if expected_source_sha256 != video_sha256:
            raise RuntimeError(
                "review source SHA256 does not match Bundle source_video: "
                f"{video_sha256} != {expected_source_sha256}"
            )

        review_frames = select_review_frames(blueprint, max_frames=max_frames)
        character_payloads: list[dict[str, Any]] = []
        for character in blueprint.get("characters", []):
            if not isinstance(character, dict):
                continue
            character_id = character.get("character_id")
            if not isinstance(character_id, str):
                continue
            bbox = _npz_array(archive, character.get("bbox_ref"))
            pose = character.get("pose") if isinstance(character.get("pose"), dict) else {}
            keypoints = _npz_array(archive, pose.get("keypoints_2d_ref"))
            confidence = _npz_array(archive, pose.get("confidence_ref"))
            character_payloads.append(
                {
                    "character_id": character_id,
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "confidence": confidence,
                }
            )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {video_path}")

        frame_rows: list[dict[str, Any]] = []
        try:
            for review_frame in review_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, review_frame.frame_idx)
                ok, source_frame = capture.read()
                if not ok or source_frame is None:
                    raise RuntimeError(f"unable to decode review frame {review_frame.frame_idx}")

                evidence_frame = source_frame.copy()
                visible_characters: list[str] = []
                for payload in character_payloads:
                    frame_idx = review_frame.frame_idx
                    bbox_series = payload["bbox"]
                    keypoint_series = payload["keypoints"]
                    confidence_series = payload["confidence"]
                    bbox = (
                        bbox_series[frame_idx]
                        if isinstance(bbox_series, np.ndarray) and frame_idx < len(bbox_series)
                        else None
                    )
                    keypoints = (
                        keypoint_series[frame_idx]
                        if isinstance(keypoint_series, np.ndarray) and frame_idx < len(keypoint_series)
                        else None
                    )
                    confidence = (
                        confidence_series[frame_idx]
                        if isinstance(confidence_series, np.ndarray) and frame_idx < len(confidence_series)
                        else None
                    )
                    if bbox is not None and np.asarray(bbox).shape == (4,) and np.isfinite(bbox).all():
                        visible_characters.append(payload["character_id"])
                    _draw_character_evidence(
                        evidence_frame,
                        character_id=payload["character_id"],
                        bbox=np.asarray(bbox) if bbox is not None else None,
                        keypoints=np.asarray(keypoints) if keypoints is not None else None,
                        confidence=np.asarray(confidence) if confidence is not None else None,
                        confidence_threshold=confidence_threshold,
                    )

                label = (
                    f"frame={review_frame.frame_idx} "
                    f"shot={review_frame.shot_id or '-'} kind={review_frame.kind}"
                )
                cv2.putText(
                    evidence_frame,
                    label,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                target_height = max(source_frame.shape[0], evidence_frame.shape[0])
                comparison = np.hstack(
                    [_fit_height(source_frame, target_height), _fit_height(evidence_frame, target_height)]
                )
                output_name = f"frame_{review_frame.frame_idx:06d}_source_vs_evidence.png"
                output_path = output_dir / output_name
                if not cv2.imwrite(str(output_path), comparison):
                    raise RuntimeError(f"unable to write review image: {output_path}")
                frame_rows.append(
                    {
                        "frame_idx": review_frame.frame_idx,
                        "time_us": review_frame.time_us,
                        "shot_id": review_frame.shot_id,
                        "kind": review_frame.kind,
                        "visible_character_ids": visible_characters,
                        "comparison_image": output_name,
                        "comparison_sha256": sha256_file(output_path),
                    }
                )
        finally:
            capture.release()

    quality = blueprint.get("quality") if isinstance(blueprint.get("quality"), dict) else {}
    manifest: dict[str, Any] = {
        "status": "review_pack_ready",
        "manual_verdict_required": True,
        "automatic_perceptual_pass": False,
        "source": {
            "path": str(video_path),
            "sha256": video_sha256,
            "file_name": source.get("file_name"),
            "duration_us": source.get("duration_us"),
            "source_frame_count": source.get("source_frame_count"),
        },
        "bundle": {"path": str(bundle_path), "sha256": bundle_sha256},
        "character_count": len(blueprint.get("characters", [])),
        "shot_count": len(blueprint.get("shots", [])),
        "module_scores": quality.get("module_scores", {}),
        "review_frames": frame_rows,
        "review_instructions": [
            "Compare the left source frame with the right evidence overlay.",
            "Reject incorrect person boxes, pose geometry, missed people, duplicated tracks, or evidence attached to the wrong subject.",
            "A green machine gate is not a perceptual acceptance result; record a manual verdict separately.",
        ],
    }
    manifest_path = output_dir / "e13_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
