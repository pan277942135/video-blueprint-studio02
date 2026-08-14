from __future__ import annotations

import hashlib
import io
import json
import pathlib
import tempfile
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

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


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
        raw_shot_id = shot.get("shot_id")
        shot_id = raw_shot_id if isinstance(raw_shot_id, str) else None
        for keyframe in shot.get("keyframes", []):
            if not isinstance(keyframe, dict):
                continue
            frame_idx = keyframe.get("frame_idx")
            if not isinstance(frame_idx, int) or frame_idx < 0:
                continue
            raw_time_us = keyframe.get("time_us")
            time_us = raw_time_us if isinstance(raw_time_us, int) else None
            raw_kind = keyframe.get("kind")
            kind = raw_kind if isinstance(raw_kind, str) else "keyframe"
            selected.setdefault(
                frame_idx,
                ReviewFrame(frame_idx=frame_idx, time_us=time_us, shot_id=shot_id, kind=kind),
            )

    if not selected:
        raw_timebase = blueprint.get("timebase")
        timebase: dict[str, Any] = raw_timebase if isinstance(raw_timebase, dict) else {}
        frame_count = timebase.get("frame_count")
        frame_duration_us = timebase.get("frame_duration_us")

        if not isinstance(frame_count, int) or frame_count <= 0:
            raw_source = blueprint.get("source_video")
            source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
            frame_count = source.get("source_frame_count")
            duration_us = source.get("duration_us")
            if isinstance(frame_count, int) and frame_count > 1 and isinstance(duration_us, int):
                frame_duration_us = duration_us / (frame_count - 1)

        if isinstance(frame_count, int) and frame_count > 0:
            fallback = sorted({0, max(0, (frame_count - 1) // 2), frame_count - 1})
            for frame_idx in fallback:
                time_us = None
                if isinstance(frame_duration_us, (int, float)):
                    time_us = round(float(frame_duration_us) * frame_idx)
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


def _normalized_analysis_media(archive: zipfile.ZipFile) -> tuple[str, bytes, str]:
    try:
        raw_manifest = archive.read("bundle_manifest.json")
    except KeyError as exc:
        raise RuntimeError("Bundle is missing bundle_manifest.json") from exc

    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, dict):
        raise TypeError("bundle_manifest.json must contain an object")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise TypeError("bundle_manifest.json is missing files[]")

    candidates: list[tuple[str, str]] = []
    for entry in raw_files:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        raw_sha256 = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_sha256, str):
            continue
        suffix = pathlib.PurePosixPath(raw_path).suffix.lower()
        if raw_path.startswith("artifacts/normalized/") and suffix in _VIDEO_SUFFIXES:
            candidates.append((raw_path, raw_sha256))

    preferred = [item for item in candidates if pathlib.PurePosixPath(item[0]).name == "analysis_cfr.mp4"]
    if len(preferred) == 1:
        uri, expected_sha256 = preferred[0]
    elif len(candidates) == 1:
        uri, expected_sha256 = candidates[0]
    elif not candidates:
        raise RuntimeError("Bundle contains no normalized analysis video")
    else:
        raise RuntimeError(f"Bundle contains ambiguous normalized analysis videos: {candidates}")

    try:
        payload = archive.read(uri)
    except KeyError as exc:
        raise RuntimeError(f"Bundle manifest references missing normalized media: {uri}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "normalized analysis media SHA256 does not match Bundle manifest: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return uri, payload, actual_sha256


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
        raw_source = blueprint.get("source_video")
        source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        expected_source_sha256 = source.get("sha256")
        if expected_source_sha256 != video_sha256:
            raise RuntimeError(
                "review source SHA256 does not match Bundle source_video: "
                f"{video_sha256} != {expected_source_sha256}"
            )

        analysis_media_uri, analysis_media_bytes, analysis_media_sha256 = _normalized_analysis_media(archive)
        review_frames = select_review_frames(blueprint, max_frames=max_frames)
        character_payloads: list[dict[str, Any]] = []
        for character in blueprint.get("characters", []):
            if not isinstance(character, dict):
                continue
            character_id = character.get("character_id")
            if not isinstance(character_id, str):
                continue
            bbox = _npz_array(archive, character.get("bbox_ref"))
            raw_pose = character.get("pose")
            pose: dict[str, Any] = raw_pose if isinstance(raw_pose, dict) else {}
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

        suffix = pathlib.PurePosixPath(analysis_media_uri).suffix or ".mp4"
        temp_path: pathlib.Path | None = None
        capture: cv2.VideoCapture | None = None
        frame_rows: list[dict[str, Any]] = []
        try:
            with tempfile.NamedTemporaryFile(
                prefix="e13_analysis_",
                suffix=suffix,
                dir=output_dir,
                delete=False,
            ) as handle:
                handle.write(analysis_media_bytes)
                temp_path = pathlib.Path(handle.name)

            capture = cv2.VideoCapture(str(temp_path))
            if not capture.isOpened():
                raise RuntimeError(f"unable to open normalized analysis video: {analysis_media_uri}")

            for review_frame in review_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, review_frame.frame_idx)
                ok, analysis_frame = capture.read()
                if not ok or analysis_frame is None:
                    raise RuntimeError(
                        f"unable to decode normalized analysis frame {review_frame.frame_idx}"
                    )

                evidence_frame = analysis_frame.copy()
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
                    f"analysis_frame={review_frame.frame_idx} "
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

                target_height = max(analysis_frame.shape[0], evidence_frame.shape[0])
                comparison = np.hstack(
                    [_fit_height(analysis_frame, target_height), _fit_height(evidence_frame, target_height)]
                )
                output_name = f"frame_{review_frame.frame_idx:06d}_analysis_vs_evidence.png"
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
            if capture is not None:
                capture.release()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    raw_quality = blueprint.get("quality")
    quality: dict[str, Any] = raw_quality if isinstance(raw_quality, dict) else {}
    raw_timebase = blueprint.get("timebase")
    timebase: dict[str, Any] = raw_timebase if isinstance(raw_timebase, dict) else {}
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
        "analysis_media": {
            "bundle_uri": analysis_media_uri,
            "sha256": analysis_media_sha256,
            "frame_count": timebase.get("frame_count"),
            "fps_num": timebase.get("fps_num"),
            "fps_den": timebase.get("fps_den"),
            "timeline": "normalized_analysis_cfr",
        },
        "bundle": {"path": str(bundle_path), "sha256": bundle_sha256},
        "character_count": len(blueprint.get("characters", [])),
        "shot_count": len(blueprint.get("shots", [])),
        "module_scores": quality.get("module_scores", {}),
        "review_frames": frame_rows,
        "review_instructions": [
            "Compare the left normalized analysis frame with the right evidence overlay from the same frame index.",
            "Reject incorrect person boxes, pose geometry, missed people, duplicated tracks, or evidence attached to the wrong subject.",
            "The raw source SHA is verified separately; evidence overlays must be reviewed on the normalized analysis timeline used by the pipeline.",
            "A green machine gate is not a perceptual acceptance result; record a manual verdict separately.",
        ],
    }
    manifest_path = output_dir / "e13_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
