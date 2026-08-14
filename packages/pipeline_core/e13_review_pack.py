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

HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_FACE_POINT_STRIDE = 8


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


def _timebase_values(blueprint: dict[str, Any]) -> tuple[int | None, float | None]:
    raw_timebase = blueprint.get("timebase")
    timebase: dict[str, Any] = raw_timebase if isinstance(raw_timebase, dict) else {}
    frame_count = timebase.get("frame_count")
    frame_duration_us = timebase.get("frame_duration_us")

    normalized_count = frame_count if isinstance(frame_count, int) and frame_count > 0 else None
    normalized_duration = (
        float(frame_duration_us)
        if isinstance(frame_duration_us, (int, float)) and float(frame_duration_us) > 0.0
        else None
    )
    if normalized_count is not None and normalized_duration is not None:
        return normalized_count, normalized_duration

    fps_num = timebase.get("fps_num")
    fps_den = timebase.get("fps_den")
    if (
        normalized_count is not None
        and isinstance(fps_num, int)
        and fps_num > 0
        and isinstance(fps_den, int)
        and fps_den > 0
    ):
        return normalized_count, 1_000_000.0 * fps_den / fps_num

    raw_source = blueprint.get("source_video")
    source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
    source_count = source.get("source_frame_count")
    duration_us = source.get("duration_us")
    if isinstance(source_count, int) and source_count > 0:
        if isinstance(duration_us, int) and duration_us > 0 and source_count > 1:
            return source_count, duration_us / (source_count - 1)
        return source_count, None
    return None, None


def _shot_id_for_frame(blueprint: dict[str, Any], frame_idx: int) -> str | None:
    for shot in blueprint.get("shots", []):
        if not isinstance(shot, dict):
            continue
        frame_start = shot.get("frame_start")
        frame_end = shot.get("frame_end")
        shot_id = shot.get("shot_id")
        if (
            isinstance(frame_start, int)
            and isinstance(frame_end, int)
            and frame_start <= frame_idx <= frame_end
            and isinstance(shot_id, str)
        ):
            return shot_id
    return None


def _supplement_review_frames(
    selected: dict[int, ReviewFrame],
    *,
    blueprint: dict[str, Any],
    frame_count: int,
    frame_duration_us: float | None,
    max_frames: int,
) -> None:
    """Fill review coverage with deterministic farthest-point timeline samples."""
    if frame_count <= 0 or len(selected) >= max_frames:
        return

    if not selected:
        initial = sorted({0, max(0, (frame_count - 1) // 2), frame_count - 1})
        for frame_idx in initial:
            time_us = round(frame_idx * frame_duration_us) if frame_duration_us is not None else None
            selected[frame_idx] = ReviewFrame(
                frame_idx=frame_idx,
                time_us=time_us,
                shot_id=_shot_id_for_frame(blueprint, frame_idx),
                kind="timeline_sample",
            )
            if len(selected) >= max_frames:
                return

    available = [frame_idx for frame_idx in range(frame_count) if frame_idx not in selected]
    while available and len(selected) < max_frames:
        existing = tuple(selected)
        best_frame = max(
            available,
            key=lambda frame_idx: (
                min(abs(frame_idx - existing_idx) for existing_idx in existing),
                -frame_idx,
            ),
        )
        time_us = round(best_frame * frame_duration_us) if frame_duration_us is not None else None
        selected[best_frame] = ReviewFrame(
            frame_idx=best_frame,
            time_us=time_us,
            shot_id=_shot_id_for_frame(blueprint, best_frame),
            kind="timeline_sample",
        )
        available.remove(best_frame)


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

    frame_count, frame_duration_us = _timebase_values(blueprint)
    if frame_count is not None:
        selected = {
            frame_idx: review_frame
            for frame_idx, review_frame in selected.items()
            if 0 <= frame_idx < frame_count
        }

    if len(selected) > max_frames:
        rows = [selected[idx] for idx in sorted(selected)]
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

    if frame_count is not None:
        _supplement_review_frames(
            selected,
            blueprint=blueprint,
            frame_count=frame_count,
            frame_duration_us=frame_duration_us,
            max_frames=max_frames,
        )

    return [selected[idx] for idx in sorted(selected)]


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


def _frame_array(series: Any, frame_idx: int) -> np.ndarray | None:
    if not isinstance(series, np.ndarray) or frame_idx >= len(series):
        return None
    return np.asarray(series[frame_idx])


def _finite_bbox(value: np.ndarray | None) -> bool:
    return value is not None and value.shape == (4,) and bool(np.isfinite(value).all())


def _finite_landmarks(value: np.ndarray | None, expected_count: int) -> bool:
    return (
        value is not None
        and value.shape == (expected_count, 2)
        and bool(np.isfinite(value).all())
    )


def _draw_bbox(frame: np.ndarray, bbox: np.ndarray | None, *, thickness: int = 1) -> None:
    if not _finite_bbox(bbox):
        return
    assert bbox is not None
    x1, y1, x2, y2 = (round(float(value)) for value in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)


def _draw_character_evidence(
    frame: np.ndarray,
    *,
    character_id: str,
    bbox: np.ndarray | None,
    keypoints: np.ndarray | None,
    confidence: np.ndarray | None,
    confidence_threshold: float,
) -> None:
    if _finite_bbox(bbox):
        assert bbox is not None
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


def _draw_face_evidence(
    frame: np.ndarray,
    *,
    bbox: np.ndarray | None,
    landmarks: np.ndarray | None,
) -> bool:
    if not _finite_landmarks(landmarks, 478):
        return False
    _draw_bbox(frame, bbox)
    assert landmarks is not None
    for point in landmarks[::_FACE_POINT_STRIDE]:
        cv2.circle(
            frame,
            (round(float(point[0])), round(float(point[1]))),
            1,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
    return True


def _draw_hand_evidence(
    frame: np.ndarray,
    *,
    bbox: np.ndarray | None,
    landmarks: np.ndarray | None,
) -> bool:
    if not _finite_landmarks(landmarks, 21):
        return False
    _draw_bbox(frame, bbox)
    assert landmarks is not None
    for first, second in HAND_EDGES:
        p1 = landmarks[first]
        p2 = landmarks[second]
        cv2.line(
            frame,
            (round(float(p1[0])), round(float(p1[1]))),
            (round(float(p2[0])), round(float(p2[1]))),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for point in landmarks:
        cv2.circle(
            frame,
            (round(float(point[0])), round(float(point[1]))),
            2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
    return True


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

            raw_face = character.get("face")
            face: dict[str, Any] = raw_face if isinstance(raw_face, dict) else {}
            face_landmarks = _npz_array(archive, face.get("landmarks_2d_ref"))
            face_bbox = _npz_array(archive, face.get("bbox_ref"))

            raw_hands = character.get("hands")
            hands: dict[str, Any] = raw_hands if isinstance(raw_hands, dict) else {}
            hand_payloads: dict[str, dict[str, np.ndarray | None]] = {}
            for side in ("left", "right"):
                raw_hand = hands.get(side)
                hand: dict[str, Any] = raw_hand if isinstance(raw_hand, dict) else {}
                hand_payloads[side] = {
                    "landmarks": _npz_array(archive, hand.get("landmarks_2d_ref")),
                    "bbox": _npz_array(archive, hand.get("bbox_ref")),
                }

            character_payloads.append(
                {
                    "character_id": character_id,
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "confidence": confidence,
                    "face_landmarks": face_landmarks,
                    "face_bbox": face_bbox,
                    "hands": hand_payloads,
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
                detail_presence = {"face": False, "left_hand": False, "right_hand": False}
                for payload in character_payloads:
                    frame_idx = review_frame.frame_idx
                    bbox = _frame_array(payload["bbox"], frame_idx)
                    keypoints = _frame_array(payload["keypoints"], frame_idx)
                    confidence = _frame_array(payload["confidence"], frame_idx)
                    if _finite_bbox(bbox):
                        visible_characters.append(str(payload["character_id"]))
                    _draw_character_evidence(
                        evidence_frame,
                        character_id=str(payload["character_id"]),
                        bbox=bbox,
                        keypoints=keypoints,
                        confidence=confidence,
                        confidence_threshold=confidence_threshold,
                    )

                    face_landmarks = _frame_array(payload["face_landmarks"], frame_idx)
                    face_bbox = _frame_array(payload["face_bbox"], frame_idx)
                    detail_presence["face"] = _draw_face_evidence(
                        evidence_frame,
                        bbox=face_bbox,
                        landmarks=face_landmarks,
                    ) or detail_presence["face"]

                    hand_payloads = payload["hands"]
                    if isinstance(hand_payloads, dict):
                        for side in ("left", "right"):
                            hand_payload = hand_payloads.get(side)
                            if not isinstance(hand_payload, dict):
                                continue
                            hand_landmarks = _frame_array(hand_payload.get("landmarks"), frame_idx)
                            hand_bbox = _frame_array(hand_payload.get("bbox"), frame_idx)
                            key = f"{side}_hand"
                            detail_presence[key] = _draw_hand_evidence(
                                evidence_frame,
                                bbox=hand_bbox,
                                landmarks=hand_landmarks,
                            ) or detail_presence[key]

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
                        "detail_presence": detail_presence,
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
    review_frame_detail_counts = {
        key: sum(1 for row in frame_rows if row["detail_presence"][key])
        for key in ("face", "left_hand", "right_hand")
    }
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
        "review_frame_detail_counts": review_frame_detail_counts,
        "review_frames": frame_rows,
        "review_instructions": [
            "Compare the left normalized analysis frame with the right evidence overlay from the same frame index.",
            "Reject incorrect person boxes, pose geometry, face landmarks, hand landmarks, missed people, duplicated tracks, or evidence attached to the wrong subject.",
            "Face and hand evidence is drawn only when measured landmarks exist; missing geometry is not interpolated or synthesized.",
            "The raw source SHA is verified separately; evidence overlays must be reviewed on the normalized analysis timeline used by the pipeline.",
            "A green machine gate is not a perceptual acceptance result; record a manual verdict separately.",
        ],
    }
    manifest_path = output_dir / "e13_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
