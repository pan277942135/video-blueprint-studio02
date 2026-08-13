from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from scenedetect import SceneManager, StatsManager, open_video
from scenedetect.detectors import ContentDetector


class ShotDetectionError(RuntimeError):
    """Raised when real shot detection or keyframe extraction cannot complete safely."""


@dataclass(frozen=True)
class ShotDetectionConfig:
    threshold: float = 27.0
    min_scene_len_frames: int = 15


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_us(frame_idx: int, fps_num: int, fps_den: int) -> int:
    return round(frame_idx * 1_000_000 * fps_den / fps_num)


def _sharpness_score(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(variance / (variance + 100.0), 6) if variance > 0 else 0.0


def _transition_score(
    stats_manager: StatsManager,
    frame_idx: int,
    threshold: float,
) -> float:
    if frame_idx == 0:
        return 1.0

    try:
        metrics = stats_manager.get_metrics(frame_idx, [ContentDetector.FRAME_SCORE_KEY])
        raw_score = metrics[0] if metrics else None
    except (KeyError, IndexError, TypeError):
        raw_score = None

    if raw_score is None:
        return 0.0

    denominator = threshold if threshold > 0 else 1.0
    return round(max(0.0, min(1.0, float(raw_score) / denominator)), 6)


def _extract_keyframe(
    capture: cv2.VideoCapture,
    frame_idx: int,
    output_path: str,
) -> float:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ShotDetectionError(f"Failed to decode keyframe at frame {frame_idx}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, frame):
        raise ShotDetectionError(f"Failed to write keyframe PNG: {output_path}")
    return _sharpness_score(frame)


def detect_shots(
    video_path: str,
    *,
    frame_count: int,
    fps_num: int,
    fps_den: int,
    output_dir: str,
    config: ShotDetectionConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect real shot boundaries and emit bundle-ready keyframe artifacts.

    The input must be the normalized CFR analysis video. Scene boundaries are
    detected with PySceneDetect ContentDetector. Every returned shot covers a
    contiguous inclusive frame interval, and every manifest keyframe URI is
    backed by a real PNG sidecar extracted from the normalized source.
    """
    cfg = config or ShotDetectionConfig()

    if not os.path.isfile(video_path):
        raise ShotDetectionError(f"Normalized analysis video does not exist: {video_path}")
    if frame_count < 1 or fps_num < 1 or fps_den < 1:
        raise ShotDetectionError("Invalid normalized timebase for shot detection")
    if cfg.min_scene_len_frames < 1:
        raise ShotDetectionError("min_scene_len_frames must be >= 1")

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)
    scene_manager.add_detector(
        ContentDetector(
            threshold=cfg.threshold,
            min_scene_len=cfg.min_scene_len_frames,
        )
    )

    video = open_video(video_path)
    try:
        scene_manager.detect_scenes(video=video, show_progress=False)
        detected = scene_manager.get_scene_list(start_in_scene=True)
    finally:
        video.reset()

    spans: list[tuple[int, int]] = []
    for start_tc, end_tc in detected:
        start = max(0, int(start_tc.get_frames()))
        end_exclusive = min(frame_count, int(end_tc.get_frames()))
        if end_exclusive <= start:
            continue
        spans.append((start, end_exclusive - 1))

    if not spans:
        spans = [(0, frame_count - 1)]

    # Enforce a contiguous, complete normalized timeline. If a detector/backend
    # ever returns inconsistent scene bounds, fail instead of silently creating
    # gaps or overlaps in the blueprint.
    expected_start = 0
    for start, end in spans:
        if start != expected_start or end < start:
            raise ShotDetectionError(
                f"Non-contiguous shot timeline: expected frame {expected_start}, got {start}-{end}"
            )
        expected_start = end + 1
    if expected_start != frame_count:
        raise ShotDetectionError(
            f"Shot timeline ended at frame {expected_start - 1}; expected {frame_count - 1}"
        )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ShotDetectionError(f"OpenCV could not open normalized analysis video: {video_path}")

    shots: list[dict[str, Any]] = []
    sidecars: dict[str, Any] = {}
    try:
        for index, (start, end) in enumerate(spans):
            shot_id = f"shot_{index:03d}"
            candidates = [
                ("first", start),
                ("middle", (start + end) // 2),
                ("last", end),
            ]

            keyframes: list[dict[str, Any]] = []
            seen_frames: set[int] = set()
            for kind, frame_idx in candidates:
                if frame_idx in seen_frames:
                    continue
                seen_frames.add(frame_idx)
                uri = f"artifacts/keyframes/{shot_id}_{frame_idx:06d}.png"
                output_path = os.path.join(output_dir, uri.replace("/", os.sep))
                score = _extract_keyframe(capture, frame_idx, output_path)
                sidecars[uri] = output_path
                keyframes.append(
                    {
                        "frame_idx": frame_idx,
                        "time_us": _time_us(frame_idx, fps_num, fps_den),
                        "kind": kind,
                        "score": score,
                        "image_uri": uri,
                    }
                )

            coverage = (end - start + 1) / frame_count
            shots.append(
                {
                    "shot_id": shot_id,
                    "index": index,
                    "frame_start": start,
                    "frame_end": end,
                    "time_start_us": _time_us(start, fps_num, fps_den),
                    "time_end_us": _time_us(end, fps_num, fps_den),
                    "cut_in_type": "start" if index == 0 else "hard_cut",
                    "cut_out_type": "end" if index == len(spans) - 1 else "hard_cut",
                    "transition_score": _transition_score(stats_manager, start, cfg.threshold),
                    "keyframes": keyframes,
                    "dominant_character_ids": [],
                    "camera_motion_id": None,
                    "quality": {
                        "score": 1.0,
                        "coverage": round(coverage, 6),
                        "warnings": [],
                        "errors": [],
                    },
                }
            )
    finally:
        capture.release()

    # Include detector provenance as a compact report artifact so the manifest
    # remains free of dense analysis data while the bundle stays auditable.
    report_uri = "artifacts/reports/shot_detection.json"
    report = {
        "backend": "pyscenedetect_content_detector",
        "threshold": cfg.threshold,
        "min_scene_len_frames": cfg.min_scene_len_frames,
        "frame_count": frame_count,
        "shot_count": len(shots),
        "normalized_video_sha256": _sha256_file(video_path),
        "shots": [
            {
                "shot_id": shot["shot_id"],
                "frame_start": shot["frame_start"],
                "frame_end": shot["frame_end"],
                "transition_score": shot["transition_score"],
            }
            for shot in shots
        ],
    }
    sidecars[report_uri] = report

    return shots, sidecars
