from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.pipeline_core.rtmdet_backend import RTMDetBackendConfig, RTMDetPersonDetector
from packages.pipeline_core.rtmpose_backend import RTMPoseBackendConfig, RTMPosePoseEstimator

POSE_SHA256 = "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"
BODY_FRAME_ANCHORS = (5, 6, 11, 12)
DISTAL_SEGMENTS = (
    ("left_forearm", 7, 9),
    ("right_forearm", 8, 10),
    ("left_lower_leg", 13, 15),
    ("right_lower_leg", 14, 16),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic E8 distal-limb periodic geometry stimulus")
    parser.add_argument("--image", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--frequency-hz", type=float, default=1.0)
    parser.add_argument("--peak-shift-px", type=float, default=6.0)
    return parser.parse_args()


def _score_person(
    bbox: tuple[float, float, float, float],
    confidence: np.ndarray,
) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1) * (y2 - y1))
    anchor_confidence = float(np.min(confidence[list(BODY_FRAME_ANCHORS)]))
    return math.sqrt(area) * max(0.0, anchor_confidence)


def _choose_distal_segment(
    keypoints: np.ndarray,
    confidence: np.ndarray,
) -> tuple[str, int, int, np.ndarray, np.ndarray, float, float]:
    body_anchors = keypoints[list(BODY_FRAME_ANCHORS)]
    candidates: list[tuple[float, str, int, int, np.ndarray, np.ndarray, float, float]] = []
    for name, start_index, end_index in DISTAL_SEGMENTS:
        endpoint_confidence = float(min(confidence[start_index], confidence[end_index]))
        if not math.isfinite(endpoint_confidence) or endpoint_confidence < 0.45:
            continue
        start = keypoints[start_index].astype(np.float64)
        end = keypoints[end_index].astype(np.float64)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            continue
        length = float(np.linalg.norm(end - start))
        if length < 24.0:
            continue
        center = 0.5 * (start + end)
        anchor_distance = float(np.min(np.linalg.norm(body_anchors - center[None, :], axis=1)))
        score = length * endpoint_confidence * (1.0 + anchor_distance / max(length, 1.0))
        candidates.append((score, name, start_index, end_index, start, end, endpoint_confidence, anchor_distance))
    if not candidates:
        raise SystemExit("selected person has no reliable distal limb segment for E8 certification")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, name, start_index, end_index, start, end, endpoint_confidence, anchor_distance = candidates[0]
    return name, start_index, end_index, start, end, endpoint_confidence, anchor_distance


def _draw_tracking_texture(
    image: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    sigma_along: float,
    sigma_across: float,
) -> tuple[np.ndarray, int]:
    textured = image.copy()
    height, width = image.shape[:2]
    radius = 1.8 * max(sigma_along, sigma_across)
    step = max(4, round(min(sigma_along, sigma_across) * 0.45))
    x_min = max(2, math.floor(float(center[0] - radius)))
    x_max = min(width - 3, math.ceil(float(center[0] + radius)))
    y_min = max(2, math.floor(float(center[1] - radius)))
    y_max = min(height - 3, math.ceil(float(center[1] + radius)))
    dot_count = 0
    for y in range(y_min, y_max + 1, step):
        for x in range(x_min, x_max + 1, step):
            offset = np.asarray([x - float(center[0]), y - float(center[1])], dtype=np.float64)
            along = float(np.dot(offset, tangent)) / max(sigma_along, 1e-6)
            across = float(np.dot(offset, normal)) / max(sigma_across, 1e-6)
            if along * along + across * across > 1.8:
                continue
            local = textured[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
            mean_value = float(np.mean(local)) if local.size else 128.0
            value = 235 if mean_value < 150 else 25
            cv2.rectangle(textured, (x - 1, y - 1), (x + 1, y + 1), (value, value, value), -1)
            dot_count += 1
    return textured, dot_count


def main() -> int:
    args = _args()
    if args.fps <= 0.0 or args.frames < 20 or args.frequency_hz <= 0.0 or args.peak_shift_px <= 0.0:
        raise SystemExit("invalid E8 stimulus parameters")
    image_path = pathlib.Path(args.image).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"could not decode human sample: {image_path}")

    # MMEngine's default scope is process-global. Finish RTMDet inference before
    # RTMPose is initialized so the pose backend cannot switch the active scope
    # while mmdet is still building its inference transform pipeline.
    detector = RTMDetPersonDetector(
        RTMDetBackendConfig(
            config_path=str(pathlib.Path(args.det_config).resolve()),
            checkpoint_path=str(pathlib.Path(args.det_checkpoint).resolve()),
            weights_license="approved project RTMDet precedent",
            device="cpu",
            score_threshold=0.30,
        ),
        weights_approved=True,
    )
    detections = detector.detect(image, 0)
    if not detections:
        raise SystemExit("certified RTMDet found no person in official human sample")

    pose_estimator = RTMPosePoseEstimator(
        RTMPoseBackendConfig(
            config_path=str(pathlib.Path(args.pose_config).resolve()),
            checkpoint_path=str(pathlib.Path(args.pose_checkpoint).resolve()),
            weights_license="approved project RTMPose precedent",
            expected_weights_sha256=POSE_SHA256,
            device="cpu",
        ),
        weights_approved=True,
    )
    people: list[tuple[float, Any, Any]] = []
    for detection in detections:
        pose = pose_estimator.estimate(image, detection.bbox_xyxy, 0, "cert_candidate")
        if pose is None:
            continue
        anchor_confidence = pose.confidence[list(BODY_FRAME_ANCHORS)]
        if not np.all(np.isfinite(anchor_confidence)) or float(np.min(anchor_confidence)) < 0.45:
            continue
        try:
            _choose_distal_segment(pose.keypoints_xy.astype(np.float64), pose.confidence)
        except SystemExit:
            continue
        people.append((_score_person(detection.bbox_xyxy, pose.confidence), detection, pose))
    if not people:
        raise SystemExit("no detected person has reliable body-frame anchors and a distal limb segment")
    people.sort(key=lambda item: item[0], reverse=True)
    _, detection, pose = people[0]

    keypoints = pose.keypoints_xy.astype(np.float64)
    segment_name, start_index, end_index, start, end, segment_confidence, anchor_distance = _choose_distal_segment(
        keypoints,
        pose.confidence,
    )
    segment_vector = end - start
    segment_length = float(np.linalg.norm(segment_vector))
    tangent = segment_vector / segment_length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    center = 0.5 * (start + end)
    sigma_along = max(7.0, segment_length * 0.18)
    sigma_across = max(4.5, segment_length * 0.10)
    textured, dot_count = _draw_tracking_texture(
        image,
        center,
        tangent,
        normal,
        sigma_along,
        sigma_across,
    )
    if dot_count < 8:
        raise SystemExit("E8 distal-limb certification patch produced too few trackable texture points")

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    offset_x = grid_x - float(center[0])
    offset_y = grid_y - float(center[1])
    along = (offset_x * float(tangent[0]) + offset_y * float(tangent[1])) / sigma_along
    across = (offset_x * float(normal[0]) + offset_y * float(normal[1])) / sigma_across
    weight = np.exp(-0.5 * (along * along + across * across)).astype(np.float32)

    # Keep the segment endpoints and the four E6 shoulder/hip body-frame anchors
    # physically stationary. The periodic warp therefore represents local surface
    # geometry rather than a body-frame change that E6 is expected to absorb.
    guard_points = [start, end, *(keypoints[index] for index in BODY_FRAME_ANCHORS)]
    guard_radius = max(5.0, 0.12 * segment_length)
    guard = np.ones_like(weight, dtype=np.float32)
    for point in guard_points:
        distance2 = (grid_x - float(point[0])) ** 2 + (grid_y - float(point[1])) ** 2
        local_guard = 1.0 - np.exp(-0.5 * distance2 / (guard_radius * guard_radius)).astype(np.float32)
        guard *= local_guard
    weight *= guard

    frames_dir = pathlib.Path(args.frames_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    for frame_idx in range(args.frames):
        t = frame_idx / args.fps
        local_shift = args.peak_shift_px * math.sin(2.0 * math.pi * args.frequency_hz * t)
        map_x = grid_x - local_shift * float(normal[0]) * weight
        map_y = grid_y - local_shift * float(normal[1]) * weight
        locally_warped = cv2.remap(
            textured,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        camera = np.asarray(
            [[1.0, 0.0, frame_idx * 0.35], [0.0, 1.0, frame_idx * 0.15]],
            dtype=np.float32,
        )
        frame = cv2.warpAffine(
            locally_warped,
            camera,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE,
        )
        if not cv2.imwrite(str(frames_dir / f"{frame_idx:03d}.png"), frame):
            raise SystemExit("failed to write E8 certification frame")

    metadata = {
        "target_frequency_hz": args.frequency_hz,
        "frames": args.frames,
        "fps": args.fps,
        "local_peak_shift_px": args.peak_shift_px,
        "global_camera_dx_per_frame": 0.35,
        "global_camera_dy_per_frame": 0.15,
        "selected_person_bbox_xyxy": [round(float(value), 4) for value in detection.bbox_xyxy],
        "selected_person_score": round(float(detection.score), 6),
        "body_frame_anchor_confidence": [round(float(pose.confidence[index]), 6) for index in BODY_FRAME_ANCHORS],
        "selected_segment": segment_name,
        "selected_segment_indices": [start_index, end_index],
        "selected_segment_confidence": round(segment_confidence, 6),
        "selected_segment_length_px": round(segment_length, 4),
        "distance_to_nearest_body_frame_anchor_px": round(anchor_distance, 4),
        "segment_start_xy": [round(float(value), 4) for value in start],
        "segment_end_xy": [round(float(value), 4) for value in end],
        "stimulus_center_xy": [round(float(value), 4) for value in center],
        "stimulus_sigma_along_across": [round(sigma_along, 4), round(sigma_across, 4)],
        "stimulus_normal_xy": [round(float(value), 6) for value in normal],
        "guard_radius_px": round(guard_radius, 4),
        "tracking_texture_dots": dot_count,
        "stimulus_semantics": "non-physiological distal-limb-local periodic geometry",
    }
    metadata_path = pathlib.Path(args.metadata).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
