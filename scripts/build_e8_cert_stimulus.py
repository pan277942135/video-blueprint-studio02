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
ANCHOR_INDICES = (5, 6, 11, 12)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic E8 torso-local periodic geometry stimulus")
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


def _score_candidate(
    bbox: tuple[float, float, float, float],
    confidence: np.ndarray,
) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1) * (y2 - y1))
    anchor_confidence = float(np.min(confidence[list(ANCHOR_INDICES)]))
    return math.sqrt(area) * max(0.0, anchor_confidence)


def _draw_tracking_texture(
    image: np.ndarray,
    center: np.ndarray,
    sigma_x: float,
    sigma_y: float,
) -> tuple[np.ndarray, int]:
    textured = image.copy()
    height, width = image.shape[:2]
    step_x = max(5, round(sigma_x * 0.30))
    step_y = max(5, round(sigma_y * 0.30))
    x_min = max(2, math.floor(float(center[0] - 1.6 * sigma_x)))
    x_max = min(width - 3, math.ceil(float(center[0] + 1.6 * sigma_x)))
    y_min = max(2, math.floor(float(center[1] - 1.6 * sigma_y)))
    y_max = min(height - 3, math.ceil(float(center[1] + 1.6 * sigma_y)))
    dot_count = 0
    for y in range(y_min, y_max + 1, step_y):
        for x in range(x_min, x_max + 1, step_x):
            dx = (x - float(center[0])) / max(sigma_x, 1e-6)
            dy = (y - float(center[1])) / max(sigma_y, 1e-6)
            if dx * dx + dy * dy > 1.8:
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
    # to mmpose while mmdet is still building its inference transform pipeline.
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
    candidates: list[tuple[float, Any, Any]] = []
    for detection in detections:
        pose = pose_estimator.estimate(image, detection.bbox_xyxy, 0, "cert_candidate")
        if pose is None:
            continue
        anchor_confidence = pose.confidence[list(ANCHOR_INDICES)]
        if not np.all(np.isfinite(anchor_confidence)) or float(np.min(anchor_confidence)) < 0.45:
            continue
        candidates.append((_score_candidate(detection.bbox_xyxy, pose.confidence), detection, pose))
    if not candidates:
        raise SystemExit("no detected person has sufficiently reliable shoulder/hip anchors")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, detection, pose = candidates[0]

    keypoints = pose.keypoints_xy.astype(np.float64)
    left_shoulder, right_shoulder, left_hip, right_hip = (keypoints[index] for index in ANCHOR_INDICES)
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    hip_mid = 0.5 * (left_hip + right_hip)
    torso_vector = hip_mid - shoulder_mid
    torso_length = float(np.linalg.norm(torso_vector))
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    hip_width = float(np.linalg.norm(left_hip - right_hip))
    body_width = max(shoulder_width, hip_width)
    if torso_length < 20.0 or body_width < 16.0:
        raise SystemExit("selected person torso is too small for deterministic E8 certification")

    # Keep the stimulus well inside the four body-frame anchors. The Gaussian
    # center sits slightly above the torso midpoint and its 2-sigma extent does
    # not reach the shoulder/hip anchor rows on a normal upright subject.
    center = shoulder_mid + 0.47 * torso_vector
    sigma_x = max(5.0, body_width * 0.16)
    sigma_y = max(6.0, torso_length * 0.11)
    textured, dot_count = _draw_tracking_texture(image, center, sigma_x, sigma_y)
    if dot_count < 8:
        raise SystemExit("E8 torso certification patch produced too few trackable texture points")

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dx = (grid_x - float(center[0])) / sigma_x
    dy = (grid_y - float(center[1])) / sigma_y
    weight = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)

    # Explicitly suppress warp near the four RTMPose anchors so E6 body-frame
    # estimation does not absorb the injected local surface motion.
    anchor_guard = np.ones_like(weight, dtype=np.float32)
    guard_radius = max(5.0, 0.10 * torso_length)
    for anchor in (left_shoulder, right_shoulder, left_hip, right_hip):
        distance2 = (grid_x - float(anchor[0])) ** 2 + (grid_y - float(anchor[1])) ** 2
        local_guard = 1.0 - np.exp(-0.5 * distance2 / (guard_radius * guard_radius)).astype(np.float32)
        anchor_guard *= local_guard
    weight *= anchor_guard

    frames_dir = pathlib.Path(args.frames_dir).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    for frame_idx in range(args.frames):
        t = frame_idx / args.fps
        local_shift = args.peak_shift_px * math.sin(2.0 * math.pi * args.frequency_hz * t)
        map_x = grid_x
        map_y = grid_y - local_shift * weight
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
        "anchor_confidence": [round(float(pose.confidence[index]), 6) for index in ANCHOR_INDICES],
        "shoulder_mid_xy": [round(float(value), 4) for value in shoulder_mid],
        "hip_mid_xy": [round(float(value), 4) for value in hip_mid],
        "stimulus_center_xy": [round(float(value), 4) for value in center],
        "stimulus_sigma_xy": [round(sigma_x, 4), round(sigma_y, 4)],
        "anchor_guard_radius_px": round(guard_radius, 4),
        "tracking_texture_dots": dot_count,
        "stimulus_semantics": "non-physiological torso-local periodic geometry",
    }
    metadata_path = pathlib.Path(args.metadata).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
