from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from packages.pipeline_core.face_hand_refinement import run_face_hand_refinement
from packages.pipeline_core.mediapipe_face_hand_backend import (
    APPROVED_FACE_TASK_SHA256,
    APPROVED_HAND_TASK_SHA256,
    MediaPipeFaceHandConfig,
    MediaPipeFaceHandRefiner,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_shape(path: str) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open certification video: {path}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError("Certification video metadata is invalid")
    return frame_count, width, height, fps


def _tracking_character(root: Path, frame_count: int, width: int, height: int) -> tuple[dict[str, Any], dict[str, Any]]:
    uri = "artifacts/timeseries/char_cert_tracking.npz"
    path = root / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    bboxes = np.tile(np.asarray([[0.0, 0.0, float(width), float(height)]], dtype=np.float32), (frame_count, 1))
    np.savez_compressed(path, bbox_xyxy=bboxes)
    checksum = _sha256_file(path)
    character = {
        "character_id": "char_cert",
        "bbox_ref": {
            "uri": uri,
            "format": "npz",
            "dtype": "float32",
            "shape": [frame_count, 4],
            "axes": ["frame", "bbox_component"],
            "unit": "px",
            "coordinate_space": "pixel_xy",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "checksum_sha256": checksum,
            "metadata": {"array_key": "bbox_xyxy"},
        },
        "privacy": {
            "identity_inference_performed": False,
            "biometric_embedding_exported": False,
        },
    }
    return character, {uri: str(path)}


def certify(args: argparse.Namespace) -> dict[str, Any]:
    face_sha256 = _sha256_file(args.face_task)
    hand_sha256 = _sha256_file(args.hand_task)
    if face_sha256 != APPROVED_FACE_TASK_SHA256:
        raise RuntimeError(f"Face task hash mismatch: {face_sha256}")
    if hand_sha256 != APPROVED_HAND_TASK_SHA256:
        raise RuntimeError(f"Hand task hash mismatch: {hand_sha256}")

    frame_count, width, height, fps = _video_shape(args.video)
    fps_num = int(round(fps * 1000))
    fps_den = 1000
    divisor = int(np.gcd(fps_num, fps_den))
    fps_num //= divisor
    fps_den //= divisor

    with tempfile.TemporaryDirectory(prefix="vbs_e3_2_cert_") as temp_dir:
        root = Path(temp_dir)
        character, sidecars = _tracking_character(root, frame_count, width, height)
        refiner = MediaPipeFaceHandRefiner(
            MediaPipeFaceHandConfig(
                face_task_path=args.face_task,
                hand_task_path=args.hand_task,
                fps_num=fps_num,
                fps_den=fps_den,
                crop_padding_ratio=0.0,
            ),
            artifacts_approved=True,
        )
        try:
            characters, refinement_sidecars, report_ref = run_face_hand_refinement(
                args.video,
                characters=[character],
                frame_count=frame_count,
                refiner=refiner,
                output_dir=str(root),
                sidecars=sidecars,
            )
        finally:
            refiner.close()

        if len(characters) != 1:
            raise RuntimeError(f"Expected one anonymous certification track, got {len(characters)}")
        result = characters[0]
        face = result["face"]
        hands = result["hands"]
        if face.get("enabled") is not True or face.get("landmark_count") != 478:
            raise RuntimeError(f"Face certification failed: {face}")
        enabled_hands = [side for side in ("left", "right") if hands[side].get("enabled") is True]
        if not enabled_hands:
            raise RuntimeError(f"Hand certification failed: no enabled hand track: {hands}")
        for side in enabled_hands:
            if hands[side].get("landmark_count") != 21:
                raise RuntimeError(f"{side} hand landmark count is not 21")
            if hands[side].get("landmarks_world_ref") is not None:
                raise RuntimeError("E3.2 must not export hand world landmarks")
        if face.get("landmarks_3d_ref") is not None:
            raise RuntimeError("E3.2 must not export face 3D landmarks")
        if face.get("blendshapes_ref") is not None or face.get("transform_ref") is not None:
            raise RuntimeError("E3.2 must not export blendshapes or facial transformation matrices")
        if result["privacy"] != {
            "identity_inference_performed": False,
            "biometric_embedding_exported": False,
        }:
            raise RuntimeError(f"Privacy invariant failed: {result['privacy']}")

        face_ref = face["landmarks_2d_ref"]
        face_path = refinement_sidecars.get(face_ref["uri"])
        if not isinstance(face_path, str) or not os.path.isfile(face_path):
            raise RuntimeError("Face sidecar is missing")
        with np.load(face_path, allow_pickle=False) as data:
            face_points = np.asarray(data["face_landmarks_2d"])
            if face_points.shape != (frame_count, 478, 2):
                raise RuntimeError(f"Unexpected face sidecar shape: {face_points.shape}")
            face_detected_frames = int(np.isfinite(face_points).all(axis=(1, 2)).sum())
            hand_detected_frames = {
                side: int(np.isfinite(data[f"{side}_hand_landmarks_2d"]).all(axis=(1, 2)).sum())
                for side in ("left", "right")
            }
        if face_detected_frames <= 0:
            raise RuntimeError("FaceLandmarker produced no finite frame-aligned output")
        if max(hand_detected_frames.values()) <= 0:
            raise RuntimeError("HandLandmarker produced no finite frame-aligned output")

        report = refinement_sidecars.get(report_ref["uri"])
        if not isinstance(report, dict):
            raise RuntimeError("Face/hands refinement report missing")
        if report.get("identity_inference_performed") is not False:
            raise RuntimeError("Certification report violated identity-inference invariant")
        if report.get("biometric_embedding_exported") is not False:
            raise RuntimeError("Certification report violated biometric-export invariant")

        return {
            "certification": "E3.2 MediaPipe anonymous face/hands 2D",
            "passed": True,
            "video": {
                "path": args.video,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps_num": fps_num,
                "fps_den": fps_den,
            },
            "runtime": {
                "mediapipe": "0.10.35",
                "mode": "VIDEO",
            },
            "artifacts": {
                "face_task_sha256": face_sha256,
                "hand_task_sha256": hand_sha256,
            },
            "outputs": {
                "face_landmark_count": 478,
                "face_detected_frames": face_detected_frames,
                "enabled_hand_sides": enabled_hands,
                "hand_landmark_count": 21,
                "hand_detected_frames": hand_detected_frames,
                "face_3d_exported": False,
                "blendshapes_exported": False,
                "facial_transform_exported": False,
                "hand_world_landmarks_exported": False,
            },
            "privacy": {
                "identity_inference_performed": False,
                "biometric_embedding_exported": False,
                "cross_video_identity_association": False,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--face-task", required=True)
    parser.add_argument("--hand-task", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = certify(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
