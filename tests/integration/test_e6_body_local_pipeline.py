from __future__ import annotations

import hashlib
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core import real_media_pipeline
from packages.pipeline_core.body_local_frame import BodyLocalFrameConfig
from packages.pipeline_core.camera_motion import CameraMotionConfig
from packages.pipeline_core.e6_media_pipeline import run_e6_media_pipeline
from packages.pipeline_core.person_mask import PersonMaskObservation
from packages.pipeline_core.person_tracking import PersonDetection
from packages.pipeline_core.pose_estimation import PoseObservation


class _Detector:
    config_sha256 = "1" * 64
    weights_sha256 = "2" * 64
    config = SimpleNamespace(score_threshold=0.3)

    def detect(self, frame, frame_idx):
        height, width = frame.shape[:2]
        return [PersonDetection(frame_idx, (60.0, 28.0, width - 60.0, height - 18.0), 0.95)]

    def provenance(self, *, code_commit, config_hash):
        return {
            "module": "person_detection",
            "tool": "test detector",
            "version": "test",
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


class _Pose:
    config_sha256 = "3" * 64
    weights_sha256 = "4" * 64

    def estimate(self, frame, bbox_xyxy, frame_idx, character_id):
        x1, y1, x2, y2 = bbox_xyxy
        center_x = (x1 + x2) * 0.5
        width = x2 - x1
        height = y2 - y1
        points = np.full((17, 2), np.nan, dtype=np.float32)
        confidence = np.zeros(17, dtype=np.float32)
        points[5] = (center_x + width * 0.18, y1 + height * 0.28)
        points[6] = (center_x - width * 0.18, y1 + height * 0.28)
        points[11] = (center_x + width * 0.10, y1 + height * 0.68)
        points[12] = (center_x - width * 0.10, y1 + height * 0.68)
        confidence[[5, 6, 11, 12]] = 0.95
        return PoseObservation(frame_idx, character_id, points, confidence)

    def provenance(self, *, code_commit, config_hash):
        return {
            "module": "pose_2d",
            "tool": "test pose",
            "version": "test",
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


class _Segmenter:
    config_sha256 = "5" * 64

    def segment(self, frame, bbox_xyxy, frame_idx, character_id):
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[25 : height - 15, 55 : width - 55] = True
        return PersonMaskObservation(frame_idx, character_id, mask, 0.95)

    def provenance(self, *, code_commit, config_hash):
        return {
            "module": "person_mask",
            "tool": "test segmenter",
            "version": "test",
            "code_commit": code_commit,
            "weights_sha256": "6" * 64,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


@pytest.fixture()
def translated_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")
    height, width = 128, 192
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(8, height, 16):
        for x in range(8, width, 16):
            value = 80 + ((x + y) % 120)
            cv2.circle(base, (x, y), 2, (value, value, value), -1)
    cv2.rectangle(base, (72, 25), (120, 110), (110, 130, 150), -1)
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(8):
        transform = np.asarray([[1.0, 0.0, index], [0.0, 1.0, index * 0.5]], dtype=np.float32)
        frame = cv2.warpAffine(base, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)
        assert cv2.imwrite(str(frames / f"{index:03d}.png"), frame)
    path = tmp_path / "e6_source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            "4",
            "-i",
            str(frames / "%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e6_wrapper_emits_valid_body_local_frames(monkeypatch, translated_video):
    monkeypatch.setattr(
        real_media_pipeline.RTMDetPersonDetector,
        "from_environment",
        classmethod(lambda cls: _Detector()),
    )
    monkeypatch.setattr(
        real_media_pipeline.RTMPosePoseEstimator,
        "from_environment",
        classmethod(lambda cls: _Pose()),
    )
    monkeypatch.setattr(
        real_media_pipeline.MediaPipeFaceHandRefiner,
        "from_environment",
        classmethod(lambda cls, *, fps_num, fps_den: None),
    )

    blueprint, sidecars = run_e6_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=translated_video.name,
        video_sha256=_sha256(translated_video),
        video_path=str(translated_video),
        person_mask_segmenter=_Segmenter(),
        camera_motion_config=CameraMotionConfig(max_points=120, min_matches=6),
        body_local_frame_config=BodyLocalFrameConfig(anchor_confidence_threshold=0.30),
    )

    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors
    assert blueprint["processing"]["pipeline_version"] == "0.6.0-e6.1"
    assert blueprint["processing"]["stages"][-1]["name"] == "body_local_frame"
    extension = blueprint["extensions"]["e6_body_local_frame"]
    assert extension["enabled"] is True
    assert extension["camera_stabilization_applied"] is True
    assert extension["coordinate_frame"] == "body_local_2d"
    assert extension["identity_inference_performed"] is False
    assert extension["biometric_embedding_exported"] is False

    assert blueprint["characters"]
    for character in blueprint["characters"]:
        ref = character["surface_motion"]["body_frame_transform_ref"]
        assert ref["coordinate_space"] == "body_local_2d"
        assert ref["interpolation_policy"] == "none"
        path_value = sidecars[ref["uri"]]
        with np.load(path_value, allow_pickle=False) as arrays:
            source_to_body = arrays["source_pixel_to_body_local"]
            inverse = arrays["body_local_to_source_pixel"]
            valid_frame = arrays["valid_frame"]
            confidence = arrays["anchor_confidence"]
        assert np.count_nonzero(valid_frame) > 0
        assert np.all(np.isfinite(source_to_body[valid_frame]))
        assert np.all(np.isfinite(inverse[valid_frame]))
        assert np.all(confidence[valid_frame] >= 0.30)
    assert any(report["kind"] == "body_local_frame_2d" for report in blueprint["artifacts"]["reports"])
    assert any(tool["module"] == "body_local_frame" for tool in blueprint["provenance"]["tools"])
