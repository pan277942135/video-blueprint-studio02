import hashlib
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core import real_media_pipeline
from packages.pipeline_core.person_tracking import PersonDetection
from packages.pipeline_core.pose_estimation import PoseObservation


class _FakeApprovedDetector:
    config_sha256 = "1" * 64
    weights_sha256 = "2" * 64
    config = SimpleNamespace(score_threshold=0.35)

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PersonDetection]:
        height, width = frame.shape[:2]
        return [
            PersonDetection(
                frame_idx=frame_idx,
                bbox_xyxy=(width * 0.25, height * 0.1, width * 0.75, height * 0.9),
                score=0.92,
            )
        ]

    def provenance(self, *, code_commit: str | None, config_hash: str):
        return {
            "module": "person_detection",
            "tool": "MMDetection RTMDet",
            "version": "test-adapter",
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


class _FakeApprovedPoseEstimator:
    config_sha256 = "3" * 64
    weights_sha256 = "4" * 64

    def estimate(self, frame, bbox_xyxy, frame_idx, character_id):
        x1, y1, x2, y2 = bbox_xyxy
        xs = np.linspace(x1, x2, 17, dtype=np.float32)
        ys = np.linspace(y1, y2, 17, dtype=np.float32)
        return PoseObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            keypoints_xy=np.stack([xs, ys], axis=1),
            confidence=np.full((17,), 0.85, dtype=np.float32),
        )

    def provenance(self, *, code_commit: str | None, config_hash: str):
        return {
            "module": "pose_2d",
            "tool": "MMPose RTMPose",
            "version": "test-adapter",
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


@pytest.fixture()
def short_pose_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for pose pipeline integration test")
    video_path = tmp_path / "pose_pipeline_source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=160x120:rate=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    return video_path


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_real_pipeline_with_enabled_pose_is_schema_valid(monkeypatch, short_pose_video):
    fake_detector = _FakeApprovedDetector()
    fake_pose = _FakeApprovedPoseEstimator()
    monkeypatch.setattr(
        real_media_pipeline.RTMDetPersonDetector,
        "from_environment",
        classmethod(lambda cls: fake_detector),
    )
    monkeypatch.setattr(
        real_media_pipeline.RTMPosePoseEstimator,
        "from_environment",
        classmethod(lambda cls: fake_pose),
    )

    blueprint, sidecars = real_media_pipeline.run_real_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=short_pose_video.name,
        video_sha256=_sha256(short_pose_video),
        video_path=str(short_pose_video),
    )

    validator = BlueprintValidator()
    valid, errors = validator.validate(blueprint)
    assert valid, errors

    assert len(blueprint["characters"]) == 1
    character = blueprint["characters"][0]
    assert character["pose"]["enabled"] is True
    assert character["pose"]["skeleton_name"] == "coco17"
    assert character["pose"]["keypoint_count"] == 17
    assert character["pose"]["keypoints_2d_ref"]["uri"] in sidecars
    assert character["pose"]["confidence_ref"]["uri"] in sidecars
    assert blueprint["extensions"]["e3_pose_2d"]["enabled"] is True
    assert blueprint["processing"]["stages"][3]["status"] == "succeeded"
    assert blueprint["processing"]["pipeline_version"] == "0.3.0-e3.1"
    assert {tool["module"] for tool in blueprint["provenance"]["tools"]} >= {
        "person_detection",
        "person_tracking",
        "pose_2d",
    }

    pose_reports = [report for report in blueprint["artifacts"]["reports"] if report["kind"] == "pose_2d"]
    assert len(pose_reports) == 1
    assert pose_reports[0]["uri"] in sidecars
    assert character["privacy"] == {
        "biometric_embedding_exported": False,
        "identity_inference_performed": False,
    }
