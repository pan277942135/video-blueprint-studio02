from __future__ import annotations

import hashlib
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from packages.blueprint_schema import BlueprintValidator
from packages.pipeline_core import real_media_pipeline
from packages.pipeline_core.camera_motion import CameraMotionConfig
from packages.pipeline_core.e5_media_pipeline import run_e5_media_pipeline
from packages.pipeline_core.person_mask import PersonMaskObservation
from packages.pipeline_core.person_tracking import PersonDetection


class _Detector:
    config_sha256 = "1" * 64
    weights_sha256 = "2" * 64
    config = SimpleNamespace(score_threshold=0.3)

    def detect(self, frame, frame_idx):
        height, width = frame.shape[:2]
        return [PersonDetection(frame_idx, (40.0, 20.0, width - 40.0, height - 20.0), 0.9)]

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


class _Segmenter:
    config_sha256 = "3" * 64

    def segment(self, frame, bbox_xyxy, frame_idx, character_id):
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[25 : height - 25, 55 : width - 55] = True
        return PersonMaskObservation(frame_idx, character_id, mask, 0.9)

    def provenance(self, *, code_commit, config_hash):
        return {
            "module": "person_mask",
            "tool": "test segmenter",
            "version": "test",
            "code_commit": code_commit,
            "weights_sha256": "4" * 64,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


@pytest.fixture()
def short_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")
    path = tmp_path / "e5_source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=1.2:size=192x128:rate=6",
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


def test_e5_wrapper_emits_schema_valid_camera_evidence(monkeypatch, short_video):
    detector = _Detector()
    monkeypatch.setattr(
        real_media_pipeline.RTMDetPersonDetector,
        "from_environment",
        classmethod(lambda cls: detector),
    )
    monkeypatch.setattr(
        real_media_pipeline.RTMPosePoseEstimator,
        "from_environment",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        real_media_pipeline.MediaPipeFaceHandRefiner,
        "from_environment",
        classmethod(lambda cls, *, fps_num, fps_den: None),
    )

    blueprint, sidecars = run_e5_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=short_video.name,
        video_sha256=_sha256(short_video),
        video_path=str(short_video),
        person_mask_segmenter=_Segmenter(),
        camera_motion_config=CameraMotionConfig(max_points=80, min_matches=6),
    )

    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors
    assert blueprint["processing"]["pipeline_version"] == "0.5.0-e5.1"
    assert blueprint["processing"]["stages"][-1]["name"] == "camera_motion"
    extension = blueprint["extensions"]["e5_camera_motion"]
    assert extension["enabled"] is True
    assert extension["foreground_exclusion"] is True
    assert extension["identity_inference_performed"] is False
    assert extension["biometric_embedding_exported"] is False

    assert len(blueprint["camera"]["per_shot"]) == len(blueprint["shots"])
    for shot, row in zip(blueprint["shots"], blueprint["camera"]["per_shot"], strict=True):
        assert shot["camera_motion_id"] == row["camera_motion_id"]
        assert row["classification"] in {"static", "pan", "tilt", "roll", "zoom", "compound", "unknown"}
        assert row["classification"] not in {"dolly", "truck", "pedestal"}
        assert row["reconstruction_backend"] == "opencv_ransac_2d"
        assert row["intrinsics"] is None
        assert row["extrinsics_ref"] is None
        assert row["affine_ref"]["uri"] in sidecars
        assert row["background_tracks_ref"]["uri"] in sidecars
        assert row["zoom_proxy_ref"]["uri"] in sidecars
    assert any(report["kind"] == "camera_motion" for report in blueprint["artifacts"]["reports"])
    assert any(tool["module"] == "camera_motion" for tool in blueprint["provenance"]["tools"])
