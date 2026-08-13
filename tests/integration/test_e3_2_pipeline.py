import hashlib
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core import real_media_pipeline
from packages.pipeline_core.face_hand_refinement import (
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    FaceHandObservation,
    FaceObservation,
    HandObservation,
)
from packages.pipeline_core.person_tracking import PersonDetection


class _FakeApprovedDetector:
    config_sha256 = "1" * 64
    weights_sha256 = "2" * 64
    config = SimpleNamespace(score_threshold=0.35)

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PersonDetection]:
        height, width = frame.shape[:2]
        return [
            PersonDetection(
                frame_idx=frame_idx,
                bbox_xyxy=(width * 0.1, height * 0.05, width * 0.9, height * 0.95),
                score=0.94,
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


class _FakeApprovedFaceHandRefiner:
    config_sha256 = "3" * 64
    face_task_sha256 = "4" * 64
    hand_task_sha256 = "5" * 64
    weights_sha256 = "6" * 64

    def __init__(self):
        self.closed = False

    def refine(self, frame, bbox_xyxy, frame_idx, character_id):
        face_x = np.linspace(40.0, 100.0, FACE_LANDMARK_COUNT, dtype=np.float32)
        face_y = np.linspace(20.0, 85.0, FACE_LANDMARK_COUNT, dtype=np.float32)
        face = FaceObservation(
            landmarks_xy=np.column_stack([face_x, face_y]),
            bbox_xyxy=(38.0, 18.0, 102.0, 88.0),
            confidence=0.91,
        )
        hand_x = np.linspace(105.0, 132.0, HAND_LANDMARK_COUNT, dtype=np.float32)
        hand_y = np.linspace(48.0, 92.0, HAND_LANDMARK_COUNT, dtype=np.float32)
        hand = HandObservation(
            side="left",
            landmarks_xy=np.column_stack([hand_x, hand_y]),
            bbox_xyxy=(103.0, 46.0, 134.0, 94.0),
            confidence=0.84,
            handedness_confidence=0.96,
        )
        return FaceHandObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            face=face,
            hands=(hand,),
        )

    def close(self):
        self.closed = True

    def provenance(self, *, code_commit: str | None, config_hash: str):
        return {
            "module": "face_hands_2d",
            "tool": "MediaPipe FaceLandmarker + HandLandmarker",
            "version": "test-adapter",
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": "TEST-ONLY",
        }


@pytest.fixture()
def short_face_hand_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for E3.2 integration test")
    video_path = tmp_path / "face_hand_pipeline_source.mp4"
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


def test_real_pipeline_with_enabled_face_hands_is_schema_valid(monkeypatch, short_face_hand_video):
    fake_detector = _FakeApprovedDetector()
    fake_refiner = _FakeApprovedFaceHandRefiner()
    monkeypatch.setattr(
        real_media_pipeline.RTMDetPersonDetector,
        "from_environment",
        classmethod(lambda cls: fake_detector),
    )
    monkeypatch.setattr(
        real_media_pipeline.RTMPosePoseEstimator,
        "from_environment",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        real_media_pipeline.MediaPipeFaceHandRefiner,
        "from_environment",
        classmethod(lambda cls, *, fps_num, fps_den: fake_refiner),
    )

    blueprint, sidecars = real_media_pipeline.run_real_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=short_face_hand_video.name,
        video_sha256=_sha256(short_face_hand_video),
        video_path=str(short_face_hand_video),
    )

    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors
    assert blueprint["schema_version"] == "1.1.0"
    assert blueprint["processing"]["pipeline_version"] == "0.3.0-e3.2"
    assert blueprint["processing"]["stages"][4]["name"] == "face_hands_2d"
    assert blueprint["processing"]["stages"][4]["status"] == "succeeded"
    assert blueprint["extensions"]["e3_face_hands_2d"]["enabled"] is True
    assert blueprint["extensions"]["e3_face_hands_2d"]["face_landmark_count"] == 478
    assert blueprint["extensions"]["e3_face_hands_2d"]["hand_landmark_count"] == 21
    assert blueprint["extensions"]["e3_face_hands_2d"]["identity_inference_performed"] is False
    assert blueprint["extensions"]["e3_face_hands_2d"]["biometric_embedding_exported"] is False

    assert len(blueprint["characters"]) == 1
    character = blueprint["characters"][0]
    assert character["face"]["enabled"] is True
    assert character["face"]["landmark_count"] == 478
    assert character["face"]["landmarks_2d_ref"]["uri"] in sidecars
    assert character["face"]["landmarks_3d_ref"] is None
    assert character["face"]["blendshapes_ref"] is None
    assert character["face"]["transform_ref"] is None
    assert character["hands"]["left"]["enabled"] is True
    assert character["hands"]["left"]["landmark_count"] == 21
    assert character["hands"]["left"]["landmarks_2d_ref"]["uri"] in sidecars
    assert character["hands"]["left"]["landmarks_world_ref"] is None
    assert character["hands"]["right"]["enabled"] is False
    assert character["hands"]["right"]["landmarks_2d_ref"] is None
    assert character["privacy"] == {
        "biometric_embedding_exported": False,
        "identity_inference_performed": False,
    }
    assert fake_refiner.closed is True

    modules = {tool["module"] for tool in blueprint["provenance"]["tools"]}
    assert "face_hands_2d" in modules
    reports = [report for report in blueprint["artifacts"]["reports"] if report["kind"] == "face_hands_refinement"]
    assert len(reports) == 1
    assert reports[0]["uri"] in sidecars
