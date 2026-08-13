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
    run_face_hand_refinement,
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
                score=0.93,
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


class _FakeFaceHandRefiner:
    def refine(self, frame, bbox_xyxy, frame_idx, character_id):
        height, width = frame.shape[:2]
        face = FaceObservation(
            landmarks_xy=np.column_stack(
                [
                    np.linspace(width * 0.35, width * 0.65, FACE_LANDMARK_COUNT, dtype=np.float32),
                    np.linspace(height * 0.15, height * 0.45, FACE_LANDMARK_COUNT, dtype=np.float32),
                ]
            ),
            bbox_xyxy=(width * 0.32, height * 0.12, width * 0.68, height * 0.48),
            confidence=0.90,
        )
        hands = []
        if frame_idx == 1:
            hands.append(self._hand(frame, side="left", confidence=0.84))
        if frame_idx == 3:
            hands.append(self._hand(frame, side="right", confidence=0.86))
        return FaceHandObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            face=face,
            hands=tuple(hands),
        )

    @staticmethod
    def _hand(frame, *, side, confidence):
        height, width = frame.shape[:2]
        if side == "left":
            x1, x2 = width * 0.15, width * 0.30
        else:
            x1, x2 = width * 0.70, width * 0.85
        y1, y2 = height * 0.48, height * 0.78
        return HandObservation(
            side=side,
            landmarks_xy=np.column_stack(
                [
                    np.linspace(x1, x2, HAND_LANDMARK_COUNT, dtype=np.float32),
                    np.linspace(y1, y2, HAND_LANDMARK_COUNT, dtype=np.float32),
                ]
            ),
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=confidence,
            handedness_confidence=0.97,
        )


@pytest.fixture()
def short_face_hands_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for face/hands integration test")
    video_path = tmp_path / "face_hands_source.mp4"
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracking_to_injected_face_hands_is_schema_valid(monkeypatch, short_face_hands_video, tmp_path):
    fake_detector = _FakeApprovedDetector()
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

    blueprint, sidecars = real_media_pipeline.run_real_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=short_face_hands_video.name,
        video_sha256=_sha256(short_face_hands_video),
        video_path=str(short_face_hands_video),
    )
    normalized_path = sidecars["artifacts/normalized/analysis_cfr.mp4"]
    frame_count = int(blueprint["timebase"]["frame_count"])

    characters, e3_2_sidecars, report_ref = run_face_hand_refinement(
        str(normalized_path),
        characters=blueprint["characters"],
        frame_count=frame_count,
        refiner=_FakeFaceHandRefiner(),
        output_dir=str(tmp_path / "e3_2_artifacts"),
        sidecars=sidecars,
    )
    sidecars.update(e3_2_sidecars)
    blueprint["characters"] = characters
    blueprint["schema_version"] = "1.1.0"
    blueprint["artifacts"]["reports"].append(report_ref)

    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors
    assert len(characters) == 1
    character = characters[0]
    assert character["face"]["enabled"] is True
    assert character["face"]["landmark_count"] == FACE_LANDMARK_COUNT
    assert character["face"]["landmarks_2d_ref"]["uri"] in sidecars
    assert character["face"]["landmarks_3d_ref"] is None
    assert character["hands"]["left"]["enabled"] is True
    assert character["hands"]["right"]["enabled"] is True
    assert character["hands"]["left"]["landmark_count"] == HAND_LANDMARK_COUNT
    assert character["hands"]["right"]["landmark_count"] == HAND_LANDMARK_COUNT
    assert character["hands"]["left"]["landmarks_world_ref"] is None
    assert character["hands"]["right"]["landmarks_world_ref"] is None
    assert report_ref["uri"] in sidecars
    report = sidecars[report_ref["uri"]]
    assert report["identity_inference_performed"] is False
    assert report["biometric_embedding_exported"] is False
