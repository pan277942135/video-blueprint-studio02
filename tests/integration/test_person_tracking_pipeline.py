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


@pytest.fixture()
def short_real_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for person tracking integration test")

    video_path = tmp_path / "person_tracking_source.mp4"
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


def test_real_pipeline_with_enabled_person_tracking_is_schema_valid(
    monkeypatch,
    short_real_video,
):
    fake_detector = _FakeApprovedDetector()
    monkeypatch.setattr(
        real_media_pipeline.RTMDetPersonDetector,
        "from_environment",
        classmethod(lambda cls: fake_detector),
    )

    blueprint, sidecars = real_media_pipeline.run_real_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=short_real_video.name,
        video_sha256=_sha256(short_real_video),
        video_path=str(short_real_video),
    )

    validator = BlueprintValidator()
    valid, errors = validator.validate(blueprint)
    assert valid, errors

    assert len(blueprint["characters"]) == 1
    character = blueprint["characters"][0]
    assert character["privacy"] == {
        "biometric_embedding_exported": False,
        "identity_inference_performed": False,
    }
    assert character["bbox_ref"]["uri"] in sidecars
    assert character["visibility_ref"]["uri"] in sidecars
    assert blueprint["shots"][0]["dominant_character_ids"] == ["char_000"]
    assert blueprint["extensions"]["e2_person_tracking"]["enabled"] is True
    assert blueprint["processing"]["stages"][2]["status"] == "succeeded"
    assert {tool["module"] for tool in blueprint["provenance"]["tools"]} >= {
        "person_detection",
        "person_tracking",
    }

    tracking_reports = [
        report for report in blueprint["artifacts"]["reports"] if report["kind"] == "person_tracking"
    ]
    assert len(tracking_reports) == 1
    assert tracking_reports[0]["uri"] in sidecars
    assert blueprint["artifacts"]["overlays"]
    assert blueprint["artifacts"]["overlays"][0]["uri"] in sidecars
