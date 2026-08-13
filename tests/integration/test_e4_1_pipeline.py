import hashlib
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core import real_media_pipeline
from packages.pipeline_core.e4_media_pipeline import run_e4_media_pipeline
from packages.pipeline_core.person_mask import PersonMaskObservation
from packages.pipeline_core.person_tracking import PersonDetection


class _Detector:
    config_sha256 = "1" * 64
    weights_sha256 = "2" * 64
    config = SimpleNamespace(score_threshold=0.3)

    def detect(self, frame, frame_idx):
        height, width = frame.shape[:2]
        return [PersonDetection(frame_idx, (20.0, 10.0, width - 20.0, height - 10.0), 0.9)]

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
        yy, xx = np.ogrid[:height, :width]
        center_x = width / 2.0
        center_y = height / 2.0
        mask = ((xx - center_x) ** 2 / 35**2 + (yy - center_y) ** 2 / 45**2) <= 1.0
        return PersonMaskObservation(frame_idx, character_id, mask, 0.87)

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
    path = tmp_path / "e4_source.mp4"
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
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e4_wrapper_emits_schema_valid_person_masks(monkeypatch, short_video):
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

    blueprint, sidecars = run_e4_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=short_video.name,
        video_sha256=_sha256(short_video),
        video_path=str(short_video),
        person_mask_segmenter=_Segmenter(),
    )

    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors
    assert blueprint["processing"]["pipeline_version"] == "0.4.0-e4.1"
    assert blueprint["processing"]["stages"][-1]["name"] == "person_mask"
    assert blueprint["processing"]["stages"][-1]["status"] == "succeeded"
    assert blueprint["extensions"]["e4_person_mask"]["enabled"] is True
    assert blueprint["extensions"]["e4_person_mask"]["encoding"] == "row_major_binary_rle_v1"
    assert blueprint["quality"]["module_scores"]["person_mask"] == pytest.approx(0.87, abs=1e-6)

    character = blueprint["characters"][0]
    ref = character["person_mask_ref"]
    assert ref["format"] == "rle_json"
    assert ref["dtype"] == "bool"
    assert ref["coordinate_space"] == "pixel_xy"
    assert ref["interpolation_policy"] == "none"
    assert ref["uri"] in sidecars
    assert any(report["kind"] == "person_mask" for report in blueprint["artifacts"]["reports"])
    assert any(tool["module"] == "person_mask" for tool in blueprint["provenance"]["tools"])
