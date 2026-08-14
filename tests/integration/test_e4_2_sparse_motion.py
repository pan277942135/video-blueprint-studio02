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
from packages.pipeline_core.sparse_motion import SparseMotionConfig


class _Detector:
    config_sha256 = "1" * 64
    weights_sha256 = "2" * 64
    config = SimpleNamespace(score_threshold=0.3)

    def detect(self, frame, frame_idx):
        height, width = frame.shape[:2]
        return [PersonDetection(frame_idx, (4.0, 4.0, width - 4.0, height - 4.0), 0.9)]

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
        mask[4 : height - 4, 4 : width - 4] = True
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
def motion_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")
    path = tmp_path / "e4_2_source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1.2:size=160x120:rate=10",
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


def test_e4_2_emits_physical_sparse_motion_sidecars(monkeypatch, motion_video):
    monkeypatch.setattr(
        real_media_pipeline.RTMDetPersonDetector,
        "from_environment",
        classmethod(lambda cls: _Detector()),
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

    config = SparseMotionConfig(max_points=32, reseed_below_ratio=0.5)
    blueprint, sidecars = run_e4_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=motion_video.name,
        video_sha256=_sha256(motion_video),
        video_path=str(motion_video),
        person_mask_segmenter=_Segmenter(),
        point_track_config=config,
    )

    valid_blueprint, validation_errors = BlueprintValidator().validate(blueprint)
    assert valid_blueprint, validation_errors
    assert blueprint["processing"]["pipeline_version"] == "0.4.0-e4.2"
    assert blueprint["processing"]["stages"][-1]["name"] == "point_tracks"
    extension = blueprint["extensions"]["e4_point_tracks"]
    assert extension["enabled"] is True
    assert extension["coordinate_space"] == "pixel_xy"
    assert extension["mask_constrained"] is True
    assert extension["shot_boundary_reset"] is True
    assert extension["interpolation"] is False

    character_id = blueprint["characters"][0]["character_id"]
    ref = extension["characters"][character_id]["track_points_ref"]
    assert ref is not None
    assert ref["shape"] == [blueprint["timebase"]["frame_count"], 32, 2]
    assert ref["interpolation_policy"] == "none"
    assert ref["uri"] in sidecars

    with np.load(sidecars[ref["uri"]], allow_pickle=False) as bundle:
        positions = bundle["positions_xy"]
        valid = bundle["valid"]
        errors = bundle["tracking_error"]
        track_id = bundle["track_id"]
    assert positions.shape == tuple(ref["shape"])
    assert valid.shape == positions.shape[:2]
    assert errors.shape == valid.shape
    assert track_id.shape == valid.shape
    assert np.count_nonzero(valid) > 0
    assert np.all(np.isnan(positions[~valid]))
    assert np.all(track_id[~valid] == -1)
    assert np.all(track_id[valid] >= 0)
    assert any(report["kind"] == "point_tracks" for report in blueprint["artifacts"]["reports"])
    assert any(tool["module"] == "point_tracks" for tool in blueprint["provenance"]["tools"])
