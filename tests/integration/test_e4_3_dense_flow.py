import hashlib
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core import real_media_pipeline
from packages.pipeline_core.dense_flow import DenseFlowConfig, run_dense_flow
from packages.pipeline_core.e4_media_pipeline import run_e4_media_pipeline
from packages.pipeline_core.person_tracking import PersonDetection


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


@pytest.fixture()
def dense_flow_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")
    path = tmp_path / "e4_3_source.mp4"
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


def test_e4_3_emits_physical_global_dense_flow(monkeypatch, dense_flow_video):
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

    blueprint, sidecars = run_e4_media_pipeline(
        job_id=str(uuid.uuid4()),
        video_file_name=dense_flow_video.name,
        video_sha256=_sha256(dense_flow_video),
        video_path=str(dense_flow_video),
        dense_flow_config=DenseFlowConfig(max_side_px=96),
    )

    valid_blueprint, validation_errors = BlueprintValidator().validate(blueprint)
    assert valid_blueprint, validation_errors
    assert blueprint["processing"]["pipeline_version"] == "0.4.0-e4.3"
    assert blueprint["processing"]["stages"][-1]["name"] == "dense_flow"
    extension = blueprint["extensions"]["e4_dense_flow"]
    assert extension["enabled"] is True
    assert extension["algorithm"] == "opencv_farneback_v1"
    assert extension["coordinate_space"] == "pixel_xy"
    assert extension["shot_boundary_reset"] is True
    assert extension["interpolation"] is False

    ref = extension["flow_ref"]
    assert ref["uri"] in sidecars
    assert ref["shape"][0] == blueprint["timebase"]["frame_count"]
    assert ref["shape"][-1] == 2
    assert ref["metadata"]["source_width"] == blueprint["source_video"]["width"]
    assert ref["metadata"]["source_height"] == blueprint["source_video"]["height"]

    with np.load(sidecars[ref["uri"]], allow_pickle=False) as bundle:
        flow = bundle["flow_xy"]
        valid_frame = bundle["valid_frame"]
    assert flow.shape == tuple(ref["shape"])
    assert valid_frame.shape == (ref["shape"][0],)
    assert not bool(valid_frame[0])
    assert np.all(np.isnan(flow[0]))
    assert np.count_nonzero(valid_frame) > 0
    assert np.all(np.isfinite(flow[valid_frame]))
    assert any(report["kind"] == "dense_flow" for report in blueprint["artifacts"]["reports"])
    assert any(tool["module"] == "dense_flow" for tool in blueprint["provenance"]["tools"])


def test_dense_flow_resets_at_hard_shot_boundary(dense_flow_video, tmp_path):
    frame_count = 12
    extension, sidecars, _, _ = run_dense_flow(
        str(dense_flow_video),
        shots=[{"frame_start": 0}, {"frame_start": 6}],
        frame_count=frame_count,
        output_dir=str(tmp_path),
        config=DenseFlowConfig(max_side_px=96),
    )
    ref = extension["flow_ref"]
    with np.load(sidecars[ref["uri"]], allow_pickle=False) as bundle:
        flow = bundle["flow_xy"]
        valid_frame = bundle["valid_frame"]

    assert not bool(valid_frame[0])
    assert not bool(valid_frame[6])
    assert np.all(np.isnan(flow[0]))
    assert np.all(np.isnan(flow[6]))
    assert bool(valid_frame[5])
    assert bool(valid_frame[7])
    assert extension["quality"]["valid_pairs"] == frame_count - 2
