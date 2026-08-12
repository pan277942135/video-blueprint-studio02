import os
import subprocess
import tempfile

import pytest

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.media_probe import (
    InvalidMediaError,
    compute_sha256,
    probe_media,
)
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline


@pytest.fixture(scope="module")
def sample_cfr_video():
    """Generates a temporary valid CFR h264 video file with no audio."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "sample_cfr_640x480_30fps.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
        "-t", "2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture(scope="module")
def sample_audio_video():
    """Generates a temporary valid video with an AAC audio stream."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "sample_with_audio.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100",
        "-t", "1",
        "-c:v", "libx264",
        "-c:a", "aac",
        video_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture(scope="module")
def invalid_text_file():
    """Generates a non-video text file with .mp4 extension."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "not_a_video.mp4")
    with open(file_path, "w") as f:
        f.write("This is plain text, not a video file.")
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)


def test_probe_media_valid_cfr_video(sample_cfr_video):
    res = probe_media(sample_cfr_video)

    assert res.width == 640
    assert res.height == 480
    assert abs(res.fps_avg - 30.0) < 0.1
    assert res.duration_us >= 1_900_000 and res.duration_us <= 2_100_000
    assert res.video_codec in ("h264", "avc1")
    assert res.has_audio is False
    assert res.audio_codec is None
    assert res.variable_frame_rate is False
    assert res.file_size_bytes > 0


def test_compute_sha256_stability(sample_cfr_video):
    hash1 = compute_sha256(sample_cfr_video)
    hash2 = compute_sha256(sample_cfr_video)

    assert len(hash1) == 64
    assert hash1 == hash2


def test_probe_media_with_audio_stream(sample_audio_video):
    res = probe_media(sample_audio_video)

    assert res.width == 320
    assert res.height == 240
    assert res.has_audio is True
    assert res.audio_codec is not None
    assert res.audio_sample_rate_hz == 44100
    assert res.audio_channels is not None and res.audio_channels > 0


def test_probe_media_invalid_media_fails(invalid_text_file):
    with pytest.raises(InvalidMediaError):
        probe_media(invalid_text_file)


def test_probe_media_missing_file_fails():
    with pytest.raises(InvalidMediaError):
        probe_media("/tmp/definitely_missing_video_99999.mp4")


def test_canonical_schema_mapping_validity(sample_cfr_video):
    validator = BlueprintValidator()
    sha256_val = compute_sha256(sample_cfr_video)
    
    blueprint, _ = run_deterministic_mock_pipeline(
        job_id="00000000-0000-0000-0000-000000000000",
        video_file_name="sample_cfr_640x480_30fps.mp4",
        video_sha256=sha256_val,
        video_path=sample_cfr_video
    )

    is_valid, errors = validator.validate(blueprint)
    assert is_valid, f"Mapped media probe output failed schema validation: {errors}"
