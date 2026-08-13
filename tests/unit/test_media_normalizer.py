import os
import shutil
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import pytest

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.media_normalizer import (
    MediaNormalizationError,
    NormalizationValidationError,
    normalize_media_to_cfr,
    validate_normalization,
)
from packages.pipeline_core.media_probe import (
    compute_sha256,
    probe_media,
)
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline


@pytest.fixture(scope="module")
def cfr_30fps_video():
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "cfr_30fps.mp4")
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
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def cfr_2997fps_video():
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "cfr_2997fps.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30000/1001",
        "-t", "2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield video_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def cfr_60fps_video():
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "cfr_60fps.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=60",
        "-t", "2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield video_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def vfr_synthetic_video():
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "vfr_synthetic.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
        "-vf", "setpts=N*N*0.01/TB",
        "-t", "1",
        "-c:v", "libx264",
        "-vsync", "vfr",
        video_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield video_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def audio_video_source():
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "audio_video.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield video_path
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_cfr_30fps_to_30fps(cfr_30fps_video):
    probe_res = probe_media(cfr_30fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_30fps_video, probe_res, temp_out)
        assert norm_res.fps_num == 30
        assert norm_res.fps_den == 1
        assert norm_res.fps == 30.0


def test_cfr_2997fps_preserves_accurate_rational(cfr_2997fps_video):
    probe_res = probe_media(cfr_2997fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_2997fps_video, probe_res, temp_out)
        assert norm_res.fps_num == 30000
        assert norm_res.fps_den == 1001
        norm_p = probe_media(norm_res.normalized_video_path)
        n_s, d_s = norm_p.r_frame_rate.split("/")
        assert Fraction(int(n_s), int(d_s)) == Fraction(30000, 1001)


def test_cfr_60fps_max_30fps_capped(cfr_60fps_video):
    probe_res = probe_media(cfr_60fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_60fps_video, probe_res, temp_out, analysis_fps_max=30)
        assert norm_res.fps_num == 30
        assert norm_res.fps_den == 1


def test_vfr_synthetic_to_normalized_cfr(vfr_synthetic_video):
    probe_res = probe_media(vfr_synthetic_video)
    assert probe_res.variable_frame_rate is True
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(vfr_synthetic_video, probe_res, temp_out)
        assert norm_res.source_variable_frame_rate is True
        norm_p = probe_media(norm_res.normalized_video_path)
        assert norm_p.variable_frame_rate is False


def test_normalized_frame_delta_stable(cfr_30fps_video):
    probe_res = probe_media(cfr_30fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_30fps_video, probe_res, temp_out)
        norm_p = probe_media(norm_res.normalized_video_path)
        deltas = [
            norm_p.frame_timestamps[i + 1] - norm_p.frame_timestamps[i]
            for i in range(len(norm_p.frame_timestamps) - 1)
        ]
        for d in deltas:
            assert abs(d - (1.0 / 30.0)) < 0.005


def test_normalized_start_pts_zero(cfr_30fps_video):
    probe_res = probe_media(cfr_30fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_30fps_video, probe_res, temp_out)
        assert norm_res.start_pts_us == 0


def test_source_pts_map_rows_and_monotonicity(cfr_30fps_video):
    probe_res = probe_media(cfr_30fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_30fps_video, probe_res, temp_out)
        npz = np.load(norm_res.source_pts_map_path)
        matrix = npz["source_pts_map"]
        assert matrix.shape[0] == norm_res.frame_count
        assert matrix.shape[1] == 2

        norm_pts = matrix[:, 0]
        source_pts = matrix[:, 1]

        for i in range(len(norm_pts) - 1):
            assert norm_pts[i + 1] > norm_pts[i]

        for i in range(len(source_pts) - 1):
            assert source_pts[i + 1] >= source_pts[i]

        assert compute_sha256(norm_res.source_pts_map_path) == norm_res.source_pts_map_sha256


def test_source_pts_map_timeseries_ref_schema_valid(cfr_30fps_video):
    validator = BlueprintValidator()
    bp, _ = run_deterministic_mock_pipeline("job-schema-test", "cfr_30fps.mp4", "sha256", video_path=cfr_30fps_video)
    is_valid, errors = validator.validate(bp)
    assert is_valid, f"Schema validation errors: {errors}"
    assert bp["timebase"]["normalized_to_cfr"] is True
    assert bp["timebase"]["source_pts_map_ref"] is not None


def test_original_source_sha256_unmodified(cfr_30fps_video):
    sha_before = compute_sha256(cfr_30fps_video)
    probe_res = probe_media(cfr_30fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        _ = normalize_media_to_cfr(cfr_30fps_video, probe_res, temp_out)
    sha_after = compute_sha256(cfr_30fps_video)
    assert sha_before == sha_after


def test_audio_preservation_and_no_audio(audio_video_source, cfr_30fps_video):
    with tempfile.TemporaryDirectory() as temp_out:
        probe_audio = probe_media(audio_video_source)
        norm_audio = normalize_media_to_cfr(audio_video_source, probe_audio, temp_out)
        assert norm_audio.audio_preserved is True
        assert norm_audio.audio_duration_us is not None
        assert norm_audio.audio_video_drift_us is not None
        assert norm_audio.audio_video_drift_us <= 50000

        probe_no_audio = probe_media(cfr_30fps_video)
        norm_no_audio = normalize_media_to_cfr(cfr_30fps_video, probe_no_audio, temp_out)
        assert norm_no_audio.audio_preserved is False
        assert norm_no_audio.audio_duration_us is None


def test_normalization_failure_raises_exception():
    with tempfile.TemporaryDirectory() as temp_out:
        fake_probe = probe_media.__wrapped__ if hasattr(probe_media, "__wrapped__") else None
        with pytest.raises(MediaNormalizationError):
            normalize_media_to_cfr("non_existent_video_path.mp4", fake_probe, temp_out)


def test_validate_normalization_bad_path():
    with pytest.raises(NormalizationValidationError):
        validate_normalization("invalid_file.mp4", None, 30, 1, np.zeros((10, 2)))


def test_mapping_error_provenance(cfr_30fps_video):
    probe_res = probe_media(cfr_30fps_video)
    with tempfile.TemporaryDirectory() as temp_out:
        norm_res = normalize_media_to_cfr(cfr_30fps_video, probe_res, temp_out)
        assert isinstance(norm_res.mapping_error_max_us, float)
        assert isinstance(norm_res.mapping_error_mean_us, float)
        assert norm_res.mapping_error_max_us >= 0.0
        assert norm_res.mapping_error_mean_us >= 0.0


def test_av_drift_measurement_and_validation_failure():
    temp_dir = tempfile.mkdtemp()
    try:
        mismatched_video = os.path.join(temp_dir, "mismatched_av.mp4")
        # Generate 2s video stream and 3.5s audio stream
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "2.0",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "3.5",
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            mismatched_video
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        probe_res = probe_media(mismatched_video)
        with tempfile.TemporaryDirectory() as temp_out:
            # High threshold allows normalization to pass and measure drift
            norm_res = normalize_media_to_cfr(mismatched_video, probe_res, temp_out, max_av_drift_us=2_000_000)
            assert norm_res.audio_video_drift_us is not None
            assert norm_res.audio_video_drift_us > 100_000

            # Exceeding tight threshold MUST fail validation
            with pytest.raises(NormalizationValidationError, match="drift"):
                validate_normalization(
                    normalized_video_path=norm_res.normalized_video_path,
                    probe_result=probe_res,
                    target_fps_num=norm_res.fps_num,
                    target_fps_den=norm_res.fps_den,
                    pts_map_matrix=np.zeros((norm_res.frame_count, 2), dtype=np.int64),
                    max_av_drift_us=10_000
                )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_validate_normalization_rational_fps_mismatch_rejected(cfr_30fps_video):
    probe_res = probe_media(cfr_30fps_video)
    # Source/normalized video is 30/1, target is 30000/1001 -> must be rejected
    with pytest.raises(NormalizationValidationError, match="rational FPS"):
        validate_normalization(
            normalized_video_path=cfr_30fps_video,
            probe_result=probe_res,
            target_fps_num=30000,
            target_fps_den=1001,
            pts_map_matrix=np.zeros((probe_res.source_frame_count or 60, 2), dtype=np.int64)
        )
