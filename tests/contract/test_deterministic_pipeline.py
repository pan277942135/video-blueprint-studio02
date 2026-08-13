import json
import os
import subprocess
import tempfile

from packages.blueprint_schema.validator import BlueprintValidator
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline


def test_mock_pipeline_determinism():
    job_id = "test_job_12345"
    file_name = "test_video.mp4"
    sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    bp1, sidecars1 = run_deterministic_mock_pipeline(job_id, file_name, sha256)
    bp2, sidecars2 = run_deterministic_mock_pipeline(job_id, file_name, sha256)

    # Must be 100% identical output
    assert json.dumps(bp1, sort_keys=True) == json.dumps(bp2, sort_keys=True)
    assert json.dumps(sidecars1, sort_keys=True) == json.dumps(sidecars2, sort_keys=True)

def test_mock_pipeline_schema_compliance():
    validator = BlueprintValidator()
    bp, _ = run_deterministic_mock_pipeline("job_schema_test", "video.mp4", "1234567890")
    
    is_valid, errors = validator.validate(bp)
    assert is_valid, f"Mock pipeline output failed schema validation: {errors}"


def test_real_media_timeline_alignment():
    validator = BlueprintValidator()
    temp_dir = tempfile.mkdtemp()
    try:
        sample_path = os.path.join(temp_dir, "real_1s_30fps.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
            "-t", "1", "-c:v", "libx264", sample_path
        ], capture_output=True, check=True)

        job_id = "job_real_timeline_test"
        sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
        bp, sidecars = run_deterministic_mock_pipeline(
            job_id=job_id,
            video_file_name="real_1s_30fps.mp4",
            video_sha256=sha256,
            video_path=sample_path
        )

        # 1. Schema Validation
        is_valid, errors = validator.validate(bp)
        assert is_valid, f"Real media blueprint failed validation: {errors}"

        # 2. Timebase assertions
        tb = bp["timebase"]
        frame_count = tb["frame_count"]
        assert frame_count > 0
        last_frame = frame_count - 1
        end_pts_us = tb["end_pts_us"]

        # 3. Shot alignment
        shot = bp["shots"][0]
        assert shot["frame_start"] == 0
        assert shot["frame_end"] == last_frame
        assert shot["time_start_us"] == 0
        assert shot["time_end_us"] == end_pts_us

        # 4. Keyframe bounds
        for kf in shot["keyframes"]:
            assert 0 <= kf["frame_idx"] < frame_count
            assert 0 <= kf["time_us"] <= end_pts_us

        # 5. Sidecar bounds
        assert "sidecars/pts_map.json" not in sidecars
        assert "artifacts/timeseries/source_pts_map.npz" in sidecars

        if "sidecars/camera_motion_affine.json" in sidecars:
            cam_affine = sidecars["sidecars/camera_motion_affine.json"]
            for transform in cam_affine.get("transforms", []):
                assert 0 <= transform["frame"] < frame_count

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

