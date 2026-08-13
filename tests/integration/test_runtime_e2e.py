import os
import time
import json
import zipfile
import tempfile
import subprocess
import pytest
import numpy as np
from httpx import Client
from packages.blueprint_schema.validator import BlueprintValidator

BASE_URL = "http://localhost:3000"

@pytest.fixture(scope="module")
def real_video_file():
    """Generates a real 8-second vertical 1080x1920 30fps MP4 video fixture (240 frames)."""
    tmp_dir = tempfile.gettempdir()
    video_path = os.path.join(tmp_dir, "test_e2e_real_8s_vertical.mp4")
    
    # Generate 8 seconds at 30 fps = 240 frames
    cmd = [
        "/usr/local/bin/ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=8:size=1080x1920:rate=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)

def test_runtime_e2e_real_media_pipeline(real_video_file):
    client = Client(base_url=BASE_URL, timeout=30.0)

    # 1. Upload Video
    with open(real_video_file, "rb") as f:
        files = {"file": ("test_e2e_real_8s_vertical.mp4", f, "video/mp4")}
        data = {
            "authorization_attested": "true",
            "adult_subject_attested": "true"
        }
        res = client.post("/api/v1/videos", files=files, data=data)
    
    assert res.status_code == 201, f"Video upload failed: {res.text}"
    video_rec = res.json()
    video_id = video_rec["video_id"]

    # Assert real probe values returned by server
    assert video_rec["file_size_bytes"] != 10485760, "Must not return mock file size"
    assert video_rec["duration_us"] != 6000000, "Must not return mock duration"
    assert video_rec["sha256"] != "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "Must not return mock sha256"
    assert video_rec["duration_us"] >= 7900000, "Duration should be ~8 seconds"

    # 2. Create Analysis Job
    analysis_req = {
        "video_id": video_id,
        "modules": ["shots", "people", "pose", "camera", "micro_motion"]
    }
    res_job = client.post("/api/v1/analyses", json=analysis_req)
    assert res_job.status_code == 202, f"Analysis creation failed: {res_job.text}"
    job_data = res_job.json()
    analysis_id = job_data["analysis_id"]

    # 3. Poll until job succeeds
    completed = False
    for _ in range(30):
        res_status = client.get(f"/api/v1/analyses/{analysis_id}")
        if res_status.status_code == 200:
            st = res_status.json().get("status")
            if st == "succeeded":
                completed = True
                break
            elif st == "failed":
                pytest.fail(f"Job failed: {res_status.text}")
        time.sleep(0.5)

    assert completed, "Job did not complete within timeout"

    # 4. Download Bundle ZIP
    res_bundle = client.get(f"/api/v1/analyses/{analysis_id}/bundle")
    assert res_bundle.status_code == 200, f"Bundle download failed: {res_bundle.text}"
    assert res_bundle.headers["content-type"] == "application/zip"

    # 5. Unzip and inspect Bundle contents
    bundle_zip_path = os.path.join(tempfile.gettempdir(), f"downloaded_bundle_{analysis_id}.zip")
    with open(bundle_zip_path, "wb") as f:
        f.write(res_bundle.content)

    with zipfile.ZipFile(bundle_zip_path, "r") as z:
        file_list = z.namelist()
        
        assert "blueprint.json" in file_list
        assert "bundle_manifest.json" in file_list
        assert "validation_report.json" in file_list
        assert "artifacts/timeseries/source_pts_map.npz" in file_list
        assert "sidecars/pts_map.json" not in file_list, "Legacy sidecars/pts_map.json must NOT exist for real media"

        # Extract and parse blueprint.json
        bp_bytes = z.read("blueprint.json")
        bp = json.loads(bp_bytes.decode("utf-8"))

        # Assert no mock values in blueprint
        timebase = bp["timebase"]
        source_vid = bp["source_video"]
        assert timebase["frame_count"] != 180, "frame_count must not be mock 180"
        assert timebase["frame_count"] == 240, "frame_count for 8s @ 30fps should be 240"
        assert timebase["end_pts_us"] != 5966667, "end_pts_us must not be mock"
        assert source_vid["file_size_bytes"] != 10485760, "file_size_bytes must not be mock"
        assert source_vid["sha256"] != "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

        # Verify source_pts_map.npz
        z.extract("artifacts/timeseries/source_pts_map.npz", path=tempfile.gettempdir())
        npz_path = os.path.join(tempfile.gettempdir(), "artifacts/timeseries/source_pts_map.npz")
        npz_data = np.load(npz_path)
        arr = npz_data["source_pts_map"]
        assert arr.shape == (240, 2), f"PTS map shape should be (240, 2), got {arr.shape}"

        # Validate blueprint using BlueprintValidator
        validator = BlueprintValidator()
        is_valid, errors = validator.validate(bp)
        assert is_valid, f"Blueprint validation failed: {errors}"
