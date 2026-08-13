import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import pytest
from httpx import Client

from packages.blueprint_schema.validator import BlueprintValidator

BASE_URL = "http://localhost:3000"
MOCK_SHA256_TEST = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


@pytest.fixture(scope="module")
def real_video_file():
    """Generate a real 8-second 30fps MP4 fixture (240 frames)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for runtime E2E")

    tmp_dir = tempfile.gettempdir()
    video_path = os.path.join(tmp_dir, "test_e2e_real_8s_vertical.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=8:size=360x640:rate=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    yield video_path
    if os.path.exists(video_path):
        os.remove(video_path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_runtime_e2e_real_media_pipeline(real_video_file):
    client = Client(base_url=BASE_URL, timeout=60.0)
    expected_sha256 = _sha256_file(real_video_file)
    expected_size = os.path.getsize(real_video_file)

    with open(real_video_file, "rb") as f:
        files = {"file": ("test_e2e_real_8s_vertical.mp4", f, "video/mp4")}
        data = {
            "authorization_attested": "true",
            "adult_subject_attested": "true",
        }
        res = client.post("/api/v1/videos", files=files, data=data)

    assert res.status_code == 201, f"Video upload failed: {res.text}"
    video_rec = res.json()
    video_id = video_rec["video_id"]

    assert video_rec["sha256"] == expected_sha256
    assert video_rec["file_size_bytes"] == expected_size
    assert video_rec["file_size_bytes"] != 10485760
    assert video_rec["duration_us"] != 6000000
    assert video_rec["sha256"] != MOCK_SHA256_TEST
    assert video_rec["duration_us"] >= 7900000

    analysis_req = {
        "video_id": video_id,
        "modules": ["shots", "people", "pose", "camera", "micro_motion"],
    }
    res_job = client.post("/api/v1/analyses", json=analysis_req)
    assert res_job.status_code == 202, f"Analysis creation failed: {res_job.text}"
    analysis_id = res_job.json()["analysis_id"]

    completed = False
    for _ in range(120):
        res_status = client.get(f"/api/v1/analyses/{analysis_id}")
        assert res_status.status_code == 200, res_status.text
        job = res_status.json()
        if job.get("status") == "succeeded":
            completed = True
            break
        if job.get("status") == "failed":
            pytest.fail(f"Job failed instead of falling back to mock: {job}")
        time.sleep(0.25)

    assert completed, "Job did not complete within timeout"

    res_bundle = client.get(f"/api/v1/analyses/{analysis_id}/bundle")
    assert res_bundle.status_code == 200, f"Bundle download failed: {res_bundle.text}"
    assert "application/zip" in res_bundle.headers["content-type"]

    bundle_zip_path = os.path.join(tempfile.gettempdir(), f"downloaded_bundle_{analysis_id}.zip")
    with open(bundle_zip_path, "wb") as f:
        f.write(res_bundle.content)

    extract_dir = tempfile.mkdtemp(prefix="vbs_runtime_e2e_")
    try:
        with zipfile.ZipFile(bundle_zip_path, "r") as z:
            file_list = z.namelist()
            assert "blueprint.json" in file_list
            assert "bundle_manifest.json" in file_list
            assert "validation_report.json" in file_list
            assert "artifacts/timeseries/source_pts_map.npz" in file_list
            assert "artifacts/reports/shot_detection.json" in file_list
            assert "sidecars/pts_map.json" not in file_list

            bp = json.loads(z.read("blueprint.json").decode("utf-8"))
            validation_report = json.loads(z.read("validation_report.json").decode("utf-8"))
            shot_report_bytes = z.read("artifacts/reports/shot_detection.json")
            shot_report = json.loads(shot_report_bytes.decode("utf-8"))

            timebase = bp["timebase"]
            source_vid = bp["source_video"]
            assert timebase["frame_count"] != 180
            assert timebase["frame_count"] == 240
            assert timebase["end_pts_us"] != 5966667
            assert source_vid["file_size_bytes"] == expected_size
            assert source_vid["sha256"] == expected_sha256
            assert source_vid["sha256"] != MOCK_SHA256_TEST
            assert validation_report["valid"] is True

            shots = bp["shots"]
            assert shots
            assert shots[0]["frame_start"] == 0
            assert shots[-1]["frame_end"] == 239
            expected_next_start = 0
            referenced_keyframes: set[str] = set()
            for shot in shots:
                assert shot["frame_start"] == expected_next_start
                expected_next_start = shot["frame_end"] + 1
                assert shot["keyframes"]
                for keyframe in shot["keyframes"]:
                    uri = keyframe["image_uri"]
                    referenced_keyframes.add(uri)
                    assert uri in file_list
                    assert uri.startswith("artifacts/keyframes/")
                    assert uri.endswith(".png")
            assert expected_next_start == 240
            assert referenced_keyframes
            assert shot_report["shot_count"] == len(shots)
            assert shot_report["frame_count"] == 240
            assert shot_report["backend"] == "pyscenedetect_content_detector"

            report_refs = bp["artifacts"]["reports"]
            assert len(report_refs) == 1
            assert report_refs[0]["uri"] == "artifacts/reports/shot_detection.json"
            assert report_refs[0]["kind"] == "shot_detection"
            assert report_refs[0]["sha256"] == hashlib.sha256(shot_report_bytes).hexdigest()

            provenance_tools = bp["provenance"]["tools"]
            assert {tool["module"] for tool in provenance_tools} == {"shot_detection", "shot_keyframes"}
            assert all(tool["version"] != "unknown" for tool in provenance_tools)
            assert all(tool["weights_sha256"] is None for tool in provenance_tools)

            z.extract("artifacts/timeseries/source_pts_map.npz", path=extract_dir)
            npz_path = os.path.join(extract_dir, "artifacts", "timeseries", "source_pts_map.npz")
            npz_data = np.load(npz_path)
            arr = npz_data["source_pts_map"]
            assert arr.shape == (240, 2), f"PTS map shape should be (240, 2), got {arr.shape}"

            ref = timebase["source_pts_map_ref"]
            assert ref is not None
            assert ref["uri"] == "artifacts/timeseries/source_pts_map.npz"
            assert ref["shape"] == [240, 2]

            validator = BlueprintValidator()
            is_valid, errors = validator.validate(bp)
            assert is_valid, f"Blueprint validation failed: {errors}"
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        if os.path.exists(bundle_zip_path):
            os.remove(bundle_zip_path)


def test_runtime_rejects_analysis_without_uploaded_video():
    client = Client(base_url=BASE_URL, timeout=10.0)
    res = client.post(
        "/api/v1/analyses",
        json={"video_id": "00000000-0000-4000-8000-000000000010"},
    )
    assert res.status_code == 400
    body = res.json()
    assert body["code"] == "REAL_MEDIA_REQUIRED"
