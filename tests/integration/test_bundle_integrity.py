import hashlib
import io
import json
import os
import subprocess
import shutil
import tempfile
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def calculate_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_zip_bundle_integrity():
    # Setup job and trigger bundle generation with valid video
    temp_dir = tempfile.mkdtemp()
    try:
        valid_video_path = os.path.join(temp_dir, "sample.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
            "-t", "1", "-c:v", "libx264", valid_video_path
        ], capture_output=True, check=True)

        with open(valid_video_path, "rb") as f:
            file_content = f.read()

        upload_resp = client.post(
            "/api/v1/videos",
            files={"file": ("sample.mp4", file_content, "video/mp4")},
            data={"authorization_attested": "true", "adult_subject_attested": "true"}
        )
        assert upload_resp.status_code == 201
        video_id = upload_resp.json()["video_id"]

        analysis_resp = client.post(
            "/api/v1/analyses",
            json={"video_id": video_id}
        )
        assert analysis_resp.status_code == 202
        analysis_id = analysis_resp.json()["analysis_id"]

        executed = client.get(f"/api/v1/analyses/{analysis_id}").json()
        assert executed["status"] == "succeeded"

        # Download bundle
        bundle_resp = client.get(f"/api/v1/analyses/{analysis_id}/bundle")
        assert bundle_resp.status_code == 200

        zip_bytes = io.BytesIO(bundle_resp.content)
        with zipfile.ZipFile(zip_bytes, 'r') as zf:
            namelist = zf.namelist()
            assert "blueprint.json" in namelist
            assert "bundle_manifest.json" in namelist
            assert "validation_report.json" in namelist
            assert "artifacts/normalized/analysis_cfr.mp4" in namelist
            assert "artifacts/timeseries/source_pts_map.npz" in namelist

            # 1. Read blueprint.json
            blueprint_bytes = zf.read("blueprint.json")
            blueprint = json.loads(blueprint_bytes.decode("utf-8"))

            frame_count = blueprint["timebase"]["frame_count"]
            canonical_npz_sha256 = blueprint["timebase"]["source_pts_map_ref"]["checksum_sha256"]

            # 2. Read bundle_manifest.json
            manifest_bytes = zf.read("bundle_manifest.json")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            assert manifest["file_count"] >= 5

            manifest_files = {item["path"]: item for item in manifest["files"]}
            assert "artifacts/normalized/analysis_cfr.mp4" in manifest_files
            assert "artifacts/timeseries/source_pts_map.npz" in manifest_files

            # 3. Read & verify source_pts_map.npz bytes
            npz_bytes = zf.read("artifacts/timeseries/source_pts_map.npz")
            npz_sha256 = calculate_sha256_bytes(npz_bytes)

            assert npz_sha256 == canonical_npz_sha256, (
                f"ZIP NPZ sha256 {npz_sha256} != blueprint sha256 {canonical_npz_sha256}"
            )
            assert npz_sha256 == manifest_files["artifacts/timeseries/source_pts_map.npz"]["sha256"], (
                f"ZIP NPZ sha256 {npz_sha256} != bundle_manifest sha256"
            )
            assert len(npz_bytes) == manifest_files["artifacts/timeseries/source_pts_map.npz"]["size"]

            # Verify numpy.load on BytesIO
            with io.BytesIO(npz_bytes) as npz_io:
                loaded_npz = np.load(npz_io)
                assert "source_pts_map" in loaded_npz
                pts_map_matrix = loaded_npz["source_pts_map"]
                assert pts_map_matrix.shape == (frame_count, 2)

            # 4. Read & verify normalized analysis_cfr.mp4
            mp4_bytes = zf.read("artifacts/normalized/analysis_cfr.mp4")
            assert len(mp4_bytes) > 0
            assert len(mp4_bytes) == manifest_files["artifacts/normalized/analysis_cfr.mp4"]["size"]
            assert calculate_sha256_bytes(mp4_bytes) == manifest_files["artifacts/normalized/analysis_cfr.mp4"]["sha256"]

            # Ensure payload is NOT a JSON string or local path string
            assert not mp4_bytes.startswith(b"/")
            assert not mp4_bytes.startswith(b"\"/")
            assert not mp4_bytes.startswith(b"{")

            # Extract MP4 to temp file and run ffprobe to confirm valid media file
            extracted_mp4_path = os.path.join(temp_dir, "extracted_cfr.mp4")
            with open(extracted_mp4_path, "wb") as f_out:
                f_out.write(mp4_bytes)

            probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", extracted_mp4_path
            ]
            probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            duration_val = float(probe_res.stdout.strip())
            assert duration_val > 0.0

            # Ensure no local absolute paths leaked into ZIP artifact payloads
            for name in namelist:
                content = zf.read(name)
                if name.endswith(".mp4") or name.endswith(".npz"):
                    assert not content.startswith(b"/app/")
                    assert not content.startswith(b"/tmp/")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

