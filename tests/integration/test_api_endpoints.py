import os
import subprocess
import tempfile
import uuid

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def _make_valid_video(duration_s: int = 1) -> tuple[str, bytes]:
    temp_dir = tempfile.mkdtemp()
    valid_video_path = os.path.join(temp_dir, f"test_valid_{duration_s}s.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
        "-t", str(duration_s), "-c:v", "libx264", valid_video_path,
    ], capture_output=True, check=True)
    with open(valid_video_path, "rb") as f:
        return valid_video_path, f.read()


def _upload_valid_video(file_content: bytes, file_name: str = "test_valid.mp4") -> str:
    response = client.post(
        "/api/v1/videos",
        files={"file": (file_name, file_content, "video/mp4")},
        data={"authorization_attested": "true", "adult_subject_attested": "true"},
    )
    assert response.status_code == 201
    return response.json()["video_id"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["epic"] == "E1"


def test_upload_and_canonical_analysis_flow():
    valid_video_path, file_content = _make_valid_video()

    response = client.post(
        "/api/v1/videos",
        files={"file": ("test_valid.mp4", file_content, "video/mp4")},
        data={"authorization_attested": "true", "adult_subject_attested": "true"},
    )
    assert response.status_code == 201
    upload_data = response.json()
    assert "video_id" in upload_data
    video_id = upload_data["video_id"]
    assert str(uuid.UUID(video_id)) == video_id

    bad_upload_resp = client.post(
        "/api/v1/videos",
        files={"file": ("test_valid.mp4", file_content, "video/mp4")},
        data={"authorization_attested": "false", "adult_subject_attested": "true"},
    )
    assert bad_upload_resp.status_code == 400

    analysis_resp = client.post("/api/v1/analyses", json={"video_id": video_id})
    assert analysis_resp.status_code == 202
    analysis_data = analysis_resp.json()
    assert analysis_data["status"] == "queued"
    analysis_id = analysis_data["analysis_id"]
    assert str(uuid.UUID(analysis_id)) == analysis_id

    get_resp = client.get(f"/api/v1/analyses/{analysis_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "succeeded"

    retry_resp = client.post(f"/api/v1/analyses/{analysis_id}/retry", json={"invalidate_downstream": True})
    assert retry_resp.status_code == 202
    assert retry_resp.json()["status"] == "queued"

    get_resp2 = client.get(f"/api/v1/analyses/{analysis_id}")
    assert get_resp2.status_code == 200
    assert get_resp2.json()["status"] == "succeeded"

    bp_resp = client.get(f"/api/v1/analyses/{analysis_id}/blueprint")
    assert bp_resp.status_code == 200
    bp_data = bp_resp.json()
    assert "blueprint_id" in bp_data
    assert str(uuid.UUID(bp_data["blueprint_id"])) == bp_data["blueprint_id"]
    assert str(uuid.UUID(bp_data["processing"]["job_id"])) == bp_data["processing"]["job_id"]
    assert bp_data["timebase"]["source_pts_map_ref"] is not None

    val_resp = client.post(f"/api/v1/analyses/{analysis_id}/validate")
    assert val_resp.status_code == 200
    assert val_resp.json()["valid"] is True

    bundle_resp = client.get(f"/api/v1/analyses/{analysis_id}/bundle")
    assert bundle_resp.status_code == 200
    assert bundle_resp.headers["content-type"] == "application/zip"

    assert client.get(f"/api/v1/analyses/{analysis_id}/artifacts").status_code == 200
    assert client.patch(f"/api/v1/analyses/{analysis_id}/annotations", json={"operations": []}).status_code == 200

    cancel_resp = client.post(f"/api/v1/analyses/{analysis_id}/cancel")
    assert cancel_resp.status_code == 202
    assert cancel_resp.json()["status"] == "cancelled"

    if os.path.exists(valid_video_path):
        os.remove(valid_video_path)


def test_analysis_requires_uploaded_real_media():
    missing_video_id = str(uuid.uuid4())
    create_resp = client.post("/api/v1/analyses", json={"video_id": missing_video_id})
    assert create_resp.status_code == 404


def test_upload_invalid_media_analysis_fails():
    file_content = b"This is plain text, not a valid video file."
    upload_resp = client.post(
        "/api/v1/videos",
        files={"file": ("invalid_text.mp4", file_content, "video/mp4")},
        data={"authorization_attested": "true", "adult_subject_attested": "true"},
    )
    assert upload_resp.status_code == 201
    video_id = upload_resp.json()["video_id"]

    create_resp = client.post("/api/v1/analyses", json={"video_id": video_id})
    assert create_resp.status_code == 202
    analysis_id = create_resp.json()["analysis_id"]

    get_resp = client.get(f"/api/v1/analyses/{analysis_id}")
    assert get_resp.status_code == 200
    job_data = get_resp.json()
    assert job_data["status"] == "failed"
    assert job_data["error"]["code"] == "MEDIA_PROBE_FAILED"

    bp_resp = client.get(f"/api/v1/analyses/{analysis_id}/blueprint")
    assert bp_resp.status_code == 400

    bundle_resp = client.get(f"/api/v1/analyses/{analysis_id}/bundle")
    assert bundle_resp.status_code == 400


def test_invalid_uuid_inputs():
    resp_body = client.post("/api/v1/analyses", json={"video_id": "invalid-uuid-string"})
    assert resp_body.status_code == 422

    resp_path1 = client.get("/api/v1/analyses/invalid-uuid-string")
    assert resp_path1.status_code == 422

    resp_path2 = client.post("/api/v1/analyses/invalid-uuid-string/cancel")
    assert resp_path2.status_code == 422

    resp_path3 = client.get("/api/v1/analyses/invalid-uuid-string/events")
    assert resp_path3.status_code == 422


def test_sse_events_stream_requires_real_uploaded_job():
    valid_video_path, file_content = _make_valid_video()
    video_id = _upload_valid_video(file_content, "sse_valid.mp4")

    create_resp = client.post("/api/v1/analyses", json={"video_id": video_id})
    assert create_resp.status_code == 202
    analysis_id = create_resp.json()["analysis_id"]

    events_resp = client.get(f"/api/v1/analyses/{analysis_id}/events")
    assert events_resp.status_code == 200
    assert "text/event-stream" in events_resp.headers["content-type"]
    assert "event: status" in events_resp.text
    assert "data: {" in events_resp.text

    if os.path.exists(valid_video_path):
        os.remove(valid_video_path)
