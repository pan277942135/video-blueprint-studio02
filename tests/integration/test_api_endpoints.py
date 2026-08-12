from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["epic"] == "E0"

def test_upload_and_job_flow():
    # 1. Upload video
    file_content = b"fake video content stream"
    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("test.mp4", file_content, "video/mp4")}
    )
    assert response.status_code == 200
    upload_data = response.json()
    assert "video_id" in upload_data
    video_id = upload_data["video_id"]

    # 2. Reject job creation without attestation
    bad_job_resp = client.post(
        "/api/v1/jobs",
        json={"video_id": video_id, "attestation_accepted": False}
    )
    assert bad_job_resp.status_code == 400

    # 3. Create job with accepted attestation
    job_resp = client.post(
        "/api/v1/jobs",
        json={"video_id": video_id, "attestation_accepted": True}
    )
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["status"] == "queued"
    job_id = job_data["job_id"]

    # 4. Get job (triggers execution -> succeeded)
    get_job_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert get_job_resp.status_code == 200
    executed_job = get_job_resp.json()
    assert executed_job["status"] == "succeeded"
    blueprint_id = executed_job["blueprint_id"]
    assert blueprint_id is not None

    # 5. Fetch Blueprint Manifest
    bp_resp = client.get(f"/api/v1/blueprints/{blueprint_id}")
    assert bp_resp.status_code == 200
    bp_data = bp_resp.json()
    assert bp_data["blueprint_id"] == blueprint_id

    # 6. Validate Blueprint
    val_resp = client.post("/api/v1/blueprints/validate", json=bp_data)
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["valid"] is True

    # 7. Download Bundle ZIP
    bundle_resp = client.get(f"/api/v1/bundles/{blueprint_id}")
    assert bundle_resp.status_code == 200
    assert bundle_resp.headers["content-type"] == "application/zip"
