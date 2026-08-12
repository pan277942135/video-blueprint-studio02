import uuid
from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["epic"] == "E0"

def test_upload_and_canonical_analysis_flow():
    # 1. Upload video with attestations
    file_content = b"fake video content stream"
    response = client.post(
        "/api/v1/videos",
        files={"file": ("test.mp4", file_content, "video/mp4")},
        data={"authorization_attested": "true", "adult_subject_attested": "true"}
    )
    assert response.status_code == 201
    upload_data = response.json()
    assert "video_id" in upload_data
    video_id = upload_data["video_id"]
    # Verify video_id is a valid UUID string
    uuid_obj = uuid.UUID(video_id)
    assert str(uuid_obj) == video_id

    # 2. Reject upload when attestations are missing/false
    bad_upload_resp = client.post(
        "/api/v1/videos",
        files={"file": ("test.mp4", file_content, "video/mp4")},
        data={"authorization_attested": "false", "adult_subject_attested": "true"}
    )
    assert bad_upload_resp.status_code == 400

    # 3. Create analysis job
    analysis_resp = client.post(
        "/api/v1/analyses",
        json={"video_id": video_id}
    )
    assert analysis_resp.status_code == 202
    analysis_data = analysis_resp.json()
    assert analysis_data["status"] == "queued"
    analysis_id = analysis_data["analysis_id"]
    # Verify analysis_id is a valid UUID string
    assert str(uuid.UUID(analysis_id)) == analysis_id

    # 4. Get analysis status (triggers execution -> succeeded)
    get_resp = client.get(f"/api/v1/analyses/{analysis_id}")
    assert get_resp.status_code == 200
    executed_job = get_resp.json()
    assert executed_job["status"] == "succeeded"

    # 5. Test retry endpoint
    retry_resp = client.post(f"/api/v1/analyses/{analysis_id}/retry", json={"invalidate_downstream": True})
    assert retry_resp.status_code == 202
    retried_job = retry_resp.json()
    assert retried_job["status"] == "queued"

    # Fetch again to execute
    get_resp2 = client.get(f"/api/v1/analyses/{analysis_id}")
    assert get_resp2.status_code == 200
    assert get_resp2.json()["status"] == "succeeded"

    # 6. Test cancel endpoint
    cancel_resp = client.post(f"/api/v1/analyses/{analysis_id}/cancel")
    assert cancel_resp.status_code == 202
    assert cancel_resp.json()["status"] == "cancelled"

    # 7. Fetch Blueprint Manifest
    bp_resp = client.get(f"/api/v1/analyses/{analysis_id}/blueprint")
    assert bp_resp.status_code == 200
    bp_data = bp_resp.json()
    assert "blueprint_id" in bp_data
    assert str(uuid.UUID(bp_data["blueprint_id"])) == bp_data["blueprint_id"]
    assert str(uuid.UUID(bp_data["processing"]["job_id"])) == bp_data["processing"]["job_id"]

    # 8. Validate Blueprint endpoint
    val_resp = client.post(f"/api/v1/analyses/{analysis_id}/validate")
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["valid"] is True

    # 9. Download Bundle ZIP
    bundle_resp = client.get(f"/api/v1/analyses/{analysis_id}/bundle")
    assert bundle_resp.status_code == 200
    assert bundle_resp.headers["content-type"] == "application/zip"

    # 10. Low-cost stubs
    assert client.get(f"/api/v1/analyses/{analysis_id}/artifacts").status_code == 200
    assert client.patch(f"/api/v1/analyses/{analysis_id}/annotations", json={"operations": []}).status_code == 200
    assert client.get(f"/api/v1/analyses/{analysis_id}/events").status_code == 200

