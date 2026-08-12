import json
import zipfile
import io
from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_zip_bundle_integrity():
    # Setup job and trigger bundle generation
    upload_resp = client.post(
        "/api/v1/videos/upload",
        files={"file": ("sample.mp4", b"dummy_content", "video/mp4")}
    )
    video_id = upload_resp.json()["video_id"]
    
    job_resp = client.post(
        "/api/v1/jobs",
        json={"video_id": video_id, "attestation_accepted": True}
    )
    job_id = job_resp.json()["job_id"]
    
    executed = client.get(f"/api/v1/jobs/{job_id}").json()
    blueprint_id = executed["blueprint_id"]

    # Download bundle
    bundle_resp = client.get(f"/api/v1/bundles/{blueprint_id}")
    assert bundle_resp.status_code == 200
    
    zip_bytes = io.BytesIO(bundle_resp.content)
    with zipfile.ZipFile(zip_bytes, 'r') as zf:
        namelist = zf.namelist()
        assert "blueprint.json" in namelist
        assert "bundle_manifest.json" in namelist
        assert "validation_report.json" in namelist
        
        # Verify bundle_manifest content
        manifest_data = json.loads(zf.read("bundle_manifest.json").decode('utf-8'))
        assert manifest_data["blueprint_id"] == blueprint_id
        assert manifest_data["file_count"] >= 4
