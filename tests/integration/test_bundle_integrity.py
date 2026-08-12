import io
import json
import zipfile

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)

def test_zip_bundle_integrity():
    # Setup job and trigger bundle generation
    upload_resp = client.post(
        "/api/v1/videos",
        files={"file": ("sample.mp4", b"dummy_content", "video/mp4")},
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
        
        # Verify bundle_manifest content
        manifest_data = json.loads(zf.read("bundle_manifest.json").decode('utf-8'))
        assert "blueprint_id" in manifest_data
        assert manifest_data["file_count"] >= 4

