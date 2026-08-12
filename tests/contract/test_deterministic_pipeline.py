import json

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
