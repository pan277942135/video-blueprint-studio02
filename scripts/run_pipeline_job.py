import sys
import os
import json
import tempfile
import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.blueprint_schema.validator import BlueprintValidator

def main():
    if len(sys.argv) < 5:
        print(json.dumps({"error": "Usage: run_pipeline_job.py <job_id> <video_path> <file_name> <sha256>"}))
        sys.exit(1)

    job_id = sys.argv[1]
    video_path = sys.argv[2]
    file_name = sys.argv[3]
    sha256 = sys.argv[4]

    temp_dir = tempfile.gettempdir()
    blueprint_out_path = os.path.join(temp_dir, f"blueprint_{job_id}.json")
    bundle_out_path = os.path.join(temp_dir, f"bundle_{job_id}.zip")
    report_out_path = os.path.join(temp_dir, f"validation_{job_id}.json")

    try:
        blueprint, sidecars = run_deterministic_mock_pipeline(
            job_id=job_id,
            video_file_name=file_name,
            video_sha256=sha256,
            video_path=video_path if os.path.exists(video_path) else None
        )

        validator = BlueprintValidator()
        is_valid, errors = validator.validate(blueprint)

        val_report = {
            "valid": is_valid,
            "schema_version": "Draft 2020-12",
            "validated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "errors": errors,
            "summary": {"passed_rules": 25 - len(errors), "failed_rules": len(errors)}
        }

        with open(blueprint_out_path, "w", encoding="utf-8") as f:
            json.dump(blueprint, f, indent=2)

        with open(report_out_path, "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

        create_bundle_zip(blueprint, sidecars, val_report, bundle_out_path)

        res = {
            "status": "succeeded",
            "blueprint_path": blueprint_out_path,
            "bundle_path": bundle_out_path,
            "validation_report_path": report_out_path,
            "valid": is_valid,
            "errors": errors
        }
        print(json.dumps(res))
    except Exception as e:  # noqa: BLE001
        res = {
            "status": "failed",
            "error": str(e)
        }
        print(json.dumps(res))
        sys.exit(1)

if __name__ == "__main__":
    main()
