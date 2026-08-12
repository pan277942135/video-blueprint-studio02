import hashlib
import json
import zipfile
from typing import Any


def calculate_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def create_bundle_zip(
    blueprint_data: dict[str, Any],
    sidecars: dict[str, Any],
    validation_report: dict[str, Any],
    output_zip_path: str
) -> str:
    """
    Assembles a verifiable Video Blueprint bundle ZIP.
    Contains:
      - blueprint.json
      - bundle_manifest.json (with SHA256 hashes of all artifacts)
      - validation_report.json
      - sidecars/*
    """
    bundle_manifest_files: list[dict[str, Any]] = []

    with zipfile.ZipFile(output_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Write blueprint.json
        blueprint_bytes = json.dumps(blueprint_data, indent=2).encode('utf-8')
        zip_file.writestr('blueprint.json', blueprint_bytes)
        bundle_manifest_files.append({
            "path": "blueprint.json",
            "size": len(blueprint_bytes),
            "sha256": calculate_sha256_bytes(blueprint_bytes)
        })

        # 2. Write validation_report.json
        val_report_bytes = json.dumps(validation_report, indent=2).encode('utf-8')
        zip_file.writestr('validation_report.json', val_report_bytes)
        bundle_manifest_files.append({
            "path": "validation_report.json",
            "size": len(val_report_bytes),
            "sha256": calculate_sha256_bytes(val_report_bytes)
        })

        # 3. Write sidecars
        for sidecar_path, sidecar_content in sidecars.items():
            s_bytes = json.dumps(sidecar_content, indent=2).encode('utf-8')
            zip_file.writestr(sidecar_path, s_bytes)
            bundle_manifest_files.append({
                "path": sidecar_path,
                "size": len(s_bytes),
                "sha256": calculate_sha256_bytes(s_bytes)
            })

        # 4. Write bundle_manifest.json
        bundle_manifest = {
            "bundle_version": "1.0.0",
            "blueprint_id": blueprint_data.get("blueprint_id", "unknown"),
            "file_count": len(bundle_manifest_files) + 1,
            "files": bundle_manifest_files
        }
        manifest_bytes = json.dumps(bundle_manifest, indent=2).encode('utf-8')
        zip_file.writestr('bundle_manifest.json', manifest_bytes)

    return output_zip_path
