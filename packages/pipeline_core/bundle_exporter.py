from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import Any

from packages.pipeline_core.artifact_integrity import (
    ArtifactIntegrityError,
    require_blueprint_artifacts,
    require_bundle_zip,
    sidecar_bytes,
)


def calculate_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2).encode("utf-8")


def create_bundle_zip(
    blueprint_data: dict[str, Any],
    sidecars: dict[str, Any],
    validation_report: dict[str, Any],
    output_zip_path: str,
) -> str:
    """Assemble and then re-verify a physical Video Blueprint bundle ZIP."""
    preflight = require_blueprint_artifacts(blueprint_data, sidecars)
    report = dict(validation_report)
    report["artifact_integrity"] = preflight
    bundle_manifest_files: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            blueprint_bytes = _json_bytes(blueprint_data)
            zip_file.writestr("blueprint.json", blueprint_bytes)
            bundle_manifest_files.append(
                {
                    "path": "blueprint.json",
                    "size": len(blueprint_bytes),
                    "sha256": calculate_sha256_bytes(blueprint_bytes),
                }
            )

            validation_bytes = _json_bytes(report)
            zip_file.writestr("validation_report.json", validation_bytes)
            bundle_manifest_files.append(
                {
                    "path": "validation_report.json",
                    "size": len(validation_bytes),
                    "sha256": calculate_sha256_bytes(validation_bytes),
                }
            )

            for sidecar_path in sorted(sidecars):
                content = sidecar_bytes(sidecars[sidecar_path])
                zip_file.writestr(sidecar_path, content)
                bundle_manifest_files.append(
                    {
                        "path": sidecar_path,
                        "size": len(content),
                        "sha256": calculate_sha256_bytes(content),
                    }
                )

            bundle_manifest = {
                "bundle_version": "1.1.0",
                "blueprint_id": blueprint_data.get("blueprint_id", "unknown"),
                "file_count": len(bundle_manifest_files) + 1,
                "files": bundle_manifest_files,
            }
            zip_file.writestr("bundle_manifest.json", _json_bytes(bundle_manifest))

        require_bundle_zip(output_zip_path)
    except Exception:
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)
        raise

    return output_zip_path


__all__ = [
    "ArtifactIntegrityError",
    "calculate_sha256_bytes",
    "create_bundle_zip",
]
