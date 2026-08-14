from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from packages.pipeline_core.artifact_integrity import (
    ArtifactIntegrityError,
    require_blueprint_artifacts,
    verify_bundle_zip,
)
from packages.pipeline_core.bundle_exporter import create_bundle_zip
from packages.pipeline_core.person_mask import encode_binary_rle


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _npz_bytes() -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        metric=np.asarray([1.0, 2.0], dtype=np.float32),
        valid=np.asarray([True, True], dtype=np.bool_),
    )
    return buffer.getvalue()


def _blueprint(npz_content: bytes, rle_content: bytes) -> dict:
    return {
        "timebase": {"frame_count": 2},
        "environment": {
            "luminance_ref": {
                "uri": "artifacts/timeseries/metric.npz",
                "format": "npz",
                "dtype": "float32",
                "shape": [2],
                "checksum_sha256": _sha(npz_content),
                "metadata": {"array_key": "metric", "valid_array_key": "valid"},
            },
            "background_mask_ref": {
                "uri": "artifacts/timeseries/background.rle.json",
                "format": "rle_json",
                "dtype": "uint8",
                "shape": [2, 2, 2],
                "checksum_sha256": _sha(rle_content),
            },
        },
        "shots": [
            {
                "keyframes": [
                    {"image_uri": "artifacts/keyframes/frame.png"},
                ]
            }
        ],
        "artifacts": {"reports": [], "overlays": []},
    }


def _fixture() -> tuple[dict, dict]:
    npz_content = _npz_bytes()
    mask = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    rle_payload = {
        "encoding": "row_major_binary_rle_v1",
        "frame_count": 2,
        "height": 2,
        "width": 2,
        "frames": [encode_binary_rle(mask), encode_binary_rle(mask)],
    }
    rle_content = json.dumps(rle_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blueprint = _blueprint(npz_content, rle_content)
    sidecars = {
        "artifacts/timeseries/metric.npz": npz_content,
        "artifacts/timeseries/background.rle.json": rle_content,
        "artifacts/keyframes/frame.png": b"PNG",
    }
    return blueprint, sidecars


def test_recursive_artifact_integrity_accepts_physical_refs() -> None:
    blueprint, sidecars = _fixture()
    result = require_blueprint_artifacts(blueprint, sidecars)
    assert result["valid"] is True
    assert result["referenced_artifact_count"] == 3


def test_recursive_artifact_integrity_rejects_hash_or_shape_drift() -> None:
    blueprint, sidecars = _fixture()
    blueprint["environment"]["luminance_ref"]["shape"] = [3]
    with pytest.raises(ArtifactIntegrityError, match="shape"):
        require_blueprint_artifacts(blueprint, sidecars)

    blueprint, sidecars = _fixture()
    blueprint["environment"]["luminance_ref"]["checksum_sha256"] = "0" * 64
    with pytest.raises(ArtifactIntegrityError, match="SHA256 mismatch"):
        require_blueprint_artifacts(blueprint, sidecars)


def test_recursive_artifact_integrity_rejects_missing_or_unsafe_uri() -> None:
    blueprint, sidecars = _fixture()
    del sidecars["artifacts/keyframes/frame.png"]
    with pytest.raises(ArtifactIntegrityError, match="missing from sidecars"):
        require_blueprint_artifacts(blueprint, sidecars)

    blueprint, sidecars = _fixture()
    blueprint["shots"][0]["keyframes"][0]["image_uri"] = "artifacts/../escape.png"
    sidecars["artifacts/../escape.png"] = b"x"
    with pytest.raises(ArtifactIntegrityError, match="unsafe artifact URI"):
        require_blueprint_artifacts(blueprint, sidecars)


def test_bundle_exporter_reverifies_zip_bytes(tmp_path) -> None:
    blueprint, sidecars = _fixture()
    blueprint["blueprint_id"] = "test-blueprint"
    output = tmp_path / "bundle.zip"
    create_bundle_zip(blueprint, sidecars, {"valid": True, "errors": []}, str(output))
    result = verify_bundle_zip(str(output))
    assert result["valid"] is True

    with zipfile.ZipFile(output, "r") as archive:
        report = json.loads(archive.read("validation_report.json"))
        assert report["artifact_integrity"]["valid"] is True
        manifest = json.loads(archive.read("bundle_manifest.json"))
        assert manifest["bundle_version"] == "1.1.0"
        assert manifest["file_count"] == len(archive.namelist())
