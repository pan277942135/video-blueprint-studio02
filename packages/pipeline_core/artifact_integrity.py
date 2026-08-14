from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from packages.pipeline_core.person_mask import decode_binary_rle


class ArtifactIntegrityError(RuntimeError):
    pass


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_candidates(value: Any) -> list[bytes]:
    candidates = [
        json.dumps(value, indent=2).encode("utf-8"),
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    ]
    candidates.extend(candidate + b"\n" for candidate in list(candidates))
    unique: list[bytes] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def sidecar_bytes(value: Any, expected_sha256: str | None = None) -> bytes:
    if isinstance(value, (str, os.PathLike)):
        path = str(value)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"sidecar file path is missing: {path}")
        with open(path, "rb") as handle:
            return handle.read()
    if isinstance(value, bytes):
        return value

    candidates = _json_candidates(value)
    if expected_sha256 is None:
        return candidates[0]
    matches = [candidate for candidate in candidates if _sha256(candidate) == expected_sha256]
    if not matches:
        raise ArtifactIntegrityError(
            "in-memory JSON sidecar cannot be deterministically materialized to its declared SHA256"
        )
    return matches[0]


def _safe_artifact_uri(uri: str) -> bool:
    if not uri.startswith("artifacts/") or "\\" in uri:
        return False
    path = PurePosixPath(uri)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _walk(value: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    nodes: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        nodes.append((path, value))
        for key, child in value.items():
            nodes.extend(_walk(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nodes.extend(_walk(child, f"{path}[{index}]"))
    return nodes


def _artifact_uri_occurrences(blueprint: dict[str, Any]) -> tuple[set[str], list[tuple[str, dict[str, Any]]]]:
    referenced: set[str] = set()
    hash_refs: list[tuple[str, dict[str, Any]]] = []
    for path, node in _walk(blueprint):
        uri = node.get("uri")
        if isinstance(uri, str) and uri.startswith("artifacts/"):
            referenced.add(uri)
            if "checksum_sha256" in node or "sha256" in node:
                hash_refs.append((path, node))
        for key, value in node.items():
            if key.endswith("_uri") and isinstance(value, str) and value.startswith("artifacts/"):
                referenced.add(value)
    return referenced, hash_refs


def resolve_sidecar_bytes(blueprint: dict[str, Any], sidecars: dict[str, Any]) -> dict[str, bytes]:
    _, hash_refs = _artifact_uri_occurrences(blueprint)
    expected_by_uri: dict[str, set[str]] = {}
    for _, ref in hash_refs:
        uri = ref.get("uri")
        if not isinstance(uri, str):
            continue
        expected = ref.get("checksum_sha256")
        if not isinstance(expected, str):
            expected = ref.get("sha256")
        if isinstance(expected, str):
            expected_by_uri.setdefault(uri, set()).add(expected)

    resolved: dict[str, bytes] = {}
    for uri, value in sidecars.items():
        expected_values = expected_by_uri.get(uri, set())
        if len(expected_values) > 1:
            raise ArtifactIntegrityError(f"conflicting declared SHA256 values for one artifact URI: {uri}")
        expected = next(iter(expected_values)) if expected_values else None
        resolved[uri] = sidecar_bytes(value, expected)
    return resolved


def _validate_npz_ref(
    content: bytes,
    ref: dict[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        return
    array_key = metadata.get("array_key")
    valid_key = metadata.get("valid_array_key")
    if not isinstance(array_key, str) and not isinstance(valid_key, str):
        return
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
            if isinstance(array_key, str):
                if array_key not in arrays.files:
                    errors.append(f"{path}: NPZ array_key {array_key!r} is missing")
                else:
                    array = np.asarray(arrays[array_key])
                    declared_shape = ref.get("shape")
                    if isinstance(declared_shape, list) and tuple(declared_shape) != array.shape:
                        errors.append(
                            f"{path}: NPZ {array_key!r} shape {list(array.shape)!r} != declared {declared_shape!r}"
                        )
                    declared_dtype = ref.get("dtype")
                    if isinstance(declared_dtype, str) and str(array.dtype) != declared_dtype:
                        errors.append(
                            f"{path}: NPZ {array_key!r} dtype {str(array.dtype)!r} != declared {declared_dtype!r}"
                        )
            if isinstance(valid_key, str):
                if valid_key not in arrays.files:
                    errors.append(f"{path}: NPZ valid_array_key {valid_key!r} is missing")
                else:
                    validity = np.asarray(arrays[valid_key])
                    declared_shape = ref.get("shape")
                    if (
                        isinstance(declared_shape, list)
                        and declared_shape
                        and (validity.ndim < 1 or validity.shape[0] != declared_shape[0])
                    ):
                        errors.append(
                            f"{path}: validity array first dimension does not match declared frame dimension"
                        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        errors.append(f"{path}: NPZ could not be decoded: {exc}")


def _validate_rle_ref(
    content: bytes,
    ref: dict[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    shape = ref.get("shape")
    if not (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(value, int) and value > 0 for value in shape)
    ):
        errors.append(f"{path}: rle_json ref must declare [frame,height,width] shape")
        return
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: rle_json could not be decoded: {exc}")
        return
    if not isinstance(payload, dict):
        errors.append(f"{path}: rle_json payload must be an object")
        return
    frame_count, height, width = shape
    expected = {"frame_count": frame_count, "height": height, "width": width}
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"{path}: rle_json {key}={payload.get(key)!r} != declared {value!r}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        errors.append(f"{path}: rle_json frames length must equal {frame_count}")
        return
    for frame_index, encoded in enumerate(frames):
        if encoded is None:
            continue
        if not isinstance(encoded, dict):
            errors.append(f"{path}: rle_json frame {frame_index} must be object or null")
            continue
        try:
            mask = decode_binary_rle(encoded)
        except (TypeError, ValueError) as exc:
            errors.append(f"{path}: rle_json frame {frame_index} decode failed: {exc}")
            continue
        if mask.shape != (height, width):
            errors.append(
                f"{path}: rle_json frame {frame_index} shape {list(mask.shape)!r} != {[height, width]!r}"
            )


def validate_blueprint_artifacts(
    blueprint: dict[str, Any],
    sidecars: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    referenced, hash_refs = _artifact_uri_occurrences(blueprint)

    for referenced_uri in sorted(referenced):
        if not _safe_artifact_uri(referenced_uri):
            errors.append(f"unsafe artifact URI: {referenced_uri!r}")
        if referenced_uri not in sidecars:
            errors.append(f"referenced artifact is missing from sidecars: {referenced_uri}")

    try:
        resolved = resolve_sidecar_bytes(blueprint, sidecars)
    except (OSError, TypeError, ValueError, ArtifactIntegrityError) as exc:
        errors.append(f"could not materialize sidecars deterministically: {exc}")
        resolved = {}

    checked_hashes: set[tuple[str, str]] = set()
    for path, ref in hash_refs:
        uri = ref.get("uri")
        if not isinstance(uri, str) or uri not in resolved:
            continue
        content = resolved[uri]
        checksum_value = ref.get("checksum_sha256")
        sha_value = ref.get("sha256")
        expected_hash: str | None
        if isinstance(checksum_value, str):
            expected_hash = checksum_value
        elif isinstance(sha_value, str):
            expected_hash = sha_value
        else:
            expected_hash = None
        if expected_hash is not None:
            key = (uri, expected_hash)
            if key not in checked_hashes:
                actual = _sha256(content)
                if actual != expected_hash:
                    errors.append(f"{path}: SHA256 mismatch for {uri}: {actual} != {expected_hash}")
                checked_hashes.add(key)
        artifact_format = ref.get("format")
        if artifact_format == "npz":
            _validate_npz_ref(content, ref, path=path, errors=errors)
        elif artifact_format == "rle_json":
            _validate_rle_ref(content, ref, path=path, errors=errors)

    for uri in sorted(sidecars):
        if not isinstance(uri, str) or not _safe_artifact_uri(uri):
            errors.append(f"sidecar path is not a safe artifacts/ URI: {uri!r}")
            continue
        value = sidecars[uri]
        if isinstance(value, (str, os.PathLike)) and not os.path.isfile(str(value)):
            errors.append(f"sidecar file path is missing for {uri}: {value}")
        if uri not in referenced and uri != "artifacts/normalized/analysis_cfr.mp4":
            warnings.append(f"unreferenced sidecar artifact: {uri}")

    return {
        "valid": not errors,
        "referenced_artifact_count": len(referenced),
        "sidecar_count": len(sidecars),
        "errors": errors,
        "warnings": warnings,
    }


def require_blueprint_artifacts(blueprint: dict[str, Any], sidecars: dict[str, Any]) -> dict[str, Any]:
    result = validate_blueprint_artifacts(blueprint, sidecars)
    if not result["valid"]:
        raise ArtifactIntegrityError("; ".join(result["errors"]))
    return result


def verify_bundle_zip(bundle_path: str) -> dict[str, Any]:
    errors: list[str] = []
    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            errors.append("bundle contains duplicate ZIP paths")
        for name in names:
            archive_path = PurePosixPath(name)
            if archive_path.is_absolute() or "\\" in name or any(
                part in {"", ".", ".."} for part in archive_path.parts
            ):
                errors.append(f"bundle contains unsafe ZIP path: {name!r}")
        required = {"blueprint.json", "validation_report.json", "bundle_manifest.json"}
        missing_core = sorted(required.difference(names))
        if missing_core:
            errors.append(f"bundle missing core files: {missing_core!r}")
            return {"valid": False, "errors": errors, "warnings": []}

        try:
            blueprint = json.loads(archive.read("blueprint.json"))
            manifest = json.loads(archive.read("bundle_manifest.json"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"bundle core JSON decode failed: {exc}")
            return {"valid": False, "errors": errors, "warnings": []}
        if not isinstance(blueprint, dict) or not isinstance(manifest, dict):
            errors.append("bundle core JSON values must be objects")
            return {"valid": False, "errors": errors, "warnings": []}

        files = manifest.get("files")
        if not isinstance(files, list):
            errors.append("bundle manifest files must be an array")
            files = []
        listed_paths: set[str] = set()
        for row in files:
            if not isinstance(row, dict):
                errors.append("bundle manifest file row must be an object")
                continue
            manifest_path = row.get("path")
            if not isinstance(manifest_path, str):
                errors.append("bundle manifest file row path must be a string")
                continue
            if manifest_path in listed_paths:
                errors.append(f"bundle manifest repeats path: {manifest_path}")
                continue
            listed_paths.add(manifest_path)
            if manifest_path not in names:
                errors.append(f"bundle manifest path missing from ZIP: {manifest_path}")
                continue
            content = archive.read(manifest_path)
            if row.get("size") != len(content):
                errors.append(f"bundle manifest size mismatch for {manifest_path}")
            if row.get("sha256") != _sha256(content):
                errors.append(f"bundle manifest SHA256 mismatch for {manifest_path}")
        if manifest.get("file_count") != len(names):
            errors.append("bundle manifest file_count does not equal ZIP entry count")
        expected_listed = set(names).difference({"bundle_manifest.json"})
        if listed_paths != expected_listed:
            errors.append("bundle manifest file list does not exactly cover non-manifest ZIP entries")

        zip_sidecars = {
            name: archive.read(name)
            for name in names
            if name.startswith("artifacts/")
        }
        artifact_result = validate_blueprint_artifacts(blueprint, zip_sidecars)
        errors.extend(artifact_result["errors"])
        warnings = list(artifact_result["warnings"])
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def require_bundle_zip(bundle_path: str) -> dict[str, Any]:
    result = verify_bundle_zip(bundle_path)
    if not result["valid"]:
        raise ArtifactIntegrityError("; ".join(result["errors"]))
    return result


__all__ = [
    "ArtifactIntegrityError",
    "require_blueprint_artifacts",
    "require_bundle_zip",
    "resolve_sidecar_bytes",
    "sidecar_bytes",
    "validate_blueprint_artifacts",
    "verify_bundle_zip",
]
