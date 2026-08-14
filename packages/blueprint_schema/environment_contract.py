from __future__ import annotations

from typing import Any

ALGORITHM = "background_photometry_v1"


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _check_metric_ref(
    ref: Any,
    *,
    path: str,
    frame_count: int,
    shape: list[int],
    axes: list[str],
    unit: str,
    array_key: str,
    valid_key: str,
    config_sha: Any,
) -> list[str]:
    if not isinstance(ref, dict):
        return [f"E9 Environment Contract Violation: {path} must be a TimeSeriesRef."]
    errors: list[str] = []
    expected = {
        "format": "npz",
        "dtype": "float32",
        "shape": shape,
        "axes": axes,
        "unit": unit,
        "coordinate_space": "none",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "nan_policy": "preserve",
        "interpolation_policy": "none",
    }
    for key, value in expected.items():
        if ref.get(key) != value:
            errors.append(f"E9 Environment Contract Violation: {path}.{key} must equal {value!r}.")
    if not _sha256(ref.get("checksum_sha256")):
        errors.append(f"E9 Environment Contract Violation: {path}.checksum_sha256 must be SHA256 hex.")
    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        return errors + [f"E9 Environment Contract Violation: {path}.metadata is required."]
    expected_metadata = {
        "array_key": array_key,
        "valid_array_key": valid_key,
        "algorithm": ALGORITHM,
        "background_definition": "inverse_union_of_e4_person_masks",
        "requires_complete_masks_for_present_tracks": True,
        "config_sha256": config_sha,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            errors.append(f"E9 Environment Contract Violation: {path}.metadata.{key} must equal {value!r}.")
    return errors


def validate_environment_contract(blueprint: dict[str, Any]) -> list[str]:
    extensions = blueprint.get("extensions")
    if not isinstance(extensions, dict) or "e9_environment" not in extensions:
        return []
    extension = extensions.get("e9_environment")
    path = "extensions.e9_environment"
    if not isinstance(extension, dict):
        return [f"E9 Environment Contract Violation: {path} must be an object."]
    errors: list[str] = []
    expected_extension = {
        "enabled": True,
        "algorithm": ALGORITHM,
        "background_definition": "inverse_union_of_e4_person_masks",
        "requires_complete_masks_for_present_tracks": True,
        "depth_emitted": False,
        "occluder_tracks_emitted": False,
        "semantic_scene_inference_performed": False,
        "weather_inference_performed": False,
        "material_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
    }
    for key, value in expected_extension.items():
        if extension.get(key) != value:
            errors.append(f"E9 Environment Contract Violation: {path}.{key} must equal {value!r}.")
    config_sha = extension.get("config_sha256")
    if not _sha256(config_sha):
        errors.append(f"E9 Environment Contract Violation: {path}.config_sha256 must be SHA256 hex.")
    coverage = extension.get("coverage")
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or not 0.0 <= float(coverage) <= 1.0:
        errors.append(f"E9 Environment Contract Violation: {path}.coverage must be within [0,1].")
    valid_count = extension.get("valid_frame_count")
    if not isinstance(valid_count, int) or isinstance(valid_count, bool) or valid_count < 0:
        errors.append(f"E9 Environment Contract Violation: {path}.valid_frame_count must be a non-negative integer.")
    frame_count = blueprint.get("timebase", {}).get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        return errors + ["E9 Environment Contract Violation: timebase.frame_count must be positive integer."]
    if isinstance(valid_count, int) and valid_count > frame_count:
        errors.append(f"E9 Environment Contract Violation: {path}.valid_frame_count cannot exceed frame_count.")
    if isinstance(valid_count, int) and isinstance(coverage, (int, float)) and not isinstance(coverage, bool):
        expected_coverage = valid_count / frame_count
        if abs(float(coverage) - expected_coverage) > 1e-6:
            errors.append(f"E9 Environment Contract Violation: {path}.coverage must equal valid_frame_count/frame_count.")
    if not isinstance(extensions.get("e4_person_mask"), dict):
        errors.append("E9 Environment Evidence Violation: e4_person_mask evidence is required for background isolation.")

    environment = blueprint.get("environment")
    if not isinstance(environment, dict):
        return errors + ["E9 Environment Contract Violation: environment must be an object."]
    if environment.get("depth_ref") is not None:
        errors.append("E9 Environment Evidence Violation: depth_ref must remain null without calibrated depth evidence.")
    if environment.get("occluder_tracks") != []:
        errors.append("E9 Environment Evidence Violation: occluder_tracks must remain empty in E9.1.")

    source_video = blueprint.get("source_video")
    if not isinstance(source_video, dict):
        return errors + ["E9 Environment Contract Violation: source_video must be an object."]
    width = source_video.get("width")
    height = source_video.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return errors + ["E9 Environment Contract Violation: source video geometry must be positive integers."]

    mask_ref = environment.get("background_mask_ref")
    if not isinstance(mask_ref, dict):
        errors.append("E9 Environment Contract Violation: environment.background_mask_ref must be a TimeSeriesRef.")
    else:
        expected_mask = {
            "format": "rle_json",
            "dtype": "uint8",
            "shape": [frame_count, height, width],
            "axes": ["frame", "y", "x"],
            "unit": "binary",
            "coordinate_space": "pixel_xy",
            "sampling": "per_frame",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "nan_policy": "preserve",
            "interpolation_policy": "none",
        }
        for key, value in expected_mask.items():
            if mask_ref.get(key) != value:
                errors.append(f"E9 Environment Contract Violation: environment.background_mask_ref.{key} must equal {value!r}.")
        if not _sha256(mask_ref.get("checksum_sha256")):
            errors.append("E9 Environment Contract Violation: environment.background_mask_ref.checksum_sha256 must be SHA256 hex.")
        metadata = mask_ref.get("metadata")
        if not isinstance(metadata, dict):
            errors.append("E9 Environment Contract Violation: environment.background_mask_ref.metadata is required.")
        else:
            expected_mask_metadata = {
                "encoding": "row_major_binary_rle_v1",
                "foreground_semantics": "background_pixel",
                "background_definition": "inverse_union_of_e4_person_masks",
                "requires_complete_masks_for_present_tracks": True,
                "invalid_frame_value": "null",
                "validity_array_uri": "artifacts/timeseries/environment_photometry.npz",
                "validity_array_key": "valid_frame",
                "config_sha256": config_sha,
            }
            for key, value in expected_mask_metadata.items():
                if metadata.get(key) != value:
                    errors.append(f"E9 Environment Contract Violation: environment.background_mask_ref.metadata.{key} must equal {value!r}.")

    metric_specs = {
        "luminance_ref": ([frame_count], ["frame"], "normalized_bt709_luma", "luminance", "valid_frame"),
        "exposure_change_ref": ([frame_count], ["frame"], "ev_proxy", "exposure_change_ev_proxy", "exposure_valid"),
        "white_balance_proxy_ref": ([frame_count, 2], ["frame", "chromaticity_component"], "chromaticity_ratio", "white_balance_chromaticity_rb", "valid_frame"),
        "blur_ref": ([frame_count], ["frame"], "normalized_luma_laplacian_variance", "blur_laplacian_variance", "valid_frame"),
    }
    refs: list[dict[str, Any]] = []
    for field, (shape, axes, unit, array_key, valid_key) in metric_specs.items():
        ref = environment.get(field)
        errors.extend(
            _check_metric_ref(
                ref,
                path=f"environment.{field}",
                frame_count=frame_count,
                shape=shape,
                axes=axes,
                unit=unit,
                array_key=array_key,
                valid_key=valid_key,
                config_sha=config_sha,
            )
        )
        if isinstance(ref, dict):
            refs.append(ref)
    if refs:
        uris = {ref.get("uri") for ref in refs}
        checksums = {ref.get("checksum_sha256") for ref in refs}
        if uris != {"artifacts/timeseries/environment_photometry.npz"} or len(checksums) != 1:
            errors.append("E9 Environment Contract Violation: all photometry refs must share one physical NPZ and checksum.")

    exposure_ref = environment.get("exposure_change_ref")
    if isinstance(exposure_ref, dict):
        metadata = exposure_ref.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("not_camera_exif_exposure") is not True or metadata.get("shot_boundary_reset") is not True:
            errors.append("E9 Environment Evidence Violation: exposure_change_ref must be an image-derived shot-reset proxy, not EXIF exposure.")
    wb_ref = environment.get("white_balance_proxy_ref")
    if isinstance(wb_ref, dict):
        metadata = wb_ref.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("proxy_only") is not True or metadata.get("not_camera_white_balance_metadata") is not True:
            errors.append("E9 Environment Evidence Violation: white_balance_proxy_ref must remain an image-derived proxy.")
    blur_ref = environment.get("blur_ref")
    if isinstance(blur_ref, dict):
        metadata = blur_ref.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("not_physical_depth_of_field_or_psf") is not True:
            errors.append("E9 Environment Evidence Violation: blur_ref cannot claim calibrated physical blur/DOF.")
    return errors


__all__ = ["validate_environment_contract"]
