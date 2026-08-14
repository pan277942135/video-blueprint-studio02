from __future__ import annotations

from typing import Any

ANCHORS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _check_ref(
    ref: dict[str, Any],
    *,
    path: str,
    frame_count: int,
    expected_unit: str,
    expected_array_key: str,
    config_sha: Any,
    source_uri: str,
    body_uri: str,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "format": "npz",
        "dtype": "float32",
        "axes": ["frame", "point_slot", "xy"],
        "unit": expected_unit,
        "coordinate_space": "body_local_2d",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "nan_policy": "preserve",
        "interpolation_policy": "none",
    }
    for key, value in expected.items():
        if ref.get(key) != value:
            errors.append(f"E7 Surface Contract Violation: {path}.{key} must equal {value!r}.")
    shape = ref.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != frame_count or shape[-1] != 2:
        errors.append(f"E7 Surface Contract Violation: {path}.shape must be [frame_count,point_slots,2].")
    if not _sha256(ref.get("checksum_sha256")):
        errors.append(f"E7 Surface Contract Violation: {path}.checksum_sha256 must be SHA256 hex.")
    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        return errors + [f"E7 Surface Contract Violation: {path}.metadata is required."]
    expected_metadata = {
        "array_key": expected_array_key,
        "algorithm": "body_local_sparse_residual_v1",
        "source_point_track_uri": source_uri,
        "body_frame_transform_uri": body_uri,
        "same_track_id_required": True,
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config_sha,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            errors.append(f"E7 Surface Contract Violation: {path}.metadata.{key} must equal {value!r}.")
    expected_valid = "body_local_valid" if expected_array_key == "body_local_positions" else "residual_valid"
    if metadata.get("valid_array_key") != expected_valid or metadata.get("track_id_array_key") != "track_id":
        errors.append(f"E7 Surface Contract Violation: {path} must declare valid and track-id array keys.")
    return errors


def _check_micro_pending(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"E7 Surface Contract Violation: {path} must be an explicit pending MicroMotion object."]
    errors: list[str] = []
    if value.get("kind") != "not_detected" or value.get("confidence") != 0.0 or value.get("usable_for_generation") is not False:
        errors.append(f"E7 Surface Evidence Violation: {path} cannot claim analyzed micro-motion before E8.")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not any("E8" in item for item in limitations if isinstance(item, str)):
        errors.append(f"E7 Surface Evidence Violation: {path}.limitations must state that E8 owns micro-motion analysis.")
    for key in (
        "signal_ref",
        "detrended_signal_ref",
        "vertical_displacement_ref",
        "radial_expansion_ratio_ref",
        "area_change_ratio_ref",
        "velocity_ref",
        "acceleration_ref",
    ):
        if value.get(key) is not None:
            errors.append(f"E7 Surface Evidence Violation: {path}.{key} must remain null before E8.")
    smoothing = value.get("smoothing")
    if smoothing != {"method": "none", "parameters": {}}:
        errors.append(f"E7 Surface Contract Violation: {path}.smoothing must remain disabled.")
    return errors


def validate_surface_motion_contract(blueprint: dict[str, Any]) -> list[str]:
    extensions = blueprint.get("extensions")
    if not isinstance(extensions, dict) or "e7_surface_motion" not in extensions:
        return []
    extension = extensions.get("e7_surface_motion")
    micro_analyzed_by_e8 = isinstance(extensions.get("e8_micro_motion"), dict)
    path = "extensions.e7_surface_motion"
    if not isinstance(extension, dict):
        return [f"E7 Surface Contract Violation: {path} must be an object."]
    errors: list[str] = []
    expected_extension = {
        "enabled": True,
        "algorithm": "body_local_sparse_residual_v1",
        "coordinate_frame": "body_local_2d",
        "source_point_tracks": "e4_point_tracks",
        "body_frame_source": "e6_body_local_frame",
        "shot_boundary_reset": True,
        "same_track_id_required": True,
        "interpolation": False,
        "dense_residual_emitted": False,
        "micro_motion_analyzed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    for key, value in expected_extension.items():
        if extension.get(key) != value:
            errors.append(f"E7 Surface Contract Violation: {path}.{key} must equal {value!r}.")
    config_sha = extension.get("config_sha256")
    if not _sha256(config_sha):
        errors.append(f"E7 Surface Contract Violation: {path}.config_sha256 must be SHA256 hex.")
    frame_count = blueprint.get("timebase", {}).get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        return errors + ["E7 Surface Contract Violation: timebase.frame_count must be positive integer."]

    point_extension = extensions.get("e4_point_tracks")
    point_rows = point_extension.get("characters") if isinstance(point_extension, dict) else None
    if not isinstance(point_rows, dict):
        return errors + ["E7 Surface Evidence Violation: e4_point_tracks character rows are required."]
    characters = blueprint.get("characters")
    if not isinstance(characters, list):
        return errors + ["E7 Surface Contract Violation: characters must be an array."]
    extension_rows = extension.get("characters")
    if not isinstance(extension_rows, list) or len(extension_rows) != len(characters):
        errors.append("E7 Surface Contract Violation: extension character rows must match characters[].")
        extension_rows = []
    ext_by_id = {str(row.get("character_id")): row for row in extension_rows if isinstance(row, dict)}

    for index, character in enumerate(characters):
        cpath = f"characters[{index}]"
        if not isinstance(character, dict):
            errors.append(f"E7 Surface Contract Violation: {cpath} must be an object.")
            continue
        character_id = str(character.get("character_id"))
        surface = character.get("surface_motion")
        if not isinstance(surface, dict):
            errors.append(f"E7 Surface Contract Violation: {cpath}.surface_motion must be an object.")
            continue
        if surface.get("coordinate_frame") != "body_local_2d":
            errors.append(f"E7 Surface Contract Violation: {cpath}.surface_motion.coordinate_frame must be body_local_2d.")
        if surface.get("global_postural_sway_ref") is not None:
            errors.append(f"E7 Surface Evidence Violation: {cpath}.surface_motion.global_postural_sway_ref is reserved for later analysis.")
        body_ref = surface.get("body_frame_transform_ref")
        if not isinstance(body_ref, dict) or body_ref.get("coordinate_space") != "body_local_2d":
            errors.append(f"E7 Surface Evidence Violation: {cpath} requires an E6 body_local_2d transform ref.")
            continue
        body_uri = body_ref.get("uri")
        point_row = point_rows.get(character_id)
        source_ref = point_row.get("track_points_ref") if isinstance(point_row, dict) else None
        source_uri = source_ref.get("uri") if isinstance(source_ref, dict) else None
        ext_row = ext_by_id.get(character_id)
        regions = surface.get("regions")
        if not isinstance(regions, list):
            errors.append(f"E7 Surface Contract Violation: {cpath}.surface_motion.regions must be an array.")
            continue
        if source_uri is None:
            if regions:
                errors.append(f"E7 Surface Evidence Violation: {cpath} cannot emit surface regions without E4.2 point tracks.")
            if not isinstance(ext_row, dict) or ext_row.get("track_points_ref") is not None or ext_row.get("residual_flow_ref") is not None:
                errors.append(f"E7 Surface Contract Violation: extension row for {character_id!r} must record null refs.")
            continue
        if len(regions) != 1:
            errors.append(f"E7 Surface Contract Violation: {cpath} must emit exactly one sparse whole-body region.")
            continue
        region = regions[0]
        if not isinstance(region, dict):
            errors.append(f"E7 Surface Contract Violation: {cpath}.surface_motion.regions[0] must be object.")
            continue
        rpath = f"{cpath}.surface_motion.regions[0]"
        if region.get("region_id") != "custom" or region.get("display_name") != "whole_body_sparse_surface":
            errors.append(f"E7 Surface Contract Violation: {rpath} must identify the whole-body sparse custom region.")
        roi = region.get("roi_definition")
        if roi != {"type": "mask_intersection", "anchor_keypoints": ANCHORS, "expansion_ratio": 1.0}:
            errors.append(f"E7 Surface Contract Violation: {rpath}.roi_definition must use the E7 whole-body anchor contract.")
        if region.get("mask_ref") != character.get("person_mask_ref"):
            errors.append(f"E7 Surface Cross-Reference Violation: {rpath}.mask_ref must equal character.person_mask_ref.")
        if region.get("deformation_modes_ref") is not None:
            errors.append(f"E7 Surface Evidence Violation: {rpath}.deformation_modes_ref is reserved for later analysis.")
        local_ref = region.get("track_points_ref")
        residual_ref = region.get("residual_flow_ref")
        if not isinstance(local_ref, dict) or not isinstance(residual_ref, dict):
            errors.append(f"E7 Surface Contract Violation: {rpath} requires track_points_ref and residual_flow_ref.")
            continue
        errors.extend(
            _check_ref(
                local_ref,
                path=f"{rpath}.track_points_ref",
                frame_count=frame_count,
                expected_unit="body_local_unit",
                expected_array_key="body_local_positions",
                config_sha=config_sha,
                source_uri=str(source_uri),
                body_uri=str(body_uri),
            )
        )
        errors.extend(
            _check_ref(
                residual_ref,
                path=f"{rpath}.residual_flow_ref",
                frame_count=frame_count,
                expected_unit="body_local_unit_per_frame",
                expected_array_key="residual_displacement",
                config_sha=config_sha,
                source_uri=str(source_uri),
                body_uri=str(body_uri),
            )
        )
        if local_ref.get("uri") != residual_ref.get("uri") or local_ref.get("checksum_sha256") != residual_ref.get("checksum_sha256"):
            errors.append(f"E7 Surface Contract Violation: {rpath} refs must share one physical NPZ sidecar.")
        if not isinstance(ext_row, dict) or ext_row.get("track_points_ref") != local_ref or ext_row.get("residual_flow_ref") != residual_ref:
            errors.append(f"E7 Surface Cross-Reference Violation: extension row for {character_id!r} must equal region refs.")
        if not micro_analyzed_by_e8:
            errors.extend(_check_micro_pending(region.get("micro_motion"), path=f"{rpath}.micro_motion"))
    return errors
