# ruff: noqa: I001
from __future__ import annotations

from typing import Any


EXPECTED_ANCHORS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def validate_body_local_contract(blueprint: dict[str, Any]) -> list[str]:
    extensions = blueprint.get("extensions")
    if not isinstance(extensions, dict) or "e6_body_local_frame" not in extensions:
        return []
    extension = extensions.get("e6_body_local_frame")
    path = "extensions.e6_body_local_frame"
    if not isinstance(extension, dict):
        return [f"E6 Body-Local Contract Violation: {path} must be an object."]
    errors: list[str] = []
    expected = {
        "enabled": True,
        "algorithm": "coco17_torso_similarity_frame_v1",
        "coordinate_frame": "body_local_2d",
        "source_coordinate_space": "pixel_xy",
        "camera_stabilization_applied": True,
        "camera_backend": "opencv_ransac_2d",
        "pose_skeleton": "coco17",
        "anchor_names": EXPECTED_ANCHORS,
        "interpolation": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    for key, value in expected.items():
        if extension.get(key) != value:
            errors.append(f"E6 Body-Local Contract Violation: {path}.{key} must equal {value!r}.")
    config_sha = extension.get("config_sha256")
    if not _sha256(config_sha):
        errors.append(f"E6 Body-Local Contract Violation: {path}.config_sha256 must be SHA256 hex.")

    frame_count = blueprint.get("timebase", {}).get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        return errors + ["E6 Body-Local Contract Violation: timebase.frame_count must be a positive integer."]
    characters = blueprint.get("characters")
    if not isinstance(characters, list):
        return errors + ["E6 Body-Local Contract Violation: characters must be an array."]
    refs = extension.get("character_refs")
    if not isinstance(refs, list):
        return errors + [f"E6 Body-Local Contract Violation: {path}.character_refs must be an array."]
    if extension.get("character_count") != len(characters) or len(refs) != len(characters):
        errors.append("E6 Body-Local Contract Violation: extension character counts must match characters[].")
    ref_by_id = {
        str(row.get("character_id")): row.get("body_frame_transform_ref")
        for row in refs
        if isinstance(row, dict) and row.get("character_id") is not None
    }
    if len(ref_by_id) != len(refs):
        errors.append("E6 Body-Local Contract Violation: character_refs must use unique character_id values.")

    for index, character in enumerate(characters):
        cpath = f"characters[{index}]"
        if not isinstance(character, dict):
            errors.append(f"E6 Body-Local Contract Violation: {cpath} must be an object.")
            continue
        character_id = str(character.get("character_id"))
        pose = character.get("pose")
        if not isinstance(pose, dict) or pose.get("enabled") is not True or pose.get("skeleton_name") != "coco17":
            errors.append(f"E6 Body-Local Evidence Violation: {cpath} must have enabled COCO-17 pose.")
        surface = character.get("surface_motion")
        if not isinstance(surface, dict) or surface.get("coordinate_frame") != "body_local_2d":
            errors.append(f"E6 Body-Local Contract Violation: {cpath}.surface_motion.coordinate_frame must be 'body_local_2d'.")
            continue
        ref = surface.get("body_frame_transform_ref")
        if not isinstance(ref, dict):
            errors.append(f"E6 Body-Local Contract Violation: {cpath}.surface_motion.body_frame_transform_ref must be a TimeSeriesRef.")
            continue
        if ref_by_id.get(character_id) != ref:
            errors.append(f"E6 Body-Local Cross-Reference Violation: extension ref for {character_id!r} must equal the character ref.")
        expected_ref = {
            "format": "npz",
            "dtype": "float32",
            "shape": [frame_count, 2, 3],
            "axes": ["frame", "matrix_row", "matrix_col"],
            "unit": "torso_length_normalized_affine",
            "coordinate_space": "body_local_2d",
            "sampling": "per_frame",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "nan_policy": "preserve",
            "interpolation_policy": "none",
        }
        rpath = f"{cpath}.surface_motion.body_frame_transform_ref"
        for key, value in expected_ref.items():
            if ref.get(key) != value:
                errors.append(f"E6 Body-Local Contract Violation: {rpath}.{key} must equal {value!r}.")
        if not _sha256(ref.get("checksum_sha256")):
            errors.append(f"E6 Body-Local Contract Violation: {rpath}.checksum_sha256 must be SHA256 hex.")
        metadata = ref.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"E6 Body-Local Contract Violation: {rpath}.metadata is required.")
            continue
        expected_metadata = {
            "array_key": "source_pixel_to_body_local",
            "inverse_array_key": "body_local_to_source_pixel",
            "origin_array_key": "body_origin_stabilized_xy",
            "scale_array_key": "torso_scale_px",
            "confidence_array_key": "anchor_confidence",
            "valid_array_key": "valid_frame",
            "source_coordinate_space": "pixel_xy",
            "camera_stabilization_applied": True,
            "camera_backend": "opencv_ransac_2d",
            "pose_skeleton": "coco17",
            "anchor_names": EXPECTED_ANCHORS,
            "origin_semantics": "midpoint_of_left_and_right_hip_after_camera_stabilization",
            "positive_x_semantics": "toward_anatomical_left",
            "positive_y_semantics": "pelvis_to_shoulder_center",
            "scale_semantics": "stabilized_shoulder_center_to_hip_center_distance_equals_1_body_local_unit",
            "config_sha256": config_sha,
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                errors.append(f"E6 Body-Local Contract Violation: {rpath}.metadata.{key} must equal {value!r}.")
    return errors
