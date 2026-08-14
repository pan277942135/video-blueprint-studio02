from __future__ import annotations

import copy

from packages.blueprint_schema.body_local_contract import validate_body_local_contract


def _blueprint() -> dict:
    config_sha = "a" * 64
    ref = {
        "uri": "artifacts/timeseries/char_000_body_local_2d.npz",
        "format": "npz",
        "dtype": "float32",
        "shape": [3, 2, 3],
        "axes": ["frame", "matrix_row", "matrix_col"],
        "unit": "torso_length_normalized_affine",
        "coordinate_space": "body_local_2d",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": 2,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": "b" * 64,
        "metadata": {
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
            "anchor_names": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            "origin_semantics": "midpoint_of_left_and_right_hip_after_camera_stabilization",
            "positive_x_semantics": "toward_anatomical_left",
            "positive_y_semantics": "pelvis_to_shoulder_center",
            "scale_semantics": "stabilized_shoulder_center_to_hip_center_distance_equals_1_body_local_unit",
            "config_sha256": config_sha,
        },
    }
    return {
        "timebase": {"frame_count": 3},
        "characters": [
            {
                "character_id": "char_000",
                "pose": {"enabled": True, "skeleton_name": "coco17"},
                "surface_motion": {
                    "coordinate_frame": "body_local_2d",
                    "body_frame_transform_ref": ref,
                },
            }
        ],
        "extensions": {
            "e6_body_local_frame": {
                "enabled": True,
                "algorithm": "coco17_torso_similarity_frame_v1",
                "coordinate_frame": "body_local_2d",
                "source_coordinate_space": "pixel_xy",
                "camera_stabilization_applied": True,
                "camera_backend": "opencv_ransac_2d",
                "pose_skeleton": "coco17",
                "anchor_names": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
                "interpolation": False,
                "config_sha256": config_sha,
                "character_count": 1,
                "character_refs": [
                    {"character_id": "char_000", "body_frame_transform_ref": ref}
                ],
                "identity_inference_performed": False,
                "biometric_embedding_exported": False,
            }
        },
    }


def test_body_local_contract_accepts_strict_2d_evidence() -> None:
    assert validate_body_local_contract(_blueprint()) == []


def test_body_local_contract_rejects_3d_or_unstabilized_claims() -> None:
    blueprint = _blueprint()
    ref = blueprint["characters"][0]["surface_motion"]["body_frame_transform_ref"]
    ref["coordinate_space"] = "body_local_3d"
    ref["metadata"]["camera_stabilization_applied"] = False
    errors = validate_body_local_contract(blueprint)
    assert any("coordinate_space" in error for error in errors)
    assert any("camera_stabilization_applied" in error for error in errors)


def test_body_local_contract_rejects_interpolation_or_ref_drift() -> None:
    blueprint = _blueprint()
    broken = copy.deepcopy(blueprint)
    ref = broken["characters"][0]["surface_motion"]["body_frame_transform_ref"]
    ref["interpolation_policy"] = "linear_short_gap"
    broken["extensions"]["e6_body_local_frame"]["character_refs"][0]["body_frame_transform_ref"] = copy.deepcopy(ref)
    broken["extensions"]["e6_body_local_frame"]["character_refs"][0]["body_frame_transform_ref"]["checksum_sha256"] = "c" * 64
    errors = validate_body_local_contract(broken)
    assert any("interpolation_policy" in error for error in errors)
    assert any("must equal the character ref" in error for error in errors)
