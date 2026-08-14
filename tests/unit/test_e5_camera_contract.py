from __future__ import annotations

import copy
import json
from pathlib import Path

from packages.blueprint_schema import BlueprintValidator


def _blueprint() -> dict:
    blueprint = json.loads(Path("contracts/example_blueprint.json").read_text(encoding="utf-8"))
    shot = blueprint["shots"][0]
    shot["camera_motion_id"] = "camera-shot_000"
    config_sha = "a" * 64
    checksum = "b" * 64
    length = shot["frame_end"] - shot["frame_start"] + 1
    common = {
        "uri": "artifacts/timeseries/camera-shot_000_camera_2d.npz",
        "format": "npz",
        "dtype": "float32",
        "sampling": "per_frame",
        "frame_start": shot["frame_start"],
        "frame_end": shot["frame_end"],
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": checksum,
    }
    metadata = {
        "algorithm": "opencv_background_lk_ransac_affine_v1",
        "config_sha256": config_sha,
        "shot_boundary_reset": True,
        "foreground_exclusion": True,
    }
    affine = {
        **common,
        "shape": [length, 2, 3],
        "axes": ["frame", "matrix_row", "matrix_col"],
        "unit": "mixed_pixel_affine",
        "coordinate_space": "pixel_xy",
        "metadata": {
            **metadata,
            "array_key": "frame_to_frame_affine",
            "stabilization_array_key": "stabilization_affine",
            "matrix_semantics": "previous_frame_to_current_frame_affine",
        },
    }
    tracks = {
        **common,
        "shape": [length, 400, 2],
        "axes": ["frame", "background_track", "xy"],
        "unit": "px",
        "coordinate_space": "pixel_xy",
        "metadata": {
            **metadata,
            "array_key": "background_points_xy",
            "valid_array_key": "background_valid",
            "point_slot_semantics": "persistent_within_shot_reseedable_slot",
        },
    }
    zoom = {
        **common,
        "shape": [length],
        "axes": ["frame"],
        "unit": "ratio",
        "coordinate_space": "none",
        "metadata": {
            **metadata,
            "array_key": "zoom_proxy",
            "inlier_ratio_array_key": "ransac_inlier_ratio",
        },
    }
    blueprint["camera"] = {
        "per_shot": [
            {
                "camera_motion_id": "camera-shot_000",
                "shot_id": "shot_000",
                "classification": "pan",
                "affine_ref": affine,
                "homography_ref": None,
                "crop_ref": None,
                "zoom_proxy_ref": zoom,
                "shake_ref": None,
                "background_tracks_ref": tracks,
                "intrinsics": None,
                "extrinsics_ref": None,
                "reconstruction_backend": "opencv_ransac_2d",
                "confidence": 0.9,
                "failure_reason": None,
            }
        ],
        "quality": {"score": 0.9, "coverage": 1.0, "warnings": [], "errors": []},
    }
    blueprint["extensions"]["e5_camera_motion"] = {
        "enabled": True,
        "algorithm": "opencv_background_lk_ransac_affine_v1",
        "reconstruction_backend": "opencv_ransac_2d",
        "foreground_exclusion": True,
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config_sha,
        "shot_count": 1,
        "supported_classifications": ["static", "pan", "tilt", "roll", "zoom", "compound", "unknown"],
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    return blueprint


def test_public_validator_accepts_e5_camera_contract() -> None:
    valid, errors = BlueprintValidator().validate(_blueprint())
    assert valid, errors


def test_validator_rejects_unsupported_3d_label_for_2d_backend() -> None:
    blueprint = _blueprint()
    blueprint["camera"]["per_shot"][0]["classification"] = "dolly"
    valid, errors = BlueprintValidator().validate(blueprint)
    assert not valid
    assert any("uncalibrated 2D-only backend" in error for error in errors)


def test_validator_rejects_calibrated_extrinsics_claim() -> None:
    blueprint = _blueprint()
    row = blueprint["camera"]["per_shot"][0]
    row["intrinsics"] = {"model": "pinhole", "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "distortion": []}
    valid, errors = BlueprintValidator().validate(blueprint)
    assert not valid
    assert any("cannot claim calibrated intrinsics/extrinsics" in error for error in errors)


def test_validator_rejects_ref_range_or_config_drift() -> None:
    blueprint = _blueprint()
    broken = copy.deepcopy(blueprint)
    ref = broken["camera"]["per_shot"][0]["affine_ref"]
    ref["frame_end"] -= 1
    ref["metadata"]["config_sha256"] = "c" * 64
    valid, errors = BlueprintValidator().validate(broken)
    assert not valid
    assert any("must cover exactly its shot frame range" in error for error in errors)
    assert any("config_sha256 must match extension config" in error for error in errors)
