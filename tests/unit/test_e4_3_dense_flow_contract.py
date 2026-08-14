import copy
import json
from pathlib import Path

from packages.blueprint_schema.validator import BlueprintValidator


def _base_blueprint():
    path = Path("contracts/example_blueprint.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _extension(blueprint):
    frame_count = blueprint["timebase"]["frame_count"]
    width = blueprint["source_video"]["width"]
    height = blueprint["source_video"]["height"]
    grid_width = min(width, 64)
    grid_height = max(1, round(height * grid_width / width))
    config_sha = "a" * 64
    return {
        "enabled": True,
        "algorithm": "opencv_farneback_v1",
        "coordinate_space": "pixel_xy",
        "vector_unit": "px",
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": config_sha,
        "flow_ref": {
            "uri": "artifacts/timeseries/source_dense_flow.npz",
            "format": "npz",
            "dtype": "float32",
            "shape": [frame_count, grid_height, grid_width, 2],
            "axes": ["frame", "grid_y", "grid_x", "xy"],
            "unit": "px",
            "coordinate_space": "pixel_xy",
            "sampling": "per_frame",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "compression": "zip",
            "nan_policy": "preserve",
            "interpolation_policy": "none",
            "checksum_sha256": "b" * 64,
            "metadata": {
                "array_key": "flow_xy",
                "valid_frame_array_key": "valid_frame",
                "algorithm": "opencv_farneback_v1",
                "vector_semantics": "previous_frame_to_current_frame_displacement",
                "source_width": width,
                "source_height": height,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "grid_to_source_scale_x": width / grid_width,
                "grid_to_source_scale_y": height / grid_height,
                "vectors_scaled_to_source_pixels": True,
                "shot_boundary_reset": True,
                "config_sha256": config_sha,
            },
        },
        "quality": {
            "coverage": 1.0,
            "valid_pairs": max(0, frame_count - 1),
            "expected_pairs": max(0, frame_count - 1),
            "median_motion_px": 1.0,
            "p95_motion_px": 1.5,
        },
    }


def test_validator_accepts_dense_flow_manifest_contract():
    blueprint = _base_blueprint()
    blueprint.setdefault("extensions", {})["e4_dense_flow"] = _extension(blueprint)
    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors


def test_validator_rejects_dense_flow_shape_and_config_mismatch():
    blueprint = _base_blueprint()
    extension = _extension(blueprint)
    broken = copy.deepcopy(extension)
    broken["flow_ref"]["shape"][0] += 1
    broken["flow_ref"]["metadata"]["config_sha256"] = "c" * 64
    blueprint.setdefault("extensions", {})["e4_dense_flow"] = broken

    valid, errors = BlueprintValidator().validate(blueprint)
    assert not valid
    assert any("shape[0] must equal frame_count" in error for error in errors)
    assert any("config_sha256 must match extension config" in error for error in errors)
