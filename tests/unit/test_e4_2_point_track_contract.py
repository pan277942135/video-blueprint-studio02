import copy
import json
from pathlib import Path

from packages.blueprint_schema.validator import BlueprintValidator


def _blueprint():
    return json.loads(Path("contracts/example_blueprint.json").read_text(encoding="utf-8"))


def _extension():
    return {
        "enabled": True,
        "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
        "coordinate_space": "pixel_xy",
        "max_points": 96,
        "mask_constrained": True,
        "shot_boundary_reset": True,
        "interpolation": False,
        "config_sha256": "a" * 64,
        "characters": {},
    }


def test_e4_2_extension_accepts_explicit_pixel_motion_semantics():
    blueprint = _blueprint()
    blueprint["extensions"]["e4_point_tracks"] = _extension()
    valid, errors = BlueprintValidator().validate(blueprint)
    assert valid, errors


def test_e4_2_extension_rejects_interpolation_or_body_local_masquerading():
    blueprint = _blueprint()
    extension = _extension()
    extension["interpolation"] = True
    extension["coordinate_space"] = "body_local_xy"
    blueprint["extensions"]["e4_point_tracks"] = extension
    valid, errors = BlueprintValidator().validate(blueprint)
    assert not valid
    assert any("interpolation" in error for error in errors)
    assert any("coordinate_space" in error for error in errors)


def test_e4_2_extension_rejects_unknown_character_reference():
    blueprint = _blueprint()
    extension = copy.deepcopy(_extension())
    extension["characters"]["char_ghost"] = {"track_points_ref": None, "quality": {}}
    blueprint["extensions"]["e4_point_tracks"] = extension
    valid, errors = BlueprintValidator().validate(blueprint)
    assert not valid
    assert any("Cross-Reference" in error for error in errors)
