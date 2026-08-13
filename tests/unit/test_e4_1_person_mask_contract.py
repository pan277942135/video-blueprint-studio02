from packages.blueprint_schema.validator import _validate_person_mask_ref


def _valid_ref() -> dict:
    return {
        "uri": "artifacts/timeseries/char_000_person_mask.rle.json",
        "format": "rle_json",
        "dtype": "bool",
        "shape": [12, 720, 1280],
        "axes": ["frame", "y", "x"],
        "unit": "binary",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": 11,
        "compression": None,
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": "0" * 64,
        "metadata": {
            "encoding": "row_major_binary_rle_v1",
            "foreground_value": 1,
            "background_value": 0,
            "missing_frame_value": None,
        },
    }


def test_person_mask_contract_accepts_binary_rle() -> None:
    errors = _validate_person_mask_ref(
        _valid_ref(), frame_count=12, height=720, width=1280, path="characters[0].person_mask_ref"
    )
    assert errors == []


def test_person_mask_contract_allows_null() -> None:
    assert _validate_person_mask_ref(
        None, frame_count=12, height=720, width=1280, path="characters[0].person_mask_ref"
    ) == []


def test_person_mask_contract_rejects_bbox_like_or_dense_semantics() -> None:
    ref = _valid_ref()
    ref["format"] = "npz"
    ref["dtype"] = "float32"
    ref["shape"] = [12, 4]
    ref["axes"] = ["frame", "bbox_component"]
    errors = _validate_person_mask_ref(
        ref, frame_count=12, height=720, width=1280, path="characters[0].person_mask_ref"
    )
    assert any("format" in error for error in errors)
    assert any("shape" in error for error in errors)
    assert any("axes" in error for error in errors)


def test_person_mask_contract_rejects_interpolation_or_wrong_coordinates() -> None:
    ref = _valid_ref()
    ref["interpolation_policy"] = "linear_short_gap"
    ref["coordinate_space"] = "normalized_uv"
    errors = _validate_person_mask_ref(
        ref, frame_count=12, height=720, width=1280, path="characters[0].person_mask_ref"
    )
    assert any("interpolation_policy" in error for error in errors)
    assert any("coordinate_space" in error for error in errors)


def test_person_mask_contract_rejects_missing_frame_fill_policy() -> None:
    ref = _valid_ref()
    ref["metadata"]["missing_frame_value"] = 0
    errors = _validate_person_mask_ref(
        ref, frame_count=12, height=720, width=1280, path="characters[0].person_mask_ref"
    )
    assert any("missing_frame_value must be null" in error for error in errors)
