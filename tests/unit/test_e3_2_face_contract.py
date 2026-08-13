from packages.blueprint_schema.validator import _validate_face_2d_ref


def _valid_face_ref() -> dict:
    return {
        "uri": "artifacts/timeseries/char_000_face.npz",
        "format": "npz",
        "dtype": "float32",
        "shape": [12, 478, 2],
        "axes": ["frame", "face_landmark", "xy"],
        "unit": "px",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": 11,
        "compression": "zip",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
        "checksum_sha256": "0" * 64,
        "metadata": {"array_key": "face_landmarks_2d"},
    }


def test_face_2d_contract_accepts_pixel_xy_geometry() -> None:
    errors = _validate_face_2d_ref(
        _valid_face_ref(),
        landmark_count=478,
        frame_count=12,
        path="characters[0].face.landmarks_2d_ref",
    )
    assert errors == []


def test_face_2d_contract_allows_explicit_null_when_disabled() -> None:
    errors = _validate_face_2d_ref(
        None,
        landmark_count=0,
        frame_count=12,
        path="characters[0].face.landmarks_2d_ref",
    )
    assert errors == []


def test_face_2d_contract_rejects_fake_camera_xyz_semantics() -> None:
    ref = _valid_face_ref()
    ref["coordinate_space"] = "camera_xyz"

    errors = _validate_face_2d_ref(
        ref,
        landmark_count=478,
        frame_count=12,
        path="characters[0].face.landmarks_2d_ref",
    )
    assert any("coordinate_space must be 'pixel_xy'" in error for error in errors)


def test_face_2d_contract_rejects_wrong_shape_and_axes() -> None:
    ref = _valid_face_ref()
    ref["shape"] = [12, 478, 3]
    ref["axes"] = ["frame", "face_landmark", "xyz"]

    errors = _validate_face_2d_ref(
        ref,
        landmark_count=478,
        frame_count=12,
        path="characters[0].face.landmarks_2d_ref",
    )
    assert any("shape must be [frame_count, landmark_count, 2]" in error for error in errors)
    assert any("axes must equal" in error for error in errors)


def test_face_2d_contract_rejects_silent_fill_or_interpolation() -> None:
    ref = _valid_face_ref()
    ref["nan_policy"] = "zero_fill"
    ref["interpolation_policy"] = "linear_short_gap"

    errors = _validate_face_2d_ref(
        ref,
        landmark_count=478,
        frame_count=12,
        path="characters[0].face.landmarks_2d_ref",
    )
    assert any("nan_policy must be 'preserve'" in error for error in errors)
    assert any("interpolation_policy must be 'none'" in error for error in errors)


def test_face_2d_contract_rejects_timeline_or_landmark_count_mismatch() -> None:
    ref = _valid_face_ref()
    ref["shape"] = [11, 477, 2]

    errors = _validate_face_2d_ref(
        ref,
        landmark_count=478,
        frame_count=12,
        path="characters[0].face.landmarks_2d_ref",
    )
    assert any("must equal timebase.frame_count" in error for error in errors)
    assert any("must equal face.landmark_count" in error for error in errors)
