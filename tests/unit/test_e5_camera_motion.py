from __future__ import annotations

import numpy as np
import pytest

from packages.pipeline_core.camera_motion import (
    CameraMotionConfig,
    CameraMotionError,
    classify_camera_motion,
    estimate_similarity_ransac,
)


def _matrix(*, tx: float = 0.0, ty: float = 0.0, angle_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle) * scale
    s = np.sin(angle) * scale
    return np.asarray([[c, -s, tx], [s, c, ty]], dtype=np.float32)


def test_ransac_recovers_translation() -> None:
    config = CameraMotionConfig(min_matches=6)
    previous = np.asarray(
        [[10, 10], [30, 10], [50, 10], [10, 40], [30, 40], [50, 40], [70, 60], [90, 80]],
        dtype=np.float32,
    )
    current = previous + np.asarray([2.0, -1.0], dtype=np.float32)
    matrix, ratio = estimate_similarity_ransac(previous, current, config)
    assert matrix is not None
    assert ratio is not None and ratio > 0.99
    assert matrix[0, 2] == pytest.approx(2.0, abs=0.05)
    assert matrix[1, 2] == pytest.approx(-1.0, abs=0.05)


def test_ransac_refuses_too_few_matches() -> None:
    config = CameraMotionConfig(min_matches=6)
    points = np.zeros((5, 2), dtype=np.float32)
    assert estimate_similarity_ransac(points, points, config) == (None, None)


def test_classification_is_limited_to_supported_2d_labels() -> None:
    config = CameraMotionConfig()
    cases = {
        "static": _matrix(),
        "pan": _matrix(tx=3.0),
        "tilt": _matrix(ty=3.0),
        "roll": _matrix(angle_deg=0.5),
        "zoom": _matrix(scale=1.01),
        "compound": _matrix(tx=3.0, angle_deg=0.5),
    }
    for expected, matrix in cases.items():
        stack = np.stack((np.full((2, 3), np.nan, dtype=np.float32), matrix, matrix))
        assert classify_camera_motion(stack, width=640, height=480, config=config) == expected
    assert not {"dolly", "truck", "pedestal"}.intersection(cases)


def test_unknown_when_no_transform_is_observed() -> None:
    values = np.full((4, 2, 3), np.nan, dtype=np.float32)
    assert classify_camera_motion(values, width=640, height=480, config=CameraMotionConfig()) == "unknown"


def test_environment_gate_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBS_E5_CAMERA_MOTION_ENABLED", "not-a-bool")
    with pytest.raises(CameraMotionError, match="must be 'true' or 'false'"):
        CameraMotionConfig.from_environment()
