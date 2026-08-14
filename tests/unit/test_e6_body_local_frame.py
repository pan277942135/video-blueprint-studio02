from __future__ import annotations

import numpy as np
import pytest

from packages.pipeline_core.body_local_frame import (
    BodyLocalFrameConfig,
    BodyLocalFrameError,
    derive_body_local_transform,
)


def _apply(matrix: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    value = np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    return matrix.astype(np.float64) @ value


def test_body_local_transform_maps_pelvis_to_origin_and_shoulders_to_unit_y() -> None:
    anchors = np.asarray(
        [
            [60.0, 40.0],
            [40.0, 40.0],
            [55.0, 80.0],
            [45.0, 80.0],
        ],
        dtype=np.float32,
    )
    stabilization = np.asarray([[1.0, 0.0, -10.0], [0.0, 1.0, -5.0]], dtype=np.float32)
    result = derive_body_local_transform(
        anchors,
        stabilization_affine=stabilization,
        min_torso_scale_px=8.0,
    )
    assert result is not None
    source_to_body, body_to_source, origin, torso_scale = result
    assert origin == pytest.approx([40.0, 75.0])
    assert torso_scale == pytest.approx(40.0)
    assert _apply(source_to_body, (50.0, 80.0)) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert _apply(source_to_body, (50.0, 40.0)) == pytest.approx([0.0, 1.0], abs=1e-6)
    assert _apply(source_to_body, (60.0, 40.0))[0] > 0.0
    recovered = _apply(body_to_source, tuple(_apply(source_to_body, (63.0, 52.0))))
    assert recovered == pytest.approx([63.0, 52.0], abs=1e-5)


def test_body_local_transform_refuses_degenerate_or_missing_evidence() -> None:
    stabilization = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    flat = np.asarray([[10.0, 10.0], [20.0, 10.0], [10.0, 10.0], [20.0, 10.0]], dtype=np.float32)
    assert derive_body_local_transform(flat, stabilization_affine=stabilization, min_torso_scale_px=8.0) is None
    missing = flat.copy()
    missing[0] = np.nan
    assert derive_body_local_transform(missing, stabilization_affine=stabilization, min_torso_scale_px=8.0) is None


def test_body_local_environment_gate_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBS_E6_BODY_LOCAL_FRAME_ENABLED", "maybe")
    with pytest.raises(BodyLocalFrameError, match="must be 'true' or 'false'"):
        BodyLocalFrameConfig.from_environment()
