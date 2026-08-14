import cv2
import numpy as np
import pytest

from packages.pipeline_core.sparse_motion import (
    SparseMotionConfig,
    SparseMotionError,
    lk_step,
    seed_features,
)


def _pattern() -> np.ndarray:
    image = np.zeros((96, 128), dtype=np.uint8)
    for y in range(20, 80, 15):
        for x in range(20, 110, 15):
            cv2.rectangle(image, (x, y), (x + 5, y + 5), 255, -1)
    return image


def test_seed_features_are_deterministic_and_mask_constrained():
    gray = _pattern()
    mask = np.zeros_like(gray, dtype=np.bool_)
    mask[10:86, 10:118] = True
    config = SparseMotionConfig(max_points=24)

    first = seed_features(gray, mask, 24, config)
    second = seed_features(gray, mask, 24, config)

    assert first.shape == second.shape
    assert np.array_equal(first, second)
    assert 0 < len(first) <= 24
    for x, y in first:
        assert mask[round(float(y)), round(float(x))]


def test_lk_step_recovers_known_translation_inside_mask():
    previous = _pattern()
    transform = np.float32([[1, 0, 2], [0, 1, 1]])
    current = cv2.warpAffine(previous, transform, (previous.shape[1], previous.shape[0]))
    mask = np.ones_like(previous, dtype=np.bool_)
    config = SparseMotionConfig(max_points=24)
    points = seed_features(previous, mask, 24, config)

    moved, valid, _ = lk_step(previous, current, points, mask, config)
    displacement = moved[valid] - points[valid]

    assert np.count_nonzero(valid) >= 8
    assert np.median(displacement[:, 0]) == pytest.approx(2.0, abs=0.2)
    assert np.median(displacement[:, 1]) == pytest.approx(1.0, abs=0.2)


def test_environment_gate_is_explicit(monkeypatch):
    monkeypatch.setenv("VBS_E4_POINT_TRACKS_ENABLED", "yes")
    with pytest.raises(SparseMotionError, match="must be 'true' or 'false'"):
        SparseMotionConfig.from_environment()

    monkeypatch.setenv("VBS_E4_POINT_TRACKS_ENABLED", "false")
    assert SparseMotionConfig.from_environment() is None

    monkeypatch.setenv("VBS_E4_POINT_TRACKS_ENABLED", "true")
    assert isinstance(SparseMotionConfig.from_environment(), SparseMotionConfig)
