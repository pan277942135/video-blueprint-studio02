import cv2
import numpy as np
import pytest

from packages.pipeline_core.dense_flow import DenseFlowConfig, DenseFlowError, farneback_step


def test_dense_flow_config_is_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("VBS_E4_DENSE_FLOW_ENABLED", raising=False)
    assert DenseFlowConfig.from_environment() is None
    monkeypatch.setenv("VBS_E4_DENSE_FLOW_ENABLED", "false")
    assert DenseFlowConfig.from_environment() is None
    monkeypatch.setenv("VBS_E4_DENSE_FLOW_ENABLED", "true")
    assert DenseFlowConfig.from_environment() == DenseFlowConfig()
    monkeypatch.setenv("VBS_E4_DENSE_FLOW_ENABLED", "yes")
    with pytest.raises(DenseFlowError):
        DenseFlowConfig.from_environment()


def test_farneback_recovers_textured_translation():
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, size=(128, 160), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    transform = np.float32([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    current = cv2.warpAffine(previous, transform, (160, 128), borderMode=cv2.BORDER_REFLECT)

    flow = farneback_step(previous, current, DenseFlowConfig(max_side_px=160))
    central = flow[20:-20, 20:-20]
    median_x = float(np.median(central[..., 0]))
    median_y = float(np.median(central[..., 1]))

    assert median_x == pytest.approx(2.0, abs=0.6)
    assert median_y == pytest.approx(1.0, abs=0.6)
    assert flow.dtype == np.float32
    assert flow.shape == (128, 160, 2)


def test_dense_flow_config_rejects_invalid_geometry():
    with pytest.raises(DenseFlowError):
        DenseFlowConfig(max_side_px=16).validate()
    with pytest.raises(DenseFlowError):
        DenseFlowConfig(winsize=14).validate()
    with pytest.raises(DenseFlowError):
        DenseFlowConfig(poly_n=9).validate()
