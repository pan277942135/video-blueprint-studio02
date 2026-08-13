from types import SimpleNamespace

import numpy as np
import pytest

from packages.pipeline_core import rtmdet_ins_mask_backend as backend
from packages.pipeline_core.rtmdet_ins_mask_backend import (
    APPROVED_RTMDET_INS_SHA256,
    RTMDetInsConfigurationError,
    RTMDetInsMaskConfig,
    RTMDetInsPersonMaskSegmenter,
)


def _files(tmp_path):
    config = tmp_path / "config.py"
    checkpoint = tmp_path / "model.pth"
    config.write_text("model = {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return config, checkpoint


def _config(tmp_path):
    config, checkpoint = _files(tmp_path)
    return RTMDetInsMaskConfig(
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        weights_license="REVIEWED",
        score_threshold=0.3,
        bbox_iou_threshold=0.2,
    )


def _fake_result():
    masks = np.zeros((2, 20, 30), dtype=np.bool_)
    masks[0, 2:16, 3:17] = True
    masks[1, 5:10, 20:25] = True
    return SimpleNamespace(
        pred_instances=SimpleNamespace(
            bboxes=np.array([[3, 2, 17, 16], [20, 5, 25, 10]], dtype=np.float32),
            scores=np.array([0.92, 0.99], dtype=np.float32),
            labels=np.array([0, 1], dtype=np.int64),
            masks=masks,
        )
    )


def _segmenter(monkeypatch, tmp_path, inference):
    monkeypatch.setattr(backend, "_sha256_file", lambda path: APPROVED_RTMDET_INS_SHA256)
    return RTMDetInsPersonMaskSegmenter(
        _config(tmp_path),
        weights_approved=True,
        init_detector_fn=lambda *args, **kwargs: object(),
        inference_detector_fn=inference,
    )


def test_rtmdet_ins_requires_explicit_approval(tmp_path) -> None:
    with pytest.raises(RTMDetInsConfigurationError, match="not approved"):
        RTMDetInsPersonMaskSegmenter(
            _config(tmp_path),
            weights_approved=False,
            init_detector_fn=lambda *args, **kwargs: object(),
            inference_detector_fn=lambda *args, **kwargs: _fake_result(),
        )


def test_rtmdet_ins_rejects_checkpoint_hash_before_init(monkeypatch, tmp_path) -> None:
    initialized = False

    def init(*args, **kwargs):
        nonlocal initialized
        initialized = True
        return object()

    monkeypatch.setattr(backend, "_sha256_file", lambda path: "0" * 64)
    with pytest.raises(RTMDetInsConfigurationError, match="SHA256 mismatch"):
        RTMDetInsPersonMaskSegmenter(
            _config(tmp_path),
            weights_approved=True,
            init_detector_fn=init,
            inference_detector_fn=lambda *args, **kwargs: _fake_result(),
        )
    assert initialized is False


def test_parse_result_keeps_only_person_masks() -> None:
    candidates = RTMDetInsPersonMaskSegmenter.parse_result(
        _fake_result(), frame_shape=(20, 30), score_threshold=0.3
    )
    assert len(candidates) == 1
    bbox, score, mask = candidates[0]
    assert bbox == (3.0, 2.0, 17.0, 16.0)
    assert score == pytest.approx(0.92, abs=1e-6)
    assert mask.shape == (20, 30)
    assert mask.dtype == np.bool_


def test_segment_matches_track_by_iou_and_caches_frame(monkeypatch, tmp_path) -> None:
    calls = 0

    def inference(model, frame):
        nonlocal calls
        calls += 1
        return _fake_result()

    segmenter = _segmenter(monkeypatch, tmp_path, inference)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    first = segmenter.segment(frame, (2.0, 1.0, 18.0, 17.0), 4, "char_000")
    second = segmenter.segment(frame, (2.5, 1.5, 18.0, 17.0), 4, "char_001")
    assert first is not None
    assert second is not None
    assert first.character_id == "char_000"
    assert first.confidence == pytest.approx(0.92, abs=1e-6)
    assert calls == 1


def test_environment_is_disabled_only_when_completely_unconfigured(monkeypatch) -> None:
    for name in (
        "VBS_RTMDET_INS_CONFIG",
        "VBS_RTMDET_INS_CHECKPOINT",
        "VBS_RTMDET_INS_WEIGHTS_LICENSE",
        "VBS_RTMDET_INS_WEIGHTS_APPROVED",
    ):
        monkeypatch.delenv(name, raising=False)
    assert RTMDetInsPersonMaskSegmenter.from_environment() is None
    monkeypatch.setenv("VBS_RTMDET_INS_CONFIG", "/tmp/config.py")
    with pytest.raises(RTMDetInsConfigurationError, match="Incomplete"):
        RTMDetInsPersonMaskSegmenter.from_environment()
