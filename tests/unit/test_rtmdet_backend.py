import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from packages.pipeline_core.rtmdet_backend import (
    RTMDetBackendConfig,
    RTMDetConfigurationError,
    RTMDetPersonDetector,
)


def _write_model_files(tmp_path):
    config_path = tmp_path / "rtmdet.py"
    checkpoint_path = tmp_path / "rtmdet.pth"
    config_path.write_text("model = dict(type='RTMDet')\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"approved-test-checkpoint")
    return config_path, checkpoint_path


def test_rtmdet_requires_explicit_weight_approval(tmp_path):
    config_path, checkpoint_path = _write_model_files(tmp_path)
    config = RTMDetBackendConfig(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        weights_license="TEST-ONLY",
    )

    with pytest.raises(RTMDetConfigurationError, match="explicitly approved"):
        RTMDetPersonDetector(
            config,
            weights_approved=False,
            init_detector_fn=lambda *_args, **_kwargs: object(),
            inference_detector_fn=lambda *_args, **_kwargs: None,
        )


def test_rtmdet_parses_only_valid_person_detections(tmp_path):
    config_path, checkpoint_path = _write_model_files(tmp_path)
    fake_result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            bboxes=np.asarray(
                [
                    [10.0, 20.0, 110.0, 220.0],
                    [15.0, 25.0, 100.0, 200.0],
                    [30.0, 40.0, 130.0, 240.0],
                    [50.0, 50.0, 40.0, 80.0],
                ],
                dtype=np.float32,
            ),
            scores=np.asarray([0.95, 0.20, 0.90, 0.99], dtype=np.float32),
            labels=np.asarray([0, 0, 2, 0], dtype=np.int64),
        )
    )
    init_calls = []

    def _init(config_file, checkpoint_file, *, device):
        init_calls.append((config_file, checkpoint_file, device))
        return "fake-model"

    detector = RTMDetPersonDetector(
        RTMDetBackendConfig(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            weights_license="TEST-ONLY",
            device="cpu",
            score_threshold=0.35,
        ),
        weights_approved=True,
        init_detector_fn=_init,
        inference_detector_fn=lambda model, frame: fake_result,
    )

    detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8), frame_idx=7)

    assert init_calls == [(str(config_path), str(checkpoint_path), "cpu")]
    assert len(detections) == 1
    assert detections[0].frame_idx == 7
    assert detections[0].bbox_xyxy == pytest.approx((10.0, 20.0, 110.0, 220.0))
    assert detections[0].score == pytest.approx(0.95)

    provenance = detector.provenance(code_commit="deadbeef", config_hash="cfg")
    assert provenance["tool"] == "MMDetection RTMDet"
    assert provenance["weights_sha256"] == hashlib.sha256(b"approved-test-checkpoint").hexdigest()
    assert provenance["license"] == "TEST-ONLY"


def test_rtmdet_environment_is_disabled_only_when_completely_unconfigured(monkeypatch):
    for key in [
        "VBS_RTMDET_CONFIG",
        "VBS_RTMDET_CHECKPOINT",
        "VBS_RTMDET_WEIGHTS_LICENSE",
        "VBS_RTMDET_WEIGHTS_APPROVED",
    ]:
        monkeypatch.delenv(key, raising=False)

    assert RTMDetPersonDetector.from_environment() is None

    monkeypatch.setenv("VBS_RTMDET_CONFIG", "/tmp/rtmdet.py")
    with pytest.raises(RTMDetConfigurationError, match="Incomplete RTMDet configuration"):
        RTMDetPersonDetector.from_environment()
