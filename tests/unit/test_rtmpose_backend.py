from types import SimpleNamespace

import numpy as np
import pytest

from packages.pipeline_core.rtmpose_backend import (
    RTMPoseBackendConfig,
    RTMPoseConfigurationError,
    RTMPosePoseEstimator,
)


def test_rtmpose_requires_explicit_weight_approval(tmp_path):
    config_path = tmp_path / "pose.py"
    checkpoint_path = tmp_path / "pose.pth"
    config_path.write_text("model = {}\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"approved-pose-weights")

    config = RTMPoseBackendConfig(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        weights_license="TEST-ONLY",
    )
    with pytest.raises(RTMPoseConfigurationError, match="explicitly approved"):
        RTMPosePoseEstimator(
            config,
            weights_approved=False,
            init_model_fn=lambda *args, **kwargs: object(),
            inference_topdown_fn=lambda *args, **kwargs: [],
        )


def test_rtmpose_parses_coco17_result():
    keypoints = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
    scores = np.linspace(0.1, 0.9, 17, dtype=np.float32).reshape(1, 17)
    result = [
        SimpleNamespace(
            pred_instances=SimpleNamespace(
                keypoints=keypoints,
                keypoint_scores=scores,
            )
        )
    ]

    observation = RTMPosePoseEstimator.parse_result(
        result,
        frame_idx=12,
        character_id="char_007",
    )

    assert observation is not None
    assert observation.frame_idx == 12
    assert observation.character_id == "char_007"
    assert observation.keypoints_xy.shape == (17, 2)
    assert observation.confidence.shape == (17,)
    np.testing.assert_allclose(observation.keypoints_xy, keypoints[0])
    np.testing.assert_allclose(observation.confidence, scores[0])


def test_rtmpose_empty_topdown_result_is_explicitly_missing():
    assert (
        RTMPosePoseEstimator.parse_result([], frame_idx=4, character_id="char_001")
        is None
    )


def test_rtmpose_environment_requires_complete_configuration(monkeypatch):
    monkeypatch.setenv("VBS_RTMPOSE_CONFIG", "/tmp/pose.py")
    monkeypatch.delenv("VBS_RTMPOSE_CHECKPOINT", raising=False)
    monkeypatch.delenv("VBS_RTMPOSE_WEIGHTS_LICENSE", raising=False)
    monkeypatch.delenv("VBS_RTMPOSE_WEIGHTS_APPROVED", raising=False)

    with pytest.raises(RTMPoseConfigurationError, match="Incomplete RTMPose configuration"):
        RTMPosePoseEstimator.from_environment()
