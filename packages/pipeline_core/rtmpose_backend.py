from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from packages.pipeline_core.pose_estimation import PoseObservation


class RTMPoseConfigurationError(RuntimeError):
    """Raised when the RTMPose runtime is incomplete or uses unapproved weights."""


@dataclass(frozen=True)
class RTMPoseBackendConfig:
    config_path: str
    checkpoint_path: str
    weights_license: str
    expected_weights_sha256: str
    device: str = "cpu"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise RTMPoseConfigurationError(
            "RTMPose expected checkpoint SHA256 must be exactly 64 hexadecimal characters"
        )
    return normalized


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


class RTMPosePoseEstimator:
    """MMPose RTMPose top-down adapter restricted to COCO-17 body pose.

    The adapter consumes anonymous person bounding boxes from E2.2 and produces
    only frame-local body keypoints/confidence. It performs no identity
    inference, face recognition, re-identification, or biometric export.
    """

    def __init__(
        self,
        config: RTMPoseBackendConfig,
        *,
        weights_approved: bool,
        init_model_fn: Callable[..., Any] | None = None,
        inference_topdown_fn: Callable[..., Any] | None = None,
    ) -> None:
        if not weights_approved:
            raise RTMPoseConfigurationError(
                "RTMPose checkpoint must be explicitly approved before inference"
            )
        if not os.path.isfile(config.config_path):
            raise RTMPoseConfigurationError(f"RTMPose config does not exist: {config.config_path}")
        if not os.path.isfile(config.checkpoint_path):
            raise RTMPoseConfigurationError(
                f"RTMPose checkpoint does not exist: {config.checkpoint_path}"
            )
        if not config.weights_license.strip():
            raise RTMPoseConfigurationError("RTMPose checkpoint license must be recorded explicitly")

        expected_weights_sha256 = _normalize_sha256(config.expected_weights_sha256)
        actual_weights_sha256 = _sha256_file(config.checkpoint_path)
        if actual_weights_sha256 != expected_weights_sha256:
            raise RTMPoseConfigurationError(
                "RTMPose checkpoint SHA256 mismatch: "
                f"expected {expected_weights_sha256}, got {actual_weights_sha256}"
            )

        self.config = config
        self.weights_sha256 = actual_weights_sha256
        self.config_sha256 = _sha256_file(config.config_path)

        if init_model_fn is None or inference_topdown_fn is None:
            try:
                from mmpose.apis import inference_topdown, init_model
            except ImportError as exc:
                raise RTMPoseConfigurationError(
                    "MMPose RTMPose runtime is not installed. Install the approved "
                    "MMPose/MMCV/MMEngine stack before enabling E3.1 pose estimation."
                ) from exc
            init_model_fn = init_model
            inference_topdown_fn = inference_topdown

        self._inference_topdown = inference_topdown_fn
        self._model = init_model_fn(
            config.config_path,
            config.checkpoint_path,
            device=config.device,
        )

    @classmethod
    def from_environment(cls) -> RTMPosePoseEstimator | None:
        config_path = os.environ.get("VBS_RTMPOSE_CONFIG")
        checkpoint_path = os.environ.get("VBS_RTMPOSE_CHECKPOINT")
        weights_license = os.environ.get("VBS_RTMPOSE_WEIGHTS_LICENSE")
        expected_weights_sha256 = os.environ.get("VBS_RTMPOSE_EXPECTED_SHA256")
        approval_raw = os.environ.get("VBS_RTMPOSE_WEIGHTS_APPROVED")

        configured_values = [
            config_path,
            checkpoint_path,
            weights_license,
            expected_weights_sha256,
            approval_raw,
        ]
        if all(value is None for value in configured_values):
            return None
        if any(value is None for value in configured_values):
            raise RTMPoseConfigurationError(
                "Incomplete RTMPose configuration: VBS_RTMPOSE_CONFIG, VBS_RTMPOSE_CHECKPOINT, "
                "VBS_RTMPOSE_WEIGHTS_LICENSE, VBS_RTMPOSE_EXPECTED_SHA256, and "
                "VBS_RTMPOSE_WEIGHTS_APPROVED are all required"
            )

        approved = str(approval_raw).strip().lower() in {"1", "true", "yes"}
        return cls(
            RTMPoseBackendConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                weights_license=str(weights_license),
                expected_weights_sha256=str(expected_weights_sha256),
                device=os.environ.get("VBS_RTMPOSE_DEVICE", "cpu"),
            ),
            weights_approved=approved,
        )

    def estimate(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        frame_idx: int,
        character_id: str,
    ) -> PoseObservation | None:
        bboxes = np.asarray([bbox_xyxy], dtype=np.float32)
        result = self._inference_topdown(self._model, frame, bboxes=bboxes)
        return self.parse_result(result, frame_idx=frame_idx, character_id=character_id)

    @staticmethod
    def parse_result(
        result: Any,
        *,
        frame_idx: int,
        character_id: str,
    ) -> PoseObservation | None:
        if not isinstance(result, (list, tuple)):
            raise RTMPoseConfigurationError("RTMPose result must be a list of pose data samples")
        if len(result) == 0:
            return None
        if len(result) != 1:
            raise RTMPoseConfigurationError(
                f"Top-down RTMPose expected one person result, received {len(result)}"
            )

        instances = getattr(result[0], "pred_instances", None)
        if instances is None:
            raise RTMPoseConfigurationError("RTMPose result is missing pred_instances")

        keypoints = _to_numpy(getattr(instances, "keypoints", []))
        confidence = _to_numpy(getattr(instances, "keypoint_scores", []))
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if confidence.ndim == 2 and confidence.shape[0] == 1:
            confidence = confidence[0]
        if keypoints.shape != (17, 2):
            raise RTMPoseConfigurationError(
                f"RTMPose COCO-17 keypoints must have shape (17, 2), got {keypoints.shape}"
            )
        if confidence.shape != (17,):
            raise RTMPoseConfigurationError(
                f"RTMPose COCO-17 confidence must have shape (17,), got {confidence.shape}"
            )
        if not np.all(np.isfinite(keypoints)):
            raise RTMPoseConfigurationError("RTMPose keypoints contain non-finite coordinates")
        if not np.all(np.isfinite(confidence)):
            raise RTMPoseConfigurationError("RTMPose confidence contains non-finite values")

        clipped_confidence = np.clip(confidence.astype(np.float32), 0.0, 1.0)
        return PoseObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            keypoints_xy=keypoints.astype(np.float32),
            confidence=clipped_confidence,
        )

    def provenance(self, *, code_commit: str | None, config_hash: str) -> dict[str, Any]:
        return {
            "module": "pose_2d",
            "tool": "MMPose RTMPose",
            "version": _package_version("mmpose"),
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": self.config.weights_license,
        }
