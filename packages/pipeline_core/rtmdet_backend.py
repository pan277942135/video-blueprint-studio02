from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import numpy as np

from packages.pipeline_core.person_tracking import PersonDetection


class RTMDetConfigurationError(RuntimeError):
    """Raised when the RTMDet runtime is missing or uses unapproved weights."""


@dataclass(frozen=True)
class RTMDetBackendConfig:
    config_path: str
    checkpoint_path: str
    weights_license: str
    device: str = "cpu"
    score_threshold: float = 0.35


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class RTMDetPersonDetector:
    """MMDetection RTMDet adapter restricted to the COCO `person` class.

    No identity or biometric inference is performed. The adapter requires a
    local config and checkpoint plus an explicit approval flag from the caller.
    It never downloads weights and never substitutes another detector.
    """

    def __init__(
        self,
        config: RTMDetBackendConfig,
        *,
        weights_approved: bool,
        init_detector_fn: Callable[..., Any] | None = None,
        inference_detector_fn: Callable[..., Any] | None = None,
    ) -> None:
        if not weights_approved:
            raise RTMDetConfigurationError(
                "RTMDet checkpoint must be explicitly approved before inference"
            )
        if not os.path.isfile(config.config_path):
            raise RTMDetConfigurationError(f"RTMDet config does not exist: {config.config_path}")
        if not os.path.isfile(config.checkpoint_path):
            raise RTMDetConfigurationError(
                f"RTMDet checkpoint does not exist: {config.checkpoint_path}"
            )
        if not config.weights_license.strip():
            raise RTMDetConfigurationError("RTMDet checkpoint license must be recorded explicitly")
        if not 0.0 <= config.score_threshold <= 1.0:
            raise RTMDetConfigurationError("RTMDet score_threshold must be within [0, 1]")

        self.config = config
        self.weights_sha256 = _sha256_file(config.checkpoint_path)
        self.config_sha256 = _sha256_file(config.config_path)

        if init_detector_fn is None or inference_detector_fn is None:
            try:
                from mmdet.apis import inference_detector, init_detector
            except ImportError as exc:
                raise RTMDetConfigurationError(
                    "MMDetection RTMDet runtime is not installed. Install the approved "
                    "MMDetection/MMCV/MMEngine stack before enabling person detection."
                ) from exc
            init_detector_fn = init_detector
            inference_detector_fn = inference_detector

        self._inference_detector = inference_detector_fn
        self._model = init_detector_fn(
            config.config_path,
            config.checkpoint_path,
            device=config.device,
        )

    @classmethod
    def from_environment(cls) -> RTMDetPersonDetector | None:
        """Create a configured detector, or return None when E2.2 is intentionally disabled.

        Supplying only part of the configuration is an error. This preserves the
        fail-closed contract: a user cannot accidentally believe person tracking
        ran when the model path, license record, or explicit approval is missing.
        """
        config_path = os.environ.get("VBS_RTMDET_CONFIG")
        checkpoint_path = os.environ.get("VBS_RTMDET_CHECKPOINT")
        weights_license = os.environ.get("VBS_RTMDET_WEIGHTS_LICENSE")
        approval_raw = os.environ.get("VBS_RTMDET_WEIGHTS_APPROVED")

        configured_values = [config_path, checkpoint_path, weights_license, approval_raw]
        if all(value is None for value in configured_values):
            return None
        if any(value is None for value in configured_values):
            raise RTMDetConfigurationError(
                "Incomplete RTMDet configuration: VBS_RTMDET_CONFIG, VBS_RTMDET_CHECKPOINT, "
                "VBS_RTMDET_WEIGHTS_LICENSE, and VBS_RTMDET_WEIGHTS_APPROVED are all required"
            )

        approved = str(approval_raw).strip().lower() in {"1", "true", "yes"}
        threshold_raw = os.environ.get("VBS_RTMDET_SCORE_THRESHOLD", "0.35")
        try:
            threshold = float(threshold_raw)
        except ValueError as exc:
            raise RTMDetConfigurationError(
                f"Invalid VBS_RTMDET_SCORE_THRESHOLD: {threshold_raw}"
            ) from exc

        return cls(
            RTMDetBackendConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                weights_license=str(weights_license),
                device=os.environ.get("VBS_RTMDET_DEVICE", "cpu"),
                score_threshold=threshold,
            ),
            weights_approved=approved,
        )

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[PersonDetection]:
        result = self._inference_detector(self._model, frame)
        return self.parse_result(result, frame_idx=frame_idx, score_threshold=self.config.score_threshold)

    @staticmethod
    def parse_result(
        result: Any,
        *,
        frame_idx: int,
        score_threshold: float,
    ) -> list[PersonDetection]:
        instances = getattr(result, "pred_instances", None)
        if instances is None:
            raise RTMDetConfigurationError("RTMDet result is missing pred_instances")

        bboxes = _to_numpy(getattr(instances, "bboxes", []))
        scores = _to_numpy(getattr(instances, "scores", []))
        labels = _to_numpy(getattr(instances, "labels", []))
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise RTMDetConfigurationError(f"Unexpected RTMDet bbox shape: {bboxes.shape}")
        if scores.ndim != 1 or labels.ndim != 1:
            raise RTMDetConfigurationError("Unexpected RTMDet score/label dimensions")
        if not (len(bboxes) == len(scores) == len(labels)):
            raise RTMDetConfigurationError("RTMDet result lengths do not match")

        detections: list[PersonDetection] = []
        for bbox, score, label in zip(bboxes, scores, labels, strict=True):
            if int(label) != 0 or float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                PersonDetection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=float(score),
                )
            )

        detections.sort(key=lambda item: (-item.score, item.bbox_xyxy))
        return detections

    def provenance(self, *, code_commit: str | None, config_hash: str) -> dict[str, Any]:
        return {
            "module": "person_detection",
            "tool": "MMDetection RTMDet",
            "version": _package_version("mmdet"),
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": self.config.weights_license,
        }
