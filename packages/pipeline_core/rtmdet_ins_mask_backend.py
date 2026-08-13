from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from packages.pipeline_core.person_mask import PersonMaskObservation
from packages.pipeline_core.person_tracking import bbox_iou

APPROVED_RTMDET_INS_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"


class RTMDetInsConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RTMDetInsMaskConfig:
    config_path: str
    checkpoint_path: str
    weights_license: str
    device: str = "cpu"
    score_threshold: float = 0.30
    bbox_iou_threshold: float = 0.20


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


class RTMDetInsPersonMaskSegmenter:
    """MMDetection RTMDet-Ins adapter for COCO person instance masks."""

    def __init__(
        self,
        config: RTMDetInsMaskConfig,
        *,
        weights_approved: bool,
        init_detector_fn: Callable[..., Any] | None = None,
        inference_detector_fn: Callable[..., Any] | None = None,
    ) -> None:
        if not weights_approved:
            raise RTMDetInsConfigurationError("RTMDet-Ins checkpoint is not approved for inference")
        if not os.path.isfile(config.config_path):
            raise RTMDetInsConfigurationError(f"RTMDet-Ins config does not exist: {config.config_path}")
        if not os.path.isfile(config.checkpoint_path):
            raise RTMDetInsConfigurationError(
                f"RTMDet-Ins checkpoint does not exist: {config.checkpoint_path}"
            )
        if not config.weights_license.strip():
            raise RTMDetInsConfigurationError("RTMDet-Ins checkpoint license posture must be recorded")
        if not 0.0 <= config.score_threshold <= 1.0:
            raise RTMDetInsConfigurationError("score_threshold must be within [0, 1]")
        if not 0.0 <= config.bbox_iou_threshold <= 1.0:
            raise RTMDetInsConfigurationError("bbox_iou_threshold must be within [0, 1]")

        self.config = config
        self.weights_sha256 = _sha256_file(config.checkpoint_path)
        if self.weights_sha256 != APPROVED_RTMDET_INS_SHA256:
            raise RTMDetInsConfigurationError(
                f"RTMDet-Ins checkpoint SHA256 mismatch: expected {APPROVED_RTMDET_INS_SHA256}, "
                f"got {self.weights_sha256}"
            )
        self.config_sha256 = _sha256_file(config.config_path)

        if init_detector_fn is None or inference_detector_fn is None:
            try:
                from mmdet.apis import inference_detector, init_detector
            except ImportError as exc:
                raise RTMDetInsConfigurationError("Approved MMDetection runtime is not installed") from exc
            init_detector_fn = init_detector
            inference_detector_fn = inference_detector

        self._model = init_detector_fn(config.config_path, config.checkpoint_path, device=config.device)
        self._inference_detector = inference_detector_fn
        self._cached_frame_idx: int | None = None
        self._cached_candidates: list[tuple[tuple[float, float, float, float], float, np.ndarray]] = []

    @classmethod
    def from_environment(cls) -> RTMDetInsPersonMaskSegmenter | None:
        config_path = os.environ.get("VBS_RTMDET_INS_CONFIG")
        checkpoint_path = os.environ.get("VBS_RTMDET_INS_CHECKPOINT")
        weights_license = os.environ.get("VBS_RTMDET_INS_WEIGHTS_LICENSE")
        approval_raw = os.environ.get("VBS_RTMDET_INS_WEIGHTS_APPROVED")
        configured = [config_path, checkpoint_path, weights_license, approval_raw]
        if all(value is None for value in configured):
            return None
        if any(value is None for value in configured):
            raise RTMDetInsConfigurationError(
                "Incomplete RTMDet-Ins configuration: config, checkpoint, license, and approval are required"
            )
        try:
            score_threshold = float(os.environ.get("VBS_RTMDET_INS_SCORE_THRESHOLD", "0.30"))
            iou_threshold = float(os.environ.get("VBS_RTMDET_INS_IOU_THRESHOLD", "0.20"))
        except ValueError as exc:
            raise RTMDetInsConfigurationError("Invalid RTMDet-Ins threshold configuration") from exc
        approved = str(approval_raw).strip().lower() in {"1", "true", "yes"}
        return cls(
            RTMDetInsMaskConfig(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                weights_license=str(weights_license),
                device=os.environ.get("VBS_RTMDET_INS_DEVICE", "cpu"),
                score_threshold=score_threshold,
                bbox_iou_threshold=iou_threshold,
            ),
            weights_approved=approved,
        )

    def _infer_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        result = self._inference_detector(self._model, frame)
        self._cached_candidates = self.parse_result(
            result,
            frame_shape=frame.shape[:2],
            score_threshold=self.config.score_threshold,
        )
        self._cached_frame_idx = frame_idx

    @staticmethod
    def parse_result(
        result: Any,
        *,
        frame_shape: tuple[int, int],
        score_threshold: float,
    ) -> list[tuple[tuple[float, float, float, float], float, np.ndarray]]:
        instances = getattr(result, "pred_instances", None)
        if instances is None:
            raise RTMDetInsConfigurationError("RTMDet-Ins result is missing pred_instances")
        bboxes = _to_numpy(getattr(instances, "bboxes", []))
        scores = _to_numpy(getattr(instances, "scores", []))
        labels = _to_numpy(getattr(instances, "labels", []))
        masks = _to_numpy(getattr(instances, "masks", []))
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise RTMDetInsConfigurationError(f"Unexpected bbox shape: {bboxes.shape}")
        if scores.ndim != 1 or labels.ndim != 1 or masks.ndim != 3:
            raise RTMDetInsConfigurationError("Unexpected RTMDet-Ins result dimensions")
        if not (len(bboxes) == len(scores) == len(labels) == len(masks)):
            raise RTMDetInsConfigurationError("RTMDet-Ins result lengths do not match")
        if tuple(masks.shape[1:]) != tuple(frame_shape):
            raise RTMDetInsConfigurationError(
                f"RTMDet-Ins mask shape {masks.shape[1:]} does not match frame {frame_shape}"
            )

        candidates: list[tuple[tuple[float, float, float, float], float, np.ndarray]] = []
        for bbox, score, label, mask in zip(bboxes, scores, labels, masks, strict=True):
            if int(label) != 0 or float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            binary = np.asarray(mask, dtype=np.bool_)
            if not np.any(binary):
                continue
            candidates.append(((x1, y1, x2, y2), float(score), binary))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates

    def segment(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        frame_idx: int,
        character_id: str,
    ) -> PersonMaskObservation | None:
        if self._cached_frame_idx != frame_idx:
            self._infer_frame(frame, frame_idx)
        matches = [
            (bbox_iou(bbox_xyxy, candidate_bbox), score, mask)
            for candidate_bbox, score, mask in self._cached_candidates
        ]
        matches = [item for item in matches if item[0] >= self.config.bbox_iou_threshold]
        if not matches:
            return None
        _, score, mask = max(matches, key=lambda item: (item[0], item[1]))
        return PersonMaskObservation(
            frame_idx=frame_idx,
            character_id=character_id,
            mask=mask,
            confidence=score,
        )

    def provenance(self, *, code_commit: str | None, config_hash: str) -> dict[str, Any]:
        return {
            "module": "person_mask",
            "tool": "MMDetection RTMDet-Ins Tiny",
            "version": _package_version("mmdet"),
            "code_commit": code_commit,
            "weights_sha256": self.weights_sha256,
            "config_hash": config_hash,
            "license": self.config.weights_license,
        }
