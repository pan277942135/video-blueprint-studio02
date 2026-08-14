from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from packages.pipeline_core.mediapipe_face_hand_backend import (
    APPROVED_FACE_TASK_SHA256,
    APPROVED_HAND_TASK_SHA256,
)

DETECTOR_SHA256 = "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
POSE_SHA256 = "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"
MASK_SHA256 = "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885"

DETECTOR_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)
POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"
)
MASK_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
)
# The approved E3.2 evidence records these exact public GCS object generations.
# Pin them instead of resolving `latest`, while retaining SHA256 verification below.
FACE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task?generation=1683136941468629"
)
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task?generation=1682480005356399"
)


@dataclass(frozen=True)
class E12RuntimePaths:
    det_config: pathlib.Path
    det_checkpoint: pathlib.Path
    pose_config: pathlib.Path
    pose_checkpoint: pathlib.Path
    mask_config: pathlib.Path
    mask_checkpoint: pathlib.Path
    face_task: pathlib.Path
    hand_task: pathlib.Path
    manifest: pathlib.Path

    def as_cli_args(self) -> list[str]:
        return [
            "--det-config",
            str(self.det_config),
            "--det-checkpoint",
            str(self.det_checkpoint),
            "--pose-config",
            str(self.pose_config),
            "--pose-checkpoint",
            str(self.pose_checkpoint),
            "--mask-config",
            str(self.mask_config),
            "--mask-checkpoint",
            str(self.mask_checkpoint),
            "--face-task",
            str(self.face_task),
            "--hand-task",
            str(self.hand_task),
        ]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_openmmlab_configs() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    try:
        mmdet: Any = import_module("mmdet")
        mmpose: Any = import_module("mmpose")
    except ImportError as exc:
        raise RuntimeError(
            "E12 runtime bootstrap requires mmdet and mmpose to be installed before asset bootstrap"
        ) from exc

    det_root = pathlib.Path(mmdet.__file__).resolve().parent / ".mim" / "configs" / "rtmdet"
    det_config = det_root / "rtmdet_tiny_8xb32-300e_coco.py"
    mask_config = det_root / "rtmdet-ins_tiny_8xb32-300e_coco.py"
    pose_config = (
        pathlib.Path(mmpose.__file__).resolve().parent
        / ".mim"
        / "configs"
        / "body_2d_keypoint"
        / "rtmpose"
        / "coco"
        / "rtmpose-t_8xb256-420e_coco-256x192.py"
    )
    for path, label in (
        (det_config, "RTMDet config"),
        (mask_config, "RTMDet-Ins config"),
        (pose_config, "RTMPose config"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} not found in installed package: {path}")
    return det_config, pose_config, mask_config


def _download_verified(
    *,
    url: str,
    destination: pathlib.Path,
    expected_sha256: str,
    label: str,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f"existing {label} SHA256 mismatch; refusing to overwrite: {actual} != {expected_sha256}"
            )
        return actual

    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(f"downloaded {label} SHA256 mismatch: {actual} != {expected_sha256}")
        temporary.replace(destination)
        return actual
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def bootstrap_e12_runtime(runtime_dir: pathlib.Path) -> E12RuntimePaths:
    """Resolve approved configs and materialize only byte-exact approved E12 runtime assets."""
    runtime_dir = runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    det_config, pose_config, mask_config = _resolve_openmmlab_configs()

    assets = (
        (
            "detector",
            DETECTOR_URL,
            runtime_dir / "rtmdet_tiny.pth",
            DETECTOR_SHA256,
            "RTMDet checkpoint",
        ),
        (
            "pose",
            POSE_URL,
            runtime_dir / "rtmpose_tiny.pth",
            POSE_SHA256,
            "RTMPose checkpoint",
        ),
        (
            "mask",
            MASK_URL,
            runtime_dir / "rtmdet_ins_tiny.pth",
            MASK_SHA256,
            "RTMDet-Ins checkpoint",
        ),
        (
            "face_task",
            FACE_TASK_URL,
            runtime_dir / "face_landmarker.task",
            APPROVED_FACE_TASK_SHA256,
            "FaceLandmarker task",
        ),
        (
            "hand_task",
            HAND_TASK_URL,
            runtime_dir / "hand_landmarker.task",
            APPROVED_HAND_TASK_SHA256,
            "HandLandmarker task",
        ),
    )

    verified: dict[str, dict[str, Any]] = {}
    for name, url, destination, expected_sha256, label in assets:
        actual = _download_verified(
            url=url,
            destination=destination,
            expected_sha256=expected_sha256,
            label=label,
        )
        verified[name] = {
            "path": str(destination),
            "url": url,
            "sha256": actual,
            "size_bytes": destination.stat().st_size,
        }

    manifest = runtime_dir / "e12_runtime_manifest.json"
    payload = {
        "status": "verified",
        "policy": "download-if-missing; refuse existing or downloaded SHA mismatch",
        "configs": {
            "detector": str(det_config),
            "pose": str(pose_config),
            "mask": str(mask_config),
        },
        "assets": verified,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return E12RuntimePaths(
        det_config=det_config,
        det_checkpoint=runtime_dir / "rtmdet_tiny.pth",
        pose_config=pose_config,
        pose_checkpoint=runtime_dir / "rtmpose_tiny.pth",
        mask_config=mask_config,
        mask_checkpoint=runtime_dir / "rtmdet_ins_tiny.pth",
        face_task=runtime_dir / "face_landmarker.task",
        hand_task=runtime_dir / "hand_landmarker.task",
        manifest=manifest,
    )


__all__ = [
    "DETECTOR_SHA256",
    "MASK_SHA256",
    "POSE_SHA256",
    "E12RuntimePaths",
    "bootstrap_e12_runtime",
    "sha256_file",
]
