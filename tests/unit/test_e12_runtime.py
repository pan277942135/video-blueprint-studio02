from __future__ import annotations

import hashlib

import pytest

from packages.pipeline_core.e12_runtime import E12RuntimePaths, _download_verified


def test_download_verified_reuses_matching_existing_asset(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"approved-e12-asset")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()

    actual = _download_verified(
        url="https://example.invalid/not-used",
        destination=asset,
        expected_sha256=expected,
        label="test asset",
    )

    assert actual == expected
    assert asset.read_bytes() == b"approved-e12-asset"


def test_download_verified_refuses_mismatched_existing_asset(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _download_verified(
            url="https://example.invalid/not-used",
            destination=asset,
            expected_sha256="0" * 64,
            label="test asset",
        )

    assert asset.read_bytes() == b"tampered"


def test_runtime_paths_emit_complete_cli_arguments(tmp_path):
    paths = E12RuntimePaths(
        det_config=tmp_path / "det.py",
        det_checkpoint=tmp_path / "det.pth",
        pose_config=tmp_path / "pose.py",
        pose_checkpoint=tmp_path / "pose.pth",
        mask_config=tmp_path / "mask.py",
        mask_checkpoint=tmp_path / "mask.pth",
        face_task=tmp_path / "face.task",
        hand_task=tmp_path / "hand.task",
        manifest=tmp_path / "manifest.json",
    )

    args = paths.as_cli_args()

    assert args[::2] == [
        "--det-config",
        "--det-checkpoint",
        "--pose-config",
        "--pose-checkpoint",
        "--mask-config",
        "--mask-checkpoint",
        "--face-task",
        "--hand-task",
    ]
    assert args[1::2] == [
        str(paths.det_config),
        str(paths.det_checkpoint),
        str(paths.pose_config),
        str(paths.pose_checkpoint),
        str(paths.mask_config),
        str(paths.mask_checkpoint),
        str(paths.face_task),
        str(paths.hand_task),
    ]
