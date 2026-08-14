from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from packages.blueprint_schema.environment_contract import validate_environment_contract
from packages.pipeline_core.environment_photometry import (
    EnvironmentPhotometryConfig,
    EnvironmentPhotometryError,
    run_environment_photometry,
)
from packages.pipeline_core.person_mask import encode_binary_rle
from packages.pipeline_core.sparse_motion import sha256_file


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 4.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("test video writer failed")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _fixture(tmp_path: Path, *, missing_mask_frame: int | None = None) -> tuple[str, dict, dict, np.ndarray]:
    frame_count = 4
    height, width = 48, 64
    person_mask = np.zeros((height, width), dtype=np.bool_)
    person_mask[12:40, 22:43] = True
    frames: list[np.ndarray] = []
    for frame_idx, background_value in enumerate((70, 90, 115, 145)):
        frame = np.full((height, width, 3), background_value, dtype=np.uint8)
        person_value = (20 + frame_idx * 50, 230 - frame_idx * 45, 40 + frame_idx * 30)
        frame[person_mask] = person_value
        frames.append(frame)
    video_path = tmp_path / "analysis.mp4"
    _write_video(video_path, frames)

    mask_uri = "artifacts/timeseries/char_000_mask.rle.json"
    mask_path = tmp_path / "mask.json"
    mask_frames = [encode_binary_rle(person_mask) for _ in range(frame_count)]
    if missing_mask_frame is not None:
        mask_frames[missing_mask_frame] = None
    payload = {
        "encoding": "row_major_binary_rle_v1",
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "frames": mask_frames,
    }
    mask_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    mask_ref = {
        "uri": mask_uri,
        "checksum_sha256": sha256_file(str(mask_path)),
    }
    blueprint = {
        "source_video": {"width": width, "height": height},
        "timebase": {"frame_count": frame_count, "fps_num": 4, "fps_den": 1},
        "shots": [{"shot_id": "shot_000", "frame_start": 0, "frame_end": frame_count - 1}],
        "characters": [
            {
                "character_id": "char_000",
                "presence": [{"frame_start": 0, "frame_end": frame_count - 1, "confidence": 1.0}],
                "person_mask_ref": mask_ref,
            }
        ],
        "extensions": {"e4_person_mask": {"enabled": True}},
    }
    return str(video_path), blueprint, {mask_uri: str(mask_path)}, person_mask


def test_background_photometry_excludes_person_and_tracks_luminance(tmp_path: Path) -> None:
    video_path, blueprint, sidecars, person_mask = _fixture(tmp_path)
    environment, emitted, _, extension, quality = run_environment_photometry(
        video_path,
        blueprint=blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=EnvironmentPhotometryConfig(mask_erosion_radius_px=1),
    )
    metrics_path = emitted[environment["luminance_ref"]["uri"]]
    with np.load(metrics_path, allow_pickle=False) as arrays:
        luminance = np.asarray(arrays["luminance"])
        exposure = np.asarray(arrays["exposure_change_ev_proxy"])
        valid = np.asarray(arrays["valid_frame"]).astype(bool)
        chroma = np.asarray(arrays["white_balance_chromaticity_rb"])
        sharpness = np.asarray(arrays["blur_laplacian_variance"])
    assert valid.tolist() == [True, True, True, True]
    assert np.all(np.diff(luminance) > 0.0)
    assert np.all(exposure[1:] > 0.0)
    assert np.allclose(chroma[:, 0], 1.0 / 3.0, atol=0.03)
    assert np.allclose(chroma[:, 1], 1.0 / 3.0, atol=0.03)
    assert np.all(np.isfinite(sharpness))
    assert extension["depth_emitted"] is False
    assert extension["semantic_scene_inference_performed"] is False
    assert quality["coverage"] == 1.0

    mask_payload = json.loads(Path(emitted[environment["background_mask_ref"]["uri"]]).read_text(encoding="utf-8"))
    first_background = mask_payload["frames"][0]
    assert first_background is not None
    assert first_background["height"] == person_mask.shape[0]
    assert first_background["width"] == person_mask.shape[1]

    blueprint["environment"] = environment
    blueprint["extensions"]["e9_environment"] = extension
    assert validate_environment_contract(blueprint) == []


def test_present_character_missing_mask_invalidates_environment_frame(tmp_path: Path) -> None:
    video_path, blueprint, sidecars, _ = _fixture(tmp_path, missing_mask_frame=2)
    environment, emitted, _, extension, quality = run_environment_photometry(
        video_path,
        blueprint=blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=EnvironmentPhotometryConfig(mask_erosion_radius_px=1),
    )
    with np.load(emitted[environment["luminance_ref"]["uri"]], allow_pickle=False) as arrays:
        luminance = np.asarray(arrays["luminance"])
        exposure = np.asarray(arrays["exposure_change_ev_proxy"])
        valid = np.asarray(arrays["valid_frame"]).astype(bool)
        exposure_valid = np.asarray(arrays["exposure_valid"]).astype(bool)
    assert valid.tolist() == [True, True, False, True]
    assert np.isnan(luminance[2])
    assert np.isnan(exposure[2]) and np.isnan(exposure[3])
    assert exposure_valid.tolist() == [False, True, False, False]
    mask_payload = json.loads(Path(emitted[environment["background_mask_ref"]["uri"]]).read_text(encoding="utf-8"))
    assert mask_payload["frames"][2] is None
    assert extension["valid_frame_count"] == 3
    assert quality["coverage"] == pytest.approx(0.75)


def test_environment_contract_rejects_depth_or_semantic_claims(tmp_path: Path) -> None:
    video_path, blueprint, sidecars, _ = _fixture(tmp_path)
    environment, _, _, extension, _ = run_environment_photometry(
        video_path,
        blueprint=blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=EnvironmentPhotometryConfig(mask_erosion_radius_px=1),
    )
    blueprint["environment"] = environment
    blueprint["extensions"]["e9_environment"] = extension
    blueprint["environment"]["depth_ref"] = {"uri": "fake-depth.npz"}
    blueprint["extensions"]["e9_environment"]["weather_inference_performed"] = True
    errors = validate_environment_contract(blueprint)
    assert any("depth_ref must remain null" in error for error in errors)
    assert any("weather_inference_performed" in error for error in errors)


def test_environment_gate_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBS_E9_ENVIRONMENT_ENABLED", "maybe")
    with pytest.raises(EnvironmentPhotometryError, match="must be 'true' or 'false'"):
        EnvironmentPhotometryConfig.from_environment()
