from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file
from packages.pipeline_core.surface_motion import SurfaceMotionConfig, SurfaceMotionError, run_surface_motion


def _fixture(tmp_path: Path, *, deform_last: bool) -> tuple[dict, dict]:
    frame_count = 3
    point_slots = 2
    positions = np.asarray(
        [
            [[10.0, 10.0], [20.0, 10.0]],
            [[11.0, 10.0], [21.0, 10.0]],
            [[12.0, 10.0], [22.5 if deform_last else 22.0, 10.0]],
        ],
        dtype=np.float32,
    )
    valid = np.ones((frame_count, point_slots), dtype=np.bool_)
    track_id = np.tile(np.asarray([0, 1], dtype=np.int64), (frame_count, 1))
    track_path = tmp_path / "tracks.npz"
    atomic_npz(str(track_path), positions_xy=positions, valid=valid, track_id=track_id, tracking_error=np.zeros_like(valid, dtype=np.float32))
    track_ref = {
        "uri": "artifacts/timeseries/char_000_point_tracks.npz",
        "checksum_sha256": sha256_file(str(track_path)),
    }

    transforms = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, -2.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    body_valid = np.ones(frame_count, dtype=np.bool_)
    body_path = tmp_path / "body.npz"
    atomic_npz(str(body_path), source_pixel_to_body_local=transforms, valid_frame=body_valid)
    body_ref = {
        "uri": "artifacts/timeseries/char_000_body_local_2d.npz",
        "coordinate_space": "body_local_2d",
        "checksum_sha256": sha256_file(str(body_path)),
    }
    mask_ref = {"uri": "artifacts/timeseries/char_000_mask.rle.json"}
    blueprint = {
        "timebase": {"frame_count": frame_count},
        "shots": [{"frame_start": 0, "frame_end": 2, "shot_id": "shot_000"}],
        "characters": [
            {
                "character_id": "char_000",
                "person_mask_ref": mask_ref,
                "surface_motion": {
                    "coordinate_frame": "body_local_2d",
                    "body_frame_transform_ref": body_ref,
                    "global_postural_sway_ref": None,
                    "regions": [],
                    "events": [],
                    "quality": {"score": 0.0, "coverage": 0.0, "warnings": [], "errors": []},
                },
            }
        ],
        "extensions": {
            "e4_point_tracks": {
                "enabled": True,
                "characters": {"char_000": {"track_points_ref": track_ref, "quality": {}}},
            },
            "e6_body_local_frame": {"enabled": True},
        },
    }
    sidecars = {track_ref["uri"]: str(track_path), body_ref["uri"]: str(body_path)}
    return blueprint, sidecars


def test_rigid_translation_becomes_zero_body_local_residual(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path, deform_last=False)
    emitted, _, extension, quality = run_surface_motion(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=SurfaceMotionConfig(),
    )
    assert extension["enabled"] is True
    assert quality["coverage"] == 1.0
    ref = blueprint["characters"][0]["surface_motion"]["regions"][0]["residual_flow_ref"]
    with np.load(emitted[ref["uri"]], allow_pickle=False) as arrays:
        residual = arrays["residual_displacement"]
        valid = arrays["residual_valid"]
    assert np.count_nonzero(valid) == 4
    assert np.nanmax(np.abs(residual)) == pytest.approx(0.0, abs=1e-6)


def test_local_deformation_survives_body_motion_removal(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path, deform_last=True)
    emitted, _, _, _ = run_surface_motion(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=SurfaceMotionConfig(),
    )
    ref = blueprint["characters"][0]["surface_motion"]["regions"][0]["residual_flow_ref"]
    with np.load(emitted[ref["uri"]], allow_pickle=False) as arrays:
        residual = arrays["residual_displacement"]
        valid = arrays["residual_valid"]
    assert valid[2, 1]
    assert residual[2, 1, 0] == pytest.approx(0.5, abs=1e-6)


def test_track_id_change_and_shot_boundary_forbid_residual(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path, deform_last=False)
    track_path = Path(sidecars["artifacts/timeseries/char_000_point_tracks.npz"])
    with np.load(track_path, allow_pickle=False) as arrays:
        positions = arrays["positions_xy"].copy()
        valid = arrays["valid"].copy()
        track_id = arrays["track_id"].copy()
        error = arrays["tracking_error"].copy()
    track_id[2, 0] = 99
    atomic_npz(str(track_path), positions_xy=positions, valid=valid, track_id=track_id, tracking_error=error)
    blueprint["extensions"]["e4_point_tracks"]["characters"]["char_000"]["track_points_ref"]["checksum_sha256"] = sha256_file(str(track_path))
    blueprint["shots"] = [
        {"frame_start": 0, "frame_end": 1, "shot_id": "shot_000"},
        {"frame_start": 2, "frame_end": 2, "shot_id": "shot_001"},
    ]
    emitted, _, _, _ = run_surface_motion(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=SurfaceMotionConfig(),
    )
    ref = blueprint["characters"][0]["surface_motion"]["regions"][0]["residual_flow_ref"]
    with np.load(emitted[ref["uri"]], allow_pickle=False) as arrays:
        residual_valid = arrays["residual_valid"]
    assert not np.any(residual_valid[2])


def test_environment_gate_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBS_E7_SURFACE_MOTION_ENABLED", "maybe")
    with pytest.raises(SurfaceMotionError, match="must be 'true' or 'false'"):
        SurfaceMotionConfig.from_environment()
