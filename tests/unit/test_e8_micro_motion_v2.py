from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from packages.blueprint_schema.micro_motion_v2_contract import validate_micro_motion_v2_contract
from packages.pipeline_core.micro_motion import MicroMotionConfig
from packages.pipeline_core.micro_motion_v2 import run_micro_motion_v2
from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    frame_count = 40
    fps = 10.0
    point_slots = 30
    periodic_slots = 6
    times = np.arange(frame_count, dtype=np.float32) / fps
    displacement = 0.025 * np.sin(2.0 * math.pi * times)
    delta = np.zeros(frame_count, dtype=np.float32)
    delta[1:] = displacement[1:] - displacement[:-1]

    residual = np.full((frame_count, point_slots, 2), np.nan, dtype=np.float32)
    residual_valid = np.zeros((frame_count, point_slots), dtype=np.bool_)
    local_positions = np.zeros((frame_count, point_slots, 2), dtype=np.float32)
    local_valid = np.ones((frame_count, point_slots), dtype=np.bool_)
    track_id = np.tile(np.arange(point_slots, dtype=np.int64), (frame_count, 1))
    rng = np.random.default_rng(7)
    for frame_idx in range(1, frame_count):
        residual_valid[frame_idx] = True
        residual[frame_idx, :periodic_slots, 0] = 0.0
        residual[frame_idx, :periodic_slots, 1] = delta[frame_idx] * np.linspace(0.9, 1.1, periodic_slots)
        noise = rng.normal(0.0, 0.0035, size=(point_slots - periodic_slots, 2)).astype(np.float32)
        residual[frame_idx, periodic_slots:] = noise
        local_positions[frame_idx] = local_positions[frame_idx - 1] + np.nan_to_num(residual[frame_idx])

    surface_path = tmp_path / "surface.npz"
    atomic_npz(
        str(surface_path),
        body_local_positions=local_positions,
        body_local_valid=local_valid,
        residual_displacement=residual,
        residual_valid=residual_valid,
        track_id=track_id,
    )
    surface_uri = "artifacts/timeseries/char_000_surface_sparse_body_local.npz"
    surface_checksum = sha256_file(str(surface_path))

    source_to_body = np.tile(
        np.asarray([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32),
        (frame_count, 1, 1),
    )
    origins = np.zeros((frame_count, 2), dtype=np.float32)
    torso_scale = np.full(frame_count, 100.0, dtype=np.float32)
    body_valid = np.ones(frame_count, dtype=np.bool_)
    body_path = tmp_path / "body.npz"
    atomic_npz(
        str(body_path),
        source_pixel_to_body_local=source_to_body,
        body_origin_stabilized_xy=origins,
        torso_scale_px=torso_scale,
        valid_frame=body_valid,
    )
    body_uri = "artifacts/timeseries/char_000_body_local_2d.npz"
    body_ref = {
        "uri": body_uri,
        "checksum_sha256": sha256_file(str(body_path)),
        "coordinate_space": "body_local_2d",
    }

    affine = np.tile(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), (frame_count, 1, 1))
    camera_path = tmp_path / "camera.npz"
    atomic_npz(str(camera_path), frame_to_frame_affine=affine)
    camera_uri = "artifacts/timeseries/camera-shot_000_camera_2d.npz"
    camera_ref = {
        "uri": camera_uri,
        "checksum_sha256": sha256_file(str(camera_path)),
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "metadata": {"array_key": "frame_to_frame_affine"},
    }

    mask_ref = {"uri": "artifacts/timeseries/char_000_mask.rle.json"}
    surface_ref = {"uri": surface_uri, "checksum_sha256": surface_checksum}
    blueprint = {
        "source_video": {"width": 192, "height": 128},
        "timebase": {"frame_count": frame_count, "fps_num": 10, "fps_den": 1},
        "shots": [{"frame_start": 0, "frame_end": frame_count - 1, "shot_id": "shot_000"}],
        "camera": {"per_shot": [{"affine_ref": camera_ref}]},
        "characters": [
            {
                "character_id": "char_000",
                "person_mask_ref": mask_ref,
                "surface_motion": {
                    "coordinate_frame": "body_local_2d",
                    "body_frame_transform_ref": body_ref,
                    "global_postural_sway_ref": None,
                    "regions": [
                        {
                            "region_id": "custom",
                            "display_name": "whole_body_sparse_surface",
                            "roi_definition": {
                                "type": "mask_intersection",
                                "anchor_keypoints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
                                "expansion_ratio": 1.0,
                            },
                            "mask_ref": mask_ref,
                            "track_points_ref": surface_ref,
                            "residual_flow_ref": surface_ref,
                            "deformation_modes_ref": None,
                            "micro_motion": {},
                            "quality": {"score": 1.0, "coverage": 1.0, "warnings": [], "errors": []},
                        }
                    ],
                    "events": [],
                    "quality": {"score": 1.0, "coverage": 1.0, "warnings": [], "errors": []},
                },
            }
        ],
        "extensions": {"e7_surface_motion": {"enabled": True}},
    }
    sidecars = {surface_uri: str(surface_path), body_uri: str(body_path), camera_uri: str(camera_path)}
    return blueprint, sidecars


def test_local_periodic_subset_survives_unrelated_whole_body_tracks(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path)
    emitted, _, extension, _ = run_micro_motion_v2(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=MicroMotionConfig(),
    )
    micro = blueprint["characters"][0]["surface_motion"]["regions"][0]["micro_motion"]
    row = extension["characters"][0]
    assert extension["algorithm"] == "body_local_sparse_periodicity_v2"
    assert row["candidate_track_count"] >= 20
    assert 4 <= row["selected_track_count"] < row["candidate_track_count"]
    assert row["consensus_frequency_hz"] == pytest.approx(1.0, abs=0.30)
    assert micro["kind"] == "periodic_micro_motion"
    assert micro["dominant_frequency_hz"] == pytest.approx(1.0, abs=0.30)
    assert micro["periodicity_score"] >= 0.50
    assert micro["usable_for_generation"] is True
    assert micro["pose_leakage_score"] < 0.60
    signal_uri = micro["signal_ref"]["uri"]
    with np.load(emitted[signal_uri], allow_pickle=False) as arrays:
        selected = np.asarray(arrays["selected_track_id"])
        assert 4 <= len(selected) < point_slots_from_blueprint(blueprint)
        assert set(selected.tolist()).issubset(set(range(6)))
    blueprint["extensions"]["e8_micro_motion"] = extension
    assert validate_micro_motion_v2_contract(blueprint) == []


def point_slots_from_blueprint(blueprint: dict) -> int:
    # The fixture owns 30 deterministic E7 point slots; keeping this helper tied
    # to the test manifest makes the selected-subset assertion intention explicit.
    assert blueprint["characters"][0]["character_id"] == "char_000"
    return 30


def test_v2_contract_rejects_global_support_or_selection_metadata_drift(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path)
    _, _, extension, _ = run_micro_motion_v2(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=MicroMotionConfig(),
    )
    blueprint["extensions"]["e8_micro_motion"] = extension
    row = extension["characters"][0]
    row["consensus_support_fraction"] = 1.0
    errors = validate_micro_motion_v2_contract(blueprint)
    assert any("consensus support must equal selected/candidate" in error for error in errors)
