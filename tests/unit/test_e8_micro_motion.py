from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from packages.blueprint_schema.micro_motion_contract import validate_micro_motion_contract
from packages.pipeline_core.micro_motion import MicroMotionConfig, MicroMotionError, run_micro_motion
from packages.pipeline_core.sparse_motion import atomic_npz, sha256_file


def _ref(uri: str, checksum: str) -> dict:
    return {"uri": uri, "checksum_sha256": checksum}


def _fixture(tmp_path: Path, *, frame_count: int = 40, frequency_hz: float = 1.0) -> tuple[dict, dict]:
    fps = 10.0
    point_slots = 8
    times = np.arange(frame_count, dtype=np.float32) / fps
    displacement = 0.02 * np.sin(2.0 * math.pi * frequency_hz * times)
    delta = np.zeros(frame_count, dtype=np.float32)
    delta[1:] = displacement[1:] - displacement[:-1]

    residual = np.full((frame_count, point_slots, 2), np.nan, dtype=np.float32)
    residual_valid = np.zeros((frame_count, point_slots), dtype=np.bool_)
    local_positions = np.zeros((frame_count, point_slots, 2), dtype=np.float32)
    local_valid = np.ones((frame_count, point_slots), dtype=np.bool_)
    track_id = np.tile(np.arange(point_slots, dtype=np.int64), (frame_count, 1))
    gains = np.linspace(0.9, 1.1, point_slots, dtype=np.float32)
    for frame_idx in range(1, frame_count):
        residual[frame_idx, :, 1] = delta[frame_idx] * gains
        residual[frame_idx, :, 0] = 0.0
        residual_valid[frame_idx] = True
        local_positions[frame_idx] = local_positions[frame_idx - 1]
        local_positions[frame_idx, :, 1] += residual[frame_idx, :, 1]

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
    surface_ref = _ref(surface_uri, sha256_file(str(surface_path)))

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
        **_ref(body_uri, sha256_file(str(body_path))),
        "coordinate_space": "body_local_2d",
    }

    affine = np.tile(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), (frame_count, 1, 1))
    camera_path = tmp_path / "camera.npz"
    atomic_npz(str(camera_path), frame_to_frame_affine=affine)
    camera_uri = "artifacts/timeseries/camera-shot_000_camera_2d.npz"
    camera_ref = {
        **_ref(camera_uri, sha256_file(str(camera_path))),
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "metadata": {"array_key": "frame_to_frame_affine"},
    }

    mask_ref = {"uri": "artifacts/timeseries/char_000_mask.rle.json"}
    pending_micro = {
        "kind": "not_detected",
        "signal_ref": None,
        "detrended_signal_ref": None,
        "vertical_displacement_ref": None,
        "radial_expansion_ratio_ref": None,
        "area_change_ratio_ref": None,
        "velocity_ref": None,
        "acceleration_ref": None,
        "axis_2d": [0.0, 0.0],
        "amplitude_px_p50": 0.0,
        "amplitude_norm_p50": 0.0,
        "dominant_frequency_hz": None,
        "phase_rad": None,
        "periodicity_score": 0.0,
        "spatial_coherence": 0.0,
        "camera_leakage_score": 0.0,
        "pose_leakage_score": 0.0,
        "occlusion_ratio": 0.0,
        "confidence": 0.0,
        "usable_for_generation": False,
        "limitations": ["E8 pending"],
        "smoothing": {"method": "none", "parameters": {}},
    }
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
                            "micro_motion": pending_micro,
                            "quality": {"score": 1.0, "coverage": 1.0, "warnings": [], "errors": []},
                        }
                    ],
                    "events": [],
                    "quality": {"score": 1.0, "coverage": 1.0, "warnings": [], "errors": []},
                },
            }
        ],
        "extensions": {
            "e7_surface_motion": {"enabled": True},
        },
    }
    sidecars = {
        surface_uri: str(surface_path),
        body_uri: str(body_path),
        camera_uri: str(camera_path),
    }
    return blueprint, sidecars


def test_periodic_geometry_is_detected_without_physiological_label(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path)
    emitted, _, extension, quality = run_micro_motion(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=MicroMotionConfig(),
    )
    micro = blueprint["characters"][0]["surface_motion"]["regions"][0]["micro_motion"]
    assert micro["kind"] == "periodic_micro_motion"
    assert micro["dominant_frequency_hz"] == pytest.approx(1.0, abs=0.26)
    assert micro["periodicity_score"] >= 0.50
    assert micro["spatial_coherence"] >= 0.90
    assert micro["usable_for_generation"] is True
    assert extension["physiological_inference_performed"] is False
    assert extension["new_model_weights_introduced"] is False
    assert quality["coverage"] == 1.0
    assert micro["signal_ref"]["uri"] in emitted
    assert all("breathing" not in str(micro["kind"]).lower() for _ in [0])


def test_short_window_cannot_claim_usable_periodicity(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path, frame_count=10)
    run_micro_motion(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=MicroMotionConfig(min_periodic_frames=8),
    )
    micro = blueprint["characters"][0]["surface_motion"]["regions"][0]["micro_motion"]
    assert micro["usable_for_generation"] is False


def test_contract_rejects_physiology_or_quality_gate_bypass(tmp_path: Path) -> None:
    blueprint, sidecars = _fixture(tmp_path)
    _, _, extension, _ = run_micro_motion(
        blueprint,
        output_dir=str(tmp_path),
        sidecars=sidecars,
        config=MicroMotionConfig(),
    )
    blueprint["extensions"]["e8_micro_motion"] = extension
    assert validate_micro_motion_contract(blueprint) == []

    blueprint["extensions"]["e8_micro_motion"]["physiological_inference_performed"] = True
    errors = validate_micro_motion_contract(blueprint)
    assert any("physiological_inference_performed" in error for error in errors)

    blueprint["extensions"]["e8_micro_motion"]["physiological_inference_performed"] = False
    micro = blueprint["characters"][0]["surface_motion"]["regions"][0]["micro_motion"]
    micro["usable_for_generation"] = True
    micro["camera_leakage_score"] = 1.0
    errors = validate_micro_motion_contract(blueprint)
    assert any("camera_leakage_score exceeds usable threshold" in error for error in errors)


def test_environment_gate_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBS_E8_MICRO_MOTION_ENABLED", "maybe")
    with pytest.raises(MicroMotionError, match="must be 'true' or 'false'"):
        MicroMotionConfig.from_environment()
