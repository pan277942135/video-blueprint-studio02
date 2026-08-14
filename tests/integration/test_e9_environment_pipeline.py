from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from packages.pipeline_core.e9_media_pipeline import run_e9_media_pipeline
from packages.pipeline_core.environment_photometry import EnvironmentPhotometryConfig
from packages.pipeline_core.person_mask import encode_binary_rle
from packages.pipeline_core.sparse_motion import sha256_file


def _write_video(path: Path, frame_count: int, width: int, height: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    assert writer.isOpened()
    try:
        for idx in range(frame_count):
            frame = np.full((height, width, 3), 80 + idx * 15, dtype=np.uint8)
            frame[10:34, 20:40] = (30, 180, 60)
            writer.write(frame)
    finally:
        writer.release()


def test_e9_wrapper_emits_environment_stage_and_physical_sidecars(tmp_path: Path, monkeypatch) -> None:
    frame_count, width, height = 4, 64, 48
    artifact_root = tmp_path / "job"
    normalized_dir = artifact_root / "artifacts" / "normalized"
    normalized_dir.mkdir(parents=True)
    normalized = normalized_dir / "analysis_cfr.mp4"
    _write_video(normalized, frame_count, width, height)

    person_mask = np.zeros((height, width), dtype=np.bool_)
    person_mask[10:34, 20:40] = True
    mask_uri = "artifacts/timeseries/char_000_mask.rle.json"
    mask_path = artifact_root / mask_uri
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_payload = {
        "encoding": "row_major_binary_rle_v1",
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "frames": [encode_binary_rle(person_mask) for _ in range(frame_count)],
    }
    mask_path.write_text(json.dumps(mask_payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    blueprint = {
        "source_video": {"width": width, "height": height},
        "timebase": {"frame_count": frame_count, "fps_num": 4, "fps_den": 1},
        "shots": [{"shot_id": "shot_000", "frame_start": 0, "frame_end": frame_count - 1}],
        "characters": [
            {
                "character_id": "char_000",
                "presence": [{"frame_start": 0, "frame_end": frame_count - 1, "confidence": 1.0}],
                "person_mask_ref": {"uri": mask_uri, "checksum_sha256": sha256_file(str(mask_path))},
            }
        ],
        "environment": {
            "background_mask_ref": None,
            "depth_ref": None,
            "luminance_ref": None,
            "exposure_change_ref": None,
            "white_balance_proxy_ref": None,
            "blur_ref": None,
            "occluder_tracks": [],
        },
        "processing": {"pipeline_version": "0.8.0-e8.1", "config_hash": "a" * 64, "stages": []},
        "artifacts": {"reports": []},
        "extensions": {"e4_person_mask": {"enabled": True}},
        "quality": {"module_scores": {}},
        "provenance": {"tools": []},
    }
    sidecars = {
        "artifacts/normalized/analysis_cfr.mp4": str(normalized),
        mask_uri: str(mask_path),
    }

    def fake_e8(**kwargs):
        del kwargs
        return blueprint, sidecars

    monkeypatch.setattr("packages.pipeline_core.e9_media_pipeline.run_e8_media_pipeline", fake_e8)
    result, emitted = run_e9_media_pipeline(
        job_id="job",
        video_file_name="input.mp4",
        video_sha256="b" * 64,
        video_path=str(normalized),
        environment_config=EnvironmentPhotometryConfig(mask_erosion_radius_px=1),
    )

    assert result["processing"]["pipeline_version"] == "0.9.0-e9.1"
    assert result["processing"]["stages"][-1]["name"] == "environment"
    assert result["extensions"]["e9_environment"]["coverage"] == 1.0
    assert result["environment"]["depth_ref"] is None
    assert result["environment"]["occluder_tracks"] == []
    assert result["provenance"]["tools"][-1]["weights_sha256"] is None
    for field in ("background_mask_ref", "luminance_ref", "exposure_change_ref", "white_balance_proxy_ref", "blur_ref"):
        uri = result["environment"][field]["uri"]
        assert uri in emitted
        assert Path(emitted[uri]).is_file()
