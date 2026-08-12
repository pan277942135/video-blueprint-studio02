import hashlib
import os
import uuid
from typing import Any

from packages.pipeline_core.media_probe import MediaProbeError, compute_sha256, probe_media


def generate_deterministic_hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()

def run_deterministic_mock_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Deterministic Mock Pipeline for Epic E0/E1.
    Produces a canonical, schema-valid Blueprint manifest matching contracts/video_blueprint.schema.json.
    Uses zero real CV models, produces zero biometric data, and embeds no dense motion vectors in the manifest.
    If video_path is provided and exists, uses real ffprobe media probing.
    """
    hash_seed = generate_deterministic_hash(f"{job_id}:{video_file_name}")
    
    probe_source_video = None
    probe_timebase = None
    if video_path and os.path.exists(video_path):
        try:
            real_sha256 = compute_sha256(video_path)
            probe_res = probe_media(video_path)
            probe_source_video = probe_res.to_source_video_dict(video_file_name, real_sha256)
            probe_timebase = probe_res.to_timebase_dict()
            valid_sha256 = real_sha256
        except (MediaProbeError, OSError, ValueError):
            valid_sha256 = video_sha256 if len(video_sha256) == 64 else hash_seed
    else:
        valid_sha256 = video_sha256 if len(video_sha256) == 64 else hash_seed
    
    # Ensure canonical UUID strings for blueprint_id and job_id
    try:
        valid_job_id = str(uuid.UUID(job_id))
    except ValueError:
        valid_job_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"job:{job_id}"))
        
    blueprint_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"blueprint:{valid_job_id}"))
    
    source_video = probe_source_video or {
        "file_name": video_file_name,
        "mime_type": "video/mp4",
        "container": "mov,mp4,m4a,3gp,3g2,mj2",
        "file_size_bytes": 10485760,
        "sha256": valid_sha256,
        "duration_us": 6000000,
        "width": 1920,
        "height": 1080,
        "display_aspect_ratio": "16:9",
        "pixel_aspect_ratio": "1:1",
        "rotation_deg": 0,
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "bit_depth": 8,
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_space": "bt709",
        "fps_avg": 30.0,
        "fps_nominal": 30.0,
        "variable_frame_rate": False,
        "source_frame_count": 180,
        "has_audio": False,
        "audio_codec": None,
        "audio_sample_rate_hz": None,
        "audio_channels": None,
        "metadata_stripped": True
    }

    timebase = probe_timebase or {
        "normalized_to_cfr": True,
        "fps_num": 30,
        "fps_den": 1,
        "frame_count": 180,
        "frame_duration_us": 33333.333,
        "start_pts_us": 0,
        "end_pts_us": 5966667,
        "source_pts_map_ref": None
    }

    # 1. Blueprint Manifest matching canonical schema
    blueprint: dict[str, Any] = {
        "schema_version": "1.0.0",
        "blueprint_id": blueprint_uuid,
        "created_at": "2026-08-12T00:00:00Z",
        "source_video": source_video,
        "timebase": timebase,
        "processing": {
            "job_id": valid_job_id,
            "status": "succeeded",
            "requested_modules": [
                "shots", "people", "pose", "face", "hands",
                "masks", "camera", "flow", "micro_motion", "environment"
            ],
            "started_at": "2026-08-12T00:00:00Z",
            "completed_at": "2026-08-12T00:01:00Z",
            "pipeline_version": "1.0.0",
            "config_hash": "cfg-mock-e0",
            "execution_mode": "offline_local",
            "hardware": {
                "gpu": "Mock GPU",
                "vram_gb": 16
            },
            "stages": []
        },
        "shots": [
            {
                "shot_id": "shot_000",
                "index": 0,
                "frame_start": 0,
                "frame_end": 179,
                "time_start_us": 0,
                "time_end_us": 5966667,
                "cut_in_type": "start",
                "cut_out_type": "end",
                "transition_score": 1.0,
                "keyframes": [
                    {
                        "frame_idx": 0,
                        "time_us": 0,
                        "kind": "first",
                        "score": 1.0,
                        "image_uri": "artifacts/keyframes/shot_000_000000.png"
                    }
                ],
                "dominant_character_ids": ["char_000"],
                "camera_motion_id": "cam_000",
                "quality": {
                    "score": 0.95,
                    "coverage": 1.0,
                    "warnings": [],
                    "errors": []
                }
            }
        ],
        "characters": [],
        "camera": {
            "per_shot": [
                {
                    "camera_motion_id": "cam_000",
                    "shot_id": "shot_000",
                    "classification": "static",
                    "affine_ref": None,
                    "homography_ref": None,
                    "crop_ref": None,
                    "zoom_proxy_ref": None,
                    "shake_ref": None,
                    "background_tracks_ref": None,
                    "intrinsics": None,
                    "extrinsics_ref": None,
                    "reconstruction_backend": "opencv_ransac_2d",
                    "confidence": 0.9,
                    "failure_reason": None
                }
            ],
            "quality": {
                "score": 0.9,
                "coverage": 1.0,
                "warnings": [],
                "errors": []
            }
        },
        "environment": {
            "background_mask_ref": None,
            "depth_ref": None,
            "luminance_ref": None,
            "exposure_change_ref": None,
            "white_balance_proxy_ref": None,
            "blur_ref": None,
            "occluder_tracks": [],
            "quality": {
                "score": 0.8,
                "coverage": 1.0,
                "warnings": [],
                "errors": []
            }
        },
        "semantics": None,
        "quality": {
            "overall_score": 0.95,
            "module_scores": {
                "shots": 0.95,
                "camera": 0.9
            },
            "warnings": [],
            "errors": [],
            "low_confidence_intervals": [],
            "reproducible": True
        },
        "artifacts": {
            "manifest_uri": "blueprint.json",
            "bundle_uri": "bundle.zip",
            "overlays": [],
            "reports": []
        },
        "provenance": {
            "tools": [],
            "system": {
                "os": "linux",
                "python": "3.11"
            }
        },
        "extensions": {}
    }

    # 2. Sidecars
    sidecars = {
        "sidecars/pts_map.json": {
            "frame_count": 180,
            "pts_timestamps": [i * (1.0 / 30.0) for i in range(180)]
        },
        "sidecars/camera_motion_affine.json": {
            "motion_type": "2d_affine",
            "transforms": [
                {"frame": i, "matrix": [[1.0, 0.0, 0.1 * i], [0.0, 1.0, 0.05 * i]]}
                for i in range(0, 180, 10)
            ]
        },
        "sidecars/environment_background_mask.json": {
            "mask_type": "rle",
            "data": "100x200:10000"
        }
    }

    return blueprint, sidecars
