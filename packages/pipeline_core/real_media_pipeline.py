import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packages.pipeline_core.media_probe import InvalidMediaError
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline
from packages.pipeline_core.shot_detection import ShotDetectionConfig, ShotDetectionError, detect_shots


def _config_hash(config: ShotDetectionConfig) -> str:
    payload = f"shots:content-detector:{config.threshold}:{config.min_scene_len_frames}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, indent=2).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_real_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real-media pipeline through E2.1 shot detection.

    E1 provides real source probing, normalization, and PTS mapping. E2.1
    replaces the previous single-shot scaffold with real PySceneDetect scene
    boundaries and real keyframe PNG artifacts extracted from the normalized
    CFR analysis video. Missing media, incomplete normalized artifacts, or shot
    detection failures are hard failures and never fall back to demo output.
    """
    if not video_path:
        raise InvalidMediaError("Real-media analysis requires a persisted uploaded video path")
    if not os.path.isfile(video_path):
        raise InvalidMediaError(f"Uploaded video path does not exist: {video_path}")

    blueprint, sidecars = run_deterministic_mock_pipeline(
        job_id=job_id,
        video_file_name=video_file_name,
        video_sha256=video_sha256,
        video_path=video_path,
    )

    normalized_path = sidecars.get("artifacts/normalized/analysis_cfr.mp4")
    if not isinstance(normalized_path, (str, os.PathLike)) or not os.path.isfile(str(normalized_path)):
        raise ShotDetectionError("Normalized CFR analysis video is required before E2 shot detection")

    timebase = blueprint["timebase"]
    config = ShotDetectionConfig()
    config_hash = _config_hash(config)
    artifact_root = os.path.join(os.path.dirname(video_path), f"vbs_artifacts_{job_id}")
    shots, shot_sidecars = detect_shots(
        str(normalized_path),
        frame_count=int(timebase["frame_count"]),
        fps_num=int(timebase["fps_num"]),
        fps_den=int(timebase["fps_den"]),
        output_dir=artifact_root,
        config=config,
    )
    sidecars.update(shot_sidecars)
    blueprint["shots"] = shots

    report_uri = "artifacts/reports/shot_detection.json"
    shot_report = shot_sidecars.get(report_uri)
    if not isinstance(shot_report, dict):
        raise ShotDetectionError("Shot detector did not emit its required provenance report")
    blueprint["artifacts"]["reports"] = [
        {
            "kind": "shot_detection",
            "uri": report_uri,
            "sha256": _json_sha256(shot_report),
            "mime_type": "application/json",
        }
    ]

    # Camera/environment analysis belongs to later Epics. Remove E0 placeholder
    # confidence claims from the real-media path rather than presenting mock
    # values as measured results.
    blueprint["camera"] = {
        "per_shot": [],
        "quality": {
            "score": 0.0,
            "coverage": 0.0,
            "warnings": ["camera estimation not run before E5"],
            "errors": [],
        },
    }
    blueprint["environment"] = {
        "background_mask_ref": None,
        "depth_ref": None,
        "luminance_ref": None,
        "exposure_change_ref": None,
        "white_balance_proxy_ref": None,
        "blur_ref": None,
        "occluder_tracks": [],
        "quality": {
            "score": 0.0,
            "coverage": 0.0,
            "warnings": ["environment analysis not run in E2"],
            "errors": [],
        },
    }

    shot_score = min((float(shot["quality"]["score"]) for shot in shots), default=0.0)
    blueprint["quality"]["overall_score"] = shot_score
    blueprint["quality"]["module_scores"] = {
        "media": 1.0,
        "shots": shot_score,
        "people": 0.0,
        "camera": 0.0,
        "environment": 0.0,
    }
    blueprint["quality"]["warnings"] = [
        "E2.1 includes real shot detection; person detection/tracking is not yet enabled",
    ]

    blueprint["processing"]["pipeline_version"] = "0.2.0-e2.1"
    blueprint["processing"]["config_hash"] = config_hash
    blueprint["processing"]["stages"] = [
        {
            "name": "media_probe_normalize",
            "status": "succeeded",
            "progress": 1.0,
            "message": "Real media probe, CFR normalization, and source PTS mapping completed",
        },
        {
            "name": "shot_detection",
            "status": "succeeded",
            "progress": 1.0,
            "message": f"PySceneDetect ContentDetector produced {len(shots)} shot(s)",
        },
        {
            "name": "person_detection_tracking",
            "status": "skipped",
            "progress": 0.0,
            "message": "Scheduled for E2.2 RTMDet + anonymous track association",
        },
    ]
    blueprint["extensions"]["e2_shot_detection"] = {
        "backend": "pyscenedetect_content_detector",
        "threshold": config.threshold,
        "min_scene_len_frames": config.min_scene_len_frames,
        "shot_count": len(shots),
        "report_uri": report_uri,
    }
    blueprint["provenance"]["tools"] = [
        {
            "module": "shot_detection",
            "tool": "PySceneDetect ContentDetector",
            "version": _package_version("scenedetect"),
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": config_hash,
            "license": "BSD-3-Clause",
        },
        {
            "module": "shot_keyframes",
            "tool": "OpenCV",
            "version": _package_version("opencv-python-headless"),
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": config_hash,
            "license": "Apache-2.0",
        },
    ]

    return blueprint, sidecars
