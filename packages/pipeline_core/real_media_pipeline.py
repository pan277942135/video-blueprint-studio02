import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packages.pipeline_core.media_probe import InvalidMediaError
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline
from packages.pipeline_core.person_tracking import PersonTrackingConfig, run_person_tracking
from packages.pipeline_core.rtmdet_backend import RTMDetPersonDetector
from packages.pipeline_core.shot_detection import ShotDetectionConfig, ShotDetectionError, detect_shots


def _shot_config_hash(config: ShotDetectionConfig) -> str:
    payload = f"shots:content-detector:{config.threshold}:{config.min_scene_len_frames}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combined_config_hash(
    shot_config: ShotDetectionConfig,
    tracking_config: PersonTrackingConfig | None = None,
    detector: RTMDetPersonDetector | None = None,
) -> str:
    parts = [
        f"shots:{shot_config.threshold}:{shot_config.min_scene_len_frames}",
    ]
    if tracking_config is not None:
        parts.append(f"tracking:{tracking_config.iou_threshold}:{tracking_config.max_gap_frames}")
    if detector is not None:
        parts.extend(
            [
                f"rtmdet-config:{detector.config_sha256}",
                f"rtmdet-weights:{detector.weights_sha256}",
                f"rtmdet-threshold:{detector.config.score_threshold}",
            ]
        )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


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
    """Run the real-media pipeline through E2 shot/people analysis.

    E1 provides real source probing, normalization, and PTS mapping. E2.1 runs
    real PySceneDetect shot detection unconditionally. E2.2 runs RTMDet person
    detection and anonymous IoU track association only when a complete,
    explicitly approved local RTMDet configuration is present. Partial or
    unapproved model configuration is a hard failure; no detector fallback is
    allowed.
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
        raise ShotDetectionError("Normalized CFR analysis video is required before E2 analysis")

    timebase = blueprint["timebase"]
    shot_config = ShotDetectionConfig()
    artifact_root = os.path.join(os.path.dirname(video_path), f"vbs_artifacts_{job_id}")
    shots, shot_sidecars = detect_shots(
        str(normalized_path),
        frame_count=int(timebase["frame_count"]),
        fps_num=int(timebase["fps_num"]),
        fps_den=int(timebase["fps_den"]),
        output_dir=artifact_root,
        config=shot_config,
    )
    sidecars.update(shot_sidecars)
    blueprint["shots"] = shots

    shot_report_uri = "artifacts/reports/shot_detection.json"
    shot_report = shot_sidecars.get(shot_report_uri)
    if not isinstance(shot_report, dict):
        raise ShotDetectionError("Shot detector did not emit its required provenance report")
    blueprint["artifacts"]["reports"] = [
        {
            "kind": "shot_detection",
            "uri": shot_report_uri,
            "sha256": _json_sha256(shot_report),
            "mime_type": "application/json",
        }
    ]

    detector = RTMDetPersonDetector.from_environment()
    tracking_config: PersonTrackingConfig | None = None
    people_enabled = detector is not None
    people_score = 0.0
    if detector is not None:
        tracking_config = PersonTrackingConfig()
        characters, tracking_sidecars, overlays, tracking_report_ref = run_person_tracking(
            str(normalized_path),
            shots=shots,
            frame_count=int(timebase["frame_count"]),
            detector=detector,
            output_dir=artifact_root,
            config=tracking_config,
        )
        sidecars.update(tracking_sidecars)
        blueprint["characters"] = characters
        blueprint["artifacts"]["overlays"] = overlays
        blueprint["artifacts"]["reports"].append(tracking_report_ref)
        people_score = min(
            (float(character["quality"]["score"]) for character in characters),
            default=1.0,
        )
    else:
        blueprint["characters"] = []
        blueprint["artifacts"]["overlays"] = []

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
    blueprint["quality"]["overall_score"] = min(shot_score, people_score) if people_enabled else shot_score
    blueprint["quality"]["module_scores"] = {
        "media": 1.0,
        "shots": shot_score,
        "people": people_score,
        "camera": 0.0,
        "environment": 0.0,
    }
    blueprint["quality"]["warnings"] = []
    if not people_enabled:
        blueprint["quality"]["warnings"].append(
            "E2.2 RTMDet person tracking is disabled until approved local model configuration is supplied"
        )

    combined_config_hash = _combined_config_hash(shot_config, tracking_config, detector)
    blueprint["processing"]["pipeline_version"] = "0.2.1-e2.2"
    blueprint["processing"]["config_hash"] = combined_config_hash
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
            "status": "succeeded" if people_enabled else "skipped",
            "progress": 1.0 if people_enabled else 0.0,
            "message": (
                f"RTMDet + anonymous IoU association produced {len(blueprint['characters'])} track(s)"
                if people_enabled
                else "RTMDet is not configured; person tracking intentionally skipped"
            ),
        },
    ]
    blueprint["extensions"]["e2_shot_detection"] = {
        "backend": "pyscenedetect_content_detector",
        "threshold": shot_config.threshold,
        "min_scene_len_frames": shot_config.min_scene_len_frames,
        "shot_count": len(shots),
        "report_uri": shot_report_uri,
    }
    blueprint["extensions"]["e2_person_tracking"] = {
        "enabled": people_enabled,
        "detector": "mmdetection_rtmdet" if people_enabled else None,
        "association": "anonymous_iou" if people_enabled else None,
        "character_count": len(blueprint["characters"]),
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }

    shot_config_hash = _shot_config_hash(shot_config)
    provenance_tools = [
        {
            "module": "shot_detection",
            "tool": "PySceneDetect ContentDetector",
            "version": _package_version("scenedetect"),
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": shot_config_hash,
            "license": "BSD-3-Clause",
        },
        {
            "module": "shot_keyframes",
            "tool": "OpenCV",
            "version": _package_version("opencv-python-headless"),
            "code_commit": os.environ.get("GITHUB_SHA"),
            "weights_sha256": None,
            "config_hash": shot_config_hash,
            "license": "Apache-2.0",
        },
    ]
    if detector is not None and tracking_config is not None:
        provenance_tools.extend(
            [
                detector.provenance(
                    code_commit=os.environ.get("GITHUB_SHA"),
                    config_hash=combined_config_hash,
                ),
                {
                    "module": "person_tracking",
                    "tool": "Video Blueprint anonymous IoU tracker",
                    "version": "0.2.1",
                    "code_commit": os.environ.get("GITHUB_SHA"),
                    "weights_sha256": None,
                    "config_hash": combined_config_hash,
                    "license": "project-internal",
                },
            ]
        )
    blueprint["provenance"]["tools"] = provenance_tools

    return blueprint, sidecars
