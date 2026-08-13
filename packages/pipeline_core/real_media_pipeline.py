import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packages.pipeline_core.face_hand_refinement import FaceHandRefinementError, run_face_hand_refinement
from packages.pipeline_core.media_probe import InvalidMediaError
from packages.pipeline_core.mediapipe_face_hand_backend import MediaPipeFaceHandRefiner
from packages.pipeline_core.mock_pipeline import run_deterministic_mock_pipeline
from packages.pipeline_core.person_tracking import PersonTrackingConfig, run_person_tracking
from packages.pipeline_core.pose_estimation import PoseEstimationError, run_pose_estimation
from packages.pipeline_core.rtmdet_backend import RTMDetPersonDetector
from packages.pipeline_core.rtmpose_backend import RTMPosePoseEstimator
from packages.pipeline_core.shot_detection import ShotDetectionConfig, ShotDetectionError, detect_shots


def _shot_config_hash(config: ShotDetectionConfig) -> str:
    payload = f"shots:content-detector:{config.threshold}:{config.min_scene_len_frames}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combined_config_hash(
    shot_config: ShotDetectionConfig,
    tracking_config: PersonTrackingConfig | None = None,
    detector: RTMDetPersonDetector | None = None,
    pose_estimator: RTMPosePoseEstimator | None = None,
    face_hand_refiner: MediaPipeFaceHandRefiner | None = None,
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
    if pose_estimator is not None:
        parts.extend(
            [
                f"rtmpose-config:{pose_estimator.config_sha256}",
                f"rtmpose-weights:{pose_estimator.weights_sha256}",
            ]
        )
    if face_hand_refiner is not None:
        parts.extend(
            [
                f"mediapipe-config:{face_hand_refiner.config_sha256}",
                f"mediapipe-face-task:{face_hand_refiner.face_task_sha256}",
                f"mediapipe-hand-task:{face_hand_refiner.hand_task_sha256}",
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


def _minimum_enabled_quality(characters: list[dict[str, Any]], component: str) -> float:
    scores: list[float] = []
    for character in characters:
        value = character.get(component)
        if isinstance(value, dict) and value.get("enabled") is True:
            quality = value.get("quality", {})
            if isinstance(quality, dict):
                scores.append(float(quality.get("score", 0.0)))
    return min(scores, default=0.0)


def _minimum_enabled_hand_quality(characters: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for character in characters:
        hands = character.get("hands", {})
        if not isinstance(hands, dict):
            continue
        for side in ("left", "right"):
            hand = hands.get(side, {})
            if isinstance(hand, dict) and hand.get("enabled") is True:
                quality = hand.get("quality", {})
                if isinstance(quality, dict):
                    scores.append(float(quality.get("score", 0.0)))
    return min(scores, default=0.0)


def run_real_media_pipeline(
    job_id: str,
    video_file_name: str,
    video_sha256: str,
    video_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the real-media pipeline through E3.2 anonymous face/hand geometry.

    E1 provides real source probing, normalization, and PTS mapping. E2.1 runs
    real PySceneDetect shot detection unconditionally. E2.2 runs RTMDet person
    detection and anonymous IoU track association only when a complete,
    explicitly approved local RTMDet configuration is present. E3.1 runs
    top-down RTMPose COCO-17 estimation only when its approved local runtime is
    present. E3.2 runs pinned MediaPipe FaceLandmarker/HandLandmarker VIDEO
    tasks only when both approved local task artifacts are present. Partial or
    unapproved model configuration is a hard failure; no model fallback or
    runtime download is allowed.
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
        raise ShotDetectionError("Normalized CFR analysis video is required before E2/E3 analysis")

    timebase = blueprint["timebase"]
    frame_count = int(timebase["frame_count"])
    fps_num = int(timebase["fps_num"])
    fps_den = int(timebase["fps_den"])
    shot_config = ShotDetectionConfig()
    artifact_root = os.path.join(os.path.dirname(video_path), f"vbs_artifacts_{job_id}")
    shots, shot_sidecars = detect_shots(
        str(normalized_path),
        frame_count=frame_count,
        fps_num=fps_num,
        fps_den=fps_den,
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
            frame_count=frame_count,
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

    pose_estimator = RTMPosePoseEstimator.from_environment()
    if pose_estimator is not None and not people_enabled:
        raise PoseEstimationError(
            "E3.1 RTMPose requires E2.2 person tracking; configure the approved RTMDet runtime first"
        )

    pose_enabled = pose_estimator is not None
    pose_score = 0.0
    if pose_estimator is not None:
        characters, pose_sidecars, pose_report_ref = run_pose_estimation(
            str(normalized_path),
            characters=blueprint["characters"],
            frame_count=frame_count,
            estimator=pose_estimator,
            output_dir=artifact_root,
            sidecars=sidecars,
        )
        sidecars.update(pose_sidecars)
        blueprint["characters"] = characters
        blueprint["artifacts"]["reports"].append(pose_report_ref)
        pose_score = min(
            (float(character["pose"]["quality"]["score"]) for character in characters),
            default=0.0,
        )

    face_hand_refiner = MediaPipeFaceHandRefiner.from_environment(fps_num=fps_num, fps_den=fps_den)
    if face_hand_refiner is not None and not people_enabled:
        raise FaceHandRefinementError(
            "E3.2 MediaPipe face/hands requires E2.2 anonymous person tracking; configure approved RTMDet first"
        )

    face_hands_enabled = face_hand_refiner is not None
    face_score = 0.0
    hands_score = 0.0
    if face_hand_refiner is not None:
        try:
            characters, refinement_sidecars, refinement_report_ref = run_face_hand_refinement(
                str(normalized_path),
                characters=blueprint["characters"],
                frame_count=frame_count,
                refiner=face_hand_refiner,
                output_dir=artifact_root,
                sidecars=sidecars,
            )
        finally:
            face_hand_refiner.close()
        sidecars.update(refinement_sidecars)
        blueprint["characters"] = characters
        blueprint["artifacts"]["reports"].append(refinement_report_ref)
        face_score = _minimum_enabled_quality(characters, "face")
        hands_score = _minimum_enabled_hand_quality(characters)
        blueprint["schema_version"] = "1.1.0"

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
            "warnings": ["environment analysis not run before its dedicated epic"],
            "errors": [],
        },
    }

    shot_score = min((float(shot["quality"]["score"]) for shot in shots), default=0.0)
    overall_components = [shot_score]
    if people_enabled:
        overall_components.append(people_score)
    if pose_enabled and blueprint["characters"]:
        overall_components.append(pose_score)
    if face_hands_enabled and blueprint["characters"]:
        if face_score > 0.0:
            overall_components.append(face_score)
        if hands_score > 0.0:
            overall_components.append(hands_score)
    blueprint["quality"]["overall_score"] = min(overall_components)
    blueprint["quality"]["module_scores"] = {
        "media": 1.0,
        "shots": shot_score,
        "people": people_score,
        "pose_2d": pose_score,
        "face_2d": face_score,
        "hands_2d": hands_score,
        "camera": 0.0,
        "environment": 0.0,
    }
    blueprint["quality"]["warnings"] = []
    if not people_enabled:
        blueprint["quality"]["warnings"].append(
            "E2.2 RTMDet person tracking is disabled until approved local model configuration is supplied"
        )
    if not pose_enabled:
        blueprint["quality"]["warnings"].append(
            "E3.1 RTMPose 2D pose is disabled until approved local model configuration is supplied"
        )
    if not face_hands_enabled:
        blueprint["quality"]["warnings"].append(
            "E3.2 MediaPipe face/hands is disabled until both approved pinned task artifacts are supplied locally"
        )

    combined_config_hash = _combined_config_hash(
        shot_config,
        tracking_config,
        detector,
        pose_estimator,
        face_hand_refiner,
    )
    blueprint["processing"]["pipeline_version"] = "0.3.0-e3.2" if face_hands_enabled else "0.3.0-e3.1"
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
        {
            "name": "pose_2d",
            "status": "succeeded" if pose_enabled else "skipped",
            "progress": 1.0 if pose_enabled else 0.0,
            "message": (
                f"RTMPose COCO-17 processed {len(blueprint['characters'])} anonymous track(s)"
                if pose_enabled
                else "RTMPose is not configured; E3.1 pose estimation intentionally skipped"
            ),
        },
        {
            "name": "face_hands_2d",
            "status": "succeeded" if face_hands_enabled else "skipped",
            "progress": 1.0 if face_hands_enabled else 0.0,
            "message": (
                f"Pinned MediaPipe VIDEO tasks refined {len(blueprint['characters'])} anonymous track(s)"
                if face_hands_enabled
                else "MediaPipe Face/Hand task artifacts are not configured; E3.2 intentionally skipped"
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
    blueprint["extensions"]["e3_pose_2d"] = {
        "enabled": pose_enabled,
        "estimator": "mmpose_rtmpose_coco17" if pose_enabled else None,
        "skeleton": "coco17" if pose_enabled else None,
        "keypoint_count": 17 if pose_enabled else 0,
        "character_count": len(blueprint["characters"]) if pose_enabled else 0,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    blueprint["extensions"]["e3_face_hands_2d"] = {
        "enabled": face_hands_enabled,
        "estimator": "mediapipe_face_hand_landmarker_video" if face_hands_enabled else None,
        "face_landmark_count": 478 if face_hands_enabled else 0,
        "hand_landmark_count": 21 if face_hands_enabled else 0,
        "character_count": len(blueprint["characters"]) if face_hands_enabled else 0,
        "face_task_sha256": face_hand_refiner.face_task_sha256 if face_hand_refiner is not None else None,
        "hand_task_sha256": face_hand_refiner.hand_task_sha256 if face_hand_refiner is not None else None,
        "blendshapes_exported": False,
        "facial_transformation_matrices_exported": False,
        "hand_world_landmarks_exported": False,
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
    if pose_estimator is not None:
        provenance_tools.append(
            pose_estimator.provenance(
                code_commit=os.environ.get("GITHUB_SHA"),
                config_hash=combined_config_hash,
            )
        )
    if face_hand_refiner is not None:
        provenance_tools.append(
            face_hand_refiner.provenance(
                code_commit=os.environ.get("GITHUB_SHA"),
                config_hash=combined_config_hash,
            )
        )
    blueprint["provenance"]["tools"] = provenance_tools

    return blueprint, sidecars
