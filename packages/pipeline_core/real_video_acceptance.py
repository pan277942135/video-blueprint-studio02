from __future__ import annotations

from collections import Counter
from typing import Any

CORE_STAGE_NAMES = (
    "media_probe_normalize",
    "shot_detection",
    "person_detection_tracking",
    "pose_2d",
    "person_mask",
    "point_tracks",
    "dense_flow",
    "camera_motion",
    "body_local_frame",
    "surface_motion",
    "micro_motion",
    "environment",
)

QUALITY_KEYS = (
    "media",
    "shots",
    "people",
    "pose_2d",
    "face_2d",
    "hands_2d",
    "person_mask",
    "point_tracks",
    "dense_flow",
    "camera",
    "body_local_frame",
    "surface_motion",
    "micro_motion",
    "environment",
)

_MICRO_THRESHOLD_RULES = (
    ("confidence", "usable_confidence_threshold", ">="),
    ("periodicity_score", "periodicity_threshold", ">="),
    ("spatial_coherence", "coherence_threshold", ">="),
    ("camera_leakage_score", "max_camera_leakage", "<="),
    ("pose_leakage_score", "max_pose_leakage", "<="),
    ("occlusion_ratio", "max_observation_dropout", "<="),
    ("amplitude_norm_p50", "min_amplitude_norm", ">="),
)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _component_quality(character: dict[str, Any], component: str) -> dict[str, Any]:
    value = character.get(component)
    if not isinstance(value, dict):
        return {"enabled": False, "score": None, "coverage": None}
    quality = value.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    return {
        "enabled": value.get("enabled") is True,
        "score": _number(quality.get("score")),
        "coverage": _number(quality.get("coverage")),
    }


def _hand_quality(character: dict[str, Any]) -> dict[str, Any]:
    hands = character.get("hands")
    if not isinstance(hands, dict):
        return {"enabled_sides": 0, "scores": [], "coverages": []}
    scores: list[float] = []
    coverages: list[float] = []
    enabled_sides = 0
    for side in ("left", "right"):
        hand = hands.get(side)
        if not isinstance(hand, dict) or hand.get("enabled") is not True:
            continue
        enabled_sides += 1
        quality = hand.get("quality")
        if not isinstance(quality, dict):
            continue
        score = _number(quality.get("score"))
        coverage = _number(quality.get("coverage"))
        if score is not None:
            scores.append(score)
        if coverage is not None:
            coverages.append(coverage)
    return {"enabled_sides": enabled_sides, "scores": scores, "coverages": coverages}


def _micro_rejection_reasons(
    micro: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    cycles_observed: float | None,
) -> list[str]:
    reasons: list[str] = []
    if micro.get("kind") != "periodic_micro_motion":
        reasons.append(f"kind={micro.get('kind')!r}")
    for metric, threshold_name, direction in _MICRO_THRESHOLD_RULES:
        value = _number(micro.get(metric))
        threshold = _number(thresholds.get(threshold_name))
        if value is None or threshold is None:
            reasons.append(f"{metric}/threshold_missing")
            continue
        if direction == ">=" and value < threshold:
            reasons.append(f"{metric}={value:.6g}<{threshold_name}={threshold:.6g}")
        elif direction == "<=" and value > threshold:
            reasons.append(f"{metric}={value:.6g}>{threshold_name}={threshold:.6g}")
    min_cycles = _number(thresholds.get("min_cycles"))
    if min_cycles is not None:
        if cycles_observed is None:
            reasons.append("cycles_observed_missing")
        elif cycles_observed < min_cycles:
            reasons.append(f"cycles_observed={cycles_observed:.6g}<min_cycles={min_cycles:.6g}")
    return reasons


def summarize_blueprint_acceptance(
    blueprint: dict[str, Any],
    *,
    integrity: dict[str, Any] | None = None,
    low_score_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build an evidence-first E12 diagnostic summary.

    This intentionally does not claim perceptual accuracy. It distinguishes
    machine/physical integrity from the manual frame-to-evidence review that a
    representative real-video acceptance decision still requires.
    """
    processing = blueprint.get("processing")
    if not isinstance(processing, dict):
        processing = {}
    stage_rows = processing.get("stages")
    if not isinstance(stage_rows, list):
        stage_rows = []
    stages = {
        str(row.get("name")): {
            "status": row.get("status"),
            "progress": row.get("progress"),
            "message": row.get("message"),
        }
        for row in stage_rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }

    quality = blueprint.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    raw_module_scores = quality.get("module_scores")
    if not isinstance(raw_module_scores, dict):
        raw_module_scores = {}
    module_scores = {
        key: _number(raw_module_scores.get(key))
        for key in QUALITY_KEYS
        if key in raw_module_scores
    }

    characters = blueprint.get("characters")
    if not isinstance(characters, list):
        characters = []
    character_rows = [row for row in characters if isinstance(row, dict)]

    track_scores: list[float] = []
    pose_scores: list[float] = []
    pose_coverages: list[float] = []
    face_scores: list[float] = []
    face_coverages: list[float] = []
    hand_scores: list[float] = []
    hand_coverages: list[float] = []
    face_enabled_characters = 0
    hand_enabled_sides = 0

    for character in character_rows:
        track_quality = character.get("quality")
        if isinstance(track_quality, dict):
            score = _number(track_quality.get("score"))
            if score is not None:
                track_scores.append(score)
        pose = _component_quality(character, "pose")
        if pose["enabled"]:
            if pose["score"] is not None:
                pose_scores.append(float(pose["score"]))
            if pose["coverage"] is not None:
                pose_coverages.append(float(pose["coverage"]))
        face = _component_quality(character, "face")
        if face["enabled"]:
            face_enabled_characters += 1
            if face["score"] is not None:
                face_scores.append(float(face["score"]))
            if face["coverage"] is not None:
                face_coverages.append(float(face["coverage"]))
        hands = _hand_quality(character)
        hand_enabled_sides += int(hands["enabled_sides"])
        hand_scores.extend(float(value) for value in hands["scores"])
        hand_coverages.extend(float(value) for value in hands["coverages"])

    extensions = blueprint.get("extensions")
    if not isinstance(extensions, dict):
        extensions = {}
    enabled_extensions = {
        key: value.get("enabled") is True
        for key, value in extensions.items()
        if isinstance(value, dict) and "enabled" in value
    }

    micro_extension = extensions.get("e8_micro_motion")
    if not isinstance(micro_extension, dict):
        micro_extension = {}
    thresholds = micro_extension.get("thresholds")
    if not isinstance(thresholds, dict):
        thresholds = {}
    micro_rows = micro_extension.get("characters")
    if not isinstance(micro_rows, list):
        micro_rows = []
    micro_row_by_id = {
        str(row.get("character_id")): row
        for row in micro_rows
        if isinstance(row, dict)
    }

    region_count = 0
    micro_count = 0
    usable_micro_count = 0
    micro_kinds: Counter[str] = Counter()
    micro_rejections: list[dict[str, Any]] = []
    micro_metrics: dict[str, list[float]] = {
        "confidence": [],
        "periodicity_score": [],
        "spatial_coherence": [],
        "camera_leakage_score": [],
        "pose_leakage_score": [],
        "occlusion_ratio": [],
        "amplitude_norm_p50": [],
    }

    for character in character_rows:
        character_id = str(character.get("character_id"))
        surface = character.get("surface_motion")
        regions = surface.get("regions") if isinstance(surface, dict) else None
        if not isinstance(regions, list):
            continue
        row = micro_row_by_id.get(character_id)
        cycles = _number(row.get("cycles_observed")) if isinstance(row, dict) else None
        for region_index, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            region_count += 1
            micro = region.get("micro_motion")
            if not isinstance(micro, dict):
                continue
            micro_count += 1
            kind = str(micro.get("kind"))
            micro_kinds[kind] += 1
            usable = micro.get("usable_for_generation") is True
            if usable:
                usable_micro_count += 1
            for metric, values in micro_metrics.items():
                value = _number(micro.get(metric))
                if value is not None:
                    values.append(value)
            if not usable:
                micro_rejections.append(
                    {
                        "character_id": character_id,
                        "region_index": region_index,
                        "kind": micro.get("kind"),
                        "reasons": _micro_rejection_reasons(
                            micro,
                            thresholds=thresholds,
                            cycles_observed=cycles,
                        ),
                    }
                )

    diagnostics: list[dict[str, Any]] = []

    integrity_valid = True
    integrity_warnings: list[Any] = []
    if integrity is not None:
        integrity_valid = integrity.get("valid") is not False
        warnings = integrity.get("warnings")
        if isinstance(warnings, list):
            integrity_warnings = warnings
        if not integrity_valid:
            diagnostics.append(
                {
                    "severity": "error",
                    "code": "artifact_integrity_failed",
                    "message": "Bundle/artifact integrity failed; do not evaluate extraction quality until physical evidence is repaired.",
                }
            )
        if integrity_warnings:
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "artifact_integrity_warnings",
                    "message": f"Artifact verification reported {len(integrity_warnings)} warning(s).",
                }
            )

    failed_core_stages = [
        name for name in CORE_STAGE_NAMES if stages.get(name, {}).get("status") != "succeeded"
    ]
    if failed_core_stages:
        diagnostics.append(
            {
                "severity": "error",
                "code": "core_stage_incomplete",
                "message": "Required E12 production stages are not all succeeded.",
                "stages": failed_core_stages,
            }
        )

    if not character_rows:
        diagnostics.append(
            {
                "severity": "error",
                "code": "no_character_tracks",
                "message": "No anonymous person tracks were extracted from the real video.",
            }
        )

    face_hands_extension = extensions.get("e3_face_hands_2d")
    face_hands_enabled = isinstance(face_hands_extension, dict) and face_hands_extension.get("enabled") is True
    if not face_hands_enabled:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "face_hands_disabled",
                "message": "E3.2 face/hand refinement is disabled; facial and hand-detail acceptance cannot be claimed for this run.",
            }
        )

    for module, score in module_scores.items():
        if score is None:
            continue
        stage_succeeded = any(
            stages.get(stage_name, {}).get("status") == "succeeded"
            for stage_name in (
                module,
                "person_detection_tracking" if module == "people" else "",
                "pose_2d" if module == "pose_2d" else "",
                "person_mask" if module == "person_mask" else "",
                "camera_motion" if module == "camera" else "",
            )
            if stage_name
        )
        if score == 0.0 and stage_succeeded:
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "zero_quality_despite_succeeded_stage",
                    "module": module,
                    "message": f"{module} stage emitted evidence but its module quality score is 0.0; success must not be interpreted as usable accuracy.",
                }
            )
        elif 0.0 < score < low_score_threshold:
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "low_module_score",
                    "module": module,
                    "score": score,
                    "threshold": low_score_threshold,
                    "message": f"{module} quality score is below the E12 review heuristic; inspect frame/evidence alignment.",
                }
            )

    if region_count == 0:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "no_surface_regions",
                "message": "No E7 surface-motion region was emitted; clothing/hair/surface-detail motion cannot be accepted from this run.",
            }
        )
    if micro_count > 0 and usable_micro_count == 0:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "no_usable_micro_motion",
                "message": "Micro-motion evidence exists, but no region passed the E8 usability gates. Review rejection reasons rather than treating stage success as usable fine motion.",
            }
        )

    privacy = {
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    for extension in extensions.values():
        if not isinstance(extension, dict):
            continue
        if extension.get("identity_inference_performed") is True:
            privacy["identity_inference_performed"] = True
        if extension.get("biometric_embedding_exported") is True:
            privacy["biometric_embedding_exported"] = True
    for character in character_rows:
        row = character.get("privacy")
        if not isinstance(row, dict):
            continue
        if row.get("identity_inference_performed") is True:
            privacy["identity_inference_performed"] = True
        if row.get("biometric_embedding_exported") is True:
            privacy["biometric_embedding_exported"] = True
    if any(privacy.values()):
        diagnostics.append(
            {
                "severity": "error",
                "code": "privacy_boundary_violation",
                "message": "Identity/biometric privacy invariant was violated.",
            }
        )

    error_count = sum(item.get("severity") == "error" for item in diagnostics)
    warning_count = sum(item.get("severity") == "warning" for item in diagnostics)
    if error_count:
        machine_gate = "failed"
    elif warning_count:
        machine_gate = "needs_review"
    else:
        machine_gate = "passed"

    return {
        "acceptance_scope": "E12 representative real-video evidence review",
        "pipeline_version": processing.get("pipeline_version"),
        "machine_gate": machine_gate,
        "final_acceptance": "manual_frame_to_evidence_review_required",
        "important_note": "A succeeded stage or structurally valid Bundle is not evidence that fine visual details were captured accurately.",
        "source": {
            "file_name": blueprint.get("source", {}).get("file_name") if isinstance(blueprint.get("source"), dict) else None,
            "sha256": blueprint.get("source", {}).get("sha256") if isinstance(blueprint.get("source"), dict) else None,
            "frame_count": blueprint.get("timebase", {}).get("frame_count") if isinstance(blueprint.get("timebase"), dict) else None,
            "fps_num": blueprint.get("timebase", {}).get("fps_num") if isinstance(blueprint.get("timebase"), dict) else None,
            "fps_den": blueprint.get("timebase", {}).get("fps_den") if isinstance(blueprint.get("timebase"), dict) else None,
        },
        "stage_statuses": stages,
        "failed_core_stages": failed_core_stages,
        "quality": {
            "overall_score": _number(quality.get("overall_score")),
            "module_scores": module_scores,
            "low_score_review_threshold": low_score_threshold,
            "pipeline_warnings": quality.get("warnings") if isinstance(quality.get("warnings"), list) else [],
        },
        "character_evidence": {
            "character_count": len(character_rows),
            "mean_track_score": _mean(track_scores),
            "mean_pose_score": _mean(pose_scores),
            "mean_pose_coverage": _mean(pose_coverages),
            "face_hands_extension_enabled": face_hands_enabled,
            "face_enabled_characters": face_enabled_characters,
            "mean_face_score": _mean(face_scores),
            "mean_face_coverage": _mean(face_coverages),
            "hand_enabled_sides": hand_enabled_sides,
            "mean_hand_score": _mean(hand_scores),
            "mean_hand_coverage": _mean(hand_coverages),
        },
        "extension_enabled": enabled_extensions,
        "surface_micro_motion": {
            "surface_region_count": region_count,
            "micro_region_count": micro_count,
            "usable_micro_region_count": usable_micro_count,
            "usable_micro_ratio": (usable_micro_count / micro_count) if micro_count else 0.0,
            "kinds": dict(micro_kinds),
            "metric_means": {key: _mean(values) for key, values in micro_metrics.items()},
            "rejections": micro_rejections,
            "thresholds": thresholds,
        },
        "artifact_integrity": {
            "valid": integrity_valid,
            "warnings": integrity_warnings,
            "referenced_artifact_count": integrity.get("referenced_artifact_count") if isinstance(integrity, dict) else None,
            "sidecar_count": integrity.get("sidecar_count") if isinstance(integrity, dict) else None,
        },
        "privacy": privacy,
        "diagnostics": diagnostics,
        "review_targets": [
            "shot boundaries and timing/PTS alignment",
            "anonymous person-track continuity across motion and occlusion",
            "pose alignment at limbs and fast motion",
            "person-mask boundaries around hair, clothing, hands, and occluders",
            "camera compensation versus true subject motion",
            "body-local frame stability",
            "surface residual motion on clothing/hair/body regions",
            "micro-motion usability gates, leakage, dropout, and rejection reasons",
            "face/hand landmark accuracy when E3.2 is enabled",
            "environment photometry under lighting/exposure changes",
        ],
    }


__all__ = ["CORE_STAGE_NAMES", "summarize_blueprint_acceptance"]
