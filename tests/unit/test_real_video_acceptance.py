from packages.pipeline_core.real_video_acceptance import CORE_STAGE_NAMES, summarize_blueprint_acceptance


def _base_blueprint() -> dict:
    stages = [
        {"name": name, "status": "succeeded", "progress": 1.0, "message": "ok"}
        for name in CORE_STAGE_NAMES
    ]
    return {
        "source_video": {"file_name": "real.mp4", "sha256": "a" * 64},
        "timebase": {"frame_count": 120, "fps_num": 30, "fps_den": 1},
        "processing": {"pipeline_version": "0.9.0-e9.1", "stages": stages},
        "quality": {
            "overall_score": 0.8,
            "module_scores": {
                "media": 1.0,
                "shots": 1.0,
                "people": 0.8,
                "pose_2d": 0.8,
                "face_2d": 0.8,
                "hands_2d": 0.8,
                "person_mask": 0.8,
                "point_tracks": 0.8,
                "dense_flow": 0.8,
                "camera": 0.8,
                "body_local_frame": 0.8,
                "surface_motion": 0.8,
                "micro_motion": 0.8,
                "environment": 0.8,
            },
            "warnings": [],
        },
        "extensions": {
            "e3_face_hands_2d": {"enabled": True},
            "e8_micro_motion": {
                "enabled": True,
                "thresholds": {
                    "usable_confidence_threshold": 0.5,
                    "periodicity_threshold": 0.5,
                    "coherence_threshold": 0.5,
                    "max_camera_leakage": 0.3,
                    "max_pose_leakage": 0.3,
                    "max_observation_dropout": 0.2,
                    "min_amplitude_norm": 0.01,
                    "min_cycles": 2.0,
                },
                "characters": [{"character_id": "c1", "cycles_observed": 4.0}],
            },
        },
        "characters": [
            {
                "character_id": "c1",
                "quality": {"score": 0.8},
                "pose": {"enabled": True, "quality": {"score": 0.8, "coverage": 0.9}},
                "face": {"enabled": True, "quality": {"score": 0.8, "coverage": 0.7}},
                "hands": {
                    "left": {"enabled": True, "quality": {"score": 0.7, "coverage": 0.6}},
                    "right": {"enabled": True, "quality": {"score": 0.75, "coverage": 0.65}},
                },
                "privacy": {
                    "identity_inference_performed": False,
                    "biometric_embedding_exported": False,
                },
                "surface_motion": {
                    "regions": [
                        {
                            "micro_motion": {
                                "kind": "periodic_micro_motion",
                                "usable_for_generation": True,
                                "confidence": 0.8,
                                "periodicity_score": 0.8,
                                "spatial_coherence": 0.8,
                                "camera_leakage_score": 0.1,
                                "pose_leakage_score": 0.1,
                                "occlusion_ratio": 0.1,
                                "amplitude_norm_p50": 0.03,
                            }
                        }
                    ]
                },
            }
        ],
    }


def test_clean_machine_evidence_still_requires_manual_review() -> None:
    report = summarize_blueprint_acceptance(
        _base_blueprint(),
        integrity={"valid": True, "warnings": [], "referenced_artifact_count": 20, "sidecar_count": 21},
    )

    assert report["machine_gate"] == "passed"
    assert report["final_acceptance"] == "manual_frame_to_evidence_review_required"
    assert report["source"]["file_name"] == "real.mp4"
    assert report["source"]["sha256"] == "a" * 64
    assert report["artifact_integrity"]["referenced_artifact_count"] == 20
    assert report["artifact_integrity"]["sidecar_count"] == 21
    assert report["surface_micro_motion"]["usable_micro_ratio"] == 1.0
    assert report["diagnostics"] == []


def test_legacy_source_fallback_remains_readable() -> None:
    blueprint = _base_blueprint()
    blueprint.pop("source_video")
    blueprint["source"] = {"file_name": "legacy.mp4", "sha256": "b" * 64}

    report = summarize_blueprint_acceptance(blueprint, integrity={"valid": True, "warnings": []})

    assert report["source"]["file_name"] == "legacy.mp4"
    assert report["source"]["sha256"] == "b" * 64


def test_succeeded_stage_with_zero_quality_is_not_treated_as_accurate() -> None:
    blueprint = _base_blueprint()
    blueprint["quality"]["module_scores"]["surface_motion"] = 0.0
    blueprint["quality"]["module_scores"]["micro_motion"] = 0.0
    micro = blueprint["characters"][0]["surface_motion"]["regions"][0]["micro_motion"]
    micro.update(
        {
            "kind": "unclassified",
            "usable_for_generation": False,
            "confidence": 0.2,
            "periodicity_score": 0.1,
            "spatial_coherence": 0.2,
            "camera_leakage_score": 0.7,
            "pose_leakage_score": 0.6,
            "occlusion_ratio": 0.5,
            "amplitude_norm_p50": 0.001,
        }
    )

    report = summarize_blueprint_acceptance(blueprint, integrity={"valid": True, "warnings": []})
    codes = [row["code"] for row in report["diagnostics"]]

    assert report["machine_gate"] == "needs_review"
    assert codes.count("zero_quality_despite_succeeded_stage") == 2
    assert "no_usable_micro_motion" in codes
    assert report["surface_micro_motion"]["usable_micro_region_count"] == 0
    assert report["surface_micro_motion"]["rejections"][0]["reasons"]


def test_missing_core_stage_or_integrity_failure_is_machine_failure() -> None:
    blueprint = _base_blueprint()
    blueprint["processing"]["stages"] = [
        row for row in blueprint["processing"]["stages"] if row["name"] != "dense_flow"
    ]

    report = summarize_blueprint_acceptance(
        blueprint,
        integrity={"valid": False, "warnings": ["bad hash"]},
    )
    codes = {row["code"] for row in report["diagnostics"]}

    assert report["machine_gate"] == "failed"
    assert "artifact_integrity_failed" in codes
    assert "core_stage_incomplete" in codes
    assert "dense_flow" in report["failed_core_stages"]


def test_disabled_face_hands_is_explicitly_flagged() -> None:
    blueprint = _base_blueprint()
    blueprint["extensions"]["e3_face_hands_2d"]["enabled"] = False

    report = summarize_blueprint_acceptance(blueprint, integrity={"valid": True, "warnings": []})
    codes = {row["code"] for row in report["diagnostics"]}

    assert report["machine_gate"] == "needs_review"
    assert "face_hands_disabled" in codes
    assert report["character_evidence"]["face_hands_extension_enabled"] is False
