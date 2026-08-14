from __future__ import annotations

from typing import Any

ALGORITHM = "body_local_sparse_periodicity_v2"
SELECTION_METHOD = "per_track_frequency_locked_consensus_v2"
_ALLOWED_KINDS = {"periodic_micro_motion", "unclassified", "not_detected"}


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _check_ref(
    ref: Any,
    *,
    path: str,
    frame_count: int,
    array_key: str,
    valid_key: str,
    unit: str,
    config_sha: Any,
    source_surface_uri: str,
    body_frame_uri: str,
) -> list[str]:
    if not isinstance(ref, dict):
        return [f"E8 v2 MicroMotion Contract Violation: {path} must be a TimeSeriesRef."]
    errors: list[str] = []
    expected = {
        "format": "npz",
        "dtype": "float32",
        "shape": [frame_count],
        "axes": ["frame"],
        "unit": unit,
        "coordinate_space": "body_local_2d",
        "sampling": "per_frame",
        "frame_start": 0,
        "frame_end": frame_count - 1,
        "nan_policy": "preserve",
        "interpolation_policy": "none",
    }
    for key, value in expected.items():
        if ref.get(key) != value:
            errors.append(f"E8 v2 MicroMotion Contract Violation: {path}.{key} must equal {value!r}.")
    if not _sha256(ref.get("checksum_sha256")):
        errors.append(f"E8 v2 MicroMotion Contract Violation: {path}.checksum_sha256 must be SHA256 hex.")
    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        return errors + [f"E8 v2 MicroMotion Contract Violation: {path}.metadata is required."]
    expected_metadata = {
        "array_key": array_key,
        "valid_array_key": valid_key,
        "algorithm": ALGORITHM,
        "selection_method": SELECTION_METHOD,
        "selected_track_id_array_key": "selected_track_id",
        "source_surface_uri": source_surface_uri,
        "body_frame_transform_uri": body_frame_uri,
        "aggregation": "winsorized_mean_selected_residual_projected_to_principal_axis",
        "shot_boundary_reset": True,
        "linear_detrend_only": True,
        "interpolation": False,
        "physiological_interpretation": False,
        "config_sha256": config_sha,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            errors.append(f"E8 v2 MicroMotion Contract Violation: {path}.metadata.{key} must equal {value!r}.")
    return errors


def validate_micro_motion_v2_contract(blueprint: dict[str, Any]) -> list[str]:
    extensions = blueprint.get("extensions")
    if not isinstance(extensions, dict):
        return []
    extension = extensions.get("e8_micro_motion")
    if not isinstance(extension, dict) or extension.get("algorithm") != ALGORITHM:
        return []
    path = "extensions.e8_micro_motion"
    errors: list[str] = []
    expected_extension = {
        "enabled": True,
        "algorithm": ALGORITHM,
        "selection_method": SELECTION_METHOD,
        "interpretation_scope": "geometry_only",
        "source_surface_motion": "e7_surface_motion",
        "body_frame_source": "e6_body_local_frame",
        "camera_source": "e5_camera_motion",
        "class_scope": ["periodic_micro_motion", "unclassified", "not_detected"],
        "radial_expansion_emitted": False,
        "area_change_emitted": False,
        "physiological_inference_performed": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
        "new_model_weights_introduced": False,
        "occlusion_metric_semantics": "observation_dropout_proxy_not_semantic_occlusion",
        "pose_leakage_semantics": "body_frame_energy_correlation_weighted_by_frequency_consensus_support",
        "track_periodicity_floor": 0.25,
        "minimum_consensus_tracks": 4,
    }
    for key, value in expected_extension.items():
        if extension.get(key) != value:
            errors.append(f"E8 v2 MicroMotion Contract Violation: {path}.{key} must equal {value!r}.")
    config_sha = extension.get("config_sha256")
    if not _sha256(config_sha):
        errors.append(f"E8 v2 MicroMotion Contract Violation: {path}.config_sha256 must be SHA256 hex.")
    thresholds = extension.get("thresholds")
    if not isinstance(thresholds, dict):
        return errors + [f"E8 v2 MicroMotion Contract Violation: {path}.thresholds is required."]
    frame_count = blueprint.get("timebase", {}).get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        return errors + ["E8 v2 MicroMotion Contract Violation: timebase.frame_count must be positive integer."]
    if not isinstance(extensions.get("e7_surface_motion"), dict):
        errors.append("E8 v2 MicroMotion Evidence Violation: e7_surface_motion is required.")

    characters = blueprint.get("characters")
    if not isinstance(characters, list):
        return errors + ["E8 v2 MicroMotion Contract Violation: characters must be an array."]
    rows = extension.get("characters")
    if not isinstance(rows, list) or len(rows) != len(characters):
        errors.append("E8 v2 MicroMotion Contract Violation: extension character rows must match characters[].")
        rows = []
    row_by_id = {str(row.get("character_id")): row for row in rows if isinstance(row, dict)}

    for index, character in enumerate(characters):
        cpath = f"characters[{index}]"
        if not isinstance(character, dict):
            errors.append(f"E8 v2 MicroMotion Contract Violation: {cpath} must be an object.")
            continue
        character_id = str(character.get("character_id"))
        row = row_by_id.get(character_id)
        surface = character.get("surface_motion")
        if not isinstance(surface, dict) or surface.get("coordinate_frame") != "body_local_2d":
            errors.append(f"E8 v2 MicroMotion Evidence Violation: {cpath} requires body_local_2d surface motion.")
            continue
        body_ref = surface.get("body_frame_transform_ref")
        body_uri = body_ref.get("uri") if isinstance(body_ref, dict) else None
        regions = surface.get("regions")
        if not isinstance(regions, list):
            errors.append(f"E8 v2 MicroMotion Contract Violation: {cpath}.surface_motion.regions must be an array.")
            continue
        if not regions:
            if not isinstance(row, dict) or row.get("kind") != "not_detected" or row.get("signal_ref") is not None:
                errors.append(f"E8 v2 MicroMotion Contract Violation: extension row for {character_id!r} must record no-region not_detected state.")
            continue
        if len(regions) != 1 or not isinstance(regions[0], dict):
            errors.append(f"E8 v2 MicroMotion Contract Violation: {cpath} must contain exactly one E7 sparse region.")
            continue
        region = regions[0]
        rpath = f"{cpath}.surface_motion.regions[0]"
        residual_ref = region.get("residual_flow_ref")
        source_uri = residual_ref.get("uri") if isinstance(residual_ref, dict) else None
        micro = region.get("micro_motion")
        if not isinstance(micro, dict):
            errors.append(f"E8 v2 MicroMotion Contract Violation: {rpath}.micro_motion must be an object.")
            continue
        kind = micro.get("kind")
        if kind not in _ALLOWED_KINDS:
            errors.append(f"E8 v2 MicroMotion Evidence Violation: {rpath}.micro_motion.kind is outside geometry-only scope.")
        if micro.get("radial_expansion_ratio_ref") is not None or micro.get("area_change_ratio_ref") is not None:
            errors.append(f"E8 v2 MicroMotion Evidence Violation: {rpath} cannot emit radial/area modes in E8.1.")
        if micro.get("smoothing") != {"method": "none", "parameters": {}}:
            errors.append(f"E8 v2 MicroMotion Contract Violation: {rpath}.micro_motion.smoothing must remain disabled.")
        limitations = micro.get("limitations")
        if not isinstance(limitations, list) or not any(
            isinstance(item, str) and "no breathing" in item and "physiological" in item for item in limitations
        ):
            errors.append(f"E8 v2 MicroMotion Evidence Violation: {rpath}.micro_motion.limitations must state the physiological inference prohibition.")
        if source_uri is None or body_uri is None:
            errors.append(f"E8 v2 MicroMotion Evidence Violation: {rpath} requires physical E7 and E6 source refs.")
            continue

        refs = {
            "signal_ref": ("signal", "signal_valid", "body_local_unit"),
            "detrended_signal_ref": ("detrended_signal", "signal_valid", "body_local_unit"),
            "vertical_displacement_ref": ("vertical_displacement", "vertical_valid", "body_local_unit"),
            "velocity_ref": ("velocity", "velocity_valid", "body_local_unit_per_frame"),
            "acceleration_ref": ("acceleration", "acceleration_valid", "body_local_unit_per_frame2"),
        }
        concrete_refs: list[dict[str, Any]] = []
        for field, (array_key, valid_key, unit) in refs.items():
            ref = micro.get(field)
            errors.extend(
                _check_ref(
                    ref,
                    path=f"{rpath}.micro_motion.{field}",
                    frame_count=frame_count,
                    array_key=array_key,
                    valid_key=valid_key,
                    unit=unit,
                    config_sha=config_sha,
                    source_surface_uri=str(source_uri),
                    body_frame_uri=str(body_uri),
                )
            )
            if isinstance(ref, dict):
                concrete_refs.append(ref)
        if concrete_refs:
            uris = {ref.get("uri") for ref in concrete_refs}
            checksums = {ref.get("checksum_sha256") for ref in concrete_refs}
            if len(uris) != 1 or len(checksums) != 1:
                errors.append(f"E8 v2 MicroMotion Contract Violation: {rpath} signal refs must share one physical NPZ.")

        if not isinstance(row, dict) or row.get("kind") != kind or row.get("signal_ref") != micro.get("signal_ref"):
            errors.append(f"E8 v2 MicroMotion Cross-Reference Violation: extension row for {character_id!r} must match MicroMotion output.")
            continue
        selected = row.get("selected_track_count")
        candidate = row.get("candidate_track_count")
        support = row.get("consensus_support_fraction")
        if not isinstance(selected, int) or selected < 0 or not isinstance(candidate, int) or candidate < 0:
            errors.append(f"E8 v2 MicroMotion Contract Violation: extension row for {character_id!r} has invalid track counts.")
        elif selected > candidate:
            errors.append(f"E8 v2 MicroMotion Contract Violation: selected tracks cannot exceed candidate tracks for {character_id!r}.")
        if not isinstance(support, (int, float)) or isinstance(support, bool) or not 0.0 <= float(support) <= 1.0:
            errors.append(f"E8 v2 MicroMotion Contract Violation: consensus_support_fraction must be within [0,1] for {character_id!r}.")
        elif isinstance(selected, int) and isinstance(candidate, int):
            expected_support = selected / candidate if candidate else 0.0
            if abs(float(support) - expected_support) > 1e-6:
                errors.append(f"E8 v2 MicroMotion Contract Violation: consensus support must equal selected/candidate for {character_id!r}.")
        if kind == "periodic_micro_motion" and (not isinstance(selected, int) or selected < 4):
            errors.append(f"E8 v2 MicroMotion Quality Violation: periodic output requires at least four frequency-locked consensus tracks for {character_id!r}.")

        usable = micro.get("usable_for_generation") is True
        if usable:
            if kind != "periodic_micro_motion" or micro.get("dominant_frequency_hz") is None:
                errors.append(f"E8 v2 MicroMotion Quality Violation: {rpath} usable output must be periodic with a dominant frequency.")
            numeric_checks = (
                ("confidence", "usable_confidence_threshold", ">="),
                ("periodicity_score", "periodicity_threshold", ">="),
                ("spatial_coherence", "coherence_threshold", ">="),
                ("camera_leakage_score", "max_camera_leakage", "<="),
                ("pose_leakage_score", "max_pose_leakage", "<="),
                ("occlusion_ratio", "max_observation_dropout", "<="),
                ("amplitude_norm_p50", "min_amplitude_norm", ">="),
            )
            for metric, threshold_name, direction in numeric_checks:
                metric_value = micro.get(metric)
                threshold_value = thresholds.get(threshold_name)
                if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
                    errors.append(f"E8 v2 MicroMotion Quality Violation: {rpath}.micro_motion.{metric} must be numeric.")
                    continue
                if not isinstance(threshold_value, (int, float)) or isinstance(threshold_value, bool):
                    errors.append(f"E8 v2 MicroMotion Contract Violation: threshold {threshold_name} must be numeric.")
                    continue
                if direction == ">=" and float(metric_value) < float(threshold_value):
                    errors.append(f"E8 v2 MicroMotion Quality Violation: {rpath}.micro_motion.{metric} is below usable threshold.")
                if direction == "<=" and float(metric_value) > float(threshold_value):
                    errors.append(f"E8 v2 MicroMotion Quality Violation: {rpath}.micro_motion.{metric} exceeds usable threshold.")
            cycles = row.get("cycles_observed")
            min_cycles = thresholds.get("min_cycles")
            if not isinstance(cycles, (int, float)) or not isinstance(min_cycles, (int, float)) or float(cycles) < float(min_cycles):
                errors.append(f"E8 v2 MicroMotion Quality Violation: {rpath} usable output must satisfy minimum observed cycles.")
    return errors


__all__ = ["validate_micro_motion_v2_contract"]
