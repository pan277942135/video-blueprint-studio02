from __future__ import annotations

from typing import Any


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _check_ref_common(ref: dict[str, Any], *, path: str, frame_start: int, frame_end: int) -> list[str]:
    errors: list[str] = []
    expected = {
        "format": "npz",
        "dtype": "float32",
        "sampling": "per_frame",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
    }
    for key, value in expected.items():
        if ref.get(key) != value:
            errors.append(f"E5 Camera Contract Violation: {path}.{key} must equal {value!r}.")
    if ref.get("frame_start") != frame_start or ref.get("frame_end") != frame_end:
        errors.append(f"E5 Camera Contract Violation: {path} must cover exactly its shot frame range.")
    if not _is_sha256(ref.get("checksum_sha256")):
        errors.append(f"E5 Camera Contract Violation: {path}.checksum_sha256 must be SHA256 hex.")
    return errors


def _check_metadata(ref: dict[str, Any], *, path: str, config_sha: Any) -> tuple[list[str], dict[str, Any] | None]:
    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        return [f"E5 Camera Contract Violation: {path}.metadata is required."], None
    errors: list[str] = []
    expected = {
        "algorithm": "opencv_background_lk_ransac_affine_v1",
        "shot_boundary_reset": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"E5 Camera Contract Violation: {path}.metadata.{key} must equal {value!r}.")
    if metadata.get("config_sha256") != config_sha:
        errors.append(f"E5 Camera Contract Violation: {path}.metadata.config_sha256 must match extension config.")
    return errors, metadata


def validate_camera_contract(blueprint: dict[str, Any]) -> list[str]:
    extensions = blueprint.get("extensions")
    if not isinstance(extensions, dict) or "e5_camera_motion" not in extensions:
        return []
    extension = extensions.get("e5_camera_motion")
    path = "extensions.e5_camera_motion"
    if not isinstance(extension, dict):
        return [f"E5 Camera Contract Violation: {path} must be an object."]
    errors: list[str] = []
    expected_extension = {
        "enabled": True,
        "algorithm": "opencv_background_lk_ransac_affine_v1",
        "reconstruction_backend": "opencv_ransac_2d",
        "shot_boundary_reset": True,
        "interpolation": False,
        "identity_inference_performed": False,
        "biometric_embedding_exported": False,
    }
    for key, value in expected_extension.items():
        if extension.get(key) != value:
            errors.append(f"E5 Camera Contract Violation: {path}.{key} must equal {value!r}.")
    config_sha = extension.get("config_sha256")
    if not _is_sha256(config_sha):
        errors.append(f"E5 Camera Contract Violation: {path}.config_sha256 must be SHA256 hex.")
    supported = ["static", "pan", "tilt", "roll", "zoom", "compound", "unknown"]
    if extension.get("supported_classifications") != supported:
        errors.append(f"E5 Camera Contract Violation: {path}.supported_classifications must equal {supported!r}.")

    shots = blueprint.get("shots")
    shot_by_id = {
        str(row.get("shot_id")): row
        for row in shots
        if isinstance(shots, list) and isinstance(row, dict) and row.get("shot_id") is not None
    } if isinstance(shots, list) else {}
    camera = blueprint.get("camera")
    if not isinstance(camera, dict) or not isinstance(camera.get("per_shot"), list):
        return errors + ["E5 Camera Contract Violation: camera.per_shot must be an array."]
    rows = camera["per_shot"]
    if extension.get("shot_count") != len(rows):
        errors.append("E5 Camera Contract Violation: extensions.e5_camera_motion.shot_count must match camera.per_shot.")

    seen_shots: set[str] = set()
    unsupported_3d = {"dolly", "truck", "pedestal"}
    for index, row in enumerate(rows):
        row_path = f"camera.per_shot[{index}]"
        if not isinstance(row, dict):
            errors.append(f"E5 Camera Contract Violation: {row_path} must be an object.")
            continue
        shot_id = str(row.get("shot_id"))
        shot = shot_by_id.get(shot_id)
        if shot is None:
            errors.append(f"E5 Camera Cross-Reference Violation: {row_path}.shot_id is not declared in shots[].")
            continue
        if shot_id in seen_shots:
            errors.append(f"E5 Camera Cross-Reference Violation: shot {shot_id!r} appears more than once in camera.per_shot.")
        seen_shots.add(shot_id)
        if shot.get("camera_motion_id") != row.get("camera_motion_id"):
            errors.append(f"E5 Camera Cross-Reference Violation: shot {shot_id!r} camera_motion_id does not match camera.per_shot.")
        if row.get("reconstruction_backend") != "opencv_ransac_2d":
            errors.append(f"E5 Camera Contract Violation: {row_path}.reconstruction_backend must be 'opencv_ransac_2d'.")
        if row.get("classification") in unsupported_3d:
            errors.append(
                f"E5 Camera Evidence Violation: {row_path}.classification cannot be {row.get('classification')!r} "
                "for an uncalibrated 2D-only backend."
            )
        if row.get("intrinsics") is not None or row.get("extrinsics_ref") is not None:
            errors.append(f"E5 Camera Evidence Violation: {row_path} cannot claim calibrated intrinsics/extrinsics.")
        for key in ("homography_ref", "crop_ref", "shake_ref"):
            if row.get(key) is not None:
                errors.append(f"E5 Camera Contract Violation: {row_path}.{key} must remain null in E5.1.")

        frame_start_value = shot.get("frame_start")
        frame_end_value = shot.get("frame_end")
        if not isinstance(frame_start_value, int) or isinstance(frame_start_value, bool):
            errors.append(f"E5 Camera Contract Violation: shot {shot_id!r}.frame_start must be an integer.")
            continue
        if not isinstance(frame_end_value, int) or isinstance(frame_end_value, bool):
            errors.append(f"E5 Camera Contract Violation: shot {shot_id!r}.frame_end must be an integer.")
            continue
        frame_start = frame_start_value
        frame_end = frame_end_value
        length = frame_end - frame_start + 1
        affine = row.get("affine_ref")
        if not isinstance(affine, dict):
            errors.append(f"E5 Camera Contract Violation: {row_path}.affine_ref must be a TimeSeriesRef.")
        else:
            affine_path = f"{row_path}.affine_ref"
            errors.extend(_check_ref_common(affine, path=affine_path, frame_start=frame_start, frame_end=frame_end))
            if affine.get("shape") != [length, 2, 3] or affine.get("axes") != ["frame", "matrix_row", "matrix_col"]:
                errors.append(f"E5 Camera Contract Violation: {affine_path} must have shape [shot_frames,2,3] and matrix axes.")
            if affine.get("unit") != "mixed_pixel_affine" or affine.get("coordinate_space") != "pixel_xy":
                errors.append(f"E5 Camera Contract Violation: {affine_path} must use mixed_pixel_affine in pixel_xy.")
            metadata_errors, metadata = _check_metadata(affine, path=affine_path, config_sha=config_sha)
            errors.extend(metadata_errors)
            if metadata is not None:
                if metadata.get("array_key") != "frame_to_frame_affine" or metadata.get("stabilization_array_key") != "stabilization_affine":
                    errors.append(f"E5 Camera Contract Violation: {affine_path} must declare affine and stabilization array keys.")
                if metadata.get("matrix_semantics") != "previous_frame_to_current_frame_affine":
                    errors.append(f"E5 Camera Contract Violation: {affine_path}.metadata.matrix_semantics is invalid.")

        tracks = row.get("background_tracks_ref")
        if not isinstance(tracks, dict):
            errors.append(f"E5 Camera Contract Violation: {row_path}.background_tracks_ref must be a TimeSeriesRef.")
        else:
            tracks_path = f"{row_path}.background_tracks_ref"
            errors.extend(_check_ref_common(tracks, path=tracks_path, frame_start=frame_start, frame_end=frame_end))
            shape = tracks.get("shape")
            if not isinstance(shape, list) or len(shape) != 3 or shape[0] != length or shape[-1] != 2:
                errors.append(f"E5 Camera Contract Violation: {tracks_path}.shape must be [shot_frames,track_slots,2].")
            if tracks.get("axes") != ["frame", "background_track", "xy"] or tracks.get("unit") != "px" or tracks.get("coordinate_space") != "pixel_xy":
                errors.append(f"E5 Camera Contract Violation: {tracks_path} must describe pixel_xy background tracks.")
            metadata_errors, metadata = _check_metadata(tracks, path=tracks_path, config_sha=config_sha)
            errors.extend(metadata_errors)
            if metadata is not None and (metadata.get("array_key") != "background_points_xy" or metadata.get("valid_array_key") != "background_valid"):
                errors.append(f"E5 Camera Contract Violation: {tracks_path} must declare background point/valid array keys.")

        zoom = row.get("zoom_proxy_ref")
        if not isinstance(zoom, dict):
            errors.append(f"E5 Camera Contract Violation: {row_path}.zoom_proxy_ref must be a TimeSeriesRef.")
        else:
            zoom_path = f"{row_path}.zoom_proxy_ref"
            errors.extend(_check_ref_common(zoom, path=zoom_path, frame_start=frame_start, frame_end=frame_end))
            if zoom.get("shape") != [length] or zoom.get("axes") != ["frame"]:
                errors.append(f"E5 Camera Contract Violation: {zoom_path} must have shape [shot_frames].")
            if zoom.get("unit") != "ratio" or zoom.get("coordinate_space") != "none":
                errors.append(f"E5 Camera Contract Violation: {zoom_path} must be a unitless ratio in coordinate_space 'none'.")
            metadata_errors, metadata = _check_metadata(zoom, path=zoom_path, config_sha=config_sha)
            errors.extend(metadata_errors)
            if metadata is not None and (metadata.get("array_key") != "zoom_proxy" or metadata.get("inlier_ratio_array_key") != "ransac_inlier_ratio"):
                errors.append(f"E5 Camera Contract Violation: {zoom_path} must declare zoom/inlier array keys.")

    if set(shot_by_id) != seen_shots:
        errors.append("E5 Camera Cross-Reference Violation: camera.per_shot must contain exactly one row for every shot.")
    return errors
