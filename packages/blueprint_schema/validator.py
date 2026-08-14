import json
import os
from typing import Any

import jsonschema

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../contracts/video_blueprint.schema.json")


def load_canonical_schema() -> dict[str, Any]:
    resolved_path = os.path.abspath(SCHEMA_PATH)
    if not os.path.exists(resolved_path):
        resolved_path = os.path.abspath("contracts/video_blueprint.schema.json")
    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_face_2d_ref(
    ref: Any,
    *,
    landmark_count: Any,
    frame_count: Any,
    path: str,
) -> list[str]:
    if ref is None:
        return []
    if not isinstance(ref, dict):
        return [f"E3.2 Face Contract Violation: {path} must be a TimeSeriesRef or null."]

    errors: list[str] = []
    if ref.get("coordinate_space") != "pixel_xy":
        errors.append(f"E3.2 Face Contract Violation: {path}.coordinate_space must be 'pixel_xy'.")
    if ref.get("unit") != "px":
        errors.append(f"E3.2 Face Contract Violation: {path}.unit must be 'px'.")
    if ref.get("axes") != ["frame", "face_landmark", "xy"]:
        errors.append(
            f"E3.2 Face Contract Violation: {path}.axes must equal "
            "['frame', 'face_landmark', 'xy']."
        )

    shape = ref.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 2:
        errors.append(
            f"E3.2 Face Contract Violation: {path}.shape must be "
            "[frame_count, landmark_count, 2]."
        )
    else:
        if isinstance(frame_count, int) and shape[0] != frame_count:
            errors.append(
                f"E3.2 Face Contract Violation: {path}.shape[0] ({shape[0]}) "
                f"must equal timebase.frame_count ({frame_count})."
            )
        if isinstance(landmark_count, int) and landmark_count > 0 and shape[1] != landmark_count:
            errors.append(
                f"E3.2 Face Contract Violation: {path}.shape[1] ({shape[1]}) "
                f"must equal face.landmark_count ({landmark_count})."
            )

    if ref.get("sampling", "per_frame") != "per_frame":
        errors.append(f"E3.2 Face Contract Violation: {path}.sampling must be 'per_frame'.")
    if ref.get("nan_policy", "preserve") != "preserve":
        errors.append(f"E3.2 Face Contract Violation: {path}.nan_policy must be 'preserve'.")
    if ref.get("interpolation_policy", "none") != "none":
        errors.append(f"E3.2 Face Contract Violation: {path}.interpolation_policy must be 'none'.")
    return errors


def _validate_person_mask_ref(
    ref: Any,
    *,
    frame_count: Any,
    height: Any,
    width: Any,
    path: str,
) -> list[str]:
    if ref is None:
        return []
    if not isinstance(ref, dict):
        return [f"E4.1 Person Mask Contract Violation: {path} must be a TimeSeriesRef or null."]

    errors: list[str] = []
    expected = {
        "format": "rle_json",
        "dtype": "bool",
        "axes": ["frame", "y", "x"],
        "unit": "binary",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
    }
    for key, value in expected.items():
        if ref.get(key) != value:
            errors.append(f"E4.1 Person Mask Contract Violation: {path}.{key} must equal {value!r}.")

    shape = ref.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        errors.append(f"E4.1 Person Mask Contract Violation: {path}.shape must be [frame_count, height, width].")
    else:
        expected_shape = [frame_count, height, width]
        if all(isinstance(value, int) for value in expected_shape) and shape != expected_shape:
            errors.append(f"E4.1 Person Mask Contract Violation: {path}.shape {shape} must equal {expected_shape}.")

    if isinstance(frame_count, int) and frame_count > 0:
        if ref.get("frame_start") != 0:
            errors.append(f"E4.1 Person Mask Contract Violation: {path}.frame_start must be 0.")
        if ref.get("frame_end") != frame_count - 1:
            errors.append(f"E4.1 Person Mask Contract Violation: {path}.frame_end must equal frame_count - 1.")

    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"E4.1 Person Mask Contract Violation: {path}.metadata is required.")
    else:
        if metadata.get("encoding") != "row_major_binary_rle_v1":
            errors.append(
                f"E4.1 Person Mask Contract Violation: {path}.metadata.encoding must be "
                "'row_major_binary_rle_v1'."
            )
        if metadata.get("foreground_value") != 1 or metadata.get("background_value") != 0:
            errors.append(f"E4.1 Person Mask Contract Violation: {path} must declare foreground=1/background=0.")
        if "missing_frame_value" not in metadata or metadata.get("missing_frame_value") is not None:
            errors.append(
                f"E4.1 Person Mask Contract Violation: {path}.metadata.missing_frame_value must be null."
            )
    return errors


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _validate_e4_point_tracks_extension(
    extension: Any,
    *,
    frame_count: Any,
    declared_character_ids: set[Any],
) -> list[str]:
    path = "extensions.e4_point_tracks"
    if not isinstance(extension, dict):
        return [f"E4.2 Point Tracks Contract Violation: {path} must be an object."]
    errors: list[str] = []
    expected = {
        "enabled": True,
        "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
        "coordinate_space": "pixel_xy",
        "mask_constrained": True,
        "shot_boundary_reset": True,
        "interpolation": False,
    }
    for key, value in expected.items():
        if extension.get(key) != value:
            errors.append(f"E4.2 Point Tracks Contract Violation: {path}.{key} must equal {value!r}.")

    max_points = extension.get("max_points")
    if not isinstance(max_points, int) or max_points <= 0:
        errors.append(f"E4.2 Point Tracks Contract Violation: {path}.max_points must be a positive integer.")
    config_sha = extension.get("config_sha256")
    if not _is_sha256(config_sha):
        errors.append(f"E4.2 Point Tracks Contract Violation: {path}.config_sha256 must be SHA256 hex.")

    rows = extension.get("characters")
    if not isinstance(rows, dict):
        return errors + [f"E4.2 Point Tracks Contract Violation: {path}.characters must be an object."]
    for character_id, row in rows.items():
        row_path = f"{path}.characters[{character_id!r}]"
        if character_id not in declared_character_ids:
            errors.append(
                f"E4.2 Point Tracks Cross-Reference Violation: {row_path} does not match characters[].character_id."
            )
        if not isinstance(row, dict):
            errors.append(f"E4.2 Point Tracks Contract Violation: {row_path} must be an object.")
            continue
        ref = row.get("track_points_ref")
        if ref is None:
            continue
        if not isinstance(ref, dict):
            errors.append(f"E4.2 Point Tracks Contract Violation: {row_path}.track_points_ref must be a ref or null.")
            continue
        ref_path = f"{row_path}.track_points_ref"
        ref_expected = {
            "format": "npz",
            "dtype": "float32",
            "axes": ["frame", "point_slot", "xy"],
            "unit": "px",
            "coordinate_space": "pixel_xy",
            "sampling": "per_frame",
            "nan_policy": "preserve",
            "interpolation_policy": "none",
        }
        for key, value in ref_expected.items():
            if ref.get(key) != value:
                errors.append(f"E4.2 Point Tracks Contract Violation: {ref_path}.{key} must equal {value!r}.")
        if isinstance(frame_count, int) and isinstance(max_points, int):
            if ref.get("shape") != [frame_count, max_points, 2]:
                errors.append(
                    f"E4.2 Point Tracks Contract Violation: {ref_path}.shape must equal "
                    f"[{frame_count}, {max_points}, 2]."
                )
            if ref.get("frame_start") != 0 or ref.get("frame_end") != frame_count - 1:
                errors.append(f"E4.2 Point Tracks Contract Violation: {ref_path} must cover the full timeline.")
        if not _is_sha256(ref.get("checksum_sha256")):
            errors.append(f"E4.2 Point Tracks Contract Violation: {ref_path}.checksum_sha256 must be SHA256 hex.")
        metadata = ref.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"E4.2 Point Tracks Contract Violation: {ref_path}.metadata is required.")
            continue
        metadata_expected = {
            "array_key": "positions_xy",
            "valid_array_key": "valid",
            "error_array_key": "tracking_error",
            "track_id_array_key": "track_id",
            "algorithm": "opencv_shi_tomasi_pyr_lk_v1",
            "mask_constraint": "character.person_mask_ref",
            "point_slot_semantics": "reusable_slot_track_id_disambiguates_generation",
            "shot_boundary_reset": True,
        }
        for key, value in metadata_expected.items():
            if metadata.get(key) != value:
                errors.append(f"E4.2 Point Tracks Contract Violation: {ref_path}.metadata.{key} must equal {value!r}.")
        if metadata.get("config_sha256") != config_sha:
            errors.append(
                f"E4.2 Point Tracks Contract Violation: {ref_path}.metadata.config_sha256 must match extension config."
            )
    return errors


def _validate_e4_dense_flow_extension(
    extension: Any,
    *,
    frame_count: Any,
    source_width: Any,
    source_height: Any,
) -> list[str]:
    path = "extensions.e4_dense_flow"
    if not isinstance(extension, dict):
        return [f"E4.3 Dense Flow Contract Violation: {path} must be an object."]
    errors: list[str] = []
    expected = {
        "enabled": True,
        "algorithm": "opencv_farneback_v1",
        "coordinate_space": "pixel_xy",
        "vector_unit": "px",
        "shot_boundary_reset": True,
        "interpolation": False,
    }
    for key, value in expected.items():
        if extension.get(key) != value:
            errors.append(f"E4.3 Dense Flow Contract Violation: {path}.{key} must equal {value!r}.")
    config_sha = extension.get("config_sha256")
    if not _is_sha256(config_sha):
        errors.append(f"E4.3 Dense Flow Contract Violation: {path}.config_sha256 must be SHA256 hex.")

    ref = extension.get("flow_ref")
    if not isinstance(ref, dict):
        return errors + [f"E4.3 Dense Flow Contract Violation: {path}.flow_ref must be a TimeSeriesRef."]
    ref_path = f"{path}.flow_ref"
    ref_expected = {
        "format": "npz",
        "dtype": "float32",
        "axes": ["frame", "grid_y", "grid_x", "xy"],
        "unit": "px",
        "coordinate_space": "pixel_xy",
        "sampling": "per_frame",
        "nan_policy": "preserve",
        "interpolation_policy": "none",
    }
    for key, value in ref_expected.items():
        if ref.get(key) != value:
            errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.{key} must equal {value!r}.")
    shape = ref.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or shape[-1] != 2:
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.shape must be [frame, grid_y, grid_x, 2].")
    elif isinstance(frame_count, int) and shape[0] != frame_count:
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.shape[0] must equal frame_count.")
    if (
        isinstance(frame_count, int)
        and frame_count > 0
        and (ref.get("frame_start") != 0 or ref.get("frame_end") != frame_count - 1)
    ):
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path} must cover the full timeline.")
    if not _is_sha256(ref.get("checksum_sha256")):
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.checksum_sha256 must be SHA256 hex.")

    metadata = ref.get("metadata")
    if not isinstance(metadata, dict):
        return errors + [f"E4.3 Dense Flow Contract Violation: {ref_path}.metadata is required."]
    metadata_expected = {
        "array_key": "flow_xy",
        "valid_frame_array_key": "valid_frame",
        "algorithm": "opencv_farneback_v1",
        "vector_semantics": "previous_frame_to_current_frame_displacement",
        "vectors_scaled_to_source_pixels": True,
        "shot_boundary_reset": True,
    }
    for key, value in metadata_expected.items():
        if metadata.get(key) != value:
            errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.metadata.{key} must equal {value!r}.")
    if metadata.get("config_sha256") != config_sha:
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.metadata.config_sha256 must match extension config.")
    if isinstance(source_width, int) and metadata.get("source_width") != source_width:
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.metadata.source_width must match source_video.width.")
    if isinstance(source_height, int) and metadata.get("source_height") != source_height:
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.metadata.source_height must match source_video.height.")
    if (
        isinstance(shape, list)
        and len(shape) == 4
        and (metadata.get("grid_height") != shape[1] or metadata.get("grid_width") != shape[2])
    ):
        errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path} grid metadata must match shape.")
    for key in ("grid_to_source_scale_x", "grid_to_source_scale_y"):
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"E4.3 Dense Flow Contract Violation: {ref_path}.metadata.{key} must be positive.")
    return errors


class BlueprintValidator:
    def __init__(self, schema: dict[str, Any] | None = None):
        if schema is None:
            schema = load_canonical_schema()
        self.schema = schema
        self.format_checker = jsonschema.FormatChecker()
        self.validator = jsonschema.Draft202012Validator(schema, format_checker=self.format_checker)

    def validate(self, blueprint_data: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []

        schema_errors = sorted(self.validator.iter_errors(blueprint_data), key=lambda e: e.path)
        for err in schema_errors:
            path_str = ".".join(str(p) for p in err.path) if err.path else "root"
            errors.append(f"Schema Error at '{path_str}': {err.message}")

        forbidden_dense_keys = [
            "dense_motion_vectors",
            "embedded_landmarks",
            "frame_by_frame_poses",
            "raw_point_cloud",
        ]
        for key in forbidden_dense_keys:
            if key in blueprint_data:
                errors.append(
                    f"Invariant Violation: '{key}' must not be embedded in blueprint.json "
                    "(Manifest + Sidecar required)."
                )

        forbidden_identity_keys = [
            "biometric_embedding",
            "facial_identity_vectors",
            "health_metrics",
            "emotion_label",
            "sexual_attributes",
        ]
        for key in forbidden_identity_keys:
            if key in blueprint_data:
                errors.append(
                    f"Invariant Violation: '{key}' is forbidden. Biometric identity embeddings "
                    "and demographic inferences are not allowed."
                )

        timebase = blueprint_data.get("timebase")
        if isinstance(timebase, dict):
            fps_num = timebase.get("fps_num")
            fps_den = timebase.get("fps_den")
            if fps_num is not None and fps_num <= 0:
                errors.append("Invariant Violation: timebase.fps_num must be a positive integer.")
            if fps_den is not None and fps_den <= 0:
                errors.append("Invariant Violation: timebase.fps_den must be a positive integer.")

        characters = blueprint_data.get("characters", [])
        declared_char_ids = {
            c.get("character_id")
            for c in characters
            if isinstance(c, dict) and "character_id" in c
        }
        shots = blueprint_data.get("shots", [])
        if isinstance(shots, list):
            for i, shot in enumerate(shots):
                if isinstance(shot, dict):
                    dom_ids = shot.get("dominant_character_ids", [])
                    if isinstance(dom_ids, list):
                        for char_id in dom_ids:
                            if char_id not in declared_char_ids:
                                errors.append(
                                    f"Cross-Reference Violation: shot[{i}] references "
                                    f"dominant_character_id '{char_id}' which is not defined "
                                    "in characters[]."
                                )

        frame_count: Any = None
        if isinstance(timebase, dict):
            frame_count = timebase.get("frame_count")
            if isinstance(frame_count, int) and frame_count > 0 and isinstance(shots, list):
                for i, shot in enumerate(shots):
                    if isinstance(shot, dict):
                        keyframes = shot.get("keyframes", [])
                        if isinstance(keyframes, list):
                            for k_idx, kf in enumerate(keyframes):
                                if isinstance(kf, dict):
                                    f_idx = kf.get("frame_idx")
                                    if isinstance(f_idx, int) and f_idx >= frame_count:
                                        errors.append(
                                            f"Timeline Bounds Violation: shot[{i}] "
                                            f"keyframe[{k_idx}] frame_idx ({f_idx}) >= "
                                            f"total frame_count ({frame_count})."
                                        )

        source_video = blueprint_data.get("source_video")
        height = source_video.get("height") if isinstance(source_video, dict) else None
        width = source_video.get("width") if isinstance(source_video, dict) else None
        if isinstance(characters, list):
            for index, character in enumerate(characters):
                if not isinstance(character, dict):
                    continue
                face = character.get("face")
                if isinstance(face, dict) and "landmarks_2d_ref" in face:
                    errors.extend(
                        _validate_face_2d_ref(
                            face.get("landmarks_2d_ref"),
                            landmark_count=face.get("landmark_count"),
                            frame_count=frame_count,
                            path=f"characters[{index}].face.landmarks_2d_ref",
                        )
                    )
                if "person_mask_ref" in character:
                    errors.extend(
                        _validate_person_mask_ref(
                            character.get("person_mask_ref"),
                            frame_count=frame_count,
                            height=height,
                            width=width,
                            path=f"characters[{index}].person_mask_ref",
                        )
                    )

        extensions = blueprint_data.get("extensions")
        if isinstance(extensions, dict) and "e4_point_tracks" in extensions:
            errors.extend(
                _validate_e4_point_tracks_extension(
                    extensions.get("e4_point_tracks"),
                    frame_count=frame_count,
                    declared_character_ids=declared_char_ids,
                )
            )
        if isinstance(extensions, dict) and "e4_dense_flow" in extensions:
            errors.extend(
                _validate_e4_dense_flow_extension(
                    extensions.get("e4_dense_flow"),
                    frame_count=frame_count,
                    source_width=width,
                    source_height=height,
                )
            )

        return (len(errors) == 0, errors)
