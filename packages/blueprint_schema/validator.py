import json
import os
from typing import Any

import jsonschema

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../contracts/video_blueprint.schema.json")


def load_canonical_schema() -> dict[str, Any]:
    resolved_path = os.path.abspath(SCHEMA_PATH)
    if not os.path.exists(resolved_path):
        # Fallback to root contracts directory
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
    """Validate E3.2 image-plane face landmark semantics beyond JSON types.

    TimeSeriesRef intentionally supports multiple coordinate spaces for other
    modules.  A Face.landmarks_2d_ref is narrower: it must be pixel XY, dense
    per-frame geometry, with explicit NaN preservation and no interpolation.
    """
    if ref is None:
        return []
    if not isinstance(ref, dict):
        return [f"E3.2 Face Contract Violation: {path} must be a TimeSeriesRef or null."]

    errors: list[str] = []
    if ref.get("coordinate_space") != "pixel_xy":
        errors.append(
            f"E3.2 Face Contract Violation: {path}.coordinate_space must be 'pixel_xy'."
        )
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
        errors.append(
            f"E3.2 Face Contract Violation: {path}.interpolation_policy must be 'none'."
        )
    return errors


class BlueprintValidator:
    def __init__(self, schema: dict[str, Any] | None = None):
        if schema is None:
            schema = load_canonical_schema()
        self.schema = schema
        self.format_checker = jsonschema.FormatChecker()
        self.validator = jsonschema.Draft202012Validator(schema, format_checker=self.format_checker)

    def validate(self, blueprint_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validates blueprint JSON against Draft 2020-12 schema and engineering invariants.
        Returns (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        # 1. Draft 2020-12 JSON Schema Validation
        schema_errors = sorted(self.validator.iter_errors(blueprint_data), key=lambda e: e.path)
        for err in schema_errors:
            path_str = ".".join(str(p) for p in err.path) if err.path else "root"
            errors.append(f"Schema Error at '{path_str}': {err.message}")

        # 2. Engineering Invariant 1: Manifest + Sidecar Separation
        # Dense arrays must NOT be embedded directly in blueprint.json
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

        # 3. Engineering Invariant 2: Biometric & Identity Protection
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

        # 4. Engineering Invariant 3: CFR Timebase
        timebase = blueprint_data.get("timebase")
        if isinstance(timebase, dict):
            fps_num = timebase.get("fps_num")
            fps_den = timebase.get("fps_den")
            if fps_num is not None and fps_num <= 0:
                errors.append("Invariant Violation: timebase.fps_num must be a positive integer.")
            if fps_den is not None and fps_den <= 0:
                errors.append("Invariant Violation: timebase.fps_den must be a positive integer.")

        # 5. Cross-Reference Validation: Character IDs
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

        # 6. Timeline Bounds Validation
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

        # 7. E3.2 Face 2D landmark semantic contract
        if isinstance(characters, list):
            for index, character in enumerate(characters):
                if not isinstance(character, dict):
                    continue
                face = character.get("face")
                if not isinstance(face, dict):
                    continue
                if "landmarks_2d_ref" not in face:
                    continue
                errors.extend(
                    _validate_face_2d_ref(
                        face.get("landmarks_2d_ref"),
                        landmark_count=face.get("landmark_count"),
                        frame_count=frame_count,
                        path=f"characters[{index}].face.landmarks_2d_ref",
                    )
                )

        return (len(errors) == 0, errors)
