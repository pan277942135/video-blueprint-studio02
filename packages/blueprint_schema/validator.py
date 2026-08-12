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

class BlueprintValidator:
    def __init__(self, schema: dict[str, Any] | None = None):
        if schema is None:
            schema = load_canonical_schema()
        self.schema = schema
        self.format_checker = jsonschema.FormatChecker()
        self.validator = jsonschema.Draft202012Validator(schema, format_checker=self.format_checker)

    def validate(self, blueprint_data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validates blueprint JSON against Draft 2020-12 schema and E0 engineering invariants.
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
        forbidden_dense_keys = ["dense_motion_vectors", "embedded_landmarks", "frame_by_frame_poses", "raw_point_cloud"]
        for key in forbidden_dense_keys:
            if key in blueprint_data:
                errors.append(f"Invariant Violation: '{key}' must not be embedded in blueprint.json (Manifest + Sidecar required).")

        # 3. Engineering Invariant 2: Biometric & Identity Protection
        forbidden_identity_keys = ["biometric_embedding", "facial_identity_vectors", "health_metrics", "emotion_label", "sexual_attributes"]
        for key in forbidden_identity_keys:
            if key in blueprint_data:
                errors.append(f"Invariant Violation: '{key}' is forbidden. Biometric identity embeddings and demographic inferences are not allowed.")

        # 4. Engineering Invariant 3: CFR Timebase
        timebase = blueprint_data.get("timebase")
        if isinstance(timebase, dict):
            fps_num = timebase.get("fps_num")
            fps_den = timebase.get("fps_den")
            if fps_num is not None and fps_num <= 0:
                errors.append("Invariant Violation: timebase.fps_num must be a positive integer.")
            if fps_den is not None and fps_den <= 0:
                errors.append("Invariant Violation: timebase.fps_den must be a positive integer.")

        return (len(errors) == 0, errors)
