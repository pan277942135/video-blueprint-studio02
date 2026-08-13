import json
import os

from packages.blueprint_schema.validator import BlueprintValidator, load_canonical_schema


def test_canonical_schema_loading():
    schema = load_canonical_schema()
    assert schema is not None
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"


def test_face_schema_exposes_optional_nullable_2d_landmark_ref():
    schema = load_canonical_schema()
    face = schema["$defs"]["Face"]
    properties = face["properties"]

    assert "landmarks_2d_ref" in properties
    assert "landmarks_2d_ref" not in face["required"]
    assert properties["landmarks_2d_ref"] == {
        "anyOf": [
            {"$ref": "#/$defs/TimeSeriesRef"},
            {"type": "null"},
        ]
    }


def test_positive_example_blueprint():
    validator = BlueprintValidator()
    example_path = os.path.abspath("contracts/example_blueprint.json")
    with open(example_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0.0"
    is_valid, errors = validator.validate(data)
    assert is_valid, f"Canonical example_blueprint.json failed validation: {errors}"


def test_invalid_fixtures():
    validator = BlueprintValidator()
    fixtures_dir = os.path.abspath("tests/contract/fixtures")

    for filename in os.listdir(fixtures_dir):
        if filename.startswith("invalid_") and filename.endswith(".json"):
            filepath = os.path.join(fixtures_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            is_valid, errors = validator.validate(data)
            assert not is_valid, f"Invalid fixture '{filename}' unexpectedly PASSED validation!"
            assert len(errors) > 0
