from __future__ import annotations

from typing import Any

from packages.blueprint_schema.body_local_contract import validate_body_local_contract
from packages.blueprint_schema.camera_contract import validate_camera_contract
from packages.blueprint_schema.micro_motion_contract import validate_micro_motion_contract
from packages.blueprint_schema.micro_motion_v2_contract import validate_micro_motion_v2_contract
from packages.blueprint_schema.surface_motion_contract import validate_surface_motion_contract
from packages.blueprint_schema.validator import BlueprintValidator as CoreBlueprintValidator


class BlueprintValidator(CoreBlueprintValidator):
    """Public validator including additive semantic contracts beyond raw JSON Schema."""

    def validate(self, blueprint_data: dict[str, Any]) -> tuple[bool, list[str]]:
        _, errors = super().validate(blueprint_data)
        errors.extend(validate_camera_contract(blueprint_data))
        errors.extend(validate_body_local_contract(blueprint_data))
        errors.extend(validate_surface_motion_contract(blueprint_data))
        extension = blueprint_data.get("extensions", {}).get("e8_micro_motion")
        if isinstance(extension, dict) and extension.get("algorithm") == "body_local_sparse_periodicity_v2":
            errors.extend(validate_micro_motion_v2_contract(blueprint_data))
        else:
            errors.extend(validate_micro_motion_contract(blueprint_data))
        return len(errors) == 0, errors
