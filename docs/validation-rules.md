# Validation Rules

Validation is layered.

## L0 Bundle/security

- required files present
- no absolute/external/traversal URI
- referenced artifacts stay inside bundle root
- ZIP extraction safe

## L1 Schema

Validate `blueprint.json` against Draft 2020-12 schema.

## L2 Time/reference integrity

- shots ordered and in bounds
- time-series frame ranges in bounds
- references resolve
- hashes match when files exist
- PTS/frame mappings are monotonic when present

## L3 Module quality

Later Epics add module-specific checks for tracking, pose, face, hands, masks, camera, and environment.

## L4 Micro-motion hard gate

Default generation-usable gate:

```text
confidence >= 0.65
spatial_coherence >= 0.55
camera_leakage_score <= 0.35
pose_leakage_score <= 0.45
occlusion_ratio <= 0.40
```

If camera estimation is unavailable/unreliable, micro-motion must not be marked usable for generation.

## E0 validator requirement

E0 only needs deterministic structural/schema validation plus clear machine-readable errors. It must not pretend to validate real CV quality.
