# Architecture

## Target components

```text
Web UI
  -> FastAPI
      -> Orchestrator / Job State
          -> CPU Worker
          -> GPU Worker (future E2+)
      -> Artifact Store
      -> Metadata DB
      -> Model Registry (future)
```

## Suggested repository shape

```text
apps/
  api/
  web/
packages/
  blueprint_schema/
  pipeline_core/
  adapters/
  quality/
  visualization/
configs/
tests/
infra/
cloud/
```

## Pipeline DAG

```text
INGEST
 -> PROBE
 -> NORMALIZE
 -> SHOT_DETECTION
 -> PERSON_DETECTION
 -> TRACK_ASSOCIATION
 -> PERSON_MASK
 -> {POSE, FACE, HANDS, POINT_TRACKS, DENSE_FLOW}
 -> CAMERA_MOTION
 -> BODY_LOCAL_FRAME
 -> SURFACE_MOTION
 -> MICRO_MOTION
 -> ENVIRONMENT
 -> [SEMANTICS optional]
 -> ASSEMBLE_BLUEPRINT
 -> VALIDATE
 -> RENDER_OVERLAYS
 -> EXPORT_BUNDLE
```

## E0 simplification

E0 uses a deterministic mock stage instead of the real analysis DAG:

```text
INGEST -> MOCK_ANALYZE -> ASSEMBLE -> VALIDATE -> EXPORT
```

The mock path must preserve the same job/artifact abstractions so later real stages can replace it without breaking the external contract.
