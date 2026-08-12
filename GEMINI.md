# Video Blueprint Studio — Gemini Project Context

## 1. Project mission

Build **Video Blueprint Studio M1**: an offline-first video analysis and verifiable-blueprint system.

The system converts an authorized source video into a structured, reproducible **Video Blueprint bundle** that preserves source-video timing, camera motion, anonymous character motion, pose, face/hand landmarks, masks, surface residual motion, micro-motion, quality, and provenance.

**M1 does not replace a person and does not generate a new video.**

## 2. Current implementation phase: E0 ONLY

The current permitted implementation scope is **Epic E0 — Contract and Engineering Skeleton**.

Do not start E1 or integrate any real computer-vision model until E0 acceptance is explicitly confirmed.

### E0 must implement

- repository skeleton
- JSON Schema contract wiring
- Pydantic v2 models or a thin schema-validation layer
- OpenAPI/FastAPI stubs
- deterministic mock analysis pipeline
- job state model
- minimal bundle export
- contract tests
- Docker Compose skeleton
- CI-ready lint/type/test commands
- README/runbook

### E0 must NOT implement

- RTMDet / MMDetection
- RTMW / MMPose
- SAM 2
- MediaPipe inference
- SEA-RAFT
- TAPIR/TAPNext
- COLMAP/GLOMAP
- Qwen semantic inference
- any identity replacement or video generation
- biometric identity embeddings
- age, health, emotion, or sexual-attribute inference

## 3. Canonical sources

Read these before changing contracts:

1. `contracts/video_blueprint.schema.json` — canonical protocol source.
2. `contracts/openapi.yaml` — HTTP contract.
3. `contracts/example_blueprint.json` — minimum contract fixture.
4. `docs/PRD_M1.md` — product requirements.
5. `docs/architecture.md` — system architecture.
6. `docs/epic-plan.md` — implementation order.
7. `docs/engineering-rules.md` — non-negotiable rules.
8. `docs/validation-rules.md` — validator and quality rules.

If prose conflicts with the JSON Schema, **do not silently change the schema**. Report the conflict and propose a versioned change.

## 4. Engineering invariants

1. Use **Manifest + Sidecar**. Dense arrays must not be embedded in `blueprint.json`.
2. Characters are anonymous `character_id` tracks only.
3. Failed or invisible data is represented explicitly with `null`, `NaN`, confidence, visibility, or errors. Never fabricate coordinates.
4. Production design is offline-first and must be capable of zero network calls.
5. Model code and model-weight licenses are reviewed separately.
6. A module failure must never create silent corruption.
7. Every artifact must be traceable with checksum/provenance once real stages exist.
8. Human edits invalidate only dependent downstream stages.
9. Micro-motion is **residual visual surface motion** after camera and body/global motion compensation.
10. Camera failure must prevent micro-motion from being marked usable for generation.
11. Do not perform biometric identity recognition or export face identity embeddings.
12. Implement one Epic at a time.

## 5. E0 target flow

```text
upload video
  -> create analysis job
  -> deterministic mock pipeline
  -> assemble legal blueprint.json
  -> validate against JSON Schema
  -> create bundle.zip
  -> expose download endpoint
```

The E0 mock may use placeholder/empty module outputs, but the manifest must be schema-valid and deterministic.

## 6. Job states

Use the contract-compatible states:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `partial`

Stage states may additionally include `pending` and `skipped` if permitted by the schema.

## 7. E0 acceptance gate

Do not claim E0 complete until all are true:

- a small video can be uploaded
- an analysis job can be created
- the mock job completes
- `blueprint.json` is produced
- `blueprint.json` validates against the canonical schema
- bundle download works
- invalid fixtures fail contract tests
- API contract tests pass
- cancellation/retry behavior is represented
- lint/type/test commands are documented
- no real CV model is integrated

## 8. Work style for Gemini

For every coding task:

1. Inspect the repository first.
2. Identify the active Epic.
3. State the smallest acceptance target.
4. Modify the minimum number of files.
5. Add or update tests with the change.
6. Run validation/tests before declaring success.
7. Summarize changed files, commands run, failures, and remaining work.
8. Stop at the current Epic boundary.

If asked to "build everything", refuse the scope expansion and continue only with the current approved Epic unless the user explicitly changes the project phase.
