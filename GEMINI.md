# Video Blueprint Studio — Gemini Project Context

## 1. Project mission

Build **Video Blueprint Studio M1**: an offline-first video analysis and verifiable-blueprint system.

The system converts an authorized source video into a structured, reproducible **Video Blueprint bundle** preserving source timing, anonymous character motion, pose/landmarks, masks, camera motion, surface residual motion, geometry-only micro-motion, environment evidence, quality, and provenance.

**M1 does not replace a person and does not generate a new video.**

## 2. Current implementation phase: E11

The repository has progressed through E0-E10. The current phase is:

**Epic E11 — Integrated Production Acceptance.**

Do not restart from E0, replace existing real stages with mocks, or disable higher stages simply to obtain a green result.

The immediate goal is to prove that the already implemented stages can execute together in one production job and produce one physically verified Bundle.

### Implemented evidence lineage

- E0: contracts, API/job skeleton, deterministic validation/bundle foundation
- E1: real media probe/normalization/timing
- E2: shot/person detection and anonymous tracking
- E3: pose plus face/hand refinement where configured
- E4: person masks, sparse point motion, dense flow
- E5: evidence-bounded 2D camera motion
- E6: body-local frame
- E7: residual surface motion
- E8: geometry-only micro-motion with leakage/quality gates
- E9: evidence-backed environment photometry
- E10: production dispatch, recursive artifact integrity, verified Bundle assembly
- E11: integrated full-stack certification, then representative real-video acceptance

Individual stage certifications are necessary but are not sufficient evidence for integrated readiness.

## 3. Canonical sources

Read these before changing contracts or stage semantics:

1. `contracts/video_blueprint.schema.json` — canonical protocol source.
2. `contracts/openapi.yaml` — HTTP contract.
3. `contracts/example_blueprint.json` — minimum contract fixture.
4. `docs/PRD_M1.md` — product requirements.
5. `docs/architecture.md` — system architecture.
6. `docs/epic-plan.md` — current implementation/acceptance order.
7. `docs/engineering-rules.md` — non-negotiable rules.
8. `docs/validation-rules.md` — validator and quality rules.

If prose conflicts with the JSON Schema, **do not silently change the schema**. Report the conflict and propose a versioned change.

## 4. Engineering invariants

1. Use **Manifest + Sidecar**. Dense arrays must not be embedded in `blueprint.json`.
2. Characters remain anonymous `character_id` tracks. Do not perform identity recognition.
3. Missing/failed evidence is explicit (`null`, NaN, validity/confidence/error fields). Never fabricate coordinates or fill gaps silently.
4. Production design remains offline-first and capable of zero network calls after approved dependencies/weights are provisioned.
5. Model code and model-weight licenses are reviewed separately; approved weight hashes are pinned for certifications.
6. A module failure must never create silent corruption or a false succeeded stage.
7. Every physical artifact must resolve safely and be traceable by checksum/provenance.
8. Human edits invalidate only dependent downstream stages.
9. Micro-motion means **residual visual geometry/surface motion** after camera/body compensation, not physiology.
10. Camera/body/observation failures must fail closed for micro-motion usability.
11. Do not export biometric identity embeddings or infer age, health, emotion, sexuality, or other sensitive personal attributes.
12. Do not interpret scene semantics, weather, or materials unless a future explicitly approved evidence source is added.
13. A green schema is not proof of extraction accuracy. Separate structural integrity from evidence quality.

## 5. Current production path

```text
source video
  -> media probe / CFR normalization / PTS mapping
  -> shot detection + anonymous people
  -> pose / optional face-hand evidence
  -> person masks + sparse point motion + dense flow
  -> 2D camera motion
  -> body-local frame
  -> residual surface motion
  -> geometry-only micro-motion
  -> environment photometry
  -> canonical schema validation
  -> recursive artifact integrity validation
  -> verified bundle.zip
```

Higher stages are opt-in/fail-closed through their environment gates. E11 certification intentionally enables the implemented E4-E9 evidence stages together.

## 6. E11 acceptance gate

Primary files:

- `scripts/certify_full_stack_e11.py`
- `.github/workflows/e11-integrated-acceptance.yml`

Do not claim E11 integrated certification complete until all are true for the **same production job**:

- certified detector, pose, and mask checkpoint hashes are verified
- point tracks succeed
- dense flow succeeds
- camera motion succeeds
- body-local frame succeeds
- surface motion succeeds
- micro-motion succeeds and emits physical signal refs
- environment photometry succeeds
- at least one anonymous character and physical surface/micro-motion region exist in the deterministic acceptance stimulus
- canonical Blueprint validation passes
- every referenced artifact resolves and hashes correctly
- the final ZIP is reopened and independently verified
- privacy/evidence boundaries remain intact

After that gate is green, the next step is representative user-supplied real-video acceptance. That phase must review both machine-readable metrics and the actual extracted evidence; it must not equate “pipeline completed” with “details were captured correctly.”

## 7. Work style for Gemini / AI Studio

For every coding task:

1. Inspect current `main` and active PR/branch first.
2. Identify the active acceptance gap from evidence, not from stale prose.
3. State the smallest target and preserve already certified behavior.
4. Modify the minimum number of files.
5. Add or strengthen tests/certification evidence with each change.
6. Run the relevant contract, unit/integration, lint/type, and stage certification gates.
7. Report exact changed files, commands, failures, outputs, and residual risks.
8. Never weaken a fail-closed gate, disable a stage, or fabricate evidence just to make CI green.
9. Stop at the current phase boundary unless a new phase is explicitly approved.
