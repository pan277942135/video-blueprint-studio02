# Video Blueprint Studio — Gemini Project Context

## 1. Project mission

Build **Video Blueprint Studio M1**: an offline-first video analysis and verifiable-blueprint system.

The system converts an authorized source video into a structured, reproducible **Video Blueprint bundle** preserving source timing, anonymous character motion, pose/landmarks, masks, camera motion, surface residual motion, geometry-only micro-motion, environment evidence, quality, and provenance.

**M1 does not replace a person and does not generate a new video.**

## 2. Current implementation phase: E12

The repository has progressed through E0-E11. The current phase is:

**Epic E12 — Representative Real-Video Acceptance.**

E11 proved the integrated production stack can execute together and produce a physically verified Bundle. It did **not** prove that representative real-video details are captured accurately. Do not restart from E0, replace existing real stages with mocks, disable stages merely to obtain green output, or treat `succeeded` as a perceptual-accuracy claim.

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
- E11: integrated E4-E9 production certification through E10 Bundle integrity
- E12: representative real-video evidence and accuracy acceptance

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
13. A green schema or green CI is not proof of extraction accuracy. Separate structural integrity from evidence quality.

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

Higher stages are opt-in/fail-closed through their environment gates.

## 6. E12 acceptance workflow

Primary files:

- `scripts/run_pipeline_job.py` — production Bundle generation
- `scripts/evaluate_real_video_e12.py` — Bundle acceptance report CLI
- `packages/pipeline_core/real_video_acceptance.py` — evidence diagnostics
- `docs/epic-plan.md` — representative-video acceptance scope

For each representative real video:

1. Run the integrated production path with the approved model/runtime configuration.
2. Preserve the source SHA, Blueprint, validation report, physical sidecars and final Bundle.
3. Run `scripts/evaluate_real_video_e12.py` on the Bundle.
4. Inspect machine diagnostics, including:
   - core stage status;
   - module scores and zero/low scores despite succeeded stages;
   - track/pose/face/hand quality where available;
   - surface-region count;
   - micro-motion usable ratio and exact rejection reasons;
   - artifact-integrity warnings;
   - privacy invariants.
5. Review source frames/video against evidence overlays/time series. Pay special attention to hair/clothing/hands/occlusion boundaries and to camera/body leakage into residual motion.
6. Keep final acceptance as `manual_frame_to_evidence_review_required` until that frame-to-evidence review is complete.

A structurally valid Bundle with physical sidecars is necessary but not sufficient. If a module score is zero/low, or micro-motion evidence is emitted but fails its usability gates, report that condition explicitly rather than calling the extractor accurate.

## 7. Work style for Gemini / AI Studio

For every coding or acceptance task:

1. Inspect current `main` and active PR/branch first.
2. Identify the active gap from evidence, not stale prose.
3. Preserve already certified behavior and fail-closed semantics.
4. Modify the minimum number of files needed.
5. Add or strengthen tests/evidence with each change.
6. Run relevant contract, unit/integration, lint/type and certification gates.
7. Report exact changed files, commands, failures, outputs and residual risks.
8. Never weaken quality gates, disable a stage, fabricate evidence, or reinterpret a zero/low score simply to obtain a pass.
9. During E12, separate machine integrity from perceptual accuracy and require representative real-video review before final acceptance.
