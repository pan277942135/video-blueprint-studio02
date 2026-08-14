# Epic Plan

This file reflects the **implemented repository lineage**, not the original E0-era placeholder plan.

## E0 — Contract and engineering skeleton [COMPLETE]

Canonical schema/OpenAPI wiring, deterministic mock foundation, job/API skeleton, validation, bundle export, tests, CI/runbook.

## E1 — Real media and jobs [COMPLETE]

Real ffprobe/FFmpeg media probing, source validation, CFR normalization, PTS mapping, persisted upload-to-analysis runtime path, cache/retry/cancel/export behavior.

## E2 — Shots and anonymous people [COMPLETE / CERTIFIED]

Shot detection, RTMDet person detection, anonymous track association, physical detection evidence and provenance.

## E3 — Pose / face / hands [COMPLETE / CERTIFIED]

RTMPose body pose plus configured MediaPipe refinement/evidence, explicit validity and physical sidecars.

## E4 — Masks / sparse motion / dense flow [COMPLETE / CERTIFIED]

RTMDet-Ins person masks plus sparse point motion and dense-flow evidence. Stages are opt-in and fail closed.

## E5 — Camera motion [COMPLETE / CERTIFIED]

Evidence-bounded 2D camera motion from background motion with quality/failure handling.

## E6 — Body-local frame [COMPLETE / CERTIFIED]

Body-local transforms derived from pose/camera evidence with explicit validity.

## E7 — Residual surface motion [COMPLETE / CERTIFIED]

Surface-region residual motion after global camera/body compensation, with physical time-series evidence.

## E8 — Geometry-only micro-motion [COMPLETE / CERTIFIED]

Residual micro-motion signals, velocity/acceleration validity, periodicity/coherence/leakage/occlusion gates and explicit non-physiological interpretation boundary.

## E9 — Environment photometry [COMPLETE / CERTIFIED]

Evidence-backed background photometry over inverse person masks. No semantic scene/weather/material inference.

## E10 — Production dispatch and physical Bundle integrity [COMPLETE / CERTIFIED]

Production API/worker path routes through the highest implemented pipeline. Recursive artifact URI/hash validation, NPZ/RLE physical checks, safe ZIP paths, manifest coverage, and post-write Bundle verification are enforced.

## E11 — Integrated Production Acceptance [COMPLETE / CERTIFIED]

E11 closed the post-E10 integration gap by enabling point tracks, dense flow, camera motion, body-local frame, surface motion, micro-motion and environment evidence together in the same production job, then performing E10 physical Bundle verification.

Primary evidence:

- `scripts/certify_full_stack_e11.py`
- `.github/workflows/e11-integrated-acceptance.yml`

The integrated certification succeeded with physical surface/micro-motion refs and zero artifact-integrity warnings. This proves the implemented stack can execute together; it **does not** prove representative real-video detail accuracy. The integrated stimulus itself demonstrated that a stage can report `succeeded` while a downstream module quality score remains low or zero.

## E12 — Representative Real-Video Acceptance [CURRENT]

### Goal

Determine what the system actually captures from representative user videos, not merely whether the pipeline completes.

### Required input classes

Representative videos should cover, over multiple samples where practical:

- realistic camera pan/tilt/translation and handheld motion;
- whole-body and limb motion;
- partial occlusion and re-entry;
- hair, loose clothing and other fine surface motion;
- facial and hand detail where E3.2 is enabled;
- lighting/exposure changes;
- multiple shots and shot boundaries;
- difficult/negative cases where evidence should fail closed.

### Primary tooling

- `scripts/run_pipeline_job.py` — run the production extraction path;
- `scripts/evaluate_real_video_e12.py` — generate the evidence-first machine acceptance report;
- `packages/pipeline_core/real_video_acceptance.py` — summarize stage status, module quality, character evidence, surface/micro-motion usability, rejection reasons, privacy and artifact integrity.

### Machine gate

The E12 report must distinguish:

- physical/structural failure (`machine_gate=failed`);
- structurally valid but quality-risk evidence (`machine_gate=needs_review`);
- machine evidence without detected structural/quality warnings (`machine_gate=passed`).

A machine pass is still **not final acceptance**.

### Manual evidence review

Final E12 acceptance remains `manual_frame_to_evidence_review_required`. Review source frames/video against extracted evidence for:

- timing and shot alignment;
- person-track continuity;
- pose alignment;
- mask boundaries around hair/clothing/hands/occluders;
- camera compensation versus true subject motion;
- body-local stability;
- surface residual motion;
- micro-motion quality gates and leakage/dropout;
- face/hand landmarks when enabled;
- environment photometry under exposure/lighting change.

### Acceptance principle

E12 must never declare detail-extraction success solely because the job completed, the schema validated, a physical sidecar exists, or a stage says `succeeded`. Accuracy claims require representative real-video evidence and frame-to-evidence review.
