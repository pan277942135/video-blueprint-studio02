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

**Important limitation discovered after E10:** the E10 Bundle smoke certification intentionally disabled E4 point/dense motion and E5-E8 motion stages. Therefore E10 proves assembly/integrity but does not by itself prove that every implemented stage can run together in one job.

## E11 — Integrated Production Acceptance [CURRENT]

### Goal

Close the integration gap before representative user-video acceptance.

### Required path

```text
real-human deterministic stimulus
  -> E1 media
  -> E2 people
  -> E3 pose
  -> E4 person masks + sparse point motion + dense flow
  -> E5 camera motion
  -> E6 body-local frame
  -> E7 residual surface motion
  -> E8 geometry-only micro-motion
  -> E9 environment photometry
  -> E10 schema + artifact integrity + verified bundle.zip
```

### Primary gate

- `scripts/certify_full_stack_e11.py`
- `.github/workflows/e11-integrated-acceptance.yml`

### Acceptance

For the same production job:

- all enabled E4-E9 stages report `succeeded`
- anonymous character evidence exists
- physical surface and micro-motion signal refs exist
- E9 environment evidence exists
- canonical schema validation passes
- every Blueprint artifact ref resolves safely and hashes correctly
- final Bundle is reopened and independently verified
- certified model hashes and privacy boundaries remain intact

## E12 — Representative real-video acceptance [NEXT, NOT STARTED]

Run the integrated production path on representative user-supplied videos covering realistic camera motion, body motion, occlusion, clothing/hair detail, lighting change, multiple shots and failure cases. Review machine-readable metrics **and** actual evidence outputs. Define accuracy/coverage regressions and acceptance thresholds from observed results.

E12 must not declare success merely because the job completes or the Bundle is structurally valid.
